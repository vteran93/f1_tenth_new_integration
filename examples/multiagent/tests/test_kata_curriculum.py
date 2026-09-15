"""Tests for the kata track generator and the curriculum wrapper plumbing."""
import pathlib
import sys
import warnings

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "training_tracks"))
warnings.filterwarnings("ignore")

import kata_trackgen as kt  # noqa: E402

MANIFEST = REPO / "maps" / "kata_curriculum.yaml"


@pytest.mark.parametrize("stage_id", [1, 3, 7, 10])
def test_generator_writes_loadable_track(tmp_path, stage_id):
    st = next(s for s in kt.STAGES if s.stage == stage_id)
    entry = kt.generate_track(st, seed=7, name="t", maps_dir=tmp_path)
    d = tmp_path / "t"
    assert (d / "t_map.png").exists() and (d / "t_map.yaml").exists() and (d / "t_centerline.csv").exists()
    assert st.length[0] * 0.9 <= entry["length_m"] <= st.length[1] * 1.1
    assert st.width[0] - 1e-6 <= entry["width_m"] <= min(st.width[1], 4.4) + 1e-6
    # hairpin stages may go below the stage floor only at the tip (width/2 + 0.4)
    assert entry["min_radius_m"] >= entry["width_m"] / 2 + 0.39
    # centerline must lie on drivable pixels
    from PIL import Image
    img = np.array(Image.open(d / "t_map.png").transpose(Image.FLIP_TOP_BOTTOM))
    spec = kt.yaml.safe_load(open(d / "t_map.yaml"))
    wps = np.loadtxt(d / "t_centerline.csv", delimiter=",")
    cols = ((wps[:, 0] - spec["origin"][0]) / spec["resolution"]).astype(int)
    rows = ((wps[:, 1] - spec["origin"][1]) / spec["resolution"]).astype(int)
    assert (img[rows, cols] > 128).all()


def test_fillet_rejects_below_floor():
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    with pytest.raises(kt.FilletError):
        kt.fillet_polygon(square, np.full(4, 8.0), r_floor=6.0)  # only r=5 fits on 10 m edges
    pts, straights, r = kt.fillet_polygon(square, np.full(4, 5.0), r_floor=4.0)
    assert abs(r - 5.0) < 1e-6 and len(straights) == 4


@pytest.mark.skipif(not MANIFEST.exists(), reason="generate maps first: python training_tracks/kata_trackgen.py")
def test_curriculum_env_stage_sampling_and_progress():
    from examples.multiagent.lib.rewards import ProgressTimePenaltyEnv

    env_config = dict(
        map="kata_01_taikyoku_shodan_01", num_agents=2, timestep=0.01, num_beams=36, integrator="rk4",
        control_input=["speed", "steering_angle"], observation_config={"type": "original"},
        reset_config={"type": "cl_grid_static"}, action_repeat=5, min_speed=0.0,
        episode_timeout={"steps": 30},
        reward_params={"progress_scale": 1.0, "time_penalty": 0.1, "crash_penalty": 20.0},
        curriculum={"manifest": str(MANIFEST), "stage": 0, "mix_prev": 0.0},
    )
    env = ProgressTimePenaltyEnv(env_config=env_config)
    assert env.num_stages == 10
    env.set_stage(6)
    env.reset()
    assert env.current_track.startswith("kata_07_") and env._effective_max_speed() == 8.0

    # windowed arclength follows the centerline, wraps, and never counts backward motion
    sp = env.env.track.centerline.spline
    L = env.track_length
    s_prev, cum = 0.0, 0.0
    for k in range(int(1.1 * L / 0.3)):
        x, y = sp.calc_position((k * 0.3) % L)
        s_w = env.arclength(x, y, s_prev)
        cum += env.progress_delta(s_w, s_prev)
        s_prev = s_w
    assert abs(cum - 1.1 * L) < 1.0
    assert env.progress_delta(5.0, 9.0) == 0.0            # backward -> 0
    assert env.progress_delta(0.2, L - 0.3) == pytest.approx(0.5, abs=1e-6)  # wrap forward
    assert env.progress_delta(50.0, 10.0) == env.MAX_STEP_PROGRESS  # projection jump is bounded

    # speed clip + timeout truncation
    obs, _ = env.reset()
    done, steps = False, 0
    while not done:
        obs, rew, term, trunc, _ = env.step({a: np.array([0.0, 20.0], np.float32) for a in obs})
        steps += 1
        done = term["__all__"] or trunc["__all__"]
    assert steps <= 30
    assert env.env.sim.agents[0].state[3] <= 8.0 + 1e-3


@pytest.mark.skipif(not MANIFEST.exists(), reason="generate maps first")
def test_speed_action_range_rescales_action_space():
    from examples.multiagent.lib.rewards import ProgressTimePenaltyEnv

    env_config = dict(
        map="kata_01_taikyoku_shodan_01", num_agents=2, timestep=0.01, num_beams=36, integrator="rk4",
        control_input=["speed", "steering_angle"], observation_config={"type": "original"},
        reset_config={"type": "cl_grid_static"}, action_repeat=5, speed_action_range=[0.5, 8.0],
        max_speed=5.0, episode_timeout={"steps": 20},
    )
    env = ProgressTimePenaltyEnv(env_config=env_config)
    assert env.action_space.low[1] == pytest.approx(0.5) and env.action_space.high[1] == pytest.approx(8.0)
    assert env.action_space.low[0] == pytest.approx(-0.4189, abs=1e-3)  # steering untouched
    obs, _ = env.reset()
    for _ in range(20):
        obs, _, term, trunc, _ = env.step({a: env.action_space.high for a in obs})
        if term["__all__"] or trunc["__all__"]:
            break
    assert env.env.sim.agents[0].state[3] <= 5.0 + 1e-3  # stage/max_speed cap still applies


@pytest.mark.skipif(not MANIFEST.exists(), reason="generate maps first")
def test_kohonda_reward_follows_track_switches():
    from examples.multiagent.lib.rewards import KohondaMultiAgentF110Env

    env_config = dict(
        map="kata_01_taikyoku_shodan_01", num_agents=2, timestep=0.01, num_beams=36, integrator="rk4",
        control_input=["speed", "steering_angle"], observation_config={"type": "original"},
        reset_config={"type": "cl_grid_static"}, action_repeat=5, speed_action_range=[0.5, 8.0],
        episode_timeout={"steps": 40},
        curriculum={"manifest": str(MANIFEST), "stage": 9, "mix_prev": 0.0},
    )
    env = KohondaMultiAgentF110Env(env_config=env_config)
    for _ in range(3):
        obs, _ = env.reset()
        # waypoint table must belong to the current track
        assert len(env._waypoints) == len(env.env.track.raceline.xs)
        first = None
        for _ in range(40):
            obs, rew, term, trunc, _ = env.step({a: np.array([0.0, 2.0], np.float32) for a in obs})
            if first is None:
                first = rew
            assert all(np.isfinite(list(rew.values())))
            assert all(r <= env.MAX_STEP_PROGRESS + 1e-6 for r in rew.values())
            if term["__all__"] or trunc["__all__"]:
                break
        # no spurious jump on the first step (2 m/s * 0.05 s = 0.1 m of progress at most ~0.3 m)
        assert all(r < 0.5 for r in first.values())
