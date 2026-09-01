"""Unit tests for the migrated Evans reward functions.

Covers CrossTrackHeadRewardEnv (CTH) and RacePerformanceRewardEnv, including the two
issues raised in the Codex review of the migration PR:
  - progress must undo BOTH finish-line wraps (a backward crossing must not score ~L of
    bogus progress),
  - overtakes must be computed from synchronized ABSOLUTE race positions seeded from the
    grid start, not from cumulative distance travelled starting at zero.

The reward logic is exercised directly against a lightweight fake env (no simulator / no
map), building the reward objects with ``__new__`` to bypass the heavy ``__init__`` that
would construct a real F110Env.
"""
import os
import sys
import types
import unittest

import numpy as np

from examples.multiagent.lib.rewards import (
    CrossTrackHeadRewardEnv,
    RacePerformanceRewardEnv,
)

sys.path.insert(0, os.path.dirname(__file__))


# --------------------------------------------------------------------------- helpers
class _FakeSpline:
    """calc_arclength_inaccurate returns the x coordinate as the arc-length, so tests
    control an agent's position along the track directly through poses_x."""

    def __init__(self, length):
        self.s = np.array([0.0, length])

    def calc_arclength_inaccurate(self, x, y, s_inds=None):
        return float(x), 0.0


def _fake_env(n, length=100.0):
    env = types.SimpleNamespace()
    env.num_agents = n
    env.config = {"timestep": 1.0}
    env.poses_x = np.zeros(n)
    env.poses_y = np.zeros(n)
    env.poses_theta = np.zeros(n)
    env.lap_counts = np.zeros(n)
    env.track = types.SimpleNamespace(
        centerline=types.SimpleNamespace(spline=_FakeSpline(length))
    )
    return env


def _make_race(n=1, length=100.0, timestep=1.0):
    """Build a RacePerformanceRewardEnv without running the real __init__, then mimic
    what reset() sets up so _compute_reward can be called deterministically."""
    r = RacePerformanceRewardEnv.__new__(RacePerformanceRewardEnv)
    r.env = _fake_env(n, length)
    r._timestep = timestep
    r.agents = [f"agent_{k}" for k in range(n)]
    return r


def _race_reset(r, xs):
    """Replicate RacePerformanceRewardEnv.reset() bookkeeping from a set of start
    positions (arc-lengths) without touching the real env."""
    n = r.env.num_agents
    r.env.poses_x = np.array(xs, dtype=float)
    r.env.poses_y = np.zeros(n)
    r.env.lap_counts = np.zeros(n)
    r._crashed_agents = set()
    r._last_action = {a: np.zeros(2, dtype=np.float32) for a in r.agents}
    r._cur_action = {a: np.zeros(2, dtype=np.float32) for a in r.agents}
    r._prev_s = [r._arclen(i) for i in range(n)]
    r._prev_xy = [(float(r.env.poses_x[i]), float(r.env.poses_y[i])) for i in range(n)]
    L0 = float(r.env.track.centerline.spline.s[-1])
    pos0 = [r._abs_pos(k, L0) for k in range(n)]
    r._prev_ahead = [[pos0[a] > pos0[b] for b in range(n)] for a in range(n)]


def _make_cth(waypoints, yaws):
    r = CrossTrackHeadRewardEnv.__new__(CrossTrackHeadRewardEnv)
    r._wp = np.asarray(waypoints, dtype=np.float32)
    r._wp_yaw = np.asarray(yaws, dtype=np.float32)
    r._crashed_agents = set()
    r.agents = ["agent_0"]
    r.env = _fake_env(1)
    return r


# --------------------------------------------------------------------- RacePerformance
class TestRacePerformanceReward(unittest.TestCase):
    def test_collision_is_large_terminal_penalty(self):
        r = _make_race(n=1)
        _race_reset(r, [10.0])
        out = r._compute_reward("agent_0", {"agent_0"}, 0)
        self.assertEqual(out, -RacePerformanceRewardEnv.COLLISION_PENALTY)

    def test_stall_penalty_when_stationary(self):
        r = _make_race(n=1, timestep=1e6)  # huge dt -> measured speed ~0 -> stall
        _race_reset(r, [10.0])
        # No movement: arc-length and position unchanged.
        out = r._compute_reward("agent_0", set(), 0)
        # progress 0, speed<MIN -> -STALL, overtake 0, jerk 0
        self.assertAlmostEqual(out, -RacePerformanceRewardEnv.STALL_PENALTY, places=5)

    def test_forward_motion_is_rewarded(self):
        r = _make_race(n=1, timestep=1.0)
        _race_reset(r, [10.0])
        r.env.poses_x = np.array([13.0])  # moved +3 along track
        out = r._compute_reward("agent_0", set(), 0)
        # progress 15*3=45 ; speed 5*(3-1)=10 ; jerk 0 -> 55
        self.assertAlmostEqual(out, 55.0, places=5)

    def test_backward_finish_line_wrap_scores_no_bogus_progress(self):
        """FIX (Codex P1): crossing the line backward (s ~0 -> ~L) must not be counted
        as ~a full lap of forward progress."""
        r = _make_race(n=1, length=100.0, timestep=1e6)
        _race_reset(r, [1.0])            # start just past the line
        r.env.poses_x = np.array([99.0])  # arc-length jumps 1 -> 99 (backward crossing)
        out = r._compute_reward("agent_0", set(), 0)
        # d = 99-1 = 98 > 0.5L -> d -= 100 -> -2 -> clamp 0. progress 0, speed~0 -> -STALL
        self.assertAlmostEqual(out, -RacePerformanceRewardEnv.STALL_PENALTY, places=5)

    def test_forward_finish_line_wrap_scores_small_progress(self):
        r = _make_race(n=1, length=100.0, timestep=1e6)
        _race_reset(r, [99.0])            # just before the line
        r.env.poses_x = np.array([1.0])   # crossed forward: 99 -> 1
        out = r._compute_reward("agent_0", set(), 0)
        # d = 1-99 = -98 < -0.5L -> d += 100 -> 2. progress 15*2=30, speed~0 -> -STALL
        self.assertAlmostEqual(out, 30.0 - RacePerformanceRewardEnv.STALL_PENALTY, places=5)

    def test_no_overtake_bonus_for_moving_while_still_behind(self):
        """FIX (Codex P1): an agent that has not passed anyone gets no bonus, even on its
        first move (old code seeded cumulative distance at 0 and fired spuriously)."""
        r = _make_race(n=2, length=100.0, timestep=1.0)
        _race_reset(r, [10.0, 20.0])       # agent_0 starts behind agent_1
        r.env.poses_x = np.array([12.0, 20.0])  # agent_0 edges forward, still behind
        out = r._compute_reward("agent_0", set(), 0)
        # progress 15*2=30, speed 5*(2-1)=5, overtake 0, jerk 0 -> 35
        self.assertAlmostEqual(out, 35.0, places=5)

    def test_leading_agent_gets_no_bonus_for_moving(self):
        """FIX (Codex P1): the agent that starts ahead must not be credited an overtake
        just for moving (seed the ahead-matrix from grid positions)."""
        r = _make_race(n=2, length=100.0, timestep=1.0)
        _race_reset(r, [10.0, 20.0])
        r.env.poses_x = np.array([10.0, 22.0])  # agent_1 (already ahead) moves
        out = r._compute_reward("agent_1", set(), 1)
        # progress 30, speed 5, overtake 0 (was already ahead) -> 35
        self.assertAlmostEqual(out, 35.0, places=5)

    def test_overtake_bonus_fires_once_on_a_real_pass(self):
        r = _make_race(n=2, length=100.0, timestep=1.0)
        _race_reset(r, [10.0, 20.0])            # agent_0 behind
        r.env.poses_x = np.array([25.0, 20.0])  # agent_0 passes agent_1
        first = r._compute_reward("agent_0", set(), 0)
        # progress 15*15=225, speed 5*(15-1)=70, overtake +50 -> 345
        self.assertAlmostEqual(first, 345.0, places=5)
        # Next step still ahead -> no repeated bonus.
        r.env.poses_x = np.array([30.0, 20.0])
        second = r._compute_reward("agent_0", set(), 0)
        # progress 15*5=75, speed 5*(5-1)=20, overtake 0 -> 95
        self.assertAlmostEqual(second, 95.0, places=5)

    def test_jerk_penalizes_action_change(self):
        r = _make_race(n=1, timestep=1e6)
        _race_reset(r, [10.0])
        r._cur_action = {"agent_0": np.array([0.4, 2.0], dtype=np.float32)}
        # stationary so only stall + jerk contribute
        out = r._compute_reward("agent_0", set(), 0)
        expected_jerk = -RacePerformanceRewardEnv.JERK_PENALTY * (abs(0.4) + abs(2.0)) / 2.0
        self.assertAlmostEqual(
            out, -RacePerformanceRewardEnv.STALL_PENALTY + expected_jerk, places=4)


# ------------------------------------------------------------------------------- CTH
class TestCrossTrackHeadReward(unittest.TestCase):
    def setUp(self):
        # Straight reference line along +x, heading 0 everywhere.
        self.wp = [[float(k), 0.0] for k in range(11)]
        self.yaws = [0.0] * 11

    def _cth(self):
        return _make_cth(self.wp, self.yaws)

    def test_collision_is_large_terminal_penalty(self):
        r = self._cth()
        out = r._compute_reward("agent_0", {"agent_0"}, 0)
        self.assertEqual(out, -CrossTrackHeadRewardEnv.COLLISION_PENALTY)

    def test_on_line_and_aligned_is_zero(self):
        r = self._cth()
        r.env.poses_x = np.array([5.0]); r.env.poses_y = np.array([0.0]); r.env.poses_theta = np.array([0.0])
        self.assertAlmostEqual(r._compute_reward("agent_0", set(), 0), 0.0, places=4)

    def test_cross_track_error_penalized(self):
        r = self._cth()
        r.env.poses_x = np.array([5.0]); r.env.poses_y = np.array([1.0]); r.env.poses_theta = np.array([0.0])
        # cte = 1.0 (< max_cte 2.0) -> -3.0*1.0
        self.assertAlmostEqual(r._compute_reward("agent_0", set(), 0), -3.0, places=4)

    def test_cross_track_error_clamped(self):
        r = self._cth()
        r.env.poses_x = np.array([5.0]); r.env.poses_y = np.array([5.0]); r.env.poses_theta = np.array([0.0])
        # cte = 5.0 clamped to max_cte 2.0 -> -3.0*2.0
        self.assertAlmostEqual(r._compute_reward("agent_0", set(), 0), -6.0, places=4)

    def test_heading_error_penalized(self):
        r = self._cth()
        r.env.poses_x = np.array([5.0]); r.env.poses_y = np.array([0.0]); r.env.poses_theta = np.array([0.5])
        # he = 0.5 (< max_he 0.7854) -> -2.0*0.5
        self.assertAlmostEqual(r._compute_reward("agent_0", set(), 0), -1.0, places=4)

    def test_heading_error_clamped_and_wrapped(self):
        r = self._cth()
        r.env.poses_x = np.array([5.0]); r.env.poses_y = np.array([0.0]); r.env.poses_theta = np.array([1.5])
        # |he| = 1.5 clamped to max_he 0.7854 -> -2.0*0.7854
        self.assertAlmostEqual(r._compute_reward("agent_0", set(), 0),
                               -2.0 * CrossTrackHeadRewardEnv.MAX_HE, places=4)


if __name__ == "__main__":
    unittest.main()
