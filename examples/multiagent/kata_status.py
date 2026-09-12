#!/usr/bin/env python
"""Print a one-screen status of the kata-curriculum experiments from their result.json.

Usage: python kata_status.py [storage_dir=models_kata] [experiment_name ...]
"""
import glob
import json
import os
import sys


def get(d, *ks, default=None):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def f(v, nd=2):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "-"


def report(result_json):
    rows = [json.loads(l) for l in open(result_json) if l.strip()]
    if not rows:
        print("   (empty result.json)")
        return
    r = rows[-1]
    er = r.get("env_runners", {}) or {}
    cm = er.get("custom_metrics", {}) or {}
    print(f"   iter={r.get('training_iteration')} steps={r.get('timesteps_total')} "
          f"time={f(r.get('time_total_s'), 0)}s stage={r.get('curriculum/stage')} "
          f"stage_steps={r.get('curriculum/stage_steps')} streak={r.get('curriculum/success_streak')}")
    print(f"   train: return={f(er.get('episode_return_mean'))} len={f(er.get('episode_len_mean'), 0)} "
          f"lap_success={f(cm.get('lap_success_mean'))} laps={f(cm.get('laps_completed_mean'))} "
          f"speed={f(cm.get('avg_speed_mean'))} m/s timed_out={f(cm.get('episode_timed_out_mean'))}")
    ev = [x for x in rows if get(x, "evaluation", "env_runners", "episode_return_mean") is not None]
    if ev:
        e = ev[-1]
        eer = e["evaluation"]["env_runners"]
        ecm = eer.get("custom_metrics", {}) or {}
        print(f"   eval (iter {e['training_iteration']}, held-out tracks): return={f(eer.get('episode_return_mean'))} "
              f"lap_success={f(ecm.get('lap_success_mean'))} laps={f(ecm.get('laps_completed_mean'))} "
              f"speed={f(ecm.get('avg_speed_mean'))} m/s")
    stages = [(x.get("training_iteration"), x.get("curriculum/stage")) for x in rows if "curriculum/stage" in x]
    changes = [(it, st) for i, (it, st) in enumerate(stages) if i == 0 or st != stages[i - 1][1]]
    if changes:
        print("   stage history (iteration -> stage): " + ", ".join(f"{it}->{int(st)}" for it, st in changes))


def main():
    storage = sys.argv[1] if len(sys.argv) > 1 else "models_kata"
    names = sys.argv[2:] or sorted(os.listdir(storage)) if os.path.isdir(storage) else []
    for name in names:
        print(f"== {name}")
        files = sorted(glob.glob(os.path.join(storage, name, "*", "result.json")), key=os.path.getmtime)
        if not files:
            print("   (no result.json yet)")
            continue
        report(files[-1])


if __name__ == "__main__":
    main()
