"""Monte Carlo over hook and port tilts (+/-45 deg roll, pitch, yaw each,
uniform and independent), run in parallel.

  closed  full closed loop: rendered wrist camera, ArUco detection, planning,
          torque-level control and contact physics, scored against ground
          truth (default).
  kin     the program's planning steps on the TRUE poses (perfect
          perception): is the job kinematically feasible at all?

Results stream to <out>/<mode>.jsonl (one JSON object per trial), so a run
can be inspected while it is going and resumed with --resume.

Usage:
  python rover_task/monte_carlo.py --n 1000 --workers 10
  python rover_task/monte_carlo.py --mode kin --n 1000 --workers 10
  python rover_task/report.py          # tables + plots from the jsonl files
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def samples(n, seed, lim=45.0):
    rng = np.random.default_rng(seed)
    return [dict(i=i, hook=tuple(np.round(rng.uniform(-lim, lim, 3), 2)),
                 port=tuple(np.round(rng.uniform(-lim, lim, 3), 2)),
                 seed=int(seed * 100000 + i)) for i in range(n)]


def _work(args):
    s, mode, errors = args
    sys.path.insert(0, HERE)
    import warnings
    warnings.filterwarnings("ignore")
    from trial import kin_trial, run_trial
    t0 = time.time()
    try:
        if mode == "kin":
            r = kin_trial(s["hook"], s["port"], seed=s["seed"])
        else:
            r = run_trial(s["hook"], s["port"], errors=errors, seed=s["seed"])
    except Exception as e:                                  # noqa: BLE001
        r = dict(result="crash", feasible=None, reason=repr(e)[:300],
                 trace=traceback.format_exc()[-1500:],
                 hook_roll=s["hook"][0], hook_pitch=s["hook"][1],
                 hook_yaw=s["hook"][2], port_roll=s["port"][0],
                 port_pitch=s["port"][1], port_yaw=s["port"][2])
    r["i"] = s["i"]
    r["wall_s"] = time.time() - t0
    return {k: (float(v) if isinstance(v, (np.floating,)) else
                int(v) if isinstance(v, (np.integer,)) else
                bool(v) if isinstance(v, np.bool_) else v)
            for k, v in r.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("closed", "kin"), default="closed")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    ap.add_argument("--errors", choices=("realistic", "none"), default="realistic")
    ap.add_argument("--lim", type=float, default=45.0, help="tilt range +/- deg")
    ap.add_argument("--out", default=os.path.join(HERE, "out", "mc"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.mode}{args.tag}.jsonl")
    todo = samples(args.n, args.seed, args.lim)
    done = set()
    if args.resume and os.path.exists(path):
        with open(path) as f:
            done = {json.loads(line)["i"] for line in f if line.strip()}
    elif os.path.exists(path):
        os.remove(path)
    todo = [s for s in todo if s["i"] not in done]
    print(f"{args.mode}: {len(todo)} trials ({len(done)} already done) on "
          f"{args.workers} workers -> {path}", flush=True)
    t0 = time.time()
    counts = {}
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, maxtasksperchild=20) as pool, open(path, "a") as f:
        for k, r in enumerate(pool.imap_unordered(
                _work, [(s, args.mode, args.errors) for s in todo]), 1):
            f.write(json.dumps(r) + "\n")
            f.flush()
            key = r.get("result") if args.mode == "closed" else (
                "feasible" if r.get("feasible") else r.get("stage", "crash"))
            counts[key] = counts.get(key, 0) + 1
            if k % 10 == 0 or k == len(todo):
                el = time.time() - t0
                print(f"  {k}/{len(todo)}  {el / 60:.1f} min  "
                      f"eta {el / k * (len(todo) - k) / 60:.1f} min  {counts}",
                      flush=True)
    print("done", counts)


if __name__ == "__main__":
    main()
