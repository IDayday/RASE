#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


def parse_gpus(text: str) -> list[str]:
    return [x.strip() for x in str(text).replace(",", " ").split() if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Small GPU job queue for independent RASE experiments.")
    parser.add_argument("--job_file", required=True)
    parser.add_argument("--gpu_ids", default=os.environ.get("RASE_GPU_IDS", "0 1 2 3 4 5 6"))
    parser.add_argument("--log_dir", default="logs/certificate_sprint")
    parser.add_argument("--prefix", default="job")
    parser.add_argument("--poll_seconds", type=float, default=5.0)
    parser.add_argument("--stop_on_failure", action="store_true")
    args = parser.parse_args()

    gpus = parse_gpus(args.gpu_ids)
    if not gpus:
        raise SystemExit("No GPUs specified. Set --gpu_ids or RASE_GPU_IDS.")
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(args.job_file, "r", encoding="utf-8") as f:
        jobs = [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]
    if not jobs:
        print("No jobs to run.")
        return 0

    print(f"[queue] jobs={len(jobs)} gpus={gpus} log_dir={log_dir}")
    pending = list(enumerate(jobs))
    running: dict[int, tuple[subprocess.Popen, str, Path, str]] = {}
    done = []
    failed = []
    next_gpu = 0

    while pending or running:
        while pending and len(running) < len(gpus):
            job_id, cmd = pending.pop(0)
            gpu = gpus[next_gpu % len(gpus)]
            next_gpu += 1
            log_path = log_dir / f"{args.prefix}_{job_id:03d}_gpu{gpu}.log"
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            # Make single-visible-GPU jobs consistently use cuda:0.
            env.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
            env.setdefault("MUJOCO_GL", "osmesa")
            env.setdefault("PYOPENGL_PLATFORM", "osmesa")
            print(f"[launch] id={job_id} gpu={gpu} log={log_path} cmd={cmd}")
            lf = open(log_path, "w", encoding="utf-8")
            proc = subprocess.Popen(["bash", "-lc", cmd], stdout=lf, stderr=subprocess.STDOUT, env=env)
            # Keep file object alive by attaching it; closed when proc exits.
            proc._rase_log_file = lf  # type: ignore[attr-defined]
            running[job_id] = (proc, cmd, log_path, gpu)

        time.sleep(float(args.poll_seconds))
        for job_id, (proc, cmd, log_path, gpu) in list(running.items()):
            ret = proc.poll()
            if ret is None:
                continue
            try:
                proc._rase_log_file.close()  # type: ignore[attr-defined]
            except Exception:
                pass
            running.pop(job_id)
            if ret == 0:
                print(f"[done] id={job_id} gpu={gpu} log={log_path}")
                done.append(job_id)
            else:
                print(f"[fail] id={job_id} gpu={gpu} ret={ret} log={log_path}")
                failed.append((job_id, ret, str(log_path), cmd))
                if args.stop_on_failure:
                    print("[queue] stop_on_failure: waiting for running jobs to finish, then exiting.")
                    pending.clear()

    print(f"[queue] complete: done={len(done)} failed={len(failed)}")
    if failed:
        print("[queue] failures:")
        for job_id, ret, log_path, cmd in failed:
            print(f"  id={job_id} ret={ret} log={log_path} cmd={cmd}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
