#!/usr/bin/env python3
import argparse, glob, os, random, subprocess, time
import pynvml

DISALLOWED_NAME_BITS = ["T1000"]
FORBIDDEN_CMD_BITS   = ["gdesmond", "icm64.bin"]

def read_cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline","rb") as f:
            return f.read().replace(b"\x00", b" ").decode(errors="ignore").lower()
    except Exception:
        return ""

def name_is_disallowed(name: str) -> bool:
    nl = name.lower()
    return any(bit.lower() in nl for bit in DISALLOWED_NAME_BITS)

def handle_has_forbidden_jobs(handle) -> bool:
    try:
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses_v3(handle)
    except Exception:
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    for p in procs:
        pid = getattr(p, "pid", None)
        if pid is None: 
            continue
        cmd = read_cmdline(pid)
        if any(cmd.endswith(bad) or bad in cmd for bad in FORBIDDEN_CMD_BITS):
            return True
    return False

def count_allowed_devices() -> int:
    pynvml.nvmlInit()
    try:
        mig_ok = all(hasattr(pynvml, fn) for fn in [
            "nvmlDeviceGetMigMode",
            "nvmlDeviceGetMaxMigDeviceCount",
            "nvmlDeviceGetMigDeviceHandleByIndex",
        ])
        count = 0
        for i in range(pynvml.nvmlDeviceGetCount()):
            gpu = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(gpu)
            if isinstance(name, bytes): name = name.decode()
            if name_is_disallowed(name):
                continue
            mode = 0
            if mig_ok:
                try: mode, _ = pynvml.nvmlDeviceGetMigMode(gpu)
                except Exception: mode = 0
            if mode == 0:
                if not handle_has_forbidden_jobs(gpu):
                    count += 1
            else:
                try: max_migs = pynvml.nvmlDeviceGetMaxMigDeviceCount(gpu)
                except Exception: max_migs = 0
                for mi in range(max_migs):
                    try: mig = pynvml.nvmlDeviceGetMigDeviceHandleByIndex(gpu, mi)
                    except Exception: continue
                    if not handle_has_forbidden_jobs(mig):
                        count += 1
        return count
    finally:
        try: pynvml.nvmlShutdown()
        except Exception: pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs-dir", required=True, help="Directory containing simple_job_*.sh")
    ap.add_argument("--max-parallel", type=int, default=0, help="Override auto device count")
    ap.add_argument("--jitter-min", type=float, default=0.5)
    ap.add_argument("--jitter-max", type=float, default=2.0)
    args = ap.parse_args()

    jobs = sorted(glob.glob(os.path.join(args.jobs_dir, "simple_job_*.sh")))
    if not jobs:
        print(f"No jobs found in {args.jobs_dir}")
        return

    for j in jobs:
        # ensure executable
        try:
            os.chmod(j, 0o755)
        except Exception:
            pass

    max_parallel = args.max_parallel or count_allowed_devices()
    if max_parallel <= 0:
        raise SystemExit("No allowed GPU/MIG devices available right now.")
    print(f"Running up to {max_parallel} jobs in parallel ({len(jobs)} total).")

    procs = []
    idx = 0
    while idx < len(jobs) or procs:
        # start new jobs while we have capacity
        while idx < len(jobs) and len(procs) < max_parallel:
            j = jobs[idx]
            log = os.path.join(args.jobs_dir, os.path.basename(j) + ".log")
            # small jitter to reduce simultaneous device picks
            time.sleep(random.uniform(args.jitter_min, args.jitter_max))
            print(f"Starting {j} → {log}")
            f = open(log, "w")
            p = subprocess.Popen([j], stdout=f, stderr=subprocess.STDOUT, close_fds=True)
            procs.append((p, f))
            idx += 1
        # reap finished
        still = []
        for p, f in procs:
            rc = p.poll()
            if rc is None:
                still.append((p, f))
            else:
                f.close()
                print(f"Finished pid {p.pid} (rc={rc})")
        procs = still
        if procs:
            time.sleep(0.5)

    print("All prediction jobs finished.")

if __name__ == "__main__":
    main()
