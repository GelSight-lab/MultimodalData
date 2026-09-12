"""Apply `policy.decide` to the running machine, once every sample period.

Two pools, moved only with SIGSTOP/SIGCONT so nothing in flight is ever lost:

    builds          disk-bound; identified by the literal `build` subcommand,
                    never by a substring. An earlier controller matched
                    `react_preprocess` plus `--task`, which also matches
                    `curate --task`, and froze a running curate for fifty
                    minutes.
    force workers   CPU-bound; they claim jobs with `mkdir`, so suspending one
                    holds its claim and resuming it continues the same job.

Stages on the critical path -- curate, the cut, the export, the upload -- are
never touched. They are short, serial, and stopping one buys nothing.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/yxma/MultimodalData")
from twm.sched.policy import Decision, Limits, State, decide  # noqa: E402
from twm.sched import publisher  # noqa: E402

FORCE_LOG = Path("/tmp/force_logs")
LOG = Path("/tmp/sched.log")
DATA = "/media/yxma/Disk1/twm/data"


def _cmdline(pid: str) -> list[str]:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")[:-1]
    except OSError:
        return []


def _comm(pid: str) -> str:
    try:
        return Path(f"/proc/{pid}/comm").read_text().strip()
    except OSError:
        return ""


def _state(pid: str) -> str:
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2]
    except (OSError, IndexError):
        return ""


def _pids():
    return [p.name for p in Path("/proc").iterdir() if p.name.isdigit()]


def builds() -> list[tuple[int, str]]:
    """(raw H5 size, pid) for every build. Size orders resume/pause."""
    out = []
    for pid in _pids():
        if not _comm(pid).startswith("python"):
            continue
        a = _cmdline(pid)
        i = next((k for k, v in enumerate(a) if v.endswith("react_preprocess")), None)
        if i is None or i + 1 >= len(a) or a[i + 1] != "build":
            continue
        def arg(flag):
            return a[a.index(flag) + 1] if flag in a else None
        t, d, e = arg("--task"), arg("--date"), arg("--episodes")
        sz = 0
        if t and d and e:
            try:
                sz = os.path.getsize(f"{DATA}/{t}/{d}/{e}.h5")
            except OSError:
                sz = 0
        out.append((sz, pid))
    return out


def force_workers() -> list[str]:
    return [p for p in _pids()
            if _comm(p).startswith("python") and "process_side" in " ".join(_cmdline(p))]


def backlog() -> int:
    """Claimed-but-unfinished plus unclaimed force jobs."""
    n = 0
    for wl in FORCE_LOG.glob("worklist*.txt"):
        for line in wl.read_text().splitlines():
            if not line.strip():
                continue
            t, d, e, s = line.split()
            if not Path(f"/media/yxma/Disk1/twm/force_recovery/{t}/{d}/{e}_{s}.npz").exists():
                n += 1
    return n


def sample(seconds: int = 20) -> tuple[int, int]:
    """(idle %, read MB/s) averaged over `seconds`, discarding vmstat's first
    line — it reports averages since boot, not the current rate."""
    out = subprocess.run(["vmstat", "5", str(max(2, seconds // 5))],
                         capture_output=True, text=True).stdout.splitlines()
    rows = [r.split() for r in out[3:] if r.strip()]
    if not rows:
        return 0, 0
    idle = sum(int(r[14]) for r in rows) / len(rows)
    bi = sum(int(r[8]) for r in rows) / len(rows) / 1000
    return int(idle), int(bi)


def main() -> int:
    lim = Limits(disk_max=int(os.environ.get("DISK_MAX", 6)),
                 cpu_max=int(os.environ.get("CPU_MAX", 6)))
    best = 0
    probing = False
    probe_pid = None
    cooldown = 0
    pub_proc: subprocess.Popen | None = None
    pub_seen = 0.0
    while True:
        idle, bi = sample()
        b = builds()
        running = sorted([x for x in b if _state(x[1]) != "T"])
        paused = sorted([x for x in b if _state(x[1]) == "T"])
        fw = force_workers()
        fw_run = [p for p in fw if _state(p) != "T"]
        fw_stop = [p for p in fw if _state(p) == "T"]
        if bi > best:
            best = bi
        if pub_proc is not None and pub_proc.poll() is not None:
            pub_proc = None
        # Asking the Hub what exists costs a round trip, so only between
        # publishes and at most every couple of minutes.
        if pub_proc is None and time.time() - pub_seen > 120:
            try:
                pending = sum(len(v) for v in publisher.ready().values())
            except Exception:                                    # noqa: BLE001
                pending = 0
            pub_seen = time.time()
        else:
            pending = 0
        st = State(idle_pct=idle, read_mbs=bi, best_read_mbs=best,
                   disk_running=len(running), disk_paused=len(paused),
                   cpu_workers=len(fw_run), cpu_backlog=backlog(),
                   probing_disk=probing, cooldown=cooldown,
                   publish_pending=pending,
                   publisher_running=pub_proc is not None)
        d: Decision = decide(st, lim)
        act = []

        if d.publish and pub_proc is None:
            # `nice`: verification decodes video, and it must not take cores
            # from the cut, which is on the critical path.
            pub_proc = subprocess.Popen(
                ["nice", "-n", "10", sys.executable, "-u",
                 "/home/yxma/MultimodalData/twm/sched/publisher.py"],
                stdout=open("/tmp/publisher.log", "a"), stderr=subprocess.STDOUT)
            act.append(f"启动发布器 pid{pub_proc.pid}（{pending} 段）")

        if d.pause_disk and probe_pid:
            os.kill(int(probe_pid), 19); act.append(f"暂停探测 pid{probe_pid}")
            probe_pid = None
        elif d.pause_disk and running:
            os.kill(int(running[0][1]), 19); act.append(f"暂停最小 pid{running[0][1]}")
        if d.resume_disk and paused:
            probe_pid = paused[-1][1]          # biggest first
            os.kill(int(probe_pid), 18); act.append(f"恢复 pid{probe_pid}")
        if d.add_cpu and fw_stop:
            os.kill(int(fw_stop[0]), 18); act.append(f"恢复 force pid{fw_stop[0]}")
        elif d.drop_cpu and fw_run:
            os.kill(int(fw_run[-1]), 19); act.append(f"暂停 force pid{fw_run[-1]}")

        probing = d.probing
        cooldown = d.cooldown if d.cooldown else max(0, cooldown - 1)
        with LOG.open("a") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] idle={idle}% 读={bi}MB/s(最佳{best}) "
                    f"构建 {len(running)}跑/{len(paused)}停  力估计 {len(fw_run)}跑/{len(fw_stop)}停 "
                    f"待办{st.cpu_backlog} 待发布{pending}  {d.why}"
                    + (f"  → {', '.join(act)}" if act else "") + "\n")
        if not b and not fw:
            with LOG.open("a") as f:
                f.write(f"[{time.strftime('%H:%M:%S')}] 无可调度任务，退出\n")
            return 0
        time.sleep(5)


if __name__ == "__main__":
    raise SystemExit(main())
