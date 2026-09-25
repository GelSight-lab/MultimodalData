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
from twm.pipeline_stages import TASKS  # one list; copies are how a task falls out

FORCE_LOG = Path("/tmp/force_logs")
FORCE_POOL = ("/tmp/claude-1004/-home-yxma-MultimodalData/"
              "d734563d-9427-48c6-a0e9-fe7c75ba0ddf/scratchpad/pub/force_pool.sh")
MAKE_WORKLIST = ("/tmp/claude-1004/-home-yxma-MultimodalData/"
                 "d734563d-9427-48c6-a0e9-fe7c75ba0ddf/scratchpad/pub/make_worklist.py")
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
    """Force jobs that exist, computed from the release tree itself.

    Not from the worklist file. That file is refreshed by the advance loop,
    which spends most of its cycle cutting -- so between refreshes the
    scheduler saw a stale list, reported backlog 0, and would not start a
    worker while real work waited. Three episodes sat that way the moment they
    finished building: motherboard/episode_007 and rope/episode_004 and _010.

    An episode counts when it is fully built, its source recording still
    exists, and a side has no npz. Same definition make_worklist writes out,
    so the two cannot disagree.
    """
    n = 0
    for task in TASKS:
        rel = Path(f"/media/yxma/Disk1/twm/release/{task}")
        for pq_path in rel.glob("meta/2026-09-*/episode_*.parquet"):
            date, ep = pq_path.parent.name, pq_path.stem
            if len(list((rel / "videos" / date / ep).glob("*.mp4"))) != 7:
                continue
            if not Path(f"/media/yxma/Disk1/twm/data/{task}/{date}/{ep}.h5").exists():
                continue
            for side in ("left", "right"):
                if not Path(f"/media/yxma/Disk1/twm/force_recovery/{task}/"
                            f"{date}/{ep}_{side}.npz").exists():
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
    quiet = 0
    while True:
        idle, bi = sample()
        b = builds()
        running = sorted([x for x in b if _state(x[1]) != "T"])
        paused = sorted([x for x in b if _state(x[1]) == "T"])
        fw = force_workers()
        fw_run = [p for p in fw if _state(p) != "T"]
        fw_stop = [p for p in fw if _state(p) == "T"]
        if not probing:
            best = bi            # the baseline this probe will be judged against
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
                   publisher_running=pub_proc is not None,
                   cpu_suspended=len(fw_stop))
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
        elif d.add_cpu:
            # Refresh the list the worker will read. The scheduler decides to
            # spawn from ground truth, so it must not hand the worker a list
            # that predates the episodes it just counted.
            subprocess.run([sys.executable, MAKE_WORKLIST],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # Nothing suspended to resume, so START one. The controller could
            # previously only resume, and the force pool exits once its queue
            # drains -- so when a later build produced new force jobs there was
            # no worker left to take them and the stage stalled silently,
            # logging "worker below floor" every tick with nothing changing.
            # force_pool workers re-read the worklist and retire themselves
            # after three idle rounds, so spawning one is self-limiting.
            spawned = subprocess.Popen(
                ["setsid", "bash", FORCE_POOL], env={**os.environ, "N": "1"},
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True)
            act.append(f"新建 force worker pid{spawned.pid}")
        elif d.drop_cpu and fw_run:
            os.kill(int(fw_run[-1]), 19); act.append(f"暂停 force pid{fw_run[-1]}")

        probing = d.probing
        cooldown = d.cooldown if d.cooldown else max(0, cooldown - 1)
        with LOG.open("a") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] idle={idle}% 读={bi}MB/s(最佳{best}) "
                    f"构建 {len(running)}跑/{len(paused)}停  力估计 {len(fw_run)}跑/{len(fw_stop)}停 "
                    f"待办{st.cpu_backlog} 待发布{pending}  {d.why}"
                    + (f"  → {', '.join(act)}" if act else "") + "\n")
        # Exit only when there is genuinely nothing left, judged AFTER this
        # tick's actions and confirmed over several ticks.
        #
        # The old check was `not builds and not force_workers`, evaluated on
        # the snapshot taken BEFORE the actions. At 10:27:55 the scheduler
        # decided to spawn a worker for two queued jobs and then, in the same
        # tick, saw the pre-spawn counts of zero and quit -- orphaning the
        # worker it had just started and taking the publisher with it, because
        # nothing else spawns one. Eleven finished segments then sat
        # unpublished for two hours.
        #
        # Queued force work and unpublished output are work too, even when no
        # process is running yet.
        nothing_left = (not builds() and not force_workers()
                        and backlog() == 0 and pending == 0
                        and pub_proc is None)
        quiet = quiet + 1 if nothing_left else 0
        if quiet >= 3:
            with LOG.open("a") as f:
                f.write(f"[{time.strftime('%H:%M:%S')}] 连续 3 次确认无任何待办，退出\n")
            return 0
        time.sleep(5)


if __name__ == "__main__":
    raise SystemExit(main())
