"""How many of each kind of job to run, given what the machine is doing.

The pipeline's stages do not compete for the same thing, and treating them as
one queue wasted hours tonight:

    build    DISK-bound. Sits in state D; barely touches a core.
    force    CPU-bound. Reads one ~30 MB tactile mp4 per episode-side.
    cut      CPU-bound. Decode plus encode, no large reads.
    upload   NETWORK. Measured at 68 MB/s; never the constraint.

So `build` and `force` can and should run at the same time, and the machine
has two independent budgets to fill rather than one.

The disk budget is not a constant and cannot be assumed. Measured on this rig:

    read_ahead_kb=128         3 readers 20 MB/s   8 readers 43 MB/s
    read_ahead_kb=2048        2 readers 72 MB/s   5 readers 35 MB/s

The two rows disagree about which direction helps, so the only safe policy is
to measure: add a reader, keep it if aggregate throughput held, put it back if
it did not. Idle CPU is specifically the WRONG signal for the disk budget --
when the disk saturates, the CPU goes idle, and a controller that reads that
as spare capacity adds readers until throughput collapses. One did.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class State:
    idle_pct: int            # mean idle CPU over the sample
    read_mbs: int            # mean blocks-in over the sample
    best_read_mbs: int       # best aggregate read seen at any concurrency
    disk_running: int
    disk_paused: int
    cpu_workers: int
    cpu_backlog: int         # queued CPU jobs (episode-sides awaiting force)
    probing_disk: bool       # last tick resumed a build to test the water
    cooldown: int            # ticks left before probing the disk again
    publish_pending: int = 0     # finished, verified-able, unpublished segments
    publisher_running: bool = False


@dataclass
class Limits:
    disk_max: int = 6
    cpu_max: int = 6
    idle_high: int = 22      # above this the machine is under-used
    idle_low: int = 6        # below this it is oversubscribed
    cpu_min: int = 3         # workers kept while any CPU work is queued
    keep_fraction: float = 0.90   # a probe must hold this much of the best


@dataclass
class Decision:
    publish: bool = False
    add_cpu: int = 0
    drop_cpu: int = 0
    resume_disk: bool = False
    pause_disk: bool = False
    cooldown: int = 0
    probing: bool = False
    why: str = ""
    notes: list = field(default_factory=list)


def decide(s: State, lim: Limits = Limits()) -> Decision:
    """One tick of control. Pure: everything it needs is in `s`.

    CPU work is considered before disk work. Force estimation is nearly free
    on the disk, so filling idle cores with it never costs the builds
    anything, while an extra build can cost every other reader.
    """
    # 0. Publishing is its own budget. The upload measured 68 MB/s and 1.36 GB
    #    in 20 seconds, so it is never the constraint, and data that is
    #    finished should not wait for a pipeline stage it does not share a
    #    resource with. It runs at most one at a time because it rewrites the
    #    shared indices.
    if s.publish_pending and not s.publisher_running:
        return Decision(publish=True,
                        why=f"{s.publish_pending} 段已完成待发布")

    # 1. A disk probe is judged before anything else is changed, or the
    #    measurement gets attributed to the wrong action.
    if s.probing_disk:
        floor = s.best_read_mbs * lim.keep_fraction
        if s.read_mbs < floor:
            return Decision(pause_disk=True, cooldown=10,
                            why=f"撤回磁盘探测：{s.read_mbs} < {floor:.0f} MB/s")
        return Decision(why=f"保留磁盘探测：{s.read_mbs} MB/s 未下降")

    # 2. Oversubscribed: give CPU back before touching the disk, because the
    #    CPU jobs are the ones that can be resumed instantly. Never below
    #    `cpu_min` while work is queued -- a suspended worker holds its claim,
    #    so shedding the pool to one stalls the stage outright. That happened:
    #    idle touched 4%, four of five workers were suspended, and because the
    #    resume threshold was idle_high the pool sat frozen at 12% idle with
    #    twenty jobs still queued and none progressing.
    floor = lim.cpu_min if s.cpu_backlog > 0 else 1
    if s.idle_pct < lim.idle_low and s.cpu_workers > floor:
        return Decision(drop_cpu=1, why=f"CPU 过载 idle={s.idle_pct}%")

    # 3. Queued CPU work. Below the floor it is restored as long as the CPU is
    #    not actually oversubscribed; above the floor it grows only when there
    #    are cores going spare. Costs the disk almost nothing either way.
    if s.cpu_backlog > 0 and s.cpu_workers < lim.cpu_max:
        if s.cpu_workers < lim.cpu_min and s.idle_pct >= lim.idle_low:
            return Decision(add_cpu=1,
                            why=f"力估计 worker 低于下限 {s.cpu_workers}<{lim.cpu_min}")
        if s.idle_pct > lim.idle_high:
            return Decision(add_cpu=1,
                            why=f"空闲 {s.idle_pct}%，力估计还有 {s.cpu_backlog} 个待办")

    # 4. Only once the CPU is busy or has nothing to do is it worth risking
    #    another reader.
    if s.cooldown > 0:
        return Decision(cooldown=s.cooldown - 1, why="磁盘探测冷却中")
    if (s.idle_pct > lim.idle_high and s.disk_paused > 0
            and s.disk_running < lim.disk_max):
        return Decision(resume_disk=True, probing=True,
                        why=f"探测磁盘 +1（当前 {s.disk_running} 个读者 {s.read_mbs} MB/s）")
    return Decision(why="保持")
