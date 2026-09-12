"""The scheduler's decisions, which are the part worth getting right.

Every rule here exists because its absence cost time tonight.
"""
from __future__ import annotations

from twm.sched.policy import Decision, Limits, State, decide

LIM = Limits()


def S(**kw) -> State:
    base = dict(idle_pct=10, read_mbs=40, best_read_mbs=40, disk_running=2,
                disk_paused=0, cpu_workers=1, cpu_backlog=0,
                probing_disk=False, cooldown=0)
    base.update(kw)
    return State(**base)


def test_idle_cpu_never_starts_a_build():
    """The failure that halved throughput: idle CPU read as spare capacity,
    when the CPU was idle BECAUSE the disk was saturated."""
    d = decide(S(idle_pct=46, disk_paused=6, cpu_backlog=0), LIM)
    # it may probe the disk, but only as a probe it is prepared to revert
    assert d.probing is True and d.resume_disk is True


def test_a_probe_that_lowers_throughput_is_reverted():
    d = decide(S(probing_disk=True, read_mbs=35, best_read_mbs=72), LIM)
    assert d.pause_disk and d.cooldown > 0


def test_a_probe_that_holds_throughput_is_kept():
    d = decide(S(probing_disk=True, read_mbs=70, best_read_mbs=72), LIM)
    assert not d.pause_disk and not d.resume_disk


def test_spare_cores_go_to_queued_cpu_work_before_the_disk():
    """Force estimation is nearly free on the disk, so it is always the
    cheaper way to spend an idle core — this is the overlap that was missed
    for hours while builds ran alone."""
    d = decide(S(idle_pct=40, cpu_backlog=20, cpu_workers=1, disk_paused=6), LIM)
    assert d.add_cpu == 1 and not d.resume_disk


def test_the_disk_is_only_probed_once_the_cpu_has_nothing_queued():
    d = decide(S(idle_pct=40, cpu_backlog=0, disk_paused=6), LIM)
    assert d.resume_disk and d.add_cpu == 0


def test_an_oversubscribed_cpu_sheds_a_worker_not_a_build():
    """CPU workers resume instantly; a suspended build keeps its place but
    stops making progress, so shed the cheap thing first."""
    d = decide(S(idle_pct=2, cpu_workers=5, disk_running=3), LIM)
    assert d.drop_cpu == 1 and not d.pause_disk


def test_it_will_not_shed_the_last_cpu_worker():
    d = decide(S(idle_pct=2, cpu_workers=1), LIM)
    assert d.drop_cpu == 0


def test_cooldown_blocks_re_probing_and_counts_down():
    d = decide(S(idle_pct=40, cpu_backlog=0, disk_paused=4, cooldown=3), LIM)
    assert not d.resume_disk and d.cooldown == 2


def test_limits_are_respected():
    assert decide(S(idle_pct=40, cpu_backlog=9, cpu_workers=LIM.cpu_max), LIM).add_cpu == 0
    assert decide(S(idle_pct=40, disk_running=LIM.disk_max, disk_paused=2), LIM).resume_disk is False


def test_finished_data_is_published_without_waiting_for_anything_else():
    """Upload is its own budget — 68 MB/s measured, never the constraint — so
    a finished segment should not queue behind a build or a force worker."""
    d = decide(S(publish_pending=3, idle_pct=2, disk_running=6, cpu_workers=6), LIM)
    assert d.publish is True


def test_only_one_publisher_at_a_time():
    """Publishing rewrites the shared indices; two at once would race."""
    d = decide(S(publish_pending=3, publisher_running=True, idle_pct=40,
                 cpu_backlog=5), LIM)
    assert d.publish is False and d.add_cpu == 1


def test_nothing_pending_means_no_publish():
    assert decide(S(publish_pending=0), LIM).publish is False
