from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from twm.scripts.benchmark_mocap_repair import _candidate_order, benchmark_task


def clean_episode(n: int = 360, phase: float = 0.0) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    xyz = np.stack([
        0.18 * t,
        0.004 * t**2 + phase * 0.0001,
        0.003 * np.sin(t + phase),
    ], axis=1)
    quat = R.from_euler(
        "zy", np.stack([18.0 * t, 2.0 * t**2], axis=1),
        degrees=True).as_quat()
    return np.concatenate([xyz, quat], axis=1)


def clean_episodes(count: int = 8) -> list[np.ndarray]:
    return [clean_episode(phase=float(i)) for i in range(count)]


def test_benchmark_reports_all_required_errors():
    report = benchmark_task(
        clean_episodes(), task="toy", anomaly_lengths=[1, 5, 20],
        max_intervals=30, seed=7)

    assert set(report.metrics) == {
        "translation_mm", "orientation_deg",
        "native_translation_mm", "native_rotation_deg",
    }
    assert report.intervals == 30
    assert all(metric.count > 0 for metric in report.metrics.values())
    assert sum(report.confidence_counts.values()) == report.intervals
    assert report.high_intervals == report.confidence_counts["HIGH"]
    assert all(report.metrics[name].count <= report.metrics_all[name].count
               for name in report.metrics)


def test_fewer_than_one_hundred_intervals_disables_high_confidence():
    report = benchmark_task(
        clean_episodes(), task="rope", anomaly_lengths=[1, 5],
        max_intervals=99, seed=7)

    assert report.intervals == 99
    assert report.gate.high_confidence_enabled is False


def test_accurate_hundred_interval_benchmark_enables_high_confidence():
    report = benchmark_task(
        clean_episodes(12), task="motherboard", anomaly_lengths=[1, 5],
        max_intervals=120, seed=11)

    assert report.intervals == 120
    assert report.high_intervals >= 100
    assert report.gate.high_confidence_enabled is True
    assert report.gate.validated_max_frames == 5


def test_calibration_is_deterministic():
    kwargs = dict(task="pushT", anomaly_lengths=[1, 5, 20],
                  max_intervals=40, seed=17)

    first = benchmark_task(clean_episodes(), **kwargs).to_dict()
    second = benchmark_task(clean_episodes(), **kwargs).to_dict()

    assert first == second


def test_candidate_pool_is_bounded_before_motion_sorting():
    ordered = _candidate_order(
        [clean_episode(5000)], (1, 5, 20, 60),
        np.random.default_rng(4), pool_size=700)

    assert len(ordered) <= 700
