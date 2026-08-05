"""Unit tests for the pure-function core (run: python -m force_recovery.test_units)."""
import numpy as np

from .dexforce import force_informed_targets, quat_to_matrix, roundtrip_force
from .evaluate import median3_fresh


def test_quat_identity():
    R = quat_to_matrix(np.array([[0, 0, 0, 1.0]]))
    assert np.allclose(R[0], np.eye(3)), R


def test_quat_90z():
    # 90 deg about z: x-axis maps to y-axis
    q = np.array([[0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)]])
    v = quat_to_matrix(q)[0] @ np.array([1.0, 0, 0])
    assert np.allclose(v, [0, 1, 0], atol=1e-12), v


def test_targets_free_space_identity():
    pose = np.zeros((5, 7)); pose[:, 6] = 1.0
    act = force_informed_targets(pose, np.zeros(5), np.array([0, 0, 1.0]))
    assert np.array_equal(act.target_pos, pose[:, :3])


def test_targets_penetration_direction():
    pose = np.zeros((1, 7)); pose[0, 6] = 1.0          # identity orientation
    act = force_informed_targets(pose, np.array([0.3]), np.array([0, 0, 1.0]),
                                 stiffness=300.0)
    assert np.allclose(act.target_pos[0], [0, 0, 0.001])   # 0.3N/300 = 1 mm


def test_roundtrip():
    rng = np.random.default_rng(0)
    pose = np.zeros((50, 7))
    pose[:, :3] = rng.normal(size=(50, 3))
    q = rng.normal(size=(50, 4)); pose[:, 3:] = q / np.linalg.norm(q, axis=1)[:, None]
    force = np.abs(rng.normal(size=50))
    act = force_informed_targets(pose, force, np.array([0.2, -0.9, -0.3]))
    assert np.allclose(roundtrip_force(act, pose[:, :3]), force, atol=1e-9)


def test_median3_kills_single_spike():
    force = np.zeros(30); is_new = np.ones(30, bool)
    force[10] = 5.0
    out = median3_fresh(force, is_new)
    assert out[10] == 0.0, out[10]


def test_median3_keeps_plateau():
    force = np.zeros(30); is_new = np.ones(30, bool)
    force[10:14] = 5.0
    out = median3_fresh(force, is_new)
    assert out[11] == 5.0 and out[12] == 5.0


def test_median3_respects_duplicates():
    # duplicated rows repeat the previous fresh value; a spike lasting one
    # FRESH frame but three ROWS must still be removed
    force = np.array([0, 0, 0, 5, 5, 5, 0, 0, 0], float)
    is_new = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0], bool)
    out = median3_fresh(force, is_new)
    assert np.all(out == 0.0), out


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("all unit tests passed")
