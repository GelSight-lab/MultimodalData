import numpy as np


def test_calib_free_contact_mask_detects_shallow_contact():
    """PushT has broad light contacts whose RGB delta sits below the old 8 DN
    cutoff; the recalibrated force mask must keep those contacts."""
    from twm.force_recovery import calib_free as cf

    dI = np.zeros((80, 80, 3), np.float32)
    dI[24:56, 24:56, 1] = 5.0

    mask = cf.contact_mask(dI)

    assert mask[40, 40]
    assert mask.sum() >= 700


def test_force_metadata_records_active_valid_di_threshold():
    from pathlib import Path

    import twm.force_recovery.run_episode as run_episode

    src = Path(run_episode.__file__).read_text()

    assert '"valid_mask_dI": float(CF.VALID_DI)' in src
    assert '"valid_mask_dI": 8.0' not in src
