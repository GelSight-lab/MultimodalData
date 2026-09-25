# Bringing the rig up on a different machine

Everything a recording depends on that is a fact about THIS hardware lives in
`twm/recorder/config.py`. Nothing else needs editing to move machines; what
follows is the order to do it in and how to check each step, because a wrong
answer here does not raise — it records, and the mistake surfaces weeks later
in a projection that is subtly off.

## 1. Which camera is which

```python
REALSENSE_SERIALS = ("143322063538", "104122062574", "217222066989")
REALSENSE_POSITIONS = ("right", "left", "middle")     # in that order
```

The pairing is positional: `SERIALS[i]` stands at `POSITIONS[i]`. This is the
single source for "which camera is this" — logs, preview labels and the
calibration file names all derive from it, so a swapped pair renames every
downstream artefact consistently and wrongly.

Read the serials off the hardware, not off this file:

```bash
python -c "from twm.sensor_camera import _attached_realsense_serials as s; print(s())"
```

## 2. Which GelSight is left and which is right

```python
GELSIGHT_SERIALS = {"left": "2DUPB53G", "right": "2BGLKZNT"}
```

Identity here is the SERIAL. (For the wrist cameras it is the USB port
instead — the generic pair both report `200901010001`, so a serial would match
both and resolving would refuse. See `sensor_camera.register_wrist_cameras`.)

List what is attached:

```bash
python -c "
from twm.sensor_camera import enumerate_capture_devices
for d in enumerate_capture_devices():
    if d.vendor_id == 0x0c45: print(d.device, d.reported_serial)"
```

Vendor `0x0c45` is the GelSight Mini. Confirm each one is alive before you
trust the mapping — a unit that enumerates can still be dead:

```bash
python -c "
import cv2, numpy as np
cap = cv2.VideoCapture(8)          # the /dev/videoN you just found
f = [cap.read()[1] for _ in range(30)]; cap.release()
print('unique frames', len({x.tobytes() for x in f}), 'of', len(f))
print('channel means', [float(np.mean([x[...,c] for x in f])) for c in range(3)])"
```

30 unique frames means it is streaming. A frozen unit returns 1. Balanced
channel means say the three LEDs are lit; one dark channel is a dead LED, and
photometric reconstruction on that sensor will be wrong without failing.

The recorder writes whatever it used into every HDF5 as
`metadata.attrs["gelsight_serials"]` (left first), so a recording always
carries the answer even if this file later changes.

## 3. Calibration

`calib_epoch.CURRENT_EPOCH` picks the epoch. A session dated on or after it
resolves without an entry in `CALIB_SESSIONS`; an earlier one must be declared.

An epoch may name the units it was solved on in `sensors.json`, and
`check_sensors` WARNS when a recording's serials differ. It warns rather than
refuses on purpose: force values never read the gel transform, and the
projection error a same-model swap introduces (about 0.13 mm at 15 N) sits
under the calibration's own 0.72 mm repeatability. Judge by whether a
projection looks wrong, not by whether a serial changed.

**Recalibrate when the geometry moved**: a sensor remounted at a different
angle, a camera repositioned, a new rigid body. Not merely because a part was
replaced in its existing mount.

## 4. Paths and OptiTrack

```python
DATA_DIR = Path("/media/yxma/Disk1/twm/data")   # needs ~120 MB/s sustained
FPS = 30
OT_TRACKERS = ("motherboard", "sensor_left", "sensor_right")
```

`OT_TRACKERS` must match the rigid-body names in Motive exactly. The recorder
writes ~117 MB/s with all streams live; a disk slower than that drops frames
under load rather than erroring.

## 5. Check before the first real recording

```bash
python -m twm.recorder validate /path/to/a/closed_episode.h5
python -m twm.visualize /path/to/a/closed_episode.h5
```

`validate` reports what the file actually contains. The viewer's projection
overlay is the fastest way to catch a wrong camera mapping or a stale
calibration: the gel centre should sit on the gel in all three views.

## 6. Two things that will bite you on a fresh machine

**The recorder tests need room on /tmp.** Each e2e case writes a synthetic
recording of a few hundred MB, and pytest keeps the last three runs. A session
that runs them repeatedly filled a 469 GB root partition to 98% here, and the
next run failed with ENOSPC reported as a test failure. Point them elsewhere:

```bash
TMPDIR=/path/with/room python -m pytest tests/recorder/ -q     --basetemp=/path/with/room/pytest
```

**`test_run_headless_records_a_valid_episode` is load-sensitive.** It asserts a
60 fps tick holds within 10%, so anything else saturating the CPU fails it on
timing alone — measured here at 24-27 ms median against 16.67 expected, while
the writer itself was fine at 248 MB/s with the queue at 2.6%. Run it on an
idle machine before believing a failure.
