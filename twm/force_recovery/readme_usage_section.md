## How to use this dataset

Three recipes, in the order most people need them. Every one is executed
against the published files by `scripts/test_readme_recipes.py`, so the code
below is code that runs, not code that reads well.

### 1. Sample training clips — start from `segments.json`, not from episodes

An episode is a raw recording and contains flagged frames. A **segment** is a
contiguous span that is already clean. Sampling clips from episodes means
re-deriving the quality filter yourself and getting it slightly different.

```python
import json, numpy as np, pyarrow.parquet as pq

segs = json.load(open("data/pushT/segments.json"))["segments"]
s = segs[0]                                     # {'source_episode', 'frame_range', ...}
date, ep = s["source_episode"].split("/")
a, b = s["frame_range"]                         # inclusive, in VIDEO frame coords

t = pq.read_table(f"data/pushT/meta/{date}/{ep}.parquet").slice(a, b - a + 1)
```

`frame_range` indexes the published MP4s and the parquet with the same origin,
so frame `i` of `view_middle.mp4` is row `i` of the parquet. No offset, no
lookup table.

### 2. Train on touch — respect the tactile rate

Rows are written at 30 Hz; the GelSight stream is slower. A row with
`tactile_{side}_is_new == False` repeats the previous tactile frame, its
contact scalars, and its force estimate, unchanged.

```python
new = t["tactile_left_is_new"].to_numpy()
# independent tactile samples only
idx = np.flatnonzero(new)
# a finite difference over ALL rows is 0 wherever is_new is False, by construction
```

Roughly 72% of rows are repeats. Ignoring this does not corrupt a model that
consumes frames independently, but it silently zeroes any temporal derivative
of a tactile channel and inflates any "how often does touch change" statistic.

### 3. Train an action that includes *how hard*

This is the part that distinguishes React from a pose-only demonstration set,
so it gets its own section: **[estimated contact
force](#estimated-contact-force-motherboard--pusht-36-episodes)**. In short:

```python
observation = np.array(t["sensor_left_pose"].to_pylist())        # where it was
action      = np.array(t["force_left_target_pose"].to_pylist())  # where to push to
```

`action` equals `observation` exactly in free space and leads it by `F/k` along
the press direction during contact. Train on `action`, deploy through an
impedance controller of stiffness `k`, and the policy commands both the reach
and the press. Read that section before choosing `k` — the shipped `k = 1 N/mm`
is a declared assumption and a soft one.

### What this dataset is not

- **No robot.** A human hand holds each sensor. There are no joint angles, no
  gripper state, and no action in the robot-command sense other than the
  force-informed target pose described above.
- **No force sensor.** Every newton in these files is estimated from tactile
  images. It is calibrated and validated, and it is still an estimate — see the
  limits in the force section before reporting absolute values.
- **Not a benchmark.** There is no train/val/test split and no success label.
  It is interaction data for dynamics and representation learning.
