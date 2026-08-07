# force_recovery — normal force from GelSight Mini, without a force sensor

The React recordings contain sensor pose and GelSight Mini images but **no
applied force**: demonstrated pose equals achieved pose, so the usual
"position error × stiffness" force channel of teleoperated data does not
exist. This package recovers it from the tactile images alone, validates it
against five public force-labelled datasets, and exports it back into the
dataset as an observation plus a force-informed action.

**Live results:** https://huggingface.co/spaces/yxma/react-force-recovery
**Module map:** [`ARCHITECTURE.md`](ARCHITECTURE.md) · **Public API:** `pipeline.py`

```python
from force_recovery.pipeline import reconstruct, virtual_target, STIFFNESS_N_PER_MM

st     = reconstruct(img, ref)                    # dI → LUT → mask → Poisson
target = virtual_target(pose, force_n, n_hat)     # pose + (F/k)·n̂ , k = 1 N/mm
```

## How it works

```
GelSight frame ─┐
                ├→ dI = img − ref ─→ 3-D RGB lookup table ─→ surface gradient
reference frame ┘                                              │
                                                    fast_poisson ↓
   force  ←── per-group calibration ←── features ←──────── depth [mm]
                                    (vol, vol², max depth, area, √area·depth)
```

The lookup table is **not learned**. It is a `(90,90,90,2)` array indexed by
the three difference-image channels (each spanning ±90 grey levels) returning
surface gradient in mm/pixel. It is filled by pressing a sphere of *unknown*
radius: the exact relation `a² = d(2R−d)` between contact-circle radius and
indentation depth recovers both `R` and the depth datum from the data, and
every contact pixel then contributes its analytic sphere slope to its colour
bin. Calibrating on a new sensor needs one set of sphere presses — ~700 frames.

## Validated on five datasets, each beside a control that could falsify it

One protocol throughout: per-group half/half least-squares + isotonic,
5 seeds, Spearman ρ. The control shuffles labels **within** each group, which
preserves that group's force range — a global shuffle proves nothing.

| dataset | gel | n | ρ | within-group shuffle | MAE |
|---|---|---|---|---|---|
| GlowTact | markerless | 201 | **0.986** | 0.171 | 0.53 N |
| Sparsh / Meta (self-calibrated LUT, in view) | markerless | 1667 | **0.968** | 0.264 | 0.042 N |
| FeelAnyForce (14 of 42 captures) | markerless | 1400 | **0.961** | 0.338 | 0.85 N |
| FoTa cnc_Mini (in view) | markerless | 337 | **0.946** | 0.056 | 0.25 N |
| FEATS | **marker dots** | 186 | 0.775 | −0.003 | 5.03 N |

Zero training frames from our rig — the table is self-calibrated from spheres.

## What it does *not* do — the measured limits

These are the numbers a reader needs before quoting any of the above.

**Accuracy depends on press depth, so never quote it without one.** Against
exact per-pixel ground truth (ray-cast meshes, 420 Tactile MNIST touches):

| press | 0.30 mm | 0.60 | 1.00 | 1.50 | 2.25 |
|---|---|---|---|---|---|
| MAE | **11 µm** | 35 | 68 | 127 | **281 µm** |
| peak recovered / true | 1.00 | 0.97 | 0.77 | 0.68 | **0.55** |

At ≤0.6 mm with zero fitting on non-spherical geometry this beats the
published per-pixel numbers for a stock Mini (3D Cal, arXiv 2511.03078,
Type-2 153–290 µm — and theirs fits both alignment and depth scale). At
2.25 mm we are an order worse. **The working range is shallow contact.**

**It interpolates; it does not extrapolate.** Fitting on the low half of a
force range and predicting the high half drops ρ 0.968 → 0.552, with the
predicted range collapsing to 5% of the true one — the isotonic stage clips
outside its training range. Plain linear recovers 0.839 but overshoots (2.18×
range), so the fix is a bounded monotone tail, not deleting the isotonic.

**ρ is a within-group rank correlation.** There is no hardness or modulus
constant anywhere; stiffness is absorbed into per-group fitted weights. These
numbers do **not** demonstrate a transferable absolute-newton scale across
gels. Only the LUT and `MM_PER_PIXEL` are shared.

**Contact must be visible.** On cnc_Mini, Sparsh and Tactile MNIST alike, a
contact clipped by the sensor border is systematically under-measured; on
cnc_Mini the all-positions ρ is 0.11 versus 0.94 strictly in view. This is
visibility, not force-range filtering — the clipped subsets carry the *higher*
median force.

**Marker gels are the weak domain.** Inpainting the dots (GelSight Wedge,
ICRA 2021) removes the dimple lattice from the depth map — lattice power
1.523 → 0.890, lower on 91% of frames, Wilcoxon p = 2.6e-19 — but does **not**
improve force (0.775 → 0.737), so it is adopted for the 3-D product only.
FEATS's real cap is its reference frame: even its lightest presses sit ~11 grey
levels off-dot, so the valid mask covers nearly the whole frame.

## Force as an observation, and a force-informed action

`export_force_columns.py` writes per-frame force back into the dataset:

| column | meaning |
|---|---|
| `force_{side}_normal_n` | estimated normal force [N] |
| `force_{side}_penetration_mm` | `F / k` |
| `force_{side}_target_pose` | observed pose pushed `F/k` along the contact normal |

`k = 1.0 N/mm` is exported as `STIFFNESS_N_PER_MM` **and written into a
sidecar next to the data** — it is an assumption about the environment, not a
measured property, so a reader of a target pose must be able to see which
stiffness produced it. Free space is exactly identity: `F = 0 → target ==
observed`, asserted in `test_units.py`.

## Running it

```bash
python -m force_recovery.run_episode              # batch force over episodes
python -m force_recovery.force_eval_all           # all datasets + controls
python -m force_recovery.improvement_study all    # 5 candidates, 2 adopted
python -m force_recovery.test_units
python -m force_recovery.design_guard             # layout gate before publishing

# anything rendering 3-D needs a display (Open3D 0.15.2 segfaults offscreen)
xvfb-run -a -s "-screen 0 1400x1000x24" python -m force_recovery.recon_study glowtact
```

Data roots are declared in `run_episode.py`; heavy artefacts live under
`/media/yxma/Disk1/twm/force_recovery/`, never in the repo.

## Negative results worth knowing

Kept because they cost time to establish and would otherwise be re-tried:

- **Small neural nets lose to linear + isotonic** on the same features and
  splits (MLP 0.967 / GBM 0.972 vs 0.974); ~400 frames per group is not enough
  for a flexible model.
- **A "PINN-lite" dimensionless contact law `F/A = g(δ/√A)`** looked decisive
  on sphere→flat transfer (MAE 0.426 → 0.150 N, beating an MLP's 0.297) until
  the control `F = g(volume)` reached 0.0736 N. The gain came from using one
  monotone feature instead of five, not from the physics form.
- **Shear features are not worth extracting**: feeding the *true* shear in as
  an oracle moves ρ 0.9749 → 0.9757.
- **LUT bin coverage does not detect out-of-domain use.** Applying the
  GlowTact table to Sparsh gave 90–97% bin coverage while its gradients sat
  **93.3°** from the analytic sphere gradient (chance is 90°). The usable
  detector is that angle, not coverage; self-calibration brings it to 4.5°.

Full ledger, including rejected variants with their numbers, in
[`../../task_plan.md`](../../task_plan.md).
