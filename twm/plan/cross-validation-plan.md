# Sensor-unit / gel-pad cross-validation — implementation plan

> **For agentic workers:** use `superpowers:subagent-driven-development` to run this
> task-by-task. Steps use `- [ ]` checkboxes. **A task may not start until its
> preflight checks are green:** `python analysis/preflight.py --task T<N>`.

**Goal:** Quantify how far a GelSight-style tactile signal moves across (a) sensor
units of one design, (b) gel pads on one unit, (c) accumulated wear in one gel —
and how much force-estimation accuracy each costs. Perception only.

**Architecture:** One statistic (RMS on mean no-contact frames) measures signal
shift; one scoring protocol (`force_eval_all.evaluate`) measures what the shift
costs. Every task reuses both, so a number in T5 is comparable to a number in T1.
Feature caches are built once per (dataset × method) and scored many times.

**Tech stack:** numpy/scipy/sklearn, torch 2.1.0+cu121 (CUDA True, 2 GPUs),
pandas+pyarrow, OpenCV. Existing package `force_recovery` supplies reconstruction,
features, loaders, and the protocol.

## Global constraints

Carried verbatim from the brief. Every task inherits these.

1. **Do NOT use Sparsh or the SITR encoder as feature extractors.** Both cancel the
   per-unit illumination offset under measurement. *(Verified absent from disk —
   `M7-banned-absent` asserts they stay absent. Sparsh **data** is still allowed.)*
2. **Do NOT invent an appearance statistic.** It is fixed: RMS pixel distance in
   uint8 between mean no-contact frames over the sensing region, plus per-channel
   medians. Implemented once in `analysis/feats_appearance_axes.py::rms`.
3. **Do NOT treat FEATS `test_diff_sensor_old_gel` / `_new_gel` as paired images.**
   Filenames match but frames sit at different force states. Distribution
   comparison only.
4. **Do NOT download FoTa (392 GB).** Use the local `fota_cnc/cnc/cnc_Mini` tars;
   sensor identity is directory-level only.
5. **Null results are deliverables.** GelSlim 2018 found wear over 3300 grasps
   "minimal" and digitally correctable. If T0b is flat, report flat.
6. **Nothing is framed as a discovery.** SITR / FeelAnyForce / FEATS already
   published the phenomenon. The contribution is quantification.
7. **n, scope, fit provenance, and exclusions on every reported number.**

## Anchor honesty — read before quoting anything

The spec that was to supply anchor numbers is **not in the linked Drive folder**
(see `plan/sensor-gel-variation.md`). Consequences, which every task must respect:

| Anchor | Status |
|---|---|
| SITR 93.77% → 38.66% across Minis with different pads | **Unverified.** Quoted only as *stated in the brief*. Label as such or omit. |
| 2.41 / 9–19 baseline ladder | **Unverified**, and no longer load-bearing — FEATS carries the gel axis directly. |
| GelSlim 2018 "wear minimal over 3300 grasps" | Cited from the brief as the T0b null hypothesis; verify against the paper before publishing. |
| FoTa 3,083,452 images / 13 sensors / 11 tasks | **Verified** in `claim-audit.md`; describes the dataset, *not* transfer. |
| GlowTact 0.209 vs 0.185 MAE at 2–20 N | **Verified** in `claim-audit.md`; GlowTact **loses** — state proactively. |

**T3 weights caveat, applies wherever T3 appears:** the released `mini.pth` is
**not on this machine**. What exists is a *downstream fine-tuned* checkpoint
(`trunk.pth` 63.79M + `encoders/wedge.pth` 22.01M ViT, `fota.models.*`). Every T3
number must be labelled "T3 downstream-finetuned trunk, not released mini.pth."
Fine-tuning may have already adapted it to a sensor population, which is a
confound for a cross-unit study and must be stated, not hidden.

---

## Verified resources (preflight-checked, 22/22 green)

**Force ground truth — 6 datasets**

| dataset | n | sensor | force source | check |
|---|---|---|---|---|
| FEATS | 22,014 npy | GelSight Mini, **markered** | `f_x/f_y/f_z` + `grid_z`, f_z ∈ [−59.9, 0] N | D2, D3 |
| cnc_mini_26 | 14,716 jpg | GelSight Mini | filename `f\|<N>`, 19 families | D4 |
| GlowTact | 14,716 jpg | **GlowTact** (2nd design) | filename `f\|<N>`, 0–20 N | D5 |
| FeelAnyForce | 3 csv (96.5k/10.7k/2.9k) | GelSight Mini | `FT` 6-axis wrench | D6 |
| Sparsh / TacBench | 49 pkl, 10 probe-batches | GelSight Mini markerless | ATI nano17 3-axis | D7 |
| FoTa cnc_Mini | 2 WebDataset tars | GelSight Mini | in-tar labels | D10 |

**The crossed grid (the whole reason both axes separate):** FEATS ships
**3000 no-contact frames = 5 sensors × 6 gels × exactly 100 repeats**, verified
balanced by `D1-feats-grid`.

**Encoders — 6 usable + 1 control**

| encoder | params | check |
|---|---|---|
| T3 trunk (downstream-finetuned) | 63.79M | M2 |
| T3 ViT encoder `wedge.pth` | 22.01M | M3 |
| DINOv2 ViT-B/14 | 86.6M | M4 |
| DINOv2 ViT-S/14 | 22.1M | M4 |
| ResNet50 / ResNet18 ImageNet | 25.6M / 11.7M | M5 |
| FEATS U-Net | 1.94M | M6 |
| raw pixels (control) | 0 | — |

**Auxiliary (no force, still load-bearing):** 3DCal (36,274 png +
`penetration_depth_mm`, a *monotone proxy* → attenuated rank correlation, say so),
TacQuad (4 sensor designs: Mini/DIGIT/DuraGel/Tac3D), Real+Sim Tactile MNIST,
GelSLAM, TactileTracking, FoTa labeled/unlabeled.

---

## Task graph

```
T0  appearance axes ──┬── T1 does shift predict cost? ──┐
     (FEATS arm DONE) │                                  ├── T4 cross-dataset matrix
T0b wear-in-one-gel ──┘                                  │      (extends existing)
T2  frozen encoders ─────────────────────────────────────┤
T3  physics reconstruction ──────────────────────────────┘
T5  cross-DESIGN (TacQuad + GlowTact) ── context for T0
```

---

### Task T0 — Appearance across units and gels

**Status: FEATS arm COMPLETE.** Results in `plan/sensor-gel-variation.md`:
noise floor **1.34**, gel axis **15.13** (11.3×), unit axis **26.36** (19.7×, an
upper bound), green the least stable channel (spread 41.1 vs R 12.0 / B 15.8).

**Files:** Modify `analysis/feats_appearance_axes.py` (add `--dataset`).
**Preflight:** `--task T0` (D1, D2).

**Interfaces produced** — T1 and T5 both consume these:
```python
rms(a: np.ndarray, b: np.ndarray, inset: int) -> float
load_cells() -> dict[tuple[int, int], np.ndarray]   # (sensor,gel) -> (n,240,320,3)
# writes plan/feats_appearance_axes.json
```

- [ ] **Step 1** — Extend to Sparsh pads. Add a loader returning the same
      `{cell_key: stack}` shape, cell = `(probe, batch)`; reuse `rms` unchanged.
- [ ] **Step 2** — Report the Sparsh table beside the FEATS one, same three rows
      (floor / gel / unit). Do **not** merge the two into one number: different
      gel type (markerless vs markered) and different capture rig.
- [ ] **Step 3** — Commit `feat(T0): appearance axes on Sparsh pads`.

---

### Task T0b — Wear accumulated within one gel

**The null hypothesis is the published one.** GelSlim 2018: wear over 3300 grasps
"minimal" and digitally correctable. A flat trend is the expected outcome and is
reported as such — it constrains what the thesis may claim.

**Wear proxy:** Sparsh `sphere/batch_1..batch_6` — 6 sequential batches on one
probe and pad (`D7` verified 6). Batch index is the only ordering available;
it is a *proxy for elapsed use*, not a grasp count, and must be labelled so.

**Files:** Create `analysis/wear_trend.py`. **Preflight:** `--task T0b` (D7).

**Interfaces produced:**
```python
wear_series(probe: str = "sphere") -> list[dict]   # {batch:int, rms_vs_batch1:float, n:int}
```

- [ ] **Step 1** — Mean no-contact (or lightest-force) frame per batch.
- [ ] **Step 2** — RMS of each batch mean vs batch_1, and the within-batch
      half-vs-half floor. **The floor decides whether any trend exists at all.**
- [ ] **Step 3** — Spearman of RMS vs batch index; report rho, n=6, and the floor.
      With n=6 this is underpowered — state the power limit, do not hide it.
- [ ] **Step 4** — If |rho| is low or RMS stays near the floor, write
      "**consistent with GelSlim 2018: no detectable wear at this resolution**"
      and stop. That is the deliverable.
- [ ] **Step 5** — Commit `feat(T0b): wear trend on 6 sequential Sparsh batches`.

---

### Task T1 — Does appearance shift predict force-estimation cost?

The scientific core: T0 says the signal moves; T1 asks what that costs.

**Files:** Create `analysis/shift_vs_cost.py`. **Preflight:** `--task T1` (D1, D4, D9, C1, C2).

**Interfaces consumed:** `rms` (T0); `force_eval_all.evaluate(X, f, groups, seeds=5)`;
`pipeline.FEATURES == ("vol","vol2","maxd","area","h1")`.

- [ ] **Step 1** — Group FEATS force frames by their `(sensor, gel)` cell.
- [ ] **Step 2** — Score **within-cell** (fit and eval same cell) and
      **across-cell** (fit cell A, eval cell B) with `evaluate`.
- [ ] **Step 3** — For each ordered cell pair, plot rho-drop against the T0 RMS
      between those two cells. Report Spearman(RMS, drop) with n = number of pairs.
- [ ] **Step 4** — Add the 3DCal arm using `penetration_depth_mm` as a **monotone
      proxy**; label every 3DCal number "attenuated rank correlation, depth proxy,
      not newtons."
- [ ] **Step 5** — Commit `feat(T1): appearance shift vs force-estimation cost`.

---

### Task T2 — Frozen encoders + linear probe (the pipeline-validation task)

The brief's T3 gate. Its purpose is to prove the pipeline reproduces a **known
order of magnitude** before new measurements are trusted.

**The brief's validation target cannot be reproduced as written:** the SITR
93.77→38.66 anchor is unverified *and* SITR/Sparsh weights are absent. So the gate
is restated honestly — a frozen encoder + linear probe must show a **large
cross-unit drop relative to within-unit**, with the **raw-pixel control** and the
**shuffle control** bracketing it. If within-unit ≈ across-unit, the split is
broken; stop and debug rather than proceeding.

**Files:** Create `analysis/encoder_probe.py`. **Preflight:** `--task T2`
(D1, D2, C1, M1–M7 = 10 checks).

**Interfaces produced:**
```python
ENCODERS: dict[str, callable]      # name -> (imgs uint8 (n,240,320,3)) -> (n,d) float32
embed(name: str, imgs: np.ndarray) -> np.ndarray
probe(feat: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict
```

- [ ] **Step 1** — Write the failing test first:

```python
def test_probe_separates_within_from_across():
    # synthetic: a per-unit offset the probe must NOT be able to transfer
    rng = np.random.default_rng(0)
    y = rng.uniform(0, 10, 200)
    unit = np.repeat([0, 1], 100)
    feat = np.c_[y + unit * 50.0, rng.normal(0, .1, 200)]   # offset only on unit 1
    within = probe(feat[unit == 0], y[unit == 0], np.zeros(100))
    across = probe(feat[unit == 0], y[unit == 0], np.ones(100), eval_on=(feat[unit == 1], y[unit == 1]))
    assert within["rho"] - across["rho"] > 0.3    # the offset must cost something
```

- [ ] **Step 2** — Run it; expect FAIL (`probe` undefined).
- [ ] **Step 3** — Implement `probe` as ridge + `evaluate`, reusing the protocol —
      do not write a second scorer.
- [ ] **Step 4** — Run; expect PASS.
- [ ] **Step 5** — Add all 7 encoders. **Freeze every backbone** (`eval()`,
      `requires_grad_(False)`); only the linear head fits. Resize 240×320 to each
      backbone's native input; record the resize per encoder — it is a confound.
- [ ] **Step 6** — Two splits on the FEATS grid: **held-out unit** (fit 4 sensors,
      eval the 5th) and **held-out gel** (fit 5 gels, eval the 6th). Both are
      leave-one-out, so n=5 and n=6 folds respectively — state n.
- [ ] **Step 7** — Gate check: within-unit vs across-unit must separate, with raw
      pixels worst and shuffle at ~0. If not, **stop and debug**.
- [ ] **Step 8** — Commit `feat(T2): frozen-encoder probe across unit and gel splits`.

---

### Task T3 — Physics reconstruction robustness (LUT vs calibration-free)

A LUT is fitted *per sensor*; the calibration-free solve is not. So the two should
degrade differently across units — directly testable, and it needs no weights.

**Files:** Create `analysis/recon_robustness.py`. **Preflight:** `--task T3`
(D1, D2, D4, C1–C4 = 7 checks).

**Interfaces consumed:**
```python
debug_gallery.stages(img, ref) -> dict            # LUT path, absolute 0.05 mm floor
calib_free.reconstruct(img, ref, ...) -> dict     # scale-free, needs relative floor
calibfree_eval.features(img, ref, method) -> dict # METHODS = ("lut","lut_native","calibfree",...)
```

- [ ] **Step 1** — Per FEATS cell, build features under `lut` and `calibfree`.
      Use `calibfree_eval.features`, which already handles the two depth floors —
      **mixing the floors is the known failure mode** (it moved contact radius
      73→100 px and rejected 6,287 of 6,308 presses once already).
- [ ] **Step 2** — Score within-cell and across-cell with `evaluate`.
- [ ] **Step 3** — Report the *drop* per method. Prediction to test, not assume:
      LUT degrades more across units. If it does not, that is the finding.
- [ ] **Step 4** — Commit `feat(T3): reconstruction robustness across units/gels`.

---

### Task T4 — Cross-dataset force matrix (extends existing work)

**Extends, does not replace,** `feature_cache/calibfree_vs_lut.json`:

| dataset | n | LUT rho | calib-free rho |
|---|---|---|---|
| cnc_mini_26 (markerless, 0–20 N) | 468 | 0.6143 | **0.7988** |
| FoTa cnc_Mini (markerless, in view) | 390 | 0.3012 | 0.3216 |
| FEATS (marker gel) | 390 | **0.7577** | 0.6154 |

**Files:** Create `analysis/force_matrix_ext.py`. **Preflight:** `--task T4`
(D3, D4, D6, D7, D10, C1, C2, C5, M6).

- [ ] **Step 1** — **Stratify the FEATS row by `(sensor, gel)`.** Those 390 frames
      come from a population spanning 15–26 RMS; the question is whether 0.7577 is
      stable across cells or an average over cells that disagree. Report per-cell
      rho with per-cell n.
- [ ] **Step 2** — Add FeelAnyForce and Sparsh rows via the existing
      `(rows, get)` loader contract (`C5`); do not write new loaders where
      `debug_gallery.load_*` already exists.
- [ ] **Step 3** — Add the FEATS U-Net (M6) as a supervised baseline column,
      beside the two physics reconstructions.
- [ ] **Step 4** — Every cell carries n and its within-group shuffle control.
- [ ] **Step 5** — Commit `feat(T4): stratified cross-dataset force matrix`.

---

### Task T5 — Cross-**design** context (how big is a unit difference, really?)

A unit gap of 26 RMS means nothing until you know what a *different sensor design*
costs. TacQuad captures 4 designs (Mini / DIGIT / DuraGel / Tac3D); GlowTact vs
cnc_mini_26 gives a force-labelled design pair.

**Files:** Create `analysis/cross_design.py`. **Preflight:** `--task T5` (D5, D8, C1).

- [ ] **Step 1** — Mean no-contact frame per TacQuad design; RMS between designs.
      Geometry differs across designs, so **resize/crop to a common sensing region
      and say exactly what was done** — this is the step that can fake a result.
- [ ] **Step 2** — Report the ladder in one table: noise floor < wear (T0b) <
      gel < unit < design. That ladder is the deliverable.
- [ ] **Step 3** — Force arm: fit on cnc_mini_26, evaluate on GlowTact. Anchor:
      GlowTact **loses** at 0.209 vs 0.185 MAE (2–20 N, verified) — state it.
- [ ] **Step 4** — Commit `feat(T5): cross-design appearance and force ladder`.

---

## Deliverables

- `plan/sensor-gel-variation.md` — results, one table per task, n everywhere. **Exists.**
- `analysis/*.py` — reusable scripts, no notebooks. **preflight.py + feats_appearance_axes.py exist.**
- Per-task anchor comparability, stated per the honesty table above.

## Ledger (continued from `plan/sensor-gel-variation.md`)

| found | evidence | fix | verified by |
|---|---|---|---|
| Plan would have used the released T3 `mini.pth` | absent from disk; only a downstream fine-tuned trunk exists | label every T3 number, treat fine-tuning as a confound | `M2`, `M3` |
| `_sd()` iterated dict **keys**, so DINOv2/ResNet param counts crashed | `'str' object has no attribute 'numel'` | `.values()` | `M4`, `M5` now report 86.6M / 25.6M |
| Sparsh probe read from the **filename** | matched 0 of 49 pkl → "0 sphere batches", silently killing T0b's axis | read probe from `parent.parent.name` | `D7`: 6 sequential sphere batches |
| 3DCal annotations path guessed one level too shallow | `missing .../3DCal/annotations/annotations.csv` | `.../3DCal/gsmini_calibration_data/annotations/` | `D9` |

**Rejected:** using Sparsh/SITR as encoders (banned — they cancel the measured
offset; also absent). Merging FEATS and Sparsh appearance numbers into one table
(different gel type and rig — the comparison would be meaningless).

## Reproduce

```bash
python analysis/preflight.py            # 22/22 must pass before any task
python analysis/preflight.py --task T2  # only the checks gating T2
python analysis/feats_appearance_axes.py
```
