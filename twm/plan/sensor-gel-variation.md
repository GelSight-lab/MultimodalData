# Cross-validation of sensor-unit and gel-pad variation

Status: **partially blocked.** Task 1 is settled and one measurement is done.
Tasks T0/T0b/T1/T2/T3/T5b cannot be started — see "Blocker" — and nothing in
this file depends on them.

---

## Blocker: the linked folder does not contain the spec

The brief says to work from the Drive folder, which "contains verified dataset
URLs, licenses, sizes, model weights, six runnable tasks with published anchor
numbers, and the failure modes."

The folder is readable. It holds **exactly one file**, `claim-audit.md`
(11,301 bytes, downloaded and read in full). It is a thesis claim audit —
GlowTact, Baby Fin Ray, GelLink, FoTa — written to record which drafted claims
the source papers actually support. It is not a study spec.

Token search over the retrieved document:

| Token the brief attributes to the spec | Occurrences in the actual file |
|---|---|
| `T0` `T0b` `T1` `T2` `T3-large` `T5b` | 0 |
| `SITR`, `93.77`, `38.66` | 0 |
| `2.41` (the baseline ladder) | 0 |
| `mini.pth`, `trunk.pth` | 0 |
| `v24_labels`, `FEATS`, `per-gel` | 0 |
| `license` | 0 |
| `http` (i.e. **any URL at all**) | 0 |

Also checked and absent: `plan/membrane-swap-experiment.md`, which the brief
says holds the ladder to be retracted. It does not exist, and `git log --all`
shows no commit has ever tracked `plan/`, `*membrane*`, or `claim-audit`.

I am not reconstructing six task definitions, dataset URLs, licenses, and
anchor numbers from memory. The brief explicitly forbids re-deriving resource
availability ("it has already been verified by fetching the actual
artifacts"), so inventing it is the one move guaranteed to be wrong — every
number would look sourced while being unsourced. **Send the real spec and
T0/T0b/T1/T2/T5b can start.**

What is *not* blocked is anything answerable from local artifacts. That is
Task 1, and one measurement whose statistic the brief fixes in full.

---

## Task 1 — FEATS per-gel labels: **present**. Retract nothing.

The brief asks that if `sensor_{0..4}_gel_{0..5}` is absent, the baseline
ladder be retracted and Tactile MNIST become the sole gel-axis source.

It is present. Source: `.../FEATS/v24_labels_24_32/` (unpacked tree, not a
zip; `convert_feats.py` consumed the archive). Example:
`100_nocontact_sensor_0_gel_2.npy`.

The design is **fully crossed and exactly balanced**:

| | value |
|---|---|
| no-contact frames carrying the convention | **3000** |
| sensor units | 5 (`sensor_0..4`) |
| gel pads per unit | 6 (`gel_0..5`) |
| distinct (sensor, gel) cells | 30 |
| frames per cell | **100, min = max** |
| repeat index prefix | 1..100, each covering all 30 cells |
| splits carrying them | train 2555, test 306, val 139 |

`f_x/f_y/f_z` and `grid_*` are all-zero on these frames, confirming true
no-contact rather than light contact.

**Consequence:** the 2.41 / 9–19 ladder does not need retracting on the
grounds raised, and FEATS supports **both** axes — unit and gel — not just
one. Tactile MNIST is not forced to be the sole gel-axis source. Two prior
passes disagreed; the disagreement is resolved in favour of "the labels are
there", and this is a fully crossed design of the kind that rarely survives
into a public release.

Script: `analysis/feats_appearance_axes.py` (enumeration printed on every run).

---

## Measurement — how far appearance moves, per axis

Statistic **as fixed by the brief**, not chosen here: RMS pixel distance in
uint8 between mean no-contact frames over the sensing region, plus per-channel
medians. No encoder is involved, so this does not depend on the unvalidated
T3 pipeline and does not jump the T3 gate.

The noise floor is computed first, because a distance between two mean images
means nothing until you know what two means of the *same* cell produce. Each
cell's 100 repeats are split into disjoint halves and the half-means compared.
A 50-frame mean is noisier than the 100-frame means compared elsewhere, so
this floor is slightly conservative.

| axis | n pairs | median RMS (full frame) | median RMS (12 px inset) | × noise floor |
|---|---|---|---|---|
| noise floor (half vs half, same cell) | 30 | **1.34** | 1.43 | 1.0 |
| **gel** — same unit, different pad | 75 | **15.13** | 16.35 | **11.3×** |
| **unit** — different unit, same gel index | 60 | **26.36** | 28.44 | **19.7×** |
| unit — different unit, different gel index | 300 | 28.71 | 30.74 | 21.5× |

Both axes clear the noise floor by an order of magnitude. Changing the unit
moves appearance **1.74×** as far as swapping a pad within one unit — so the
gel axis is not a rounding error against the unit axis; it is well over half
of it. The 12 px inset changes no ordering, so this is not a vignetting or
mount-shadow artefact.

### The unit figure is an upper bound, and here is the test

Whether `gel_2` on sensor 0 is the *same physical pad* as `gel_2` on sensor 1
is not knowable from filenames, and it decides how the unit row reads. It is
still testable: if the index were one pad travelling between units, holding it
fixed across units would remove a real source of variation and score well
below pairs that also differ in index.

Holding it fixed buys **26.36 vs 28.71, about 8%**. Under independent additive
variance a travelling pad would predict sqrt(26.36² + 15.13²) = 30.4 for the
differing-index case; the observed 28.71 sits much nearer the
"index is a per-unit slot number" end.

So the index is *mostly* a slot number, the two populations of pads are
largely disjoint, and **26.36 is an upper bound on the pure unit effect** —
it carries some pad change inside it. The gel row (15.13) has no such
problem: same unit, definitely a different pad.

### Per-channel medians — green is the unstable channel

Median (R, G, B) per unit, taken over that unit's six gels:

| unit | R | G | B |
|---|---|---|---|
| sensor_0 | 59.53 | 114.88 | 115.69 |
| sensor_1 | 65.31 | 117.19 | 110.67 |
| sensor_2 | 71.54 | 98.66 | 105.35 |
| sensor_3 | 68.40 | **76.06** | 99.86 |
| sensor_4 | 65.78 | 116.22 | 107.78 |

Across-unit spread: **G = 41.1**, B = 15.8, R = 12.0. Green varies **3.4× more
than red** and 2.6× more than blue, driven mainly by sensor_3.

This is worth flagging against `claim-audit.md`, the one document actually in
the folder: its retracted chromatic finding turned on green/red ratios, and it
concluded that a *global* R:G offset of 0.75 ± 0.20 stops survives as
white-balance or LED-power difference. The present measurement is consistent
with that reading and sharpens it — the green channel is where units differ
most. Any method leaning on green transfers worst across units. Unverified
here: whether this is LED power, camera white balance, or gel dye, which needs
a colour target the dataset does not contain.

---

## Comparability to published anchors

Stated plainly: **none of the numbers above has a published anchor I could
verify**, because the document that was supposed to carry the anchors is not
in the folder. The SITR 93.77% → 38.66% figure and the 2.41 / 9–19 ladder are
quoted here only as *stated in the brief*; I could not check them against a
source and they must not be reported as verified.

The phenomenon itself is published — SITR, FeelAnyForce and FEATS all report
cross-sensor and cross-gel degradation. Nothing here is offered as a
discovery. The contribution is quantification on a balanced crossed design,
and separating the gel axis from the unit axis, which needs the 5×6 grid
confirmed in Task 1.

---

## Existing cross-dataset force estimation eval

Present at `feature_cache/calibfree_vs_lut.json` and unchanged by this work:

| dataset | n | LUT rho | calib-free rho |
|---|---|---|---|
| cnc_mini_26 (markerless, 0–20 N) | 468 | 0.6143 | **0.7988** |
| FoTa cnc_Mini (markerless, in view) | 390 | 0.3012 | 0.3216 |
| FEATS (marker gel) | 390 | **0.7577** | 0.6154 |

The natural extension — now unblocked by Task 1 — is to **stratify the FEATS
row by unit and by gel**, since those 390 frames come from a population whose
no-contact appearance spans 15–26 RMS. That is a real question about whether
the 0.7577 is stable across the grid or an average over cells that disagree.
It needs no missing spec. I have not run it, because the brief orders T3
pipeline validation before new measurements and T3 is blocked; say the word
and I will either run it or wait for the spec.

---

## Ledger

| found | evidence | fix | verified by |
|---|---|---|---|
| Drive folder assumed to hold the spec; holds a thesis claim audit | 1 file, 11,301 B; 0 occurrences of `T0`, `SITR`, `mini.pth`, `license`, or even `http` | stop; request the real spec rather than reconstruct it | token table above |
| `plan/membrane-swap-experiment.md` assumed to exist | absent from disk; `git log --all --name-only` never tracked `plan/` or `*membrane*` | reported, not recreated | git history |
| FEATS per-gel labels in doubt across two passes | 3000 files match `sensor_{0..4}_gel_{0..5}`, 30 cells, exactly 100 frames each | resolved: present; nothing retracted | `analysis/feats_appearance_axes.py` |
| Unit axis would be confounded if gel index were a travelling pad | same-index 26.36 vs differing-index 28.71 (~8%); independence predicts 30.4 | report the unit axis as an upper bound | same script, `unit_axis_diff_gel_index` |
| Border optics could fake an appearance difference | 12 px inset: 1.43 / 16.35 / 28.44 — no ordering change | conclusion stands on the interior | same script, both regions |

**Rejected:** substituting Tactile MNIST as the gel-axis source — unnecessary,
FEATS carries a balanced 5×6 grid (Task 1). **Rejected:** reconstructing the
six tasks from memory — every resource would appear sourced while being
invented, which is the specific failure the brief warns against.

## Reproduce

```bash
python analysis/feats_appearance_axes.py      # -> plan/feats_appearance_axes.json
```
Reads only `.../FEATS/v24_labels_24_32/`. No network, no model weights, no
Sparsh, no SITR encoder.
