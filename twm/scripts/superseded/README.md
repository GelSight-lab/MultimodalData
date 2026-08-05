# superseded/

Scripts whose logic now lives in `twm/react_preprocess/`. Kept for provenance —
they are the code that actually produced the published release, so they are the
reference the package was checked against.

**Do not run these.** They import each other by bare module name and by
`sys.path` injection, which no longer resolves after the scripts directory was
reorganised. Use the package.

| script | replaced by | how the port was checked |
|---|---|---|
| `detect_bad_intervals.py` | `react_preprocess.detect` | every interval of all 5 detector outputs, all 36 published episodes: **0 differences** |
| `build_release_curation.py` | `react_preprocess.curation` | rebuilt `segments.json` compared field by field against the published file: **identical**, 76 + 17 segments |

```bash
python -m react_preprocess curate --task motherboard
```

The port also fixed a latent break: `build_release_curation.py` imported
`find_clean_segments` from `build_segments.py`, which was archived to
`legacy_pt/` as part of the `.pt`-era cleanup. That function now has a live
home in `react_preprocess.detect`.
