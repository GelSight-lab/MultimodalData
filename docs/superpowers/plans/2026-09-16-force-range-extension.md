# Force Range Extension Implementation Plan

**Goal:** Extend the approved v7 sensitivity to a 0-15 N output range using measured high-load calibration data.

**Architecture:** Freeze v7's reconstruction, reference-noise gate, spatial gain,
linear score and low-force isotonic knots. Append a continuous monotone tail,
trained on 8-15 N round presses above the original score ceiling. Exclude all
original held-out positions from tail fitting; reserve one third of previously
unseen positions as additional holdout. Bound predictions at measured support,
never manufacture a 15 N endpoint. Retain explicit v7 fitting for comparisons.

Full-range refitting was considered but changed the low-force curve substantially
(exploratory <1 N MAE 0.436 -> 1.100 N). Merely changing the output cap would
not remove isotonic saturation. The anchored tail matches the user's request
to preserve sensitivity; high loads whose score falls below the join can still
be underestimated and must be counted in the evaluation.

**Tech Stack:** Existing NumPy/SciPy/sklearn model, canonical preview renderer,
pytest, Hugging Face Hub and Playwright.

1. Add failing tests for anchored-tail continuity, preservation of low scores,
   bounded outputs, insufficient calibration support and position isolation.
   Run `python -m pytest tests/test_force_range_extension.py -q`.
2. Implement the tail in `twm/force_recovery/react_calib.py`; reconstruct an
   isolated high-force cache from raw images with the production feature code.
   Keep the existing low-force cache unchanged; bump pipeline version to 8.
3. Report paired errors by true-force band on original held-out positions and
   additional held-out positions, including all high-load samples below the
   join. Confirm all original low-range held-out outputs are unchanged.
4. Recompute the same seven review windows with v7/v8, unchanged references,
   contact masks and source mappings. Use separate artifact folders and update
   plots/force legend to 15 N. Keep the v7 webpage available.
5. Run focused force/preview tests, named-frame checks, full video decodes,
   desktop/mobile screenshots and remote playback. Publish isolated review
   artifacts only; do not overwrite full-episode release data. Document measured
   errors and remaining React accuracy/domain-transfer limitations.
