# TWM Recorder Soak — Five Verification Iterations (ledger)

**Goal:** prove on the rig that every stream is recorded in the right format, per-stream timestamps are synchronized, and a 10-minute episode has no dropped or late ticks. Up to five debug→verify iterations; each iteration = a headless `soak` run + `validate` + root-cause any failure + fix.

**Key facts**
- Branch `feature/twm-soak-test` in `.worktrees/twm-arducam`. Commands: `python -m twm.recorder soak --task <t> --duration <s> [--data_dir D] [--no_optitrack] [--realsense_serials ...] [--bandwidth_margin M] [--min_free_gb G]`, `python -m twm.recorder validate <h5> --expected-duration <s> --report out.json`, `python -m twm.recorder bench --dir D --seconds 5 --arducams 2`.
- Rig (2026-09-06): 3× D415 (`143322063538`, `104122062574`, `217222066989`), GelSight `2DUPB53G` left / `2BKRDTAD` right, Arducam `TWML0001` left (cam0) / `TWMR0001` right (cam1). No ROS master → `--no_optitrack`.
- Disks: `/media/yxma/Disk1` HDD, 239 GB free, **106 MB/s sustained (fsync)**; root NVMe 41 GB free. `vm.dirty_ratio=20` (≈12 GB of 60 GB RAM absorbs write bursts).
- Real stored size (lz4 SHUFFLE, June episode): 4.85 MB/tick for 3 RS + 2 GS; Arducam adds ~1.75 MB → ~6.6 MB/tick ≈ 198 MB/s at 30 Hz. With BITSHUFFLE (Task 4) ≈ 5.4 MB/tick ≈ 162 MB/s.

**Phases**
- [x] Iteration 1 — 10 s smoke (1b: without Arducams; Arducam hub faulty, see ledger), full rig, `/tmp`: does it record; T, dt stats, per-stream lag by hand.
- [x] Iteration 2 — validator available: 60 s on `/tmp`, every check green.
- [x] Iteration 3 — 120 s on Disk1 with `--bandwidth_margin 1.0`: watch queue peak and file MB/s.
- [x] Iteration 4 — 600 s on Disk1 (expected to overload at ~4 min at `vm.dirty_ratio=20`); measure the break point.
- [x] Iteration 5a — 600 s on Disk1 with BITSHUFFLE, no Arducams: PASS. [ ] 5b — full rig incl. Arducams: blocked on the hub rewiring (+ sysctl for the extra ~40 MB/s) (dirty-page budget or storage change); all checks green.

## Ledger

| found | evidence | fix | verified by |
|---|---|---|---|
| Recorder refused with `--no_optitrack` absent on this host | no ROS master: `rostopic list` → "Unable to communicate with master" | `--no_optitrack` flag + `DummyOptitrack` (Task 1) | test_rig `test_no_optitrack_installs_dummy…`, smoke run |
| Startup preflight refuses Disk1 at the default margin | bench 46 ticks/s < 57.9 required (1.5 × 1.29) | `--bandwidth_margin` flag (Task 2); the soak run's `queue_peak_fraction` is the real verdict | iteration 3 |
| Bench numbers were page-cache illusions | 5 s bench 259 MB/s, 20 s bench 478 MB/s, fsync probe 106 MB/s | none in code; documented; validator/soak judge by queue occupancy | iteration 4 |
| Smoke run failed "No device connected" for `143322063538` | pyrealsense2 enumerated only `217222066989`, `104122062574` at that moment; operator was re-cabling | none (operator reconnected all three) | iteration 1 |

| Iteration 1: soak aborted at rig open — `Arducam cam1 (/dev/video6) timed out waiting for first frame` | 3 RS + cam0 started; cam1 no frame in 15 s | pending (hardware) | — |
| Arducam first-frame failures are intermittent and device-level | isolation via `ArducamVideoStream`: cam1-alone FAIL, cam0-alone FAIL, cam0+cam1 → cam0 FAIL / cam1 OK 0.5 s; raw `v4l2-ctl`: TWML0001 3/3 tries no frames in 12 s, TWMR0001 1 OK then 2× `VIDIOC_STREAMON` EPROTO | none in code; both cameras share USB 2.0 hub 1-12 with both GelSights → suspect hub power/link | — |

| Hub 1-12 hosts both Arducams + both GelSights, all bMaxPower=500mA; kernel log shows repeated `reset high-speed USB device` on 1-12.3/1-12.4 during runs; hub descriptor says Self Powered, MaxPower 100mA | dmesg 43814/44185 resets; lsusb -v 0xe0 | hardware: move the Arducams to direct host ports or a hub with an adequate supply — operator action | iteration 2 |
| Iteration 1b (3 RS + 2 GS, `--no_arducam`, /tmp, 10 s): PASS by hand | T=289, median dt 33.4 ms, max 78 ms, late 0.69 %, strictly increasing; GelSight lag median 52 ms / max 171 ms, 17.7–18.3 Hz distinct, non-decreasing; shapes/dtypes correct; frames vary; stored 4.71 MB/tick; queue peak 5.6 % | — | validator (iteration 2) |

| Iteration 3 (3 RS + 2 GS, Disk1, 120 s, `--bandwidth_margin 1.0`): PASS by hand | T=3562 (29.75 Hz), late 0.06 % (2 ticks: 68/107 ms), p99 dt 38 ms, queue peak 11 %, writer 350 MB/s (cached), 16.9 GB = 4.74 MB/tick; GelSight lag median 55 ms / max 160 ms, 15.5 Hz distinct, non-decreasing; valid | — | validator (iteration 2, on this file) |

| Iteration 2: validator on the iteration-3 file: all 8 checks ok | report `2026-09-06-soak-iter3-validate.json` (metadata, shapes 13 datasets, tick_rate, duration 3562 ≥ 3492, sensor_sync, content, optitrack none, writer) | validator implemented (Task 3) | this run |
| `validate --expected-duration 10` fails on the 10 s smoke (T=289 < 291) | warm-up drops 10 frames + ~0.1 s start latency; 0.97 tolerance cannot cover that at 10 s | pending review decision (subtract warm-up from the requirement) | Task 3 fix round |

| Iteration 4 (3 RS + 2 GS, Disk1, 600 s): auto-ended INVALID at 521.4 s — `overload: writer queue full (90 ticks, 581 MB)`; fail-fast behaved exactly as designed | 15641 frames intact, late 0.20 %, max gap 178 ms, GelSight lag ≤ 160 ms @ 18.4 Hz; intake 140 MB/s vs disk drain ~91 MB/s; Dirty 9.5 GB at exit (limit ≈ 12 GB = vm.dirty_ratio 20 % × 60 GB); report `2026-09-06-soak-iter4-validate.json` | disk-bound, not code: (a) BITSHUFFLE (Task 4) cuts ~15 % bytes; (b) raise the dirty budget (`sudo sysctl -w vm.dirty_ratio=60 vm.dirty_background_ratio=5`, ~36 GB of 54 GB available) — operator action | iteration 5 |

| Iteration 5a (3 RS + 2 GS, Disk1, 600 s, BITSHUFFLE, default vm.dirty_ratio=20): **PASS** — 10 minutes, no drops | T=17839 (594.6 s), late 0.16 % (29 ticks), max gap 85 ms, queue peak 13 %, 69.6 GB = 3.90 MB/tick (−17 % vs SHUFFLE), GelSight lag 18–157 ms @ 16.7–17.1 Hz; validator 8/8 ok with `--expected-duration 600`; report `2026-09-06-soak-iter5a-validate.json` | BITSHUFFLE (Task 4) closed the gap for this rig without OS tuning | this run |

## Rejected ideas
- zstd-3 (shuffle or bitshuffle): GelSight 1.45–1.71×, color 1.47–1.59× but 6–14 ms/frame → ~10–20 ticks/s single-threaded; too slow for 30 Hz. Rejected.
- JPEG q95 for color/GelSight: 8.5× / 18.7×, 5 ms/frame — would solve the disk gap but is lossy; needs the user's decision, not taken.
