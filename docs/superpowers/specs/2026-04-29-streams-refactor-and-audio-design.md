# Design — `streams/` package refactor + contact-microphone audio integration

**Date:** 2026-04-29
**Status:** Approved (pending file review)
**Owner:** Yuxiang

## 1. Goal

1. Add 2-channel contact-microphone audio capture to the `twm/` multimodal data
   collection pipeline.
2. Refactor the four sibling modality folders (`camera_stream/`, `ft_sensor/`,
   `optitrack/`, `audio_stream/`) into a single `streams/` package with
   consistent internal shape.
3. Fix a small set of bugs and dead-import-style issues that fall out of the
   refactor.

Out of scope: wiring the FT sensor into `twm/data_collection.py` (refactor +
bugfix only this round), `print` → `logging` migration, a unified `BaseStream`
ABC across modalities, broad `probing_panda/` cleanup beyond import-path
rewrites forced by the rename.

## 2. Module layout — `streams/` package

```
streams/
  __init__.py                  # empty (subpackages explicit)
  base.py                      # BaseStream(start, stop) — marker only
  camera/
    __init__.py                # re-exports concrete classes
    base.py                    # BaseVideoStream — kept verbatim (probing_panda depends on JPG-recording API)
    realsense.py               # RealsenseStream
    usb.py                     # USBVideoStream
    digit.py                   # DigitVideoStream
    raspi.py                   # RaspiVideoStream
  ft_sensor/
    __init__.py
    base.py                    # BaseFTStream
    mms101.py                  # MMS101FTStream (bugfixes per §5)
    mms101/                    # vendor SDK + example — moved verbatim
      example.py
      mms101.py
      README.md
  audio/
    __init__.py
    base.py                    # BaseAudioStream (NEW — see §3)
    contact_mic.py             # ContactMicStream (NEW)
    contact_mic/
      viz.py                   # current audio_stream/viz.py — vendor reference
  optitrack/
    __init__.py
    base.py                    # BaseOptitrackStream (NEW thin ABC)
    motive.py                  # was optitrack_stream.py; class stays OptitrackStream
```

**Naming convention:**
- Folder name = modality. Files inside drop the `_stream` suffix.
- Class names unchanged (`RealsenseStream`, `MMS101FTStream`, `OptitrackStream`,
  `ContactMicStream`) — call sites change only in the import line.
- Vendor SDK / example code lives in a same-named subfolder under each
  implementation (matches the existing `ft_sensor/mms101/` precedent).

**`__init__.py` policy:** explicit re-exports of public classes; no
`try/except ImportError: print(...)` swallow. Broken imports of in-package
modules are bugs and should fail loudly.

**Caller-side import rewrites** (done with grep + replace, then run tests):
- `twm/data_collection.py`
- `twm/visualize.py`, `twm/visualize_projection.py`
- `twm/calibration/mocap_to_cam.py`
- `tests/test_optitrack_stream.py`, `tests/test_realsense_stream.py`
- `probing_panda/scripts/example_gelsight_stream.py`,
  `probing_panda/displacement_data_collection.py`

`pyproject.toml` package list updated if it declares packages explicitly.

## 3. New `streams/audio/` module

### 3.1 `streams/audio/base.py` — `BaseAudioStream`

Thin abstract that fixes the surface every concrete audio source must expose.
Mirrors `OptitrackStream`'s shape (background producer + `flush_buffer()`).

```python
class BaseAudioStream:
    def __init__(self, channel_name: str, sample_rate_hz: int, dtype=np.int16):
        self.channel_name   = channel_name        # "left" / "right"
        self.sample_rate_hz = sample_rate_hz      # nominal
        self.dtype          = dtype
        self._buffer        = collections.deque() # (timestamp_s: float, sample: int)
        self._lock          = threading.Lock()
        self._running       = False

    def start(self):           raise NotImplementedError
    def stop(self):            raise NotImplementedError

    def flush_buffer(self) -> list[tuple[float, int]]:
        """Atomically swap & return all buffered (t, sample) tuples since last call."""
        with self._lock:
            out, self._buffer = list(self._buffer), collections.deque()
        return out

    def get_recent_samples(self, n: int) -> np.ndarray:
        """Non-destructive — for live preview level meter. Returns up to last n samples."""
        with self._lock:
            return np.fromiter((s for _, s in list(self._buffer)[-n:]), dtype=self.dtype)
```

### 3.2 `streams/audio/contact_mic.py` — `ContactMicStream(BaseAudioStream)`

Concrete impl for one Arduino-USB-serial contact microphone.

- Constructor: `ContactMicStream(channel_name, device_path, baud=115200, sample_rate_hz=1000)`.
- `device_path` must be a stable identifier under `/dev/serial/by-id/...` —
  see §5.3 for why and how.
- `start()` opens the serial port and spawns a daemon producer thread that
  `readline()`s as fast as possible, parses an `int`, timestamps it with
  `time.time()`, and appends to `_buffer` under `_lock`.
- Reader robust to malformed lines (skip + continue) and brief disconnects
  (log + retry-with-backoff). Same defensive pattern as the existing camera
  streams.
- `stop()` clears `_running`, joins the thread, closes the port.
- No internal cap on `_buffer` size: 1 kHz × 16 B/tuple ≈ 60 KB/min/mic;
  hour-scale episodes are not anticipated.

### 3.3 Why a base + concrete split

A second audio source (USB sound card, I²S DAC, network mic) is foreseeable.
`BaseAudioStream` lets a future `SounddeviceMicStream` slot in with no
caller-side changes.

## 4. Refactor of existing modules

### 4.1 Camera (`camera_stream/` → `streams/camera/`)

- File renames only. Class names unchanged.
- `BaseVideoStream` kept **as-is** including JPG-recording surface
  (`prepare_recording`, `start_recording`, `pause_recording`, `stop_recording`,
  `write_frame`). It is actively used by `probing_panda/displacement_data_collection.py`
  and called from inside `digit/raspi/usb` `update()` loops.
- `__init__.py` rewritten to explicit re-exports.

### 4.2 FT sensor (`ft_sensor/` → `streams/ft_sensor/`)

- File renames: `base_ft_stream.py` → `base.py`,
  `mms101_stream.py` → `mms101.py`. Vendor `mms101/` subfolder moves verbatim.
- Bugfixes per §5.1, §5.2.
- Not wired into `twm/data_collection.py` this round.

### 4.3 OptiTrack (`optitrack/` → `streams/optitrack/`)

- File rename: `optitrack_stream.py` → `motive.py` (file = vendor; class =
  interface).
- Class stays `OptitrackStream`.
- New `base.py` with a thin `BaseOptitrackStream` ABC defining only the
  existing public surface (`start`, `stop`, `get_latest_pose`, `flush_buffer`).
  No internal-method reshuffling.

### 4.4 Audio (`audio_stream/` → `streams/audio/`)

- New module per §3.
- `audio_stream/viz.py` moves verbatim to `streams/audio/contact_mic/viz.py`
  as vendor reference.

## 5. Bugfixes + USB-serial port collision

### 5.1 Wrong import names + over-broad `except ImportError`

`ft_sensor/__init__.py`, `camera_stream/__init__.py`, `optitrack/__init__.py`
all currently use:

```python
try:
    from .x import Wrong_Class_Name
except ImportError:
    print("...")
```

Fix: explicit imports of the actual class names; no `try/except`. Specifically
in FT, the existing names `Base_FT_Stream` and `MMS101_FT_Stream` are wrong —
the actual classes are `BaseFTStream` and `MMS101FTStream`.

### 5.2 `MMS101FTStream` broken `Timer` + non-looping reader

Today (`mms101_stream.py:249`):

```python
self.thread = threading.Timer(0.001, target=_update_ft, args=[self])
```

Broken twice over: `Timer.__init__` does not take a `target=` kwarg (that's
`Thread`), and a `Timer` fires once after the delay then stops.

Replace with the same daemon-thread loop pattern OptiTrack uses today:

```python
def _reader_loop(self):
    while self._running:
        self.get_ft()
        self.medfilt_ft()

def start(self):
    self.initialize()
    self._running = True
    self.thread = threading.Thread(target=self._reader_loop, daemon=True)
    self.thread.start()

def stop(self):
    self._running = False
    if self.thread is not None:
        self.thread.join(timeout=1.0)
    self.serialPortClose()
```

Out of scope: rewriting the byte-level MMS101 protocol code.

### 5.3 USB-serial device collision (`/dev/ttyUSB1` × 3)

The bench will have **3** USB-serial devices: FT sensor + 2 contact mics. All
default to `/dev/ttyUSB1` today; enumeration is plug-order dependent.

Fix: every USB-serial stream takes a `device_path` constructor arg.
`twm/data_collection.py` populates from a top-level config dict pointing at
stable `/dev/serial/by-id/` symlinks:

```python
USB_SERIAL_DEVICES = {
    "audio_left":  "/dev/serial/by-id/usb-Arduino_LLC_Arduino_Uno_<serial-A>-if00",
    "audio_right": "/dev/serial/by-id/usb-Arduino_LLC_Arduino_Uno_<serial-B>-if00",
    # documented for completeness; FT not wired this round (FT-ii):
    # "ft_sensor": "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_<serial-C>-if00-port0",
}
```

The hardcoded module-level `DEV_SERIAL_PATH = '/dev/ttyUSB1'` in
`mms101.py` becomes a non-default constructor arg.

Discovery helper in `streams/audio/contact_mic.py`:
`list_arduino_devices()` greps `/dev/serial/by-id/usb-Arduino*` so first-time
labeling is `python -c "from streams.audio.contact_mic import list_arduino_devices; print(list_arduino_devices())"`.
One-time human step: physically unplug one Arduino, see which serial
disappears, label the remaining one. Documented in
`streams/audio/contact_mic/README.md`.

## 6. Audio integration into `twm/data_collection.py`

### 6.1 HDF5 schema additions (in `create_episode_file`)

```python
meta.attrs["audio_channels"]    = ["left", "right"]
meta.attrs["audio_sample_rate"] = 1000     # nominal Hz; actual timestamps in dataset

for name in ["left", "right"]:
    g = f.create_group(f"audio/{name}")
    g.create_dataset("timestamps", shape=(0,), maxshape=(None,), dtype=np.float64)
    g.create_dataset("samples",    shape=(0,), maxshape=(None,), dtype=np.int16)
```

`int16` fits the Arduino's 10-bit ADC range with headroom for future ADCs.
No BLOSC compression on these 1D streams (negligible benefit, low bytes total).

### 6.2 Sensor wiring (in `main()`)

```python
print("Initializing contact microphones...")
audio_streams = {
    "left":  ContactMicStream("left",  USB_SERIAL_DEVICES["audio_left"],  sample_rate_hz=1000),
    "right": ContactMicStream("right", USB_SERIAL_DEVICES["audio_right"], sample_rate_hz=1000),
}
for s in audio_streams.values():
    s.start()
```

Same `_try_start_*` graceful-fallback pattern as GelSight: if a mic is
unplugged, fall back to a `_DummyAudioStream` that returns empty
`flush_buffer()` and zero `get_recent_samples()`.

### 6.3 Per-tick coupling — none

Main loop does **not** poll audio per camera tick. Audio runs entirely
off-thread. The only main-loop interaction is `get_recent_samples(n=200)` for
the VU-meter — non-destructive, microsecond cost.

### 6.4 Episode-boundary drain

```python
def flush_audio_to_hdf5(f, audio_data):
    """audio_data: {channel_name: list[(timestamp, sample)]}"""
    for name, samples in audio_data.items():
        if not samples:
            continue
        ts   = np.array([s[0] for s in samples], dtype=np.float64)
        vals = np.array([s[1] for s in samples], dtype=np.int16)
        ds_t = f[f"audio/{name}/timestamps"]
        ds_s = f[f"audio/{name}/samples"]
        n = ds_t.shape[0]
        ds_t.resize(n + len(ts),   axis=0); ds_t[n:] = ts
        ds_s.resize(n + len(vals), axis=0); ds_s[n:] = vals
```

- `'s'` (start episode): call `flush_buffer()` on each audio stream to discard
  pre-episode samples. Mirrors the OptiTrack `flush_buffer(name)` pattern.
- `'e'` and `'q'`: harvest via `flush_buffer()`, write via
  `flush_audio_to_hdf5`.

### 6.5 `log_episode` adds an `audio` column

`"yes"` / `"no"` based on `any(len(v) > 0 ...)`. If a `dataset_log.csv`
already exists in the data dir, the implementation reads the existing rows,
adds the new column with empty values, and rewrites. Otherwise the new file
just has the new header.

### 6.6 Episode size estimate

2 mics × 1 kHz × (8 B timestamp + 2 B sample) = ~20 KB/s/mic = ~40 KB/s total.
A 30-min episode adds ~72 MB — small relative to camera bytes.

## 7. Preview UX (VU-meter)

`make_audio_panel(audio_streams, w=320, h=240)` — new helper in
`twm/preview.py`, sibling of `make_optitrack_panel`. Pure cv2.

For each of the two mics:
- Pull last ~200 samples (≈200 ms at 1 kHz) via `get_recent_samples(n=200)`.
- Compute `peak = np.max(np.abs(samples - mid))` (mid ≈ 512 for 10-bit
  Arduino ADC) and `rms = np.sqrt(np.mean((samples - mid)**2))`.
- Render two stacked horizontal bars:
  - Bar fill width = `clamp(peak / 512, 0, 1) × bar_width`.
  - Color: green < 70%, amber ≥ 70%, red ≥ 95% (clip warning).
  - Numeric overlay: `"left  peak=372 rms=180"` / `"right peak=410 rms=205"`.
  - Greyed "no signal" when stream is `_DummyAudioStream` or buffer empty.

Layout in `make_preview` (row 2, replaces `blank`):

```
Row 1: [cam0]    [cam1]    [cam2]    [optitrack]    — 4 × 320 = 1280px
Row 2: [gs_l]    [gs_l_d]  [gs_r]    [gs_r_d]    [audio-VU]
       240×240   240×240   240×240   240×240     320×240
```

`streams/audio/contact_mic/viz.py` (the moved standalone matplotlib viz)
remains unchanged — vendor-reference only, not on the hot path.

## 8. `data_collection.py` decomposition

Pure code move; no behavior change.

```
twm/
  data_collection.py    # entry point + main loop (~150 lines after split)
  episode_io.py         # create_episode_file, append_camera_frame(s_batch),
                        # flush_optitrack_to_hdf5, flush_audio_to_hdf5,
                        # HDF5Writer, log_episode, next_episode_number
  preview.py            # make_preview, make_optitrack_panel, make_audio_panel
  sensors.py            # _try_start_gelsight, _DummyGelSight, _DummyAudioStream,
                        # init_sensors(...) returning a sensors dataclass
                        # (rs_streams, gs_left, gs_right, audio_streams, optitrack)
```

`main()` shrinks to: parse args → `init_sensors()` → loop → key handler →
cleanup.

Constants hoisted to `data_collection.py` top:
- `STARTUP_TIMEOUT = 15.0`
- `RS_START_STAGGER_S = 0.5`
- `TIMING_REPORT_INTERVAL = 60`

`_DummyGelSight` promoted out of `main()` into `twm/sensors.py` as a
module-level class. `_DummyAudioStream` joins it.

## 9. Testing

- Existing tests must still pass after import-path rewrites:
  `tests/test_optitrack_stream.py`, `tests/test_realsense_stream.py`,
  `tests/test_hdf5_writer.py`, `tests/test_visualize.py`. These are the
  regression net for the file moves.
- New `tests/test_audio_stream.py` — `ContactMicStream` against a fake serial
  port (`serial.serial_for_url("loop://")`); write fake int lines, assert
  `flush_buffer()` returns the expected ordered tuples, assert thread shuts
  down cleanly on `stop()`. No real hardware required for CI.
- Extension to `tests/test_hdf5_writer.py` — round-trip the new
  `audio/{left,right}/{timestamps,samples}` datasets.
- Manual bench smoke test (documented):
  - Run `data_collection.py` with audio + 3 RealSense + 2 GS + OptiTrack.
  - Record 10 s episode.
  - Verify HDF5 has `audio/{left,right}/{timestamps,samples}` populated at
    ~10 000 samples/mic.
  - Verify VU-meters track when each mic is tapped.

Out of scope for testing this round: `MMS101FTStream` bench tests (FT path
not wired in), end-to-end "all sensors" CI test (too brittle without
hardware fixtures).

## 10. Risk register

- **Arduino sample rate is nominal, not guaranteed.** The `1000 Hz` figure is
  the loop rate of the firmware; serial RTT, USB scheduling, and Python's GIL
  can drop this. Storing per-sample timestamps (rather than assuming a fixed
  rate) means downstream tools that need exact timing have it.
- **Per-Arduino serial labeling is a manual one-time step.** If someone
  swaps Arduino boards without updating `USB_SERIAL_DEVICES`, "left" and
  "right" silently flip. Mitigated by labeling the physical USB cables and
  documenting in `streams/audio/contact_mic/README.md`.
- **`MMS101FTStream` was likely never working as a stream.** The Timer bug
  meant the reader loop never ran. The bugfix may surface latent issues
  (protocol timing, MMS101 hardware quirks) that weren't visible before.
  Acceptable for this round because FT is not wired into `twm/`; the failure
  mode if anyone tries to use it is a clean exception, not a silent bad
  recording.
- **External consumers of the renamed packages** (`probing_panda/`) will
  fail to import until their import lines are updated. This refactor updates
  the call sites it knows about; if there are out-of-tree consumers they
  will need a one-line fix.
