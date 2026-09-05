# TWM Sensor-Camera Arducam Integration Design

## Objective

Add the two Arducam B0578 RGB cameras mounted beside the tactile sensors to
the TWM recorder as additional streams. The recorder must work before their
physical left/right assignment is known, preserve stable camera identity
across reboots, retain the existing RealSense/GelSight/OptiTrack behavior, and
produce recordings whose structure and timing can be verified automatically
and against the attached hardware.

## Constraints and observed hardware

- Both Arducams report the same model, serial (`SN001`), supported modes, and
  USB vendor/product identity. Serial-number lookup cannot distinguish them.
- Their current stable topology paths are `1-12.1` (`/dev/video6`) and
  `1-12.3` (`/dev/video10`). `/dev/videoN` numbers are not stable identities.
- Each camera supports MJPEG at 640x480 and 30 FPS. This matches the recorder's
  canonical frame size and tick rate.
- The existing recorder assumes three RealSense streams, two GelSight streams,
  three OptiTrack rigid bodies, and a 30 FPS main timeline. Existing HDF5 files
  and downstream readers must remain valid.
- Only one RealSense is attached during development, so real-hardware
  verification of the new cameras must be runnable without requiring the
  complete legacy rig. Full writer integration can use simulated legacy
  streams plus both real Arducams.

## Chosen approach

Use the Linux USB topology path as each camera's acquisition identity and keep
physical position as separate metadata. The two streams are named `cam0` and
`cam1` while their sides are unknown. `twm/config/arducam.json` maps slots to
USB paths and optionally to `left` or `right`; a preview/identification command
will update that mapping after physical inspection.

This is preferred to hard-coded `/dev/videoN` numbers, which change across
boots, and to custom udev rules, which require root-level machine setup. If
unique camera serials become available later, the resolver can gain a serial
selector without changing the HDF5 schema or capture pipeline.

## Components

### Camera configuration and resolution

A small TWM-specific configuration defines exactly two logical slots. Each
entry contains:

- logical slot: `cam0` or `cam1`;
- USB topology identifier (`ID_PATH` or its stable `/dev/v4l/by-path` link);
- optional physical position: `left`, `right`, or `unknown`;
- requested width, height, FPS, and pixel format.

At startup, a resolver enumerates V4L2 capture devices, ignores metadata-only
nodes, and matches each configured topology path to one capture node. It fails
with a clear inventory when a configured path is absent, duplicated, or
ambiguous. It never silently falls back to a numeric device index.

`python -m twm.sensor_camera identify` opens both resolved streams side by side, labels
them by slot and topology path, and permits assigning left/right. It validates
that both sides occur exactly once before persisting a known mapping. Until
then, both cameras remain recordable with `position=unknown`.

### Arducam stream

A dedicated `ArducamVideoStream` uses OpenCV's V4L2 backend and configures
MJPEG, 640x480, 30 FPS, and a one-frame driver buffer. Its background thread
continuously grabs and decodes frames and atomically publishes the newest BGR
frame with the post-grab wall-clock timestamp. The API mirrors the existing
USB stream API (`start`, `stop`, `get_frame`, and
`get_frame_with_timestamp`) without inheriting GelSight-specific reduced
8-megapixel decoding assumptions.

Startup verifies the negotiated resolution, pixel format when observable, and
that a non-empty first frame arrives within a timeout. Runtime read failure is
reported with slot and topology identity. The stream shuts down and releases
its V4L2 handle deterministically.

### Capture and HDF5 writing

The two Arducam streams are additive inputs to `CaptureLoop`. At every 30 FPS
recorder tick, the loop snapshots the newest frame and its actual capture
timestamp for each slot. The frames travel through the existing non-blocking,
batched `HDF5Writer` queue with the RealSense and GelSight payloads, so all
modalities share the recorder's queue/drop/flush lifecycle.

New episode files contain:

```text
arducam/
  cam0/
    frames       uint8 [T, 480, 640, 3], chunked and BLOSC-compressed
    timestamps   float64 [T]
  cam1/
    frames       uint8 [T, 480, 640, 3], chunked and BLOSC-compressed
    timestamps   float64 [T]
```

Each camera group records acquisition identity and settings as attributes:
`usb_path`, `reported_serial`, `position`, `width`, `height`, `fps`, and
`pixel_format`. `position` may be `unknown`; pixel data remains identified by
slot and USB path and therefore does not need rewriting when the mapping is
learned.

The root metadata records the two-slot camera configuration. Existing function
callers remain valid by making Arducam configuration and frames optional. Old
recordings lack the `arducam` group and continue to be accepted. Existing
downstream preprocessors only consume named RealSense and GelSight groups, so
the additive group is ignored until a future release feature explicitly opts
into it.

Arducam timestamp fallback follows the GelSight precedent: if a test double or
legacy-compatible stream cannot provide a capture timestamp, the recorder tick
timestamp is written. Normal real-camera operation must supply a timestamp.

### Preview and operator workflow

The canonical live preview accepts an optional pair of sensor-camera frames.
When present, it adds a third 240-pixel row containing the two 320x240
thumbnails. Labels show `cam0`/`cam1`, topology path, and known physical side.
When absent, the existing 1280x480 layout is byte-compatible in shape and
behavior for current callers and offline renderers.

Normal operation is:

1. Run the identification preview whenever USB wiring changes.
2. Optionally assign left/right; leaving both unknown is permitted.
3. Start the normal TWM recorder. Both sensor cameras are required by default
   when configured, and their readiness is checked before episode creation.
4. Record, stop, and close through the existing keyboard workflow. Both camera
   handles are also stopped in the `finally` cleanup path.

`python -m twm.sensor_camera verify --duration 5 --output <path>` is the timed,
headless verification command. It uses the same stream, schema, and batch
writer code, records both Arducams without requiring RealSense, GelSight, or
OptiTrack hardware, and validates its output. This is a diagnostic entry point,
not a second recording implementation.

## Failure handling

- Duplicate factory serials are expected and never used as the primary key.
- A missing or ambiguous configured USB topology path prevents recording and
  prints the discovered V4L2 inventory.
- Failure to negotiate or receive valid 640x480 frames prevents recording.
- Queue saturation retains existing behavior: drop the entire multimodal tick,
  count it, and report it, preserving row alignment among per-tick datasets.
- An unavailable Arducam does not generate silent black frames. Sensor-camera
  data is either explicitly disabled for legacy operation or complete for both
  configured slots.
- Periodic HDF5 metadata flushes and clean shutdown cover the new datasets
  through the existing writer lifecycle.
- Unknown physical side is valid metadata, not an error. An invalid mapping
  (duplicate left, duplicate right, unsupported value) is rejected.

## Testing and verification

Automated tests cover:

- topology-path inventory parsing and resolution, including duplicate serials,
  missing paths, metadata nodes, and ambiguous matches;
- mapping validation for two unknown slots and for one-to-one left/right;
- HDF5 creation with and without Arducams, exact dataset shapes/dtypes,
  attributes, compression-compatible writes, and backward compatibility;
- batched appends, timestamp propagation/fallback, equal row counts, and known
  pixel values for both camera slots;
- `CaptureLoop` behavior with stream doubles and clean shutdown;
- preview dimensions and labels with and without sensor-camera frames;
- the headless verifier's pass/fail checks using deterministic stream doubles.

Real attached-hardware verification records both cameras concurrently for a
short timed interval and reopens the resulting HDF5. It must prove:

- both topology paths resolve to distinct capture nodes;
- each dataset has nonzero and equal frame count;
- every frame has shape 480x640x3 and nonzero variance;
- timestamps are finite, monotonic nondecreasing, and span the recording;
- measured distinct-frame cadence is compatible with the requested 30 FPS;
- the two streams are not byte-identical; and
- the file closes cleanly and can be read in a new process.

Because only one of the three configured RealSense devices is currently
attached, complete legacy hardware verification is not claimed. A test harness
exercises `CaptureLoop`, `HDF5Writer`, and episode finalization with simulated
RealSense, GelSight, and OptiTrack inputs plus both real Arducams. Repeating the
normal interactive run on the complete rig remains an operator acceptance
check, not a prerequisite for proving the new pipeline records correctly on
this host.

## Non-goals

- Calibrating Arducam intrinsics or extrinsics.
- Publishing sensor-camera videos in the current React release pipeline.
- Changing the three-camera RealSense schema or existing calibration mapping.
- Guessing left/right from USB port order or scene appearance.
- Installing machine-global udev rules or modifying camera firmware.
