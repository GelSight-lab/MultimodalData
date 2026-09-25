"""Live wrist-camera tuning: see the picture, the histogram and the byte cost
while you turn the knobs, then save what you chose.

Tuning by editing JSON and re-recording is how the last set of values ended
up washed out: they were chosen to hit a mean brightness, which lifting the
black point and the mid-tones will do while flattening the picture. The two
numbers that matter are on screen here — the contrast the picture actually
has, and the kilobytes each frame costs, because the wrist stream competes
with everything else for the disk.

    python -m twm.sensor_camera tune                 # both cameras, same values
    python -m twm.sensor_camera tune --camera cam0   # one of them
"""
from __future__ import annotations

import re
import subprocess
from typing import Dict, Optional, Tuple

# key -> (control, step). Upper case steps down.
KEY_BINDINGS = {
    "e": ("exposure_time_absolute", 10),
    "b": ("brightness", 5),
    "g": ("gamma", 25),
    "c": ("contrast", 5),
    "s": ("sharpness", 5),
    "w": ("white_balance_temperature", 100),
    "f": ("focus_absolute", 5),
}
TOGGLES = {"a": ("auto_exposure", 1, 3), "W": ("white_balance_automatic", 0, 1),
           "F": ("focus_automatic_continuous", 0, 1)}
# Each automatic must be written before the value it locks: the manual value
# is read-only while its automatic is on.
SAVE_ORDER = ("exposure_dynamic_framerate", "power_line_frequency",
              "auto_exposure", "exposure_time_absolute", "gamma", "brightness",
              "contrast", "sharpness", "white_balance_automatic",
              "white_balance_temperature", "focus_automatic_continuous",
              "focus_absolute")

_CTRL_RE = re.compile(
    r"^\s*(\w+)\s+0x[0-9a-f]+\s+\((\w+)\)\s*:\s*(.*)$", re.M)


def parse_control_ranges(text: str) -> Dict[str, Tuple[int, int, int]]:
    """{control: (min, max, step)} from `v4l2-ctl --list-ctrls` output.

    Read from the camera rather than hardcoded: the ranges differ per model,
    and a tuner that clamps to the wrong ones writes values the driver
    silently ignores.
    """
    out = {}
    for name, _kind, rest in _CTRL_RE.findall(text):
        fields = dict(re.findall(r"(\w+)=(-?\d+)", rest))
        if "min" in fields and "max" in fields:
            out[name] = (int(fields["min"]), int(fields["max"]),
                         int(fields.get("step", 1)))
    return out


def read_control_ranges(device: str) -> Dict[str, Tuple[int, int, int]]:
    r = subprocess.run(["v4l2-ctl", "-d", str(device), "--list-ctrls"],
                       capture_output=True, text=True, timeout=10)
    return parse_control_ranges(r.stdout)


def read_defaults(device: str) -> Dict[str, int]:
    r = subprocess.run(["v4l2-ctl", "-d", str(device), "--list-ctrls"],
                       capture_output=True, text=True, timeout=10)
    out = {}
    for name, _kind, rest in _CTRL_RE.findall(r.stdout):
        m = re.search(r"default=(-?\d+)", rest)
        if m:
            out[name] = int(m.group(1))
    return out


class ControlModel:
    """The values being tuned, and what each key does to them."""

    def __init__(self, values: Dict[str, int], ranges: Dict[str, Tuple[int, int, int]],
                 defaults: Optional[Dict[str, int]] = None, fps: int = 0):
        self.values = dict(values)
        self.ranges = dict(ranges)
        self.defaults = dict(defaults or {})
        # Exposure is in units of 0.1 ms, and a frame period at `fps` is the
        # ceiling: past it the camera delivers fewer frames than the recorder
        # asks for, which is a worse trade than a darker picture.
        self.exposure_cap = int(10_000 / fps) if fps else None

    def _clamp(self, name: str, value: int) -> int:
        lo, hi, _ = self.ranges.get(name, (-10 ** 9, 10 ** 9, 1))
        if name == "exposure_time_absolute" and self.exposure_cap:
            hi = min(hi, self.exposure_cap)
        return max(lo, min(hi, value))

    def press(self, key: str):
        """Apply one keystroke. Returns (control, new value) or None."""
        if key in TOGGLES:
            name, a, b = TOGGLES[key]
            cur = self.values.get(name, a)
            self.values[name] = b if cur == a else a
            return name, self.values[name]
        low = key.lower()
        if low not in KEY_BINDINGS:
            return None
        name, step = KEY_BINDINGS[low]
        if name not in self.values:
            return None
        # Returns the value the control now holds, changed or not: a key that
        # is bound but at its limit still answers, so the display shows the
        # limit rather than going quiet.
        self.values[name] = self._clamp(
            name, self.values[name] + (step if key == low else -step))
        return name, self.values[name]

    def reset(self) -> None:
        for name, value in self.defaults.items():
            if name in self.values:
                self.values[name] = value

    def to_config(self) -> Dict[str, int]:
        """The values, ordered so each automatic precedes what it locks."""
        ordered = {k: self.values[k] for k in SAVE_ORDER if k in self.values}
        ordered.update({k: v for k, v in self.values.items() if k not in ordered})
        return ordered


def contrast_of(image) -> Tuple[float, float, float]:
    """(mean, p1, p99). A washed-out picture has a high p1: its blacks are
    grey. Chasing the mean alone is what produced one."""
    import numpy as np
    p1, p99 = np.percentile(image, [1, 99])
    return float(image.mean()), float(p1), float(p99)


HELP = [
    "e/E exposure   b/B brightness   g/G gamma   c/C contrast",
    "s/S sharpness  w/W white-bal    f/F focus",
    "a auto-exposure   r reset to camera defaults",
    "S save to config   q quit without saving",
]


def _apply(devices, name, value):
    from camera_stream.arducam_video_stream import _v4l2_set_control
    for d in devices:
        _v4l2_set_control(d, name, value)


def tune(config_path, camera=None, fps=30) -> int:
    """Live tuning window. Both cameras take the same values, always: a pair
    that disagrees puts a per-camera bias into the dataset that no downstream
    step can tell from the scene."""
    import json
    import time

    import cv2
    import numpy as np

    from camera_stream.arducam_video_stream import ArducamVideoStream
    from twm.recorder.frames import decode_arducam
    from twm.sensor_camera import load_config, resolve_slots

    slots = load_config(config_path)
    resolved = [c for c in resolve_slots(slots)
                if camera is None or c.slot == camera]
    if not resolved:
        print(f"no camera {camera!r} in {config_path}")
        return 1

    devices = [c.device for c in resolved]
    ranges = read_control_ranges(devices[0])
    defaults = read_defaults(devices[0])
    start = dict(resolved[0].config.controls) or dict(defaults)
    model = ControlModel(start, ranges, defaults, fps=fps)
    for name, value in model.to_config().items():
        _apply(devices, name, value)

    streams = [ArducamVideoStream(c.config, c.device, encoding="mjpeg")
               for c in resolved]
    for st in streams:
        st.start()
    win = "wrist camera tuning"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    saved = False
    try:
        last, shown = {}, 0.0
        while True:
            tiles, notes = [], []
            for c, st in zip(resolved, streams):
                buf, ts = st.get_frame_with_timestamp(timeout=2.0)
                img = decode_arducam(buf)
                mean, p1, p99 = contrast_of(img)
                notes.append(f"{c.slot} {c.config.position}: mean {mean:5.1f}  "
                             f"p1 {p1:3.0f}  p99 {p99:3.0f}  {len(buf)/1024:5.1f} KB")
                tiles.append(cv2.resize(img, (480, 360)))
            panel = np.hstack(tiles)
            bar = np.zeros((150, panel.shape[1], 3), np.uint8)
            lines = notes + [""] + [
                "  ".join(f"{k}={model.values[k]}" for k in
                          ("auto_exposure", "exposure_time_absolute", "gamma",
                           "brightness", "contrast") if k in model.values)] + HELP
            for i, text in enumerate(lines[:9]):
                cv2.putText(bar, text, (8, 16 + i * 16), cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, (200, 220, 200), 1, cv2.LINE_AA)
            cv2.imshow(win, np.vstack([panel, bar]))

            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                model.reset()
                for name, value in model.to_config().items():
                    _apply(devices, name, value)
                continue
            if key == ord("S"):
                raw = json.loads(Path(config_path).read_text())
                for entry in raw["cameras"]:
                    if camera is None or entry["slot"] == camera:
                        entry["controls"] = model.to_config()
                Path(config_path).write_text(json.dumps(raw, indent=2) + "\n")
                saved = True
                print(f"saved to {config_path}")
                continue
            if key == 255:
                continue
            hit = model.press(chr(key))
            if hit:
                _apply(devices, *hit)
    finally:
        for st in streams:
            st.stop()
        cv2.destroyAllWindows()
    if not saved:
        print("nothing saved (press S in the window to save)")
    return 0
