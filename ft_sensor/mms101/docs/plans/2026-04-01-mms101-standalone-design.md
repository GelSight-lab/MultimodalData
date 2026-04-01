# MMS101 Standalone Driver — Design Document

**Date:** 2026-04-01  
**Author:** Yuxiang  

---

## Goal

Create a self-contained, shareable driver for the NMB MMS101 6-axis force/torque sensor that others can drop into any Python project without depending on the MultimodalData package.

---

## Output Files

```
ft_sensor/mms101/
├── mms101.py           # Standalone driver class
├── example_mms101.py   # Usage example
└── README_mms101.md    # Documentation
```

---

## `mms101.py` — Class Design

### Constructor Parameters

| Parameter     | Default          | Description                                                  |
|---------------|------------------|--------------------------------------------------------------|
| `port`        | `'/dev/ttyUSB0'` | Serial port (`'COM5'` on Windows)                            |
| `medfilt_num` | `5`              | Median filter window size (set to 1 to disable)              |
| `log_csv`     | `None`           | Path to CSV file; if set, logs all readings with timestamps  |
| `verbose`     | `True`           | Print status messages to stdout                              |

### Public API

| Method           | Description                                                  |
|------------------|--------------------------------------------------------------|
| `start()`        | Initialize sensor, start background reading thread           |
| `stop()`         | Stop thread, close serial port, flush/close CSV              |
| `get_ft()`       | Return latest median-filtered `[Fx, Fy, Fz, Mx, My, Mz]`   |
| `tare()`         | Zero sensor against current reading                          |
| `get_ft_tared()` | Return `get_ft()` minus tare offset                          |

### Background Thread

- Runs a `while` loop at ~1 kHz (1000 µs interval)
- Reads raw 6-axis data from serial, stores in a `deque(maxlen=medfilt_num)`
- If `log_csv` is set, appends each reading with a UTC timestamp to CSV

### Serial Configuration

- Baudrate: 1,000,000
- Default port: `/dev/ttyUSB0` (Linux), `COM5` (Windows)
- Initialization sequence: boardSelect → powerSwitch → axisSelectAndIdle → bootload → setIntervalMeasure → setIntervalRestart

### Sensor Frequency

- Raw sampling: ~1000 Hz (INTERVAL_MEASURE_TIME = 1000 µs)
- Datasheet ADC conversion time: ~781 µs (~1280 Hz max)
- Effective output rate: same as raw rate; median filter does not reduce rate, only smooths values

### CSV Log Format

```
timestamp_utc, Fx, Fy, Fz, Mx, My, Mz
2026-04-01T12:00:00.001Z, 0.123, -0.045, 9.810, 0.001, -0.002, 0.000
...
```

---

## `example_mms101.py`

Shows:
1. Basic connection and reading loop (5 seconds)
2. Tare then read tared values
3. Optional CSV logging (via `log_csv` parameter)

---

## `README_mms101.md` — Sections

1. **Hardware Setup** — identify serial port, wiring overview
2. **⚠ FPC Connector Warning** — prominent section quoting datasheet, handling rules
3. **⏱ Warm-up Notice** — 3–5 min stabilization, 1–2 N drift expected before that
4. **Installation** — `pip install pyserial numpy`
5. **Quick Start** — minimal code snippet
6. **Constructor Options** — parameter table
7. **Sensor Frequency** — ~1000 Hz raw, median filter explanation
8. **Tare / Zeroing** — when and how to use
9. **CSV Logging** — how to enable

---

## Key Constraints

- No dependencies outside `pyserial` and `numpy`
- No relative imports — fully self-contained
- No changes to existing `mms101_stream.py` in the parent package
