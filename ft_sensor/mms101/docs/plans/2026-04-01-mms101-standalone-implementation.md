# MMS101 Standalone Driver Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a self-contained MMS101 force/torque sensor driver in `ft_sensor/mms101/` with no dependencies on the parent MultimodalData package.

**Architecture:** Single driver class `MMS101` in `mms101.py` with a continuous background thread for reading, optional CSV logging, median filtering, and tare. Accompanied by a usage example and README documentation.

**Tech Stack:** Python 3, `pyserial`, `numpy`, `threading`, `csv`, `collections.deque`

---

### Task 1: Create `mms101.py` — class skeleton and serial init

**Files:**
- Create: `ft_sensor/mms101/mms101.py`

**Step 1: Create the file with imports and constants**

```python
import threading
import time
import csv
import os
from collections import deque
import numpy as np
import serial

DEV_SERIAL_PATH_LINUX = '/dev/ttyUSB0'
DEV_SERIAL_PATH_WINDOWS = 'COM5'
SERIAL_BAUDRATE = 1_000_000
INTERVAL_MEASURE_TIME = 1000       # microseconds → ~1000 Hz
INTERVAL_RESTART_TIME = 1_000_000  # microseconds → temperature update every ~1 min
```

**Step 2: Write the class constructor**

```python
class MMS101:
    def __init__(self, port=None, medfilt_num=5, log_csv=None, verbose=True):
        """
        MMS101 6-axis force/torque sensor driver.

        Args:
            port (str): Serial port. Defaults to '/dev/ttyUSB0' (Linux) or 'COM5' (Windows).
            medfilt_num (int): Median filter window size. Set to 1 to disable filtering.
            log_csv (str): Path to CSV log file. If None, logging is disabled.
            verbose (bool): Print status messages.
        """
        if port is None:
            port = DEV_SERIAL_PATH_WINDOWS if os.name == 'nt' else DEV_SERIAL_PATH_LINUX

        self.port = port
        self.baudrate = SERIAL_BAUDRATE
        self.medfilt_num = medfilt_num
        self.log_csv = log_csv
        self.verbose = verbose

        self._serial = None
        self._serial_open = False
        self._board_selected = False

        self._data = [0.0] * 6        # latest raw reading [Fx,Fy,Fz,Mx,My,Mz]
        self._data_que = deque(maxlen=medfilt_num)
        self._zeros = [0.0] * 6       # tare offsets

        self._thread = None
        self._running = False

        self._csv_file = None
        self._csv_writer = None

        self._elaps_time = 0.0
        self._interval_measure = INTERVAL_MEASURE_TIME
        self._interval_restart = INTERVAL_RESTART_TIME
```

**Step 3: Add serial open/close helpers**

```python
    def _serial_port_open(self):
        self._serial = serial.Serial(self.port, self.baudrate, timeout=2)
        self._serial_open = True
        if self.verbose:
            print(f"[MMS101] Serial port opened: {self.port}")

    def _serial_port_close(self):
        if self._serial_open:
            self._stop_measure()
            self._serial.close()
            self._serial_open = False
            if self.verbose:
                print("[MMS101] Serial port closed.")
```

**Step 4: Commit**

```bash
git add ft_sensor/mms101/mms101.py
git commit -m "feat: mms101 standalone - class skeleton and serial init"
```

---

### Task 2: Add low-level serial communication methods

**Files:**
- Modify: `ft_sensor/mms101/mms101.py`

**Step 1: Add `_read_all` (reads a response packet from the sensor)**

```python
    def _read_all(self):
        j = self._serial.read(2)
        if j[0] == 0 and j[1] > 0:
            data = self._serial.read(j[1])
            j = j + data
        return j
```

**Step 2: Add initialization command sequence**

```python
    def _board_select(self):
        self._serial.write(bytes([0x54, 0x02, 0x10, 0x00]))
        self._serial.flush()
        rsp = self._read_all()
        if rsp[0] != 0:
            raise RuntimeError("MMS101: Board Select failed")
        self._board_selected = True

    def _power_switch(self):
        self._serial.write([0x54, 0x03, 0x36, 0x00, 0xFF])
        self._serial.flush()
        rsp = self._read_all()
        if rsp[0] != 0 and rsp[0] != 0x10:
            raise RuntimeError("MMS101: Power Switch 1-2 failed")
        self._serial.write([0x54, 0x03, 0x36, 0x05, 0xFF])
        self._serial.flush()
        rsp = self._read_all()
        if rsp[0] != 0 and rsp[0] != 0x10:
            raise RuntimeError("MMS101: Power Switch 4-5 failed")

    def _axis_select_and_idle(self):
        for axis in range(6):
            self._serial.write([0x54, 0x02, 0x1C, axis])
            self._serial.flush()
            rsp = self._read_all()
            if rsp[0] != 0 and rsp[0] != 0x10:
                raise RuntimeError(f"MMS101: Axis Select failed for axis {axis}")
            self._serial.write([0x53, 0x02, 0x57, 0x94])
            self._serial.flush()
            rsp = self._read_all()
            if rsp[0] != 0 and rsp[0] != 0x10:
                raise RuntimeError(f"MMS101: IDLE failed for axis {axis}")
        time.sleep(0.01)

    def _bootload(self):
        self._serial.write([0x54, 0x01, 0xB0])
        self._serial.flush()
        rsp = self._read_all()
        if rsp[0] != 0:
            raise RuntimeError("MMS101: Bootload failed")

    def _set_interval_measure(self, interval_us):
        self._interval_measure = interval_us
        b1 = (interval_us >> 16) & 0xFF
        b2 = (interval_us >> 8) & 0xFF
        b3 = interval_us & 0xFF
        self._serial.write([0x54, 0x04, 0x43, b1, b2, b3])
        self._serial.flush()
        rsp = self._read_all()
        if rsp[0] != 0:
            raise RuntimeError("MMS101: Set Interval Measure failed")

    def _set_interval_restart(self, interval_us):
        self._interval_restart = interval_us
        b1 = (interval_us >> 16) & 0xFF
        b2 = (interval_us >> 8) & 0xFF
        b3 = interval_us & 0xFF
        self._serial.write([0x54, 0x04, 0x44, b1, b2, b3])
        self._serial.flush()
        rsp = self._read_all()
        if rsp[0] != 0:
            raise RuntimeError("MMS101: Set Interval Restart failed")

    def _start_measure(self):
        self._serial.write([0x54, 0x02, 0x23, 0x00])
        j = self._serial.read(2)
        if len(j) != 2 or j[0] != 0 or j[1] != 0:
            raise RuntimeError("MMS101: Start Measure failed")
        time.sleep(0.01)

    def _stop_measure(self):
        if self._board_selected:
            self._serial.write([0x54, 0x01, 0x33])
            self._board_selected = False
```

**Step 3: Add raw data read**

```python
    def _read_data_raw(self):
        """Sync to packet header and read one 23-byte data packet."""
        prev = b'\xff'
        while True:
            curr = self._serial.read(1)
            if prev == b'\x00' and curr == b'\x17':
                break
            prev = curr
        return self._serial.read(23)

    def _parse_data(self, rdata):
        """Parse raw 23-byte packet into [Fx, Fy, Fz, Mx, My, Mz] in N / N·m."""
        if len(rdata) != 23 or rdata[0] != 0x80 or rdata[1] != 0x00:
            return None
        data = [0.0] * 6
        self._elaps_time += ((rdata[20] << 16) + (rdata[21] << 8) + rdata[22]) / 1_000_000
        for axis in range(6):
            val = (rdata[axis*3+2] << 16) + (rdata[axis*3+3] << 8) + rdata[axis*3+4]
            if val >= 0x00800000:
                val -= 0x1000000
            data[axis] = val
        # Scale: forces in N (divide by 1000), moments in N·m (divide by 100000)
        data[0] /= 1000
        data[1] /= 1000
        data[2] /= 1000
        data[3] /= 100000
        data[4] /= 100000
        data[5] /= 100000
        return data
```

**Step 4: Commit**

```bash
git add ft_sensor/mms101/mms101.py
git commit -m "feat: mms101 standalone - serial communication and packet parsing"
```

---

### Task 3: Add background thread, CSV logging, and public API

**Files:**
- Modify: `ft_sensor/mms101/mms101.py`

**Step 1: Add the background thread loop**

```python
    def _thread_loop(self):
        while self._running:
            rdata = self._read_data_raw()
            parsed = self._parse_data(rdata)
            if parsed is not None:
                self._data = parsed
                self._data_que.append(parsed)
                if self._csv_writer is not None:
                    ts = time.time()
                    self._csv_writer.writerow([f"{ts:.6f}"] + [f"{v:.6f}" for v in parsed])
```

**Step 2: Add `start()` and `stop()`**

```python
    def start(self):
        """Initialize sensor and start background reading thread."""
        self._serial_port_open()
        self._board_select()
        self._power_switch()
        self._axis_select_and_idle()
        self._bootload()
        self._set_interval_measure(INTERVAL_MEASURE_TIME)
        self._set_interval_restart(INTERVAL_RESTART_TIME)
        self._start_measure()

        if self.log_csv is not None:
            self._csv_file = open(self.log_csv, 'w', newline='')
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(['timestamp', 'Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'])

        self._running = True
        self._thread = threading.Thread(target=self._thread_loop, daemon=True)
        self._thread.start()
        if self.verbose:
            print("[MMS101] Started. NOTE: Allow 3-5 minutes warm-up before recording data.")

    def stop(self):
        """Stop background thread and release resources."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._csv_file is not None:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
        self._serial_port_close()
        if self.verbose:
            print("[MMS101] Stopped.")
```

**Step 3: Add public read/tare methods**

```python
    def get_ft(self):
        """
        Return latest median-filtered [Fx, Fy, Fz, Mx, My, Mz] in N / N·m.
        Filter window = medfilt_num (set to 1 in constructor to disable).
        """
        if len(self._data_que) == 0:
            return self._data
        return list(np.median(np.array(self._data_que), axis=0))

    def tare(self, n_samples=50):
        """
        Zero the sensor. Collects n_samples readings and sets them as the offset.
        Call this after warm-up (3-5 min) with no load applied.
        """
        if self.verbose:
            print(f"[MMS101] Taring over {n_samples} samples...")
        samples = []
        for _ in range(n_samples):
            time.sleep(0.002)
            samples.append(self.get_ft())
        self._zeros = list(np.mean(np.array(samples), axis=0))
        if self.verbose:
            print(f"[MMS101] Tare complete. Offsets: {[f'{v:.4f}' for v in self._zeros]}")

    def get_ft_tared(self):
        """Return get_ft() minus tare offsets."""
        ft = self.get_ft()
        return [ft[i] - self._zeros[i] for i in range(6)]
```

**Step 4: Add `__del__` for safety**

```python
    def __del__(self):
        if self._running:
            self.stop()
```

**Step 5: Commit**

```bash
git add ft_sensor/mms101/mms101.py
git commit -m "feat: mms101 standalone - background thread, CSV logging, public API"
```

---

### Task 4: Create `example_mms101.py`

**Files:**
- Create: `ft_sensor/mms101/example_mms101.py`

**Step 1: Write the example**

```python
"""
MMS101 Force/Torque Sensor — Example Usage

IMPORTANT: Allow 3-5 minutes warm-up before recording data.
           Expect 1-2 N drift during warm-up.
"""
import time
from mms101 import MMS101

# --- Basic usage (no logging) ---
sensor = MMS101(
    port='/dev/ttyUSB0',   # Change to 'COM5' on Windows
    medfilt_num=5,         # Median filter over 5 samples (set to 1 to disable)
    verbose=True,
)
sensor.start()

print("Warming up... wait 3-5 minutes before recording.")
time.sleep(5)  # In real use, wait 3-5 minutes

# Zero the sensor with no load applied
sensor.tare()

# Read loop
print("Reading for 5 seconds...")
t_end = time.time() + 5
while time.time() < t_end:
    ft = sensor.get_ft_tared()
    print(f"Fx={ft[0]:.3f}N  Fy={ft[1]:.3f}N  Fz={ft[2]:.3f}N  "
          f"Mx={ft[3]:.5f}Nm  My={ft[4]:.5f}Nm  Mz={ft[5]:.5f}Nm")
    time.sleep(0.05)

sensor.stop()

# --- With CSV logging ---
# sensor = MMS101(port='/dev/ttyUSB0', log_csv='ft_data.csv', verbose=True)
# sensor.start()
# time.sleep(300)   # warm-up
# sensor.tare()
# time.sleep(60)    # record 60 seconds of data
# sensor.stop()
# # Data saved to ft_data.csv
```

**Step 2: Commit**

```bash
git add ft_sensor/mms101/example_mms101.py
git commit -m "feat: mms101 standalone - usage example"
```

---

### Task 5: Create `README_mms101.md`

**Files:**
- Create: `ft_sensor/mms101/README_mms101.md`

**Step 1: Write the README**

Content sections (write in full):

```markdown
# MMS101 6-Axis Force/Torque Sensor — Driver

Standalone Python driver for the [NMB MMS101](https://cdn.nmbtc.com/uploads/2023/02/mms101_datasheet_en_rev5.0.pdf) miniature 6-axis force/torque sensor.

---

## ⚠ FPC Connector Warning

**The FPC (Flexible Printed Circuit) connector is fragile. Read this before handling.**

From the MMS101 datasheet:

> "The FPC must NOT be strongly pulled in a lateral or the upper direction while the sensor body is fixed with screws. Otherwise, load is applied to the base of the FPC, and the wiring on the FPC might be snapped."

> "In the FPC termination part, a level difference exists between the FPC and the reinforcing plate. Bending the FPC at this level difference part could cut the wiring on the FPC."

> "Do not bend the FPC at a sharp angle or pull it hard so that the load is concentrated. Otherwise, the wiring on the FPC may be broken, resulting in operation failure."

> "Do not mount the product on moving parts that are bent repeatedly."

**Rules:**
- Never pull the FPC laterally or upward while the sensor is screwed down
- Never bend the FPC at the junction between the FPC and the reinforcing plate
- Fix the PCB connected to the FPC with a screw so the FPC is not repeatedly flexed
- Insert and remove cables only when the FPC is secured to the attachment

---

## ⏱ Warm-Up Notice

**Wait 3–5 minutes after starting before recording data.**

From the datasheet:

> "Immediately after AD converter starts, the built-in AFE heats up and deforms the structure. This causes the output to drift. Therefore, it is recommended to wait for stabilization about 5 min before acquiring data."

Expect **1–2 N of drift** during the initial warm-up period. Always call `tare()` after warm-up and before recording.

---

## Installation

```bash
pip install pyserial numpy
```

Copy `mms101.py` into your project directory.

---

## Hardware Setup

1. Connect the sensor to the controller board via the FPC (see warning above)
2. Connect the controller board to your PC via USB
3. Identify the serial port:
   - **Linux:** Run `ls /dev/ttyUSB*` before and after plugging in. New entry is your port (e.g. `/dev/ttyUSB0`)
   - **Windows:** Check Device Manager → Ports (COM & LPT). Note the `COMx` number.
4. On Linux, you may need to add yourself to the `dialout` group:
   ```bash
   sudo usermod -aG dialout $USER
   # Log out and back in for this to take effect
   ```

---

## Quick Start

```python
from mms101 import MMS101
import time

sensor = MMS101(port='/dev/ttyUSB0')  # 'COM5' on Windows
sensor.start()

time.sleep(300)   # Wait 3-5 minutes for warm-up
sensor.tare()     # Zero with no load applied

ft = sensor.get_ft_tared()
print(f"Fx={ft[0]:.3f}N, Fy={ft[1]:.3f}N, Fz={ft[2]:.3f}N")

sensor.stop()
```

---

## Constructor Options

| Parameter     | Default          | Description                                                  |
|---------------|------------------|--------------------------------------------------------------|
| `port`        | `/dev/ttyUSB0`   | Serial port. Use `'COM5'` on Windows.                        |
| `medfilt_num` | `5`              | Median filter window size. Set to `1` to disable.            |
| `log_csv`     | `None`           | Path to CSV file. If set, all readings are logged.           |
| `verbose`     | `True`           | Print status messages.                                       |

---

## Sensor Frequency

The sensor samples at approximately **1000 Hz** (1000 µs interval, configurable in code).

The datasheet specifies a minimum ADC conversion time of ~781 µs (~1280 Hz theoretical max).

The median filter (`medfilt_num`) does **not** reduce the sampling rate — it smooths values over the last N samples. A window of 5 means each call to `get_ft()` returns the median of the 5 most recent readings.

Output format: `[Fx, Fy, Fz, Mx, My, Mz]`
- Forces in **N** (Newtons), range ±40 N (rated), ±200 N (absolute max)
- Moments in **N·m**, range ±0.4 N·m (rated), ±1.8 N·m (absolute max)

---

## Tare / Zeroing

```python
sensor.tare()              # Average 50 samples as zero offset
ft = sensor.get_ft_tared() # Returns reading minus tare offset
ft_raw = sensor.get_ft()   # Returns raw (un-tared) reading
```

Always tare **after warm-up** and with **no load applied** to the sensor.

---

## CSV Logging

```python
sensor = MMS101(port='/dev/ttyUSB0', log_csv='ft_data.csv')
sensor.start()
# ... run your experiment ...
sensor.stop()  # Flushes and closes the CSV file
```

CSV format:
```
timestamp,Fx,Fy,Fz,Mx,My,Mz
1743480000.123456,0.001234,-0.000512,9.810000,0.000010,-0.000020,0.000003
...
```

`timestamp` is Unix time in seconds (UTC).
```

**Step 2: Commit**

```bash
git add ft_sensor/mms101/README_mms101.md
git commit -m "docs: mms101 standalone - README with setup, FPC warning, warm-up, usage"
```

---

## Final Checklist

- [ ] `mms101.py` imports only `serial`, `numpy`, `threading`, `csv`, `os`, `time`, `collections` — no package-relative imports
- [ ] `start()` prints warm-up reminder
- [ ] `stop()` always flushes CSV and closes serial even if called multiple times
- [ ] README FPC warning section is prominent (near the top)
- [ ] Example script runs standalone from `ft_sensor/mms101/` directory with `python example_mms101.py`
