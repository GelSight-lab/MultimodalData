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
