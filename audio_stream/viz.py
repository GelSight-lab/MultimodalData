import serial
import matplotlib.pyplot as plt
from collections import deque
import time

PORT = "/dev/ttyUSB1"
BAUD = 115200

BUFFER_SIZE = 1000
PLOT_INTERVAL = 0.01  # seconds (~100 FPS)

ser = serial.Serial(PORT, BAUD, timeout=1)

data = deque([0]*BUFFER_SIZE, maxlen=BUFFER_SIZE)

plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(data)

ax.set_ylim(0, 1023)
ax.set_title("Contact Mic Signal")

last_plot = time.time()

print("Streaming... Ctrl+C to stop.")

try:
    while True:
        # read as fast as possible
        raw = ser.readline().decode('utf-8', errors='ignore').strip()

        if raw:
            try:
                value = int(raw)
                data.append(value)
            except:
                pass

        # update plot at limited FPS
        if time.time() - last_plot > PLOT_INTERVAL:
            line.set_ydata(data)
            line.set_xdata(range(len(data)))
            plt.pause(0.001)
            last_plot = time.time()

except KeyboardInterrupt:
    ser.close()