#!/usr/bin/env python3
"""
Script to find serial numbers of connected GelSight Mini sensors.
"""

import pyudev


def find_gelsight_sensors():
    """Find all connected GelSight Mini sensors and their serial numbers."""
    context = pyudev.Context()
    devices = context.list_devices(subsystem="video4linux")

    gelsight_devices = {}

    for device in devices:
        device_name = device.get("ID_MODEL", "")
        serial = device.get("ID_SERIAL", "")

        # Filter for GelSight Mini devices
        if "GelSight" in device_name or "GelSight" in serial:
            video_num = int(device.sys_number)

            # Extract just the serial number (remove manufacturer prefix if present)
            serial_short = serial.split("_")[-1] if "_" in serial else serial

            # Group by serial number (each camera has multiple /dev/video entries)
            if serial_short not in gelsight_devices:
                gelsight_devices[serial_short] = {
                    "serial": serial,
                    "model": device_name,
                    "video_devices": [],
                    "device_node": device.device_node,
                }

            gelsight_devices[serial_short]["video_devices"].append(video_num)

    return gelsight_devices


def main():
    print("Searching for GelSight Mini sensors...\n")

    sensors = find_gelsight_sensors()

    if not sensors:
        print("No GelSight Mini sensors found.")
        print("\nTroubleshooting:")
        print("- Make sure the sensors are plugged in")
        print("- Check USB connection")
        print("- Run: v4l2-ctl --list-devices")
        return

    print(f"Found {len(sensors)} GelSight Mini sensor(s):\n")
    print("=" * 70)

    for i, (serial_short, info) in enumerate(sorted(sensors.items()), 1):
        print(f"\nSensor #{i}:")
        print(f"  Serial Number: {serial_short}")
        print(f"  Full ID:       {info['serial']}")
        print(f"  Model:         {info['model']}")
        print(f"  Video Devices: {sorted(info['video_devices'])}")
        print(f"  Primary:       /dev/video{min(info['video_devices'])}")

    print("\n" + "=" * 70)
    print("\nTo use in your code:")
    print("```python")
    print("from camera_stream.usb_video_stream import USBVideoStream")
    print()

    for i, serial_short in enumerate(sorted(sensors.keys()), 1):
        print(f"# Sensor #{i}")
        print(f'camera_{i} = USBVideoStream(serial="{serial_short}", resolution=(640, 480))')
        print(f"camera_{i}.start()")
        print()
    print("```")


if __name__ == "__main__":
    main()
