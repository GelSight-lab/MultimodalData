#!/usr/bin/env python3
"""
Example script for streaming GelSight Mini images using USBVideoStream.
Supports switching between raw image and difference image modes.

Controls:
- 'd': Switch to difference mode
- 'r': Switch to raw mode
- 's': Set/reset reference image (for difference mode)
- 'q': Quit
"""

import argparse
import cv2
import numpy as np
from camera_stream.usb_video_stream import USBVideoStream


def compute_difference_image(current_frame, reference_frame):
    """Compute the difference between current frame and reference frame."""
    if reference_frame is None:
        return current_frame

    # Convert to float for better precision
    diff = current_frame.astype(np.float32) - reference_frame.astype(np.float32)

    # Normalize to 0-255 range (adding 128 to center around middle gray)
    diff_normalized = np.clip(diff + 128, 0, 255).astype(np.uint8)

    return diff_normalized


def main():
    parser = argparse.ArgumentParser(description='Stream GelSight Mini images')
    parser.add_argument('--serial', type=str, default='28YUTA6Z',
                        help='GelSight sensor serial number (default: 28YUTA6Z)')
    args = parser.parse_args()

    # Initialize the USB video stream for GelSight Mini
    camera = USBVideoStream(serial=args.serial, resolution=(640, 480), format="BGR")

    # Start streaming
    print("Starting GelSight Mini stream...")
    camera.start()

    # Mode settings
    mode = "raw"  # "raw" or "difference"
    reference_frame = None

    print("\n=== GelSight Mini Stream ===")
    print("Controls:")
    print("  'd' - Switch to difference mode")
    print("  'r' - Switch to raw mode")
    print("  's' - Set/reset reference image")
    print("  'q' - Quit")
    print("\nCurrent mode: RAW")

    try:
        while True:
            # Get the current frame
            frame = camera.get_frame(wait=True)

            # Process based on current mode
            if mode == "difference":
                display_frame = compute_difference_image(frame, reference_frame)
            else:
                display_frame = frame

            # Add mode indicator on the image
            mode_text = f"Mode: {mode.upper()}"
            cv2.putText(display_frame, mode_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if mode == "difference" and reference_frame is None:
                ref_text = "Press 's' to set reference"
                cv2.putText(display_frame, ref_text, (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Display the frame
            cv2.imshow("GelSight Mini Stream", display_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('d'):
                mode = "difference"
                print("Switched to DIFFERENCE mode")
            elif key == ord('r'):
                mode = "raw"
                print("Switched to RAW mode")
            elif key == ord('s'):
                reference_frame = frame.copy()
                print("Reference image set/reset")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        # Clean up
        camera.stop()
        cv2.destroyAllWindows()
        print("Stream stopped")


if __name__ == "__main__":
    main()
