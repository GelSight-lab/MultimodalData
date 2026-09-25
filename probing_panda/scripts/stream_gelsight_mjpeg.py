#!/usr/bin/env python3
"""
Stream GelSight Mini frames as MJPEG over HTTP for remote viewing (e.g. over
VS Code SSH, where cv2.imshow has no local X11 display).

Usage:
    python3 probing_panda/scripts/stream_gelsight_mjpeg.py --serial 2BGLKZNT --port 8765

Then open http://localhost:8765 in a local browser. VS Code auto-forwards the
port when it detects the server start; if it doesn't, add it manually in the
"Ports" tab.

To view both sensors at once, run this script twice with different --port
values (e.g. 8765 for left, 8766 for right).
"""

import argparse
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

from camera_stream.usb_video_stream import USBVideoStream

BOUNDARY = "frame"


def make_handler(camera):
    class MJPEGHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # silence per-request access logging

        def do_GET(self):
            if self.path != "/":
                self.send_response(404)
                self.end_headers()
                return

            self.send_response(200)
            self.send_header(
                "Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}"
            )
            self.end_headers()

            try:
                while True:
                    frame = camera.get_frame(wait=True)
                    ok, jpg = cv2.imencode(".jpg", frame)
                    if not ok:
                        continue
                    body = jpg.tobytes()
                    self.wfile.write(f"--{BOUNDARY}\r\n".encode())
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(body)}\r\n\r\n".encode())
                    self.wfile.write(body)
                    self.wfile.write(b"\r\n")
                    time.sleep(1 / 30)
            except (BrokenPipeError, ConnectionResetError):
                pass  # client closed the tab

    return MJPEGHandler


def main():
    parser = argparse.ArgumentParser(description="Stream GelSight Mini over HTTP MJPEG")
    parser.add_argument("--serial", type=str, required=True,
                         help="GelSight sensor serial number")
    parser.add_argument("--port", type=int, default=8765,
                         help="HTTP port to serve on (default: 8765)")
    args = parser.parse_args()

    camera = USBVideoStream(serial=args.serial, resolution=(640, 480), format="BGR")
    print(f"Starting GelSight Mini stream (serial={args.serial})...")
    camera.start()

    server = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(camera))
    print(f"Serving at http://localhost:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        server.server_close()
        camera.stop()
        print("Stream stopped")


if __name__ == "__main__":
    main()
