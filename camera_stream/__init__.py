try:
    from usb_video_stream import USBVideoStream
    from raspi_video_stream import RaspiVideoStream
    from digit_video_stream import DigitVideoStream
except ImportError:
    print("Error importing online modules in camera_stream module. Skipping.")