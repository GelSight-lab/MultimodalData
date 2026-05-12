for _name in ("usb_video_stream", "raspi_video_stream",
              "digit_video_stream", "realsense_stream"):
    try:
        _mod = __import__(f"{__name__}.{_name}", fromlist=["*"])
        for _cls in ("USBVideoStream", "RaspiVideoStream",
                     "DigitVideoStream", "RealsenseStream"):
            if hasattr(_mod, _cls):
                globals()[_cls] = getattr(_mod, _cls)
    except ImportError as e:
        print(f"camera_stream: skipped {_name} ({e})")