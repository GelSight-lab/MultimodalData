"""TWM multimodal recorder as a library.

Modules
  config    RecorderConfig and the CLI that builds it
  frames    Tick — one synchronized multimodal sample
  schema    the HDF5 episode layout (create / append / finalize)
  writer    EpisodeWriter — byte-bounded, batched, fail-fast background writer
  rig       SensorRig — hardware startup, grab(), ordered shutdown
  capture   CaptureLoop — strict-rate capture thread
  preflight startup / episode-start checks
  monitor   health line for the preview
  episode   EpisodeStore (paths, numbering, CSV log) and EpisodeSummary
  app       Recorder controller, cv2 GUI loop, main()
"""
from twm.recorder.config import (DATA_DIR, FPS, GELSIGHT_SERIALS,  # noqa: F401
                                 OT_TRACKERS, REALSENSE_SERIALS, DiskConfig,
                                 RecorderConfig, WriterConfig, parse_args)
