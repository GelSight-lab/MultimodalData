"""React dataset preprocessing — source recordings to the published release.

This is the *producer* side of the dataset; ``react_toolbox`` is the consumer
side. Both ship with the data so the release is reproducible.

    from react_preprocess import pipeline, config
    pipeline.build_episode(h5_path, task="pushT")

Or from the command line:

    python -m react_preprocess build --task pushT
    python -m react_preprocess audit --task pushT
    python -m react_preprocess backfill-flags --root /path/to/data/pushT

Time alignment
--------------
Recordings from 2026-06-27 onward carry per-sensor GelSight capture
timestamps and are resampled onto the camera clock during the build. Earlier
recordings are index-aligned and still need the constant latency shift; the
pipeline reports which kind it saw and refuses to shift a timestamped one
twice (see ``h5io.TactileAlignment.needs_legacy_shift``).
"""
from __future__ import annotations

__version__ = "0.2.0"

from . import config, contact, encode, h5io, meta, pipeline, tactile  # noqa: F401

__all__ = ["config", "contact", "encode", "h5io", "meta", "pipeline", "tactile"]
