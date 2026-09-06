"""`python -m twm.recorder run --task X` or `python -m twm.recorder bench --dir D`."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from twm.recorder.config import WriterConfig, build_parser, config_from_namespace
from twm.recorder.preflight import check_write_bandwidth


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)-5s %(message)s",
                        datefmt="%H:%M:%S")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "soak":
        from twm.recorder.app import run_headless
        p = build_parser()
        p.add_argument("--duration", type=float, required=True,
                       help="Recording duration in seconds.")
        a = p.parse_args(argv[1:])
        if a.duration <= 0:
            p.error("--duration must be positive")
        cfg = config_from_namespace(a)
        _configure_logging()
        return run_headless(cfg, a.duration)
    if argv and argv[0] == "bench":
        p = argparse.ArgumentParser(prog="python -m twm.recorder bench",
                                    description="Measure sustained writer throughput "
                                                "into a directory using the real HDF5 path.")
        p.add_argument("--dir", required=True)
        p.add_argument("--seconds", type=float, default=5.0)
        p.add_argument("--fps", type=int, default=30)
        p.add_argument("--margin", type=float, default=1.5)
        p.add_argument("--arducams", type=int, default=2)
        a = p.parse_args(argv[1:])
        r = check_write_bandwidth(Path(a.dir), a.fps, a.seconds, a.margin, WriterConfig(),
                                  n_arducam=a.arducams)
        print(("ok   " if r.ok else "FAIL ") + r.detail)
        return 0 if r.ok else 1
    if argv and argv[0] == "validate":
        from twm.recorder.validate import validate_episode
        p = argparse.ArgumentParser(prog="python -m twm.recorder validate",
                                    description="Validate a recorded episode's format, "
                                                "timing, and content.")
        p.add_argument("path")
        p.add_argument("--fps", type=int, default=30)
        p.add_argument("--expected-duration", type=float, default=None)
        p.add_argument("--max-tick-gap", type=float, default=0.5)
        p.add_argument("--report", default=None,
                       help="Also write the JSON report to this path.")
        a = p.parse_args(argv[1:])
        report = validate_episode(a.path, fps=a.fps, expected_duration=a.expected_duration,
                                  max_tick_gap_s=a.max_tick_gap)
        text = json.dumps(report.to_dict(), indent=2)
        print(text)
        if a.report:
            Path(a.report).write_text(text)
        return 0 if report.ok else 1
    if argv and argv[0] == "run":
        argv = argv[1:]
    from twm.recorder.app import main as run_main
    return run_main(argv)


if __name__ == "__main__":
    sys.exit(main())
