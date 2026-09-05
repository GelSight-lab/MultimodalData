"""`python -m twm.recorder run --task X` or `python -m twm.recorder bench --dir D`."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from twm.recorder.config import WriterConfig
from twm.recorder.preflight import check_write_bandwidth


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
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
    if argv and argv[0] == "run":
        argv = argv[1:]
    from twm.recorder.app import main as run_main
    return run_main(argv)


if __name__ == "__main__":
    sys.exit(main())
