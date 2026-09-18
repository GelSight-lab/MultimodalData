"""Synthetic, hardware-free preview benchmark; equality is mandatory, speed is not.

Run: python -m twm.scripts.benchmark_visualization --iterations 100
"""
import argparse
import json
from time import perf_counter

import numpy as np

from twm.viz import build_preview_panel
from twm.visualization import render_preview


def benchmark(iterations=100):
    if iterations < 1:
        raise ValueError("iterations must be positive")
    rng = np.random.default_rng(0)
    colors = [rng.integers(0, 256, (480, 640, 3), dtype=np.uint8) for _ in range(3)]
    gels = [image.copy() for image in colors[:2]]
    args = (colors, gels, gels, {}, False, 0, 0.)
    np.testing.assert_array_equal(build_preview_panel(*args), render_preview(*args))
    results = {"iterations": iterations, "pixel_equal": True}
    for name, render in (("legacy", build_preview_panel), ("shared", render_preview)):
        for _ in range(5):
            render(*args)
        times = []
        for _ in range(iterations):
            start = perf_counter()
            render(*args)
            times.append((perf_counter() - start) * 1000)
        results[name] = {"median_ms": float(np.median(times)),
                         "p95_ms": float(np.percentile(times, 95))}
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args(argv)
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    print(json.dumps(benchmark(args.iterations), indent=2))


if __name__ == "__main__":
    main()
