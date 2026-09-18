"""Test collection must not swap dependencies or modify import search paths."""
import subprocess
import sys
from pathlib import Path


def test_legacy_preview_tests_do_not_mutate_import_state():
    test_file = Path(__file__).resolve().parents[1] / "test_visualize.py"
    script = """
import runpy, sys, cv2, h5py
before = list(sys.path)
runpy.run_path(sys.argv[1])
assert sys.modules['cv2'] is cv2
assert sys.modules['h5py'] is h5py
assert sys.path == before
"""
    subprocess.run([sys.executable, "-c", script, str(test_file)], check=True)
