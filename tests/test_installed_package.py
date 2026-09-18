"""Build and import the distribution away from checkout and hardware."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


def test_wheel_contains_twm_subpackages_and_calibration(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    source = tmp_path / "source"
    source.mkdir()
    shutil.copy2(repo / "pyproject.toml", source)
    for name in ("probing_panda", "camera_stream", "optitrack", "ft_sensor", "misc", "twm"):
        shutil.copytree(repo / name, source / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    wheel_dir = tmp_path / "wheels"
    subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
                    "--wheel-dir", str(wheel_dir), str(source)], check=True,
                   capture_output=True, text=True)
    installed = tmp_path / "installed"
    with zipfile.ZipFile(next(wheel_dir.glob("*.whl"))) as wheel:
        wheel.extractall(installed)
    env = dict(os.environ, PYTHONPATH=str(installed))
    # Avoid a host calibration override hiding missing package data.
    env.pop("REACT_CALIB", None)
    env.pop("REACT_RELEASE", None)
    smoke = """
import sys
from pathlib import Path
import twm.visualization
import twm.recorder.config
import twm.react_preprocess.encode
import twm.react_toolbox
from twm.calib_epoch import current_epoch_dir, EPOCH_DIRS
assert Path(twm.visualization.__file__).is_relative_to(Path(sys.argv[1]))
for directory in EPOCH_DIRS.values():
    assert (directory / 'T_mocap_to_cam_middle.json').is_file(), directory
assert (Path(twm.recorder.config.__file__).parents[1] / 'config/wrist_usb.json').is_file()
assert not {'pyrealsense2', 'frankapy', 'gsdevice'} & sys.modules.keys()
"""
    subprocess.run([sys.executable, "-c", smoke, str(installed)], cwd=tmp_path,
                   env=env, check=True)
    subprocess.run([sys.executable, "-m", "twm.visualization", "--help"], cwd=tmp_path,
                   env=env, check=True, capture_output=True, text=True)
