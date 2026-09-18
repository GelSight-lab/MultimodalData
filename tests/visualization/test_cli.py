import subprocess
import sys


def test_module_help_is_compatible():
    result = subprocess.run([sys.executable, "-m", "twm.visualization", "--help"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for option in ("--fps", "--save_video", "--no_projection"):
        assert option in result.stdout


def test_explicit_arguments_are_forwarded(monkeypatch):
    from twm import visualize
    from twm.visualization.__main__ import main
    calls = []
    monkeypatch.setattr(visualize, "main", lambda argv: calls.append(argv))
    main(["example.h5", "--no_projection"])
    assert calls == [["example.h5", "--no_projection"]]
