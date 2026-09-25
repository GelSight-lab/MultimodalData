"""Call sites that go stale when a shared helper changes shape.

Both classes here shipped broken code: `_load_proj_calibs` grew a fourth
return value and four scripts still unpacked three (each dies on its first
statement), and `build_preview_panel` grew a status strip while modules kept
feeding ffmpeg a hardcoded 1280x480 (rawvideo desynchronises, and the mp4 is
sheared garbage that ffmpeg still exits 0 on).
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TWM = ROOT / "twm"


def _py_files():
    return [p for p in TWM.rglob("*.py") if "__pycache__" not in p.parts]


def test_every_load_proj_calibs_call_unpacks_four_values():
    from twm.scripts import build_episode_previews as BEP
    src = ast.parse(Path(BEP.__file__).read_text())
    fn = next(n for n in ast.walk(src)
              if isinstance(n, ast.FunctionDef) and n.name == "_load_proj_calibs")
    arities = {len(r.value.elts) for r in ast.walk(fn)
               if isinstance(r, ast.Return) and isinstance(r.value, ast.Tuple)}
    assert arities == {4}, f"the helper itself returns {arities}"

    bad = []
    for p in _py_files():
        for node in ast.walk(ast.parse(p.read_text())):
            if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
                continue
            f = node.value.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            if name != "_load_proj_calibs":
                continue
            tgt = node.targets[0]
            if not isinstance(tgt, ast.Tuple):
                continue                      # storing the whole tuple is fine
            n = len(tgt.elts)
            if n != 4:
                bad.append(f"{p.relative_to(ROOT)}:{node.lineno}: unpacks {n}, helper returns 4")
    assert not bad, "\n".join(bad)


def test_nothing_hardcodes_the_old_panel_height():
    """The panel is 528 tall (two rows plus the status strip) and 768 with
    wrist cameras. A module that says 480 either feeds ffmpeg a wrong frame
    size or documents a layout that no longer exists."""
    from twm import viz
    assert (viz.PANEL_W, viz.PANEL_H) == (1280, 528)
    bad = []
    for p in _py_files():
        text = p.read_text()
        if "build_preview_panel" not in text:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if "1280, 480" in line or "1280x480" in line:
                bad.append(f"{p.relative_to(ROOT)}:{i}: {line.strip()}")
    assert not bad, "\n".join(bad)


def test_every_wrist_frame_shown_to_a_human_goes_through_the_tone_curve():
    """Three places decode `arducam/*/frames` for display: the recorder's live
    preview, the replay viewer and the dataset preview panels. They must agree,
    or the operator aims the cameras against one picture, reviews a second and
    ships a third. The recorder preview was wired first and the other two were
    left raw — this is the guard against that recurring.
    """
    offenders = []
    for p in _py_files():
        src = p.read_text()
        if "arducam" not in src or "decode_arducam" not in src:
            continue
        # camera_tuner shows the camera's OWN output — judging a V4L2 control
        # through a post curve is the one thing it must not do. validate reads
        # pixels to check them, not to show them. frames.py is the decoder.
        if p.name in {"frames.py", "camera_tuner.py", "validate.py"}:
            continue
        tree = ast.parse(src)
        decodes = [n for n in ast.walk(tree)
                   if isinstance(n, ast.Call)
                   and getattr(n.func, "id", getattr(n.func, "attr", "")) == "decode_arducam"]
        if not decodes:
            continue
        if "apply_tone_curve" not in src:
            offenders.append(p.relative_to(ROOT))
    assert not offenders, (
        "these decode wrist frames for display without the published tone "
        f"curve: {offenders}")
