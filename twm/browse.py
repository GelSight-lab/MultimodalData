"""Walk a folder and show every recording in it, so naming can be checked by eye.

Verifying that an episode is filed under the right task means LOOKING at it.
One `python -m twm.visualize <path>` at a time does not scale to a date with
twelve recordings, and an automated colour check is not evidence: a throwaway
frame grab that flipped BGR turned a blue T orange and a wooden table
blue-grey, and reported two correctly-filed recordings as "a different setup".

Point this at a folder, get every recording under it as a labelled thumbnail,
open the ones that look wrong in the full viewer. Both tree shapes are found,
because the question comes up on both: the published tree
(`<task>/videos/<date>/<episode>/view_middle.mp4`) and the raw one
(`<task>/<date>/episode_NNN.h5`) -- and the raw tree is where catching it
saves the eight hours of building.

    python -m twm.browse                  # pick a folder
    python -m twm.browse /media/.../data  # or name one

Keys: arrows/hjkl move, n/p page, Enter open in the viewer, f flag,
      w write the flag list, q quit.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# The view a mislabelled task is obvious in: the overhead camera sees the
# object. The gel and wrist streams see a fingertip either way.
PREFERRED_STREAM = "view_middle.mp4"
THUMB_SECONDS = 3.0        # past the settle, into the manipulation


@dataclass(frozen=True)
class Episode:
    task: str
    date: str
    episode: str
    kind: str                 # "video" | "h5"
    path: Path                # the mp4, or the .h5

    @property
    def key(self) -> str:
        return f"{self.task}/{self.date}/{self.episode}"


def _from_video(p: Path) -> Episode | None:
    # <task>/videos/<date>/<episode>/view_middle.mp4
    parts = p.parts
    try:
        i = len(parts) - 1 - parts[::-1].index("videos")
    except ValueError:
        return None
    task = parts[i - 1] if i >= 1 else "?"
    return Episode(task, p.parent.parent.name, p.parent.name, "video", p)


def _from_h5(p: Path) -> Episode:
    # <task>/<date>/episode_NNN.h5
    return Episode(p.parent.parent.name, p.parent.name, p.stem, "h5", p)


def find_episodes(root) -> list[Episode]:
    """Every recording under `root`, in the order a date reads in.

    A folder handed over by a file picker can be any level of either tree, so
    the shapes are recognised by what is IN them rather than by how deep the
    path is.
    """
    root = Path(root)
    out: list[Episode] = []
    for p in root.rglob(PREFERRED_STREAM):
        e = _from_video(p)
        if e is not None:
            out.append(e)
    if not out:
        # A single episode directory, handed over directly.
        for p in root.rglob("view_*.mp4"):
            if p.name == PREFERRED_STREAM:
                continue
        pass
    for p in root.rglob("episode_*.h5"):
        out.append(_from_h5(p))
    return sorted(out, key=lambda e: (e.task, e.date, e.episode))


def label_of(e: Episode) -> str:
    """Enough to spot a wrong task without opening anything."""
    return f"{e.task} {e.date} {e.episode.replace('episode_', '')}"


def thumbnail(e: Episode, cell=(320, 240)) -> np.ndarray:
    """One frame, or a marked placeholder.

    A file that cannot be read is itself a finding, so it gets a cell saying
    so rather than taking the whole sheet down.
    """
    import cv2
    w, h = cell
    img = None
    try:
        if e.kind == "video":
            cap = cv2.VideoCapture(str(e.path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            want = min(int(THUMB_SECONDS * fps), max(0, n - 1))
            if want:
                cap.set(cv2.CAP_PROP_POS_FRAMES, want)
            ok, frame = cap.read()
            cap.release()
            if ok:
                img = frame
        else:
            import hdf5plugin  # noqa: F401
            import h5py
            with h5py.File(e.path, "r") as f:
                cams = sorted(f["realsense"].keys())
                ds = f[f"realsense/{cams[len(cams) // 2]}/color"]
                i = min(int(THUMB_SECONDS * 30), ds.shape[0] - 1)
                # Stored the way the pipeline writes it; NOT re-ordered here.
                # Flipping the channels in an ad-hoc reader is what produced a
                # false report of a mislabelled session.
                img = np.asarray(ds[i])
    except Exception:                                  # noqa: BLE001
        img = None
    if img is None:
        img = np.full((h, w, 3), 40, np.uint8)
        cv2.putText(img, "unreadable", (8, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)
        return img
    return cv2.resize(img, (w, h))


def contact_sheet(episodes, cols: int = 4, cell=(320, 240), flagged=(),
                  selected: int | None = None):
    """(sheet, cells) where cells[i] is the (x, y, w, h) of episode i."""
    import cv2
    w, h = cell
    pad, bar = 4, 26
    n = max(1, len(episodes))
    rows = (n + cols - 1) // cols
    H = rows * (h + bar + pad) + pad
    W = cols * (w + pad) + pad
    sheet = np.full((H, W, 3), 245, np.uint8)
    cells = []
    for i, e in enumerate(episodes):
        r, c = divmod(i, cols)
        x = pad + c * (w + pad)
        y = pad + r * (h + bar + pad)
        sheet[y:y + h, x:x + w] = thumbnail(e, cell)
        colour = (0, 0, 200) if e.key in flagged else (30, 30, 30)
        cv2.putText(sheet, label_of(e), (x + 4, y + h + 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)
        if i == selected:
            cv2.rectangle(sheet, (x - 2, y - 2), (x + w + 1, y + h + bar - 4),
                          (0, 140, 255), 3)
        cells.append((x, y, w, h))
    return sheet, cells


def hit_test(cells, x: int, y: int) -> int | None:
    for i, (cx, cy, cw, ch) in enumerate(cells):
        if cx <= x < cx + cw and cy <= y < cy + ch:
            return i
    return None


def pick_folder() -> str | None:
    """The stdlib picker. No new dependency, and the repo has no GUI toolkit."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        r = tk.Tk()
        r.withdraw()
        d = filedialog.askdirectory(title="Folder to browse")
        r.destroy()
        return d or None
    except Exception:                                  # noqa: BLE001
        return None


# ── the window ──────────────────────────────────────────────────────────────
FLAG_FILE = "suspect_episodes.txt"
WIN = "twm browse — arrows move, Enter open, f flag, w write, q quit"


def _open_in_viewer(e: Episode) -> None:
    """Hand the episode to the existing viewer, in its own process.

    Its own, because the viewer owns an OpenCV window and a key loop of its
    own; running it inline would leave two loops fighting over waitKey.
    """
    import subprocess
    import sys
    target = str(e.path) if e.kind == "h5" else e.key
    subprocess.Popen([sys.executable, "-m", "twm.visualize", target],
                     cwd=str(Path(__file__).resolve().parents[1]))


def browse(root, cols: int = 4, per_page: int = 12, cell=(320, 240)) -> int:
    import cv2
    episodes = find_episodes(root)
    if not episodes:
        print(f"no recordings under {root}")
        return 1
    print(f"{len(episodes)} recording(s) under {root}")
    flagged: set[str] = set()
    page, sel = 0, 0
    pages = (len(episodes) + per_page - 1) // per_page
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)

    state = {"click": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x, y)

    cv2.setMouseCallback(WIN, on_mouse)

    while True:
        chunk = episodes[page * per_page:(page + 1) * per_page]
        sel = min(sel, len(chunk) - 1)
        sheet, cells = contact_sheet(chunk, cols=cols, cell=cell,
                                     flagged=flagged, selected=sel)
        cv2.putText(sheet, f"page {page + 1}/{pages}   flagged {len(flagged)}",
                    (6, sheet.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (90, 90, 90), 1, cv2.LINE_AA)
        cv2.imshow(WIN, sheet)
        k = cv2.waitKey(30) & 0xFF

        if state["click"] is not None:
            i = hit_test(cells, *state["click"])
            state["click"] = None
            if i is not None:
                sel = i
                _open_in_viewer(chunk[i])

        if k in (ord('q'), 27):
            break
        elif k in (ord('n'), 83, ord('l')) and page + 1 < pages:
            page, sel = page + 1, 0
        elif k in (ord('p'), 81, ord('h')) and page:
            page, sel = page - 1, 0
        elif k in (82, ord('k')):
            sel = max(0, sel - cols)
        elif k in (84, ord('j')):
            sel = min(len(chunk) - 1, sel + cols)
        elif k == ord('f'):
            key = chunk[sel].key
            flagged.symmetric_difference_update({key})
            print(("flagged " if key in flagged else "unflagged ") + key)
        elif k in (13, 10):
            _open_in_viewer(chunk[sel])
        elif k == ord('w'):
            out = Path(root) / FLAG_FILE
            out.write_text("".join(sorted(k_ + "\n" for k_ in flagged)))
            print(f"wrote {len(flagged)} flagged key(s) to {out}")

    cv2.destroyAllWindows()
    if flagged:
        out = Path(root) / FLAG_FILE
        out.write_text("".join(sorted(k_ + "\n" for k_ in flagged)))
        print(f"wrote {len(flagged)} flagged key(s) to {out}")
    return 0


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", nargs="?", default=None,
                    help="folder to walk (default: ask)")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--per-page", type=int, default=12)
    ap.add_argument("--cell", type=int, nargs=2, default=(320, 240))
    a = ap.parse_args()
    root = a.root or pick_folder()
    if not root:
        print("no folder chosen")
        return 1
    return browse(root, cols=a.cols, per_page=a.per_page, cell=tuple(a.cell))


if __name__ == "__main__":
    raise SystemExit(main())
