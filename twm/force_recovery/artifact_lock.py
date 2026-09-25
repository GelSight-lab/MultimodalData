"""One writer per artifact.

Twice now, two runs of the same module have been in flight together and both
have written the same JSON:

  * `error_analysis` — a leftover background job plus a new one. The artifact
    ended up holding a mix of two samplings (n=201 in the file against n=303 in
    the log), and a headline finding computed from the contaminated file went
    into a docstring and onto the website before the mix was noticed.
  * `force_recon_matrix` — a poller shell left over from an earlier session
    fired its follow-up command hours later, launching a second run with
    DIFFERENT arguments (`--per-group 0`) alongside a fresh one.

Neither was noticed by anything except reading the log closely, because the
failure is silent: both processes succeed, and the file is simply not what
either of them computed. So the fix is not "remember to check" — it is a lock
around the write, applied by every module that owns an artifact.

A lock file whose owning process is gone is stale and taken over, so a killed
run does not wedge the next one.

    with one_writer(OUT_JSON):
        ...
"""
from __future__ import annotations

import contextlib
import os
from pathlib import Path


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True                 # exists, owned by someone else
    return True


@contextlib.contextmanager
def one_writer(artifact: Path):
    """Refuse to run while another live process is writing `artifact`."""
    lock = Path(artifact).with_suffix(".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            break
        except FileExistsError:
            try:
                held = int(lock.read_text().strip())
            except (ValueError, OSError):
                held = -1
            if held != -1 and _alive(held):
                raise SystemExit(
                    f"{Path(artifact).name}: pid {held} is already writing it. "
                    f"Wait for that run, or kill it — two writers produce a "
                    f"file neither of them computed.")
            # owner is gone; take the lock over and say so, because a stale
            # lock usually means the previous run was killed mid-write and the
            # artifact may be half of an older one.
            print(f"  {lock.name}: taking over from dead pid {held}",
                  flush=True)
            with contextlib.suppress(OSError):
                lock.unlink()
    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            lock.unlink()
