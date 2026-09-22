"""Give a newly built episode the published schema: raw + repaired + actions.

The React dataset publishes 60 columns. A tree built by `react_preprocess`
carries 29 of them, and the missing 31 are the two families produced on this
branch: the pose-repair provenance (raw preserved in `sensor_<side>_pose`, the
repair beside it in `sensor_<side>_pose_repaired`, with method, confidence,
event id, validity and the operator-policy acceptance flag) and the action
channel (`action[t]` is the next row's repaired left/right pose pair, with the
masks that say when that transition may be trained on).

`accept_mocap_repairs.stage` cannot do this for new data: it pins itself to
`action_publication_manifest.json` and verifies every base file against a HF
revision, which is right for revising the published snapshot and wrong for an
episode that has never been published.

So this script bridges the two trees that already exist and then hands the
result to the SAME authority the published data went through:

    release/<task>/meta/<date>/<ep>.parquet          raw poses
    <candidates>/<task>/meta/<date>/<ep>.parquet     repaired poses + flags
    <candidates>/<task>/repair_events/<date>/<ep>.json

`accepted_repairs.accept_table` then applies
`operator_short33_and_supported_return_branches_v1` -- the operator's own
codified rule, the one every published episode was accepted under -- and
writes the action series. Nothing about acceptance is decided here.

Events from a whole-episode candidate carry no `local_*` rows because the
episode IS the local frame; they are filled in from `start`/`end`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from twm.react_preprocess.accepted_repairs import accept_table
from twm.react_preprocess.mocap_candidate import _write_actions

SIDES = ("left", "right")


def _put(table, name, values):
    arr = pa.array(values.tolist() if getattr(values, "ndim", 1) > 1 else values)
    if name in table.column_names:
        i = table.schema.get_field_index(name)
        return table.set_column(i, table.schema.field(name), arr)
    return table.append_column(name, arr)


def merge_candidate(raw_table, cand_table, events):
    """The input `accept_table` expects: raw kept, repair beside it.

    The candidate tree writes the REPAIRED pose into `sensor_<side>_pose`,
    because for its own purposes that column is the answer. Published data
    keeps the raw measurement there instead and puts the repair in
    `_repaired`. Getting this backwards would publish an interpolated value
    under the name readers use for the measurement.
    """
    if raw_table.num_rows != cand_table.num_rows:
        raise ValueError(f"row mismatch: raw {raw_table.num_rows} "
                         f"vs candidate {cand_table.num_rows}")
    out = raw_table
    for side in SIDES:
        rep = np.asarray(cand_table[f"sensor_{side}_pose"].to_pylist(), float)
        out = _put(out, f"sensor_{side}_pose_repaired", rep)
        for col, default in ((f"pose_{side}_repaired", False),
                             (f"pose_{side}_valid", True),
                             (f"pose_{side}_repair_confidence", 0),
                             (f"pose_{side}_repair_event_id", "")):
            if col in cand_table.column_names:
                out = _put(out, col, np.asarray(cand_table[col].to_pylist()))
            else:
                out = _put(out, col, np.full(out.num_rows, default))
        # method is per ROW and lives in the events, not in the candidate table
        method = np.full(out.num_rows, None, dtype=object)
        for e in events:
            if e["side"] != side:
                continue
            method[e["start"]:e["end"] + 1] = e.get("method")
        out = _put(out, f"pose_{side}_repair_method", method)
    return out


def localise(events, n_rows):
    """A whole-episode candidate's event rows ARE its local rows."""
    out = []
    for e in events:
        e = dict(e)
        e.setdefault("local_start_row", int(e["start"]))
        e.setdefault("local_end_row", int(e["end"]))
        if not (0 <= e["local_start_row"] <= e["local_end_row"] < n_rows):
            raise ValueError(f"event outside the episode: {e['event_id']}")
        out.append(e)
    return out


def apply_episode(release_root: Path, cand_root: Path, out_root: Path,
                  task: str, date: str, ep: str) -> dict:
    raw = pq.read_table(str(release_root / task / "meta" / date / f"{ep}.parquet"))
    cand = pq.read_table(str(cand_root / task / "meta" / date / f"{ep}.parquet"))
    doc = json.loads((cand_root / task / "repair_events" / date /
                      f"{ep}.json").read_text())
    events = localise(doc.get("events", []), raw.num_rows)
    merged = merge_candidate(raw, cand, events)
    after, sides, updated = accept_table(merged, events)

    dst = out_root / task / "meta" / date / f"{ep}.parquet"
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(after, str(dst), compression="zstd")
    for side, rates in sides.items():
        for rate, series in rates.items():
            sp = (out_root / task / f"actions_{rate}" / date /
                  f"{ep}_{side}.npz")
            _write_actions(sp, series)
    (out_root / task / "action_repair_events" / date).mkdir(parents=True,
                                                            exist_ok=True)
    (out_root / task / "action_repair_events" / date / f"{ep}.json").write_text(
        json.dumps({**doc, "events": updated}, indent=2, default=str) + "\n")
    accepted = sum(1 for e in updated if e.get("policy_accepted"))
    return {"task": task, "date": date, "episode": ep, "rows": after.num_rows,
            "columns": len(after.column_names), "events": len(updated),
            "accepted": accepted, "skipped": len(updated) - accepted}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", type=Path,
                    default=Path("/media/yxma/Disk1/twm/release"))
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--date", action="append", required=True)
    args = ap.parse_args()
    rows = []
    for date in args.date:
        meta = args.candidates / args.task / "meta" / date
        for p in sorted(meta.glob("*.parquet")):
            r = apply_episode(args.release, args.candidates, args.out,
                              args.task, date, p.stem)
            rows.append(r)
            print(f"  {r['task']}/{r['date']}/{r['episode']}: {r['rows']} rows, "
                  f"{r['columns']} cols, {r['accepted']}/{r['events']} events "
                  f"accepted", flush=True)
    print(f"\n{len(rows)} episode(s); "
          f"{sum(r['accepted'] for r in rows)} accepted / "
          f"{sum(r['events'] for r in rows)} events")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
