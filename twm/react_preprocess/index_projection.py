"""The published index files are projections of the published episode set.

`episodes.jsonl`, `segments.json`, `bad_frames.json`, `segment_provenance.json`
and `splits.json` all have the same shape: a per-episode body plus a summary
computed over that body. None of them needs cross-episode information, so none
of them is state -- each is a pure function of "which episodes are published".

Treating them as accumulated state is what put this on the Hub:

    data/motherboard/segment_provenance.json
      "summary": {"episodes": 0, "raw_frames": 0, "kept_minutes": 0.0,
                  "skipped_already_cut": 11}

The body listed 14 segments while the summary described the LAST `segment`
invocation, which had nothing left to cut. rope published the same shape, and
pushT published `episodes: 30` against 31 published segments. Every one of
those numbers is a fact about a process run, not about the dataset.

Every function here takes the on-disk document plus the set of published keys
(`"<date>/<episode>"`) and returns the document that describes exactly those
episodes. Local-only episodes -- pushT's 2026-06-18 segments are cut but not
published -- drop out, and the summary is recomputed rather than carried.
"""
from __future__ import annotations

import json
from pathlib import Path

# A published key is "<date>/<episode>". `dropped_spans` is keyed by the SOURCE
# recording instead, which for an already-segmented tree is the published key
# with its `_segNN` suffix removed -- and for an uncut tree is the key itself.
# Both must match, so compare against both forms.


def _source_forms(keys) -> set[str]:
    out = set()
    for k in keys:
        out.add(k)
        head, sep, tail = k.rpartition("_seg")
        if sep and tail.isdigit():
            out.add(head)
    return out


def project_episodes(rows: list[dict], keys) -> list[dict]:
    """`episodes.jsonl`, keyed by the `episode` field."""
    keep = set(keys)
    return sorted((r for r in rows if r["episode"] in keep),
                  key=lambda r: r["episode"])


def project_bad_frames(doc: dict, keys) -> dict:
    keep = set(keys)
    eps = {k: v for k, v in doc["episodes"].items() if k in keep}
    total_frames = sum(v["n_frames"] for v in eps.values())
    total_bad = sum(v.get("total_bad_frames", 0) for v in eps.values())
    out = dict(doc)
    out["episodes"] = dict(sorted(eps.items()))
    out["summary"] = {
        "n_episodes": len(eps),
        "total_frames": total_frames,
        "total_bad_frames": total_bad,
        "bad_fraction_overall": round(total_bad / total_frames, 6) if total_frames else 0.0,
    }
    return out


def project_segments(doc: dict, keys) -> dict:
    keep = set(keys)
    segs = [s for s in doc["segments"] if s["source_episode"] in keep]
    segs.sort(key=lambda s: (s["source_episode"], s["segment_idx"]))
    frames = sum(s["n_frames"] for s in segs)
    out = dict(doc)
    out["segments"] = segs
    out["n_segments"] = len(segs)
    out["total_frames"] = frames
    out["total_duration_min"] = round(sum(s["duration_s"] for s in segs) / 60, 2)
    return out


def project_provenance(doc: dict, keys) -> dict:
    """Recompute the summary from the projection instead of carrying it.

    `skipped_already_cut` is dropped: it counts episodes one invocation of
    `segment` chose not to redo, which says nothing about the dataset and is
    exactly the field that made a fully-cut task report `episodes: 0`.
    """
    keep, sources = set(keys), _source_forms(keys)
    segs = sorted((s for s in doc["segments"] if s["episode"] in keep),
                  key=lambda s: s["episode"])
    dropped = [d for d in doc["dropped_spans"] if d["episode"] in sources]
    kept_frames = sum(s["n_frames"] for s in segs)
    discarded = sum(d["n_frames"] for d in dropped)
    raw = kept_frames + discarded
    out = dict(doc)
    out["segments"] = segs
    out["dropped_spans"] = dropped
    out["summary"] = {
        "task": doc["summary"]["task"],
        "published_segments": len(segs),
        "source_recordings": len({s["episode"].rpartition("_seg")[0] or s["episode"]
                                  for s in segs}),
        "raw_frames": raw,
        "kept_frames": kept_frames,
        "kept_minutes": round(kept_frames / 30 / 60, 2),
        "discarded_frames": discarded,
        "kept_fraction": round(kept_frames / raw, 4) if raw else 0.0,
        "min_publish_seconds": doc["summary"].get("min_publish_seconds"),
        "note": "Computed over the PUBLISHED segments, not over one run of the "
                "cutter. raw/discarded cover the source recordings that "
                "contributed at least one published segment; a recording "
                "dropped in full contributes nothing to either. "
                "See twm/react_preprocess/index_projection.py",
    }
    return out


def project_splits(doc: dict, keys) -> dict:
    """`splits.json`'s `episodes` map, plus whatever `stats` it carries."""
    keep = set(keys)
    eps = {k: v for k, v in doc["episodes"].items() if k in keep}
    out = dict(doc)
    out["episodes"] = dict(sorted(eps.items()))
    out["stats"] = splits_stats(out["episodes"])
    return out


def splits_stats(episodes: dict) -> dict:
    """Frame counts per split, recomputed from the projected episode map."""
    n_train = n_test = 0
    for v in episodes.values():
        n_test += int(v.get("n_test_frames", len(v.get("test_frames", []))))
        n_train += int(v.get("n_train_frames", 0))
    total = n_train + n_test
    return {"n_episodes": len(episodes), "n_train_frames": n_train,
            "n_test_frames": n_test,
            "test_fraction": round(n_test / total, 4) if total else 0.0}


def uncovered(keys, doc: dict) -> list[str]:
    """Published keys a splits document does not mention.

    `ReactVideoDataset._split_filter` returns `self.split == "train"` for an
    unlisted episode, so an episode missing here leaks silently into training
    rather than raising.
    """
    return sorted(set(keys) - set(doc.get("episodes", {})))


def write_json(path: Path, doc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n")
    tmp.replace(path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tmp.replace(path)
