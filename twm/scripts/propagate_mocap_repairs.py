"""Map repaired source poses to every published segment without altering raw data.

Inventory pins remote membership/digests and records a verified coordinate
conversion. Export keeps every non-action base column, including force columns.
"""
from __future__ import annotations

import argparse
from collections import Counter
from html import escape
import json
from pathlib import Path
import re
from urllib.parse import quote

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from twm.react_toolbox.frames import convert_poses
from twm.react_preprocess.mocap_candidate import _write_actions
from twm.react_preprocess.repaired_actions import native_actions, actions_fps15
from twm.scripts.export_old_motherboard_actions import (
    build_action_episode, _atomic_parquet, _columns_equal, _sha256)


SIDES = ('left', 'right')
POSES = tuple(f'sensor_{s}_pose' for s in SIDES)


def source_relative_path(remote_path: str | Path) -> Path:
    parts = Path(remote_path).parts
    if len(parts) != 5 or parts[0] not in ('data', 'old_data') or parts[2] != 'meta':
        raise ValueError(f'unsupported remote path: {remote_path}')
    return Path(*parts[1:-1], re.sub(r'_seg\d+(?=\.parquet$)', '', parts[-1]))


def _values(table, column):
    array = table[column].combine_chunks()
    if (pa.types.is_list(array.type) or pa.types.is_fixed_size_list(array.type)) and not array.null_count:
        if len(array) and len(array.values) % len(array) == 0:
            return array.values.to_numpy(zero_copy_only=False).reshape(len(array), -1)
    return np.asarray(array.to_pylist())


def _coordinate_mode(raw: pa.Table, selected: pa.Table) -> str:
    if all(np.array_equal(_values(raw, c), _values(selected, c), equal_nan=True)
           for c in POSES):
        return 'identity'
    if all(np.array_equal(_values(raw, c), convert_poses(_values(selected, c)),
                          equal_nan=True) for c in POSES):
        return 'yup_to_zup'
    raise ValueError('sensor pose mismatch: neither identity nor verified Y-up to Z-up')


def align_rows(raw: pa.Table, source: pa.Table) -> np.ndarray:
    """Require unique source IDs, consecutive rows, exact times and known poses."""
    if raw.num_rows == 0:
        raise ValueError('empty segment')
    ids = source['source_h5_frame'].to_pylist()
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate source_h5_frame in source')
    index = {value: i for i, value in enumerate(ids)}
    try:
        rows = np.asarray([index[value] for value in raw['source_h5_frame'].to_pylist()],
                          dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f'missing source_h5_frame: {exc}') from exc
    if np.any(np.diff(rows) != 1):
        raise ValueError('segment source rows must be contiguous and unique')
    selected = source.take(pa.array(rows))
    if not np.array_equal(_values(raw, 'timestamp'), _values(selected, 'timestamp')):
        raise ValueError('timestamp mismatch')
    _coordinate_mode(raw, selected)
    return rows


def _derived(name):
    return (name == 'action' or name.startswith('action_')
            or name in ('sensor_left_pose_repaired', 'sensor_right_pose_repaired')
            or name.startswith(('pose_left_', 'pose_right_')))


def build_segment(raw: pa.Table, source: pa.Table, candidate: pa.Table,
                  source_rows: np.ndarray, events: list[dict]):
    """Return augmented parquet and native/fps15 ActionSeries for each side.

    Recompute transitions from candidate pose validity after selecting the
    published segment, so neither actions nor 15fps phase cross a cut boundary.
    """
    rows = align_rows(raw, source)
    if not np.array_equal(rows, source_rows):
        raise ValueError('provided source rows disagree with verified mapping')
    if candidate.num_rows != source.num_rows:
        raise ValueError('candidate/source row counts differ')
    for c in ('source_h5_frame', 'timestamp'):
        if not _columns_equal(source[c], candidate[c]):
            raise ValueError(f'candidate/source {c} mismatch')
    segment = candidate.take(pa.array(rows))
    mode = _coordinate_mode(raw, source.take(pa.array(rows)))
    sidecars = {}
    for side in SIDES:
        required = [f'pose_{side}_{key}' for key in
                    ('valid', 'repaired', 'repair_confidence', 'repair_event_id')]
        for name in required:
            if name not in segment.column_names:
                raise ValueError(f'missing candidate provenance: {name}')
        column = f'sensor_{side}_pose'
        pose = _values(segment, column)
        if mode == 'yup_to_zup':
            pose = convert_poses(pose)
            segment = segment.set_column(segment.schema.get_field_index(column),
                                         column, pa.array(pose.tolist()))
        native = native_actions(pose, _values(segment, required[0]),
                                _values(segment, required[1]),
                                _values(segment, required[3]))
        sidecars[side] = {'native': native, 'fps15': actions_fps15(pose, native)}
    # Keep original event duration; negative local starts are intentional and
    # retain long-loss classification for events clipped by the segment edge.
    local_events = [dict(event, start=int(event['start']) - int(rows[0]),
                        end=int(event['end']) - int(rows[0]))
                    for event in events
                    if int(event['end']) >= rows[0] - 1
                    and int(event['start']) <= rows[-1] + 1]
    base = raw.select([name for name in raw.column_names if not _derived(name)])
    out = build_action_episode(base, segment,
        {k: getattr(sidecars['left']['native'], k) for k in ('valid', 'repaired')},
        {k: getattr(sidecars['right']['native'], k) for k in ('valid', 'repaired')},
        local_events)
    for side in SIDES:
        for key in ('valid', 'repaired', 'repair_confidence', 'repair_event_id'):
            name = f'pose_{side}_{key}'
            out = out.append_column(name, segment[name])
        method_name = f'pose_{side}_repair_method'
        if method_name in segment.column_names:
            out = out.append_column(method_name, segment[method_name])
        else:
            methods = {event['event_id']: event.get('method') for event in events
                       if event.get('side') == side and 'event_id' in event}
            values = [methods.get(event_id) for event_id in
                      segment[f'pose_{side}_repair_event_id'].to_pylist()]
            out = out.append_column(method_name, pa.array(values, type=pa.string()))
    for name in base.column_names:
        if out.schema.field(name) != base.schema.field(name) or not _columns_equal(out[name], base[name]):
            raise ValueError(f'preserved column changed: {name}')
    return out, sidecars


def segment_reviews(remote_path: str, raw: pa.Table, output: pa.Table,
                    rows: np.ndarray, events: list[dict], video_files: set[str],
                    revision: str, include_high_charts: bool = False) -> list[dict]:
    """Clip review coordinates to the published video, preserving source evidence."""
    remote = Path(remote_path)
    capture_times = _values(raw, 'timestamp').astype(float)
    capture_times -= capture_times[0]
    # Published videos contain one frame per parquet row at constant 30 fps.
    # Capture timestamps include acquisition stalls and are not media time.
    times = np.arange(raw.num_rows, dtype=float) / 30.
    records = []
    for event in events:
        first, last = int(event['start']), int(event['end'])
        if last < rows[0] or first > rows[-1]:
            continue
        lo, hi = max(0, first - int(rows[0])), min(len(rows) - 1, last - int(rows[0]))
        side = event['side']
        event_id = event.get('event_id', f'{side}:{first}-{last}')
        # Video filenames are included only when present in the frozen HF listing.
        links = {}
        for view in ('view_middle', f'wrist_{side}', 'view_left', 'view_right'):
            video = str(Path(*remote.parts[:2], 'videos', remote.parts[3],
                             remote.stem, view + '.mp4'))
            if video in video_files:
                links[view] = (f'https://huggingface.co/datasets/yxma/React/resolve/'
                               f'{quote(revision or "main", safe="")}/{quote(video, safe="/")}'
                               f'#t={max(0., times[lo] - 1):.3f},{times[hi] + 1:.3f}')
        record = dict(event, qualified_event_id=f'{remote_path}::{event_id}',
                      remote_path=remote_path, source_start_row=first,
                      source_end_row=last, local_start_row=lo, local_end_row=hi,
                      local_start_seconds=float(times[lo]), local_end_seconds=float(times[hi]),
                      capture_start_seconds=float(capture_times[lo]),
                      capture_end_seconds=float(capture_times[hi]),
                      video_time_basis='local_row / 30 fps',
                      video_links=links)
        if include_high_charts or str(event.get('confidence', '')).upper() != 'HIGH':
            a, b = max(0, lo - 30), min(len(rows), hi + 31)
            sample = np.unique(np.r_[np.linspace(a, b - 1, min(160, b - a), dtype=int), lo, hi])
            chart = {'seconds': times[sample].tolist(), 'sampled': b - a > 160}
            for label, column in [('raw', f'sensor_{side}_pose'),
                                  ('repaired', f'sensor_{side}_pose_repaired')]:
                context_start = max(0, a - 1)
                pose = _values((output if label == 'repaired' else raw).slice(
                    context_start, b - context_start), column).astype(float)
                # Angular displacement between adjacent native frames; no unwrap
                # or interpolation is invented for missing/zero quaternions.
                q = pose[:, 3:]
                norm = np.linalg.norm(q, axis=1)
                usable = np.isfinite(q).all(axis=1) & (norm > 1e-12)
                unit = np.full(q.shape, np.nan)
                unit[usable] = q[usable] / norm[usable, None]
                angle = np.r_[np.nan, np.degrees(2 * np.arccos(np.clip(
                    np.abs(np.sum(unit[1:] * unit[:-1], axis=1)), 0, 1)))]
                data = np.column_stack((pose[:, :3] * 1000., angle))[sample - context_start]
                chart[label] = [[float(v) if np.isfinite(v) else None for v in row] for row in data]
            record['chart'] = chart
        records.append(record)
    return records


def _json_safe(value):
    """Represent unavailable nonfinite evidence as JSON null, never as a number."""
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _review_svg(chart: dict) -> str:
    times = np.asarray(chart['seconds'])
    left, width, height = 70., 740., 64.
    x = left + width * (times - times[0]) / max(1e-9, times[-1] - times[0])
    parts = ['<svg viewBox="0 0 850 340" role="img" aria-label="Raw and repaired pose traces">']
    for k, title in enumerate(('x (mm)', 'y (mm)', 'z (mm)', 'step (deg)')):
        arrays = [np.asarray(chart[label], dtype=float)[:, k] for label in ('raw', 'repaired')]
        finite = np.concatenate([a[np.isfinite(a)] for a in arrays])
        top = 10. + k * 78.
        parts.append(f'<text x="0" y="{top + 30}">{title}</text>')
        if not len(finite):
            parts.append(f'<text x="80" y="{top + 30}">Missing pose</text>')
            continue
        low, high = float(finite.min()), float(finite.max())
        span = max(1e-6, high - low)
        for values, color in zip(arrays, ('#bd3342', '#147cbb')):
            y = top + height - (values - low) / span * height
            # Separate path pieces at missing values instead of bridging gaps.
            points, active = [], False
            for px, py in zip(x, y):
                if np.isfinite(py):
                    points.append(f'{"L" if active else "M"}{px:.1f},{py:.1f}')
                    active = True
                else:
                    active = False
            parts.append(f'<path d="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="1.4"/>')
        parts.append(f'<text x="815" y="{top + 15}" font-size="9">{high:.2f}</text>')
        parts.append(f'<text x="815" y="{top + 63}" font-size="9">{low:.2f}</text>')
    parts.append(f'<text x="70" y="335">{times[0]:.2f}s</text><text x="745" y="335">{times[-1]:.2f}s</text></svg>')
    return ''.join(parts)


def choose_review_samples(records: list[dict], count: int = 25) -> list[dict]:
    """Prefer 20 algorithmic proposals + 5 other events, balancing all scopes.

    One source event appears at most once even if it overlaps several segments.
    Selection uses the final candidate event records, never candidate labels from
    an earlier build. A shortfall is reported by the exported selection manifest.
    """
    unique = {}
    for record in records:
        source = source_relative_path(record['remote_path'])
        scope = Path(record['remote_path']).parts[0]
        key = (scope, str(source), record.get('event_id', record['qualified_event_id']))
        prior = unique.get(key)
        if prior is None or record['local_end_row'] - record['local_start_row'] > prior['local_end_row'] - prior['local_start_row']:
            unique[key] = record
    def features(record):
        length = record['source_end_row'] - record['source_start_row'] + 1
        duration = 'short_1_4' if length <= 4 else 'flicker_5_20' if length <= 20 else 'medium_21_60' if length <= 60 else 'long_over_60'
        return ('/'.join(Path(record['remote_path']).parts[:2]), duration, record.get('kind', 'unknown'))
    selected, used = [], Counter()
    def take(pool, n):
        pool = list(pool)
        for _ in range(min(n, len(pool))):
            def score(record):
                group, duration, kind = features(record)
                ambiguity = 'ambiguous' in str(record.get('method', '')) or 'incompatible' in str(record.get('evidence', {}))
                return (100 / (1 + used[('scope', group)]) +
                        20 / (1 + used[('duration', duration)]) +
                        10 / (1 + used[('kind', kind)]) +
                        (15 if ambiguity and not used[('ambiguous', True)] else 0))
            choice = max(sorted(pool, key=lambda r: r['qualified_event_id']), key=score)
            pool.remove(choice)
            selected.append(choice)
            group, duration, kind = features(choice)
            for name, value in [('scope', group), ('duration', duration), ('kind', kind)]:
                used[(name, value)] += 1
            if 'ambiguous' in str(choice.get('method', '')) or 'incompatible' in str(choice.get('evidence', {})):
                used[('ambiguous', True)] += 1
    def proposed(record):
        evidence = record.get('evidence', {})
        return (str(record.get('confidence', '')).upper() == 'HIGH'
                or str(evidence.get('prior_confidence', record.get('prior_confidence', ''))).upper() == 'HIGH'
                or evidence.get('independent_loss_evidence') is False)
    high = [r for r in unique.values() if proposed(r)]
    other = [r for r in unique.values() if not proposed(r)]
    take(high, max(0, count - 5))
    take(other, min(5, count))
    selected_ids = {r['qualified_event_id'] for r in selected}
    take([r for r in unique.values() if r['qualified_event_id'] not in selected_ids], count - len(selected))
    return [dict(record, human_verification_status='UNVERIFIED',
                 sample_group='algorithmic_proposal' if proposed(record) else 'other_uncertain')
            for record in selected]


def review_html(records: list[dict], *, include_high: bool = False,
                title: str = 'Mocap event review') -> str:
    review = [r for r in records if include_high or str(r.get('confidence', '')).upper() != 'HIGH']
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8">',
             '<title>Mocap event review</title><style>body{font:15px system-ui;max-width:1100px;margin:30px auto;padding:0 16px}details{border:1px solid #ccc;margin:10px 0;padding:12px}summary{cursor:pointer;overflow-wrap:anywhere}svg{width:100%;max-width:900px}a{margin-right:15px}pre{white-space:pre-wrap}</style>',
             f'<h1>{escape(title)}</h1><p>{len(review)} {"selected" if include_high else "non-HIGH"} event occurrences across published segments. ',
             'Raw traces are red; repaired traces are blue. Missing values are gaps. ',
             'Each chart includes one second of context; long intervals are sampled. ',
             'Video links and charts use segment-local frame / 30 fps; capture-time offsets are listed separately. No approval is implied. ',
             'The measured camera footage is unchanged; before/after repair is shown in the red/blue pose charts. ',
             'These examples support human verification and are not proof that every jump is tracking loss. ',
             'Check visible hand/object motion against the measured jump, the surrounding frames, and the repair.</p>']
    for record in review:
        eid = escape(record['qualified_event_id'])
        confidence = escape(str(record.get('confidence', 'UNKNOWN')))
        method = escape(str(record.get('method', 'unavailable')))
        parts.append(f'<details data-clip-start="{max(0., record["local_start_seconds"] - 1):.3f}" '
                     f'data-clip-end="{record["local_end_seconds"] + 1:.3f}"><summary>{eid} — {confidence} — {method} — '
                     f'{record["local_start_seconds"]:.3f}–{record["local_end_seconds"]:.3f}s'
                     f'{" — UNVERIFIED" if include_high else ""}</summary>')
        parts.append('<p>' + ''.join(f'<a href="{escape(url, quote=True)}" target="_blank" rel="noopener">{escape(view)}</a>'
                                     for view, url in record['video_links'].items()) + '</p>')
        if not record['video_links']:
            parts.append('<p>No matching video in the frozen repository inventory.</p>')
        if include_high:
            if record['video_links']:
                parts.append('<p><button type="button" onclick="playSample(this)">Play synced</button> '
                             '<button type="button" onclick="pauseSample(this)">Pause</button> '
                             '<span class="play-status" aria-live="polite"></span></p>')
            for view, url in record['video_links'].items():
                if view == 'view_middle' or view.startswith('wrist_'):
                    parts.append(f'<p>{escape(view)}</p><video controls preload="none" width="480" src="{escape(url, quote=True)}"></video>')
        if 'chart' in record:
            parts.append(_review_svg(record['chart']))
        parts.append('<pre>' + escape(json.dumps({key: record.get(key) for key in
            ('source_start_row', 'source_end_row', 'local_start_row', 'local_end_row', 'capture_start_seconds', 'capture_end_seconds', 'video_time_basis', 'kind', 'evidence')}, indent=2)) + '</pre></details>')
    if include_high:
        parts.append('''<script>
let reviewPlayToken = 0;
function pauseSample(button) {
  reviewPlayToken++;
  button.closest('details').querySelectorAll('video').forEach(v => v.pause());
}
async function playSample(button) {
  const token = ++reviewPlayToken;
  const box = button.closest('details');
  const videos = Array.from(box.querySelectorAll('video'));
  const status = box.querySelector('.play-status');
  const start = Number(box.dataset.clipStart), end = Number(box.dataset.clipEnd);
  document.querySelectorAll('video').forEach(v => v.pause());
  status.textContent = 'Loading video metadata…';
  try {
    await Promise.all(videos.map(v => v.readyState >= 1 ? Promise.resolve() : new Promise((resolve, reject) => {
      v.addEventListener('loadedmetadata', resolve, {once:true});
      v.addEventListener('error', () => reject(new Error('Video unavailable; use the direct link.')), {once:true});
      v.load();
    })));
    if (token !== reviewPlayToken) return;
    videos.forEach(v => {v.currentTime = Math.min(start, Math.max(0, v.duration - 0.05));});
    const master = videos[0];
    master.ontimeupdate = () => {
      if (master.currentTime >= end) {videos.forEach(v => v.pause()); return;}
      if (!master.paused) videos.slice(1).forEach(v => {
        if (Math.abs(v.currentTime - master.currentTime) > 0.15) v.currentTime = master.currentTime;
      });
    };
    videos.forEach(v => {v.onended = () => videos.forEach(other => other.pause());});
    await Promise.all(videos.map(v => v.play()));
    status.textContent = 'Playing the same segment interval.';
  } catch (error) {
    videos.forEach(v => v.pause());
    status.textContent = error.message + ' Click Play synced again if playback was blocked.';
  }
}
</script>''')
    return ''.join(parts) + '</html>'


def inventory(base_root: Path, source_root: Path, validation_source: Path,
              revision: str = '') -> dict:
    entries, errors = [], []
    cache = {}
    for path in sorted(base_root.glob('*/*/meta/*/*.parquet')):
        relative = path.relative_to(base_root)
        source_relative = source_relative_path(relative)
        root = validation_source if relative.parts[1] == 'validation' else source_root
        source_path = root / source_relative
        entry = {'remote_path': str(relative), 'base_sha256': _sha256(path),
                 'source_path': str(source_path), 'candidate_relative': str(source_relative)}
        try:
            raw = pq.read_table(path)
            if source_path not in cache:
                cache[source_path] = pq.read_table(source_path)
            source = cache[source_path]
            rows = align_rows(raw, source)
            entry.update(rows=len(rows), start_row=int(rows[0]), end_row=int(rows[-1]),
                         coordinate_mode=_coordinate_mode(raw, source.take(pa.array(rows))),
                         source_sha256=_sha256(source_path))
            entries.append(entry)
        except (ValueError, FileNotFoundError, KeyError) as exc:
            errors.append(dict(entry, error=str(exc)))
    return {'schema_version': 1, 'revision': revision, 'base_root': str(base_root),
            'counts': dict(Counter('/'.join(Path(e['remote_path']).parts[:2]) for e in entries)),
            'episodes': entries, 'errors': errors}


def export_inventory(manifest: dict, candidate_root: Path, validation_candidate: Path,
                     output_root: Path, base_root: Path | None = None,
                     video_inventory: dict | None = None) -> dict:
    if manifest['errors']:
        raise ValueError('inventory has unresolved mapping errors')
    base_root = Path(base_root or manifest['base_root'])
    if output_root.resolve() == base_root.resolve():
        raise ValueError('output must differ from frozen base')
    records, cache, review_records = [], {}, []
    video_inventory = video_inventory or {}
    video_files = set(video_inventory.get('files', []))
    for entry in manifest['episodes']:
        relative = Path(entry['remote_path'])
        base_path = base_root / relative
        # A refreshed base is allowed only after the caller freezes a new
        # inventory. This prevents silently overwriting concurrently changed force.
        if _sha256(base_path) != entry['base_sha256']:
            raise ValueError(f'base digest changed: {relative}; regenerate inventory')
        source_path = Path(entry['source_path'])
        if _sha256(source_path) != entry['source_sha256']:
            raise ValueError(f'source digest changed: {source_path}')
        root = validation_candidate if relative.parts[1] == 'validation' else candidate_root
        candidate_path = root / entry['candidate_relative']
        if candidate_path not in cache:
            task, _, date, filename = Path(entry['candidate_relative']).parts
            event_path = root / task / 'repair_events' / date / (Path(filename).stem + '.json')
            cache[candidate_path] = (pq.read_table(source_path), pq.read_table(candidate_path),
                                    json.loads(event_path.read_text())['events'])
        source, candidate, events = cache[candidate_path]
        raw = pq.read_table(base_path)
        rows = np.arange(entry['start_row'], entry['end_row'] + 1)
        out, sidecars = build_segment(raw, source, candidate, rows, events)
        dest = output_root / relative
        _atomic_parquet(out, dest)
        for side, rates in sidecars.items():
            for rate, series in rates.items():
                side_path = output_root / relative.parts[0] / relative.parts[1] / ('actions_' + rate) / relative.parts[3] / (relative.stem + '_' + side + '.npz')
                _write_actions(side_path, series)
        reviews = segment_reviews(str(relative), raw, out, rows, events, video_files,
                                  video_inventory.get('revision', manifest.get('revision', 'main')))
        review_records.extend(reviews)
        review_path = output_root / relative.parts[0] / relative.parts[1] / 'action_repair_events' / relative.parts[3] / (relative.stem + '.json')
        review_path.parent.mkdir(parents=True, exist_ok=True)
        review_path.write_text(json.dumps(_json_safe({'schema_version': 1, 'remote_path': str(relative),
                                          'source_path': str(source_path), 'events': reviews}), indent=2, allow_nan=False) + '\n')
        records.append(dict(entry, output_sha256=_sha256(dest),
                            valid_left=int(sum(out['action_valid_left'].to_pylist())),
                            valid_right=int(sum(out['action_valid_right'].to_pylist()))))
    result = dict(manifest, episodes=records, output_root=str(output_root))
    result['review_event_occurrences'] = len(review_records)
    result['non_high_review_occurrences'] = sum(str(r.get('confidence', '')).upper() != 'HIGH' for r in review_records)
    (output_root / 'action_review.html').write_text(review_html(review_records))
    samples = choose_review_samples(review_records)
    output_cache = {}
    for i, sample in enumerate(samples):
        if 'chart' in sample:
            continue
        relative = sample['remote_path']
        if relative not in output_cache:
            output_cache[relative] = pq.read_table(output_root / relative)
        table = output_cache[relative]
        entry = next(e for e in manifest['episodes'] if e['remote_path'] == relative)
        rows = np.arange(entry['start_row'], entry['end_row'] + 1)
        samples[i] = segment_reviews(relative, table, table, rows, [sample], video_files,
                                     video_inventory.get('revision', manifest.get('revision', 'main')),
                                     include_high_charts=True)[0]
    sample_manifest = {'requested_count': 25, 'actual_count': len(samples),
                       'high_count': sum(str(s.get('confidence', '')).upper() == 'HIGH' for s in samples),
                       'sample_group_counts': dict(Counter(s.get('sample_group', 'unknown') for s in samples)),
                       'scope_counts': dict(Counter('/'.join(Path(s['remote_path']).parts[:2]) for s in samples)),
                       'selection': 'Prefer 20 algorithmic proposals (HIGH/prior HIGH or explicitly lacking independent loss evidence) and 5 other uncertain; balance scopes, durations, kinds; deduplicate source events. All require human verification.',
                       'limitation': 'Human verification examples, not proof that every jump is tracking loss.',
                       'events': samples}
    result['sample_review_count'] = len(samples)
    (output_root / 'action_sample25.json').write_text(json.dumps(_json_safe(sample_manifest), indent=2, allow_nan=False) + '\n')
    (output_root / 'action_sample25.html').write_text(review_html(samples, include_high=True,
        title=f'{len(samples)} examples for human verification'))
    (output_root / 'action_publication_manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('inventory', 'export'))
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--validation-source', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--revision', default='')
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--validation-candidate', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--remote-videos', type=Path, help='Frozen JSON with revision and MP4 files list')
    args = parser.parse_args()
    if args.command == 'inventory':
        result = inventory(args.base, args.source, args.validation_source, args.revision)
        args.inventory.parent.mkdir(parents=True, exist_ok=True)
        args.inventory.write_text(json.dumps(result, indent=2) + '\n')
    else:
        if not all((args.candidate, args.validation_candidate, args.output)):
            parser.error('export requires --candidate --validation-candidate --output')
        result = export_inventory(json.loads(args.inventory.read_text()), args.candidate,
                                  args.validation_candidate, args.output, args.base,
                                  json.loads(args.remote_videos.read_text()) if args.remote_videos else None)
    print(json.dumps({'episodes': len(result['episodes']), 'counts': result['counts'],
                      'errors': result['errors']}, indent=2))
    return bool(result['errors'])


if __name__ == '__main__':
    raise SystemExit(main())
