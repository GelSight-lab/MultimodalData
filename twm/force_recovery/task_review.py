"""Reproducible four-task force review, isolated from release artifacts.

python -m twm.force_recovery.task_review extract|render|publish [--clip ID]
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import html
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import cv2
import h5py
import hdf5plugin  # noqa: F401
import joblib
import numpy as np
import pyarrow.parquet as pq

from . import calib_free as CF, react_calib as RC
from .lut_calibration import crop
from .run_episode import (DATA_ROOT, STAGE_ROOT, OUT_ROOT, _reference_rows,
                          reference_stack, reference_noise_area, open_episode)

PREVIOUS_ROOT = OUT_ROOT / 'task_review_2026-09-16'
ROOT = OUT_ROOT / 'task_review_2026-09-16_range15'
DATASET_PATH = 'data/force_calibration_review/2026-09-16-range15'
SPACE_PATH = 'task-review-2026-09-16-range15'
REPO = 'yxma/React'
SPACE = 'yxma/react-force-recovery'
SPACE_HOST = 'https://yxma-react-force-recovery.static.hf.space'
EPISODES = [
    ('pushT', '2026-09-12', 'episode_000'),
    ('pushT', '2026-09-11', 'episode_001'),
    ('pushT', '2026-09-11', 'episode_003'),
    ('pushT', '2026-09-10', 'episode_004'),
    ('motherboard', '2026-09-11', 'episode_000'),
    ('rope', '2026-09-14', 'episode_002'),
    ('toy', '2026-09-14', 'episode_000'),
]


def renderer():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    import build_episode_previews
    return build_episode_previews


def save_cache(payload, path):
    """Readers can see either complete version, never a partially written array."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name+'.', delete=False) as f:
        temp = Path(f.name)
        try:
            joblib.dump(payload, f, compress=3)
            f.flush()
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)


@contextmanager
def threshold(value):
    old = CF.VALID_DI
    try:
        CF.VALID_DI = value
        yield
    finally:
        CF.VALID_DI = old


def models():
    old_cache = RC.CACHE
    try:
        RC.CACHE = old_cache.with_name('glowtact_round_mm.pre_dI4_2026-09-16.json')
        before, held_old = RC.fit(report=False, holdout=True, legacy=True)
    finally:
        RC.CACHE = old_cache
    intermediate = RC.fit(report=False, extend_range=False)
    after, held_new = RC.fit(report=False, holdout=True)
    low = held_new['f'] <= 8
    high = ~low
    return before, intermediate, after, {
        'historical_di8_low_range_mae_n': float(np.abs(held_old['pred'] - held_old['f']).mean()),
        'historical_di8_n_heldout': len(held_old['f']),
        'v7_full_range_mae_n': float(np.abs(held_new['before_pred'] - held_new['f']).mean()),
        'v8_full_range_mae_n': float(np.abs(held_new['pred'] - held_new['f']).mean()),
        'n_heldout': len(held_new['f']), 'range_n': [0, RC.F_MAX_N],
        'after_force_ceiling_n': after.force_ceiling_n,
        'before_v7_force_ceiling_n': intermediate.force_ceiling_n,
        'low_range_mae_n': float(np.abs(held_new['pred'][low]-held_new['f'][low]).mean()),
        'high_range_mae_n': float(np.abs(held_new['pred'][high]-held_new['f'][high]).mean()),
        'tail': after.tail_info,
        'react_force_ground_truth': False,
        'note': 'Historical dI8 gain fit includes held-out z; after uses train-only gain. '
                'v8 preserves v7 below the old score ceiling and adds a measured 8-15 N tail. '
                'React absolute MAE remains unvalidated.'}


def jobs():
    path = ROOT / 'plan.json'
    if path.exists():
        return json.loads(path.read_text())
    previous = PREVIOUS_ROOT / 'plan.json'
    if previous.exists():
        ROOT.mkdir(parents=True, exist_ok=True)
        path.write_text(previous.read_text())
        return json.loads(path.read_text())
    result = []
    bep = renderer()
    for task, date, episode in EPISODES:
        t = pq.read_table(STAGE_ROOT / task / 'meta' / date / f'{episode}.parquet')
        score = np.zeros(len(t))
        for side in ('left', 'right'):
            x = t[f'tactile_{side}_intensity'].to_numpy()
            fresh = t[f'tactile_{side}_is_new'].to_numpy()
            light = (x >= 3) & (x < 6)
            # Choose visible transitions and weak contact, independent of force predictions.
            score += fresh * (light + 0.3 * (x >= 6))
            score[1:] += 2 * (light[1:] != light[:-1])
        bad = np.zeros(len(t), bool)
        for a, b, _ in bep._flagged_intervals(task, date, episode):
            bad[max(a, 0):min(b + 1, len(t))] = True
        candidates = np.arange(0, len(t) - 899, 30)
        if not len(candidates):
            raise ValueError(f'{task}/{episode}: less than 30 seconds')
        clean = np.array([not bad[i:i+900].any() for i in candidates])
        if clean.any():
            candidates = candidates[clean]
        scores = np.array([score[i:i+900].sum() for i in candidates])
        start = int(candidates[np.argmax(scores)])
        result.append(dict(id=f'{task}_{date}_{episode}', task=task, date=date,
                           episode=episode, row_start=start, n_frames=900,
                           source_start=int(t['source_h5_frame'][start].as_py()),
                           flagged_frames=int(bad[start:start+900].sum())))
    ROOT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    return result


def summarize(rows, before, intermediate, after, intensity, fresh, ceiling):
    rows = np.asarray(rows)
    mask = np.asarray(fresh, bool)
    x, a, b, c = intensity[mask], before[mask], intermediate[mask], after[mask]
    report = {'n_rows': len(rows), 'n_fresh': int(mask.sum()),
              'finite': bool(np.isfinite(after).all()),
              'candidate_range_n': [float(after.min()), float(after.max())],
              'candidate_ceiling_fraction': float(np.mean(c >= ceiling - 1e-6)),
              'bands': {}}
    for label, pick in [('low_intensity_lt3', x < 3), ('light_proxy_3_6', (x >= 3) & (x < 6)),
                        ('strong_proxy_ge6', x >= 6)]:
        n = int(pick.sum())
        report['bands'][label] = {'n': n}
        if n:
            report['bands'][label].update({
                'before_nonzero_fraction': float(np.mean(a[pick] > 0.02)),
                'v7_nonzero_fraction': float(np.mean(b[pick] > 0.02)),
                'after_nonzero_fraction': float(np.mean(c[pick] > 0.02)),
                'after_gt_0_1_fraction': float(np.mean(c[pick] > 0.1)),
                'after_median_n': float(np.median(c[pick])),
                'after_p95_n': float(np.quantile(c[pick], 0.95))})
    duplicates = ~mask
    duplicates[0] = False
    report['duplicate_force_mismatches'] = int(np.sum(
        duplicates[1:] & (after[1:] != after[:-1])))
    return report


def extension_metrics(v7, after, area, fresh, noise, ceiling):
    weights = np.array([RC.contact_weight(v, noise) for v in area])
    # The last v7 knot is its unique maximum. Undo the gate when deciding
    # whether a score lies below that knot; a weak gated plateau is not low-score.
    below_join = fresh & (weights > 0) & (v7 < ceiling*weights-1e-9)
    return {
        'v7_ceiling_frames': int(np.sum(fresh & (v7 >= ceiling-1e-6))),
        'above_8_n_frames': int(np.sum(fresh & (after > 8))),
        'nonzero_decision_changes': int(np.sum(fresh & ((v7 > .02) != (after > .02)))),
        'n_below_old_score_ceiling': int(below_join.sum()),
        'max_change_below_old_score_ceiling_n': float(np.max(np.abs(
            after[below_join]-v7[below_join]), initial=0))}


def extract(job):
    out = ROOT / 'cache' / f"{job['id']}.joblib"
    if out.exists():
        print('cached', job['id'], flush=True)
        return
    before, intermediate, after, calibration = models()
    previous_path = PREVIOUS_ROOT / 'cache' / f"{job['id']}.joblib"
    previous = joblib.load(previous_path) if previous_path.exists() else None
    task, date, episode = (job[k] for k in ('task', 'date', 'episode'))
    t = pq.read_table(STAGE_ROOT / task / 'meta' / date / f'{episode}.parquet')
    raw = DATA_ROOT / task / date / f'{episode}.h5'
    meta = open_episode(raw, task)
    start, count = job['row_start'], job['n_frames']
    output = {'job': job, 'calibration': calibration, 'sides': {}}
    with h5py.File(raw, 'r') as h5:
        for side in ('left', 'right'):
            frames = h5[f'gelsight/{side}/frames']
            idx = np.asarray(meta.align[side].index_map, int)
            intensity = t[f'tactile_{side}_intensity'].to_numpy()
            fresh = t[f'tactile_{side}_is_new'].to_numpy()
            refs = reference_stack(frames, idx, intensity, fresh)
            cropped = np.stack([crop(im).astype(np.float32) for im in refs])
            ref = np.median(cropped, axis=0)
            noise = reference_noise_area(cropped)
            ref_rows = _reference_rows(intensity, fresh)
            probe = ref_rows[12:]
            probe_force = [after(RC.force_stages(crop(frames[int(idx[r])]).astype(np.float32), ref),
                                 noise_area_mm2=noise) for r in probe]
            # Uniform fresh-frame audit outside the selected window limits selection bias.
            available = np.flatnonzero(fresh)
            audit_rows = np.unique(available[np.linspace(0, len(available)-1, 96).astype(int)])
            desired = set(range(start, start+count)) | set(audit_rows)
            trace = {k: [] for k in ('before', 'v7', 'after', 'area', 'source')}
            previous_side = previous['sides'][side] if previous else None
            prior_audit = {int(a[0]): a[1] for a in previous_side['audit_rows']} if previous else {}
            depths, masks = [], []
            audit = []
            last_source, last = None, None
            for n, row in enumerate(sorted(desired)):
                # Repeated captures use the last fresh source even at a clip boundary.
                preceding = available[available <= row]
                source_row = int(preceding[-1]) if len(preceding) else row
                source = int(idx[source_row])
                if source != last_source:
                    im = crop(frames[source]).astype(np.float32)
                    st = RC.force_stages(im, ref)
                    if previous_side is not None:
                        old_force = (previous_side['before'][row-start] if start <= row < start+count
                                     else prior_audit[row])
                    else:
                        with threshold(8):
                            old_st = RC.force_stages(im, ref)
                        old_force = before(old_st)
                    middle_force = intermediate(st, noise_area_mm2=noise)
                    new_force = after(st, noise_area_mm2=noise)
                    depth = cv2.resize(st['depth'].astype(np.float32), (160, 120))
                    mask = cv2.resize(st['contact'].astype(np.uint8)*255,
                                      (160, 120), interpolation=cv2.INTER_NEAREST)
                    last = (old_force, middle_force, new_force, st['feats']['area'],
                            source, depth, mask)
                    last_source = source
                if row in audit_rows:
                    audit.append((row, *last[:4]))
                if start <= row < start+count:
                    for key, value in zip(trace, last[:5]):
                        trace[key].append(value)
                    depths.append(last[5])
                    masks.append(last[6])
                if n % 200 == 0:
                    print(job['id'], side, n, '/', len(desired), flush=True)
            trace = {k: np.asarray(v) for k, v in trace.items()}
            rows = np.arange(start, start+count)
            if previous_side is not None:
                np.testing.assert_allclose(trace['v7'], previous_side['after'], atol=1e-10, rtol=0)
                np.testing.assert_array_equal(trace['source'], previous_side['source'])
                np.testing.assert_array_equal(masks, previous_side['mask'])
                np.testing.assert_allclose(trace['area'], previous_side['area'], atol=0, rtol=0)
            metric = summarize(rows, trace['before'], trace['v7'], trace['after'],
                               intensity[rows], fresh[rows], after.force_ceiling_n)
            metric.update(extension_metrics(trace['v7'], trace['after'], trace['area'],
                                            fresh[rows], noise, intermediate.force_ceiling_n))
            ar = np.array(audit)
            audit_metric = summarize(ar[:, 0], ar[:, 1], ar[:, 2], ar[:, 3],
                                     intensity[ar[:, 0].astype(int)], np.ones(len(ar), bool),
                                     after.force_ceiling_n)
            output['sides'][side] = dict(**trace, depth=np.asarray(depths), mask=np.asarray(masks),
                intensity=intensity[rows], fresh=fresh[rows], metric=metric,
                audit_metric=audit_metric, audit_rows=ar.tolist(), noise_area_mm2=noise,
                reference_rows=ref_rows[:12].tolist(), baseline_probe_rows=probe.tolist(),
                baseline_probe_force_n=probe_force,
                reference_intensity_range=[float(intensity[ref_rows[:12]].min()),
                                           float(intensity[ref_rows[:12]].max())])
            # Full row indexing is required by the existing renderer. Outside-window
            # values stay NaN and the file is explicitly NOT an exportable episode.
            force = np.full(len(t), np.nan, np.float32)
            force[rows] = trace['after']
            source_indices = np.full(len(t), -1, np.int32)
            source_indices[rows] = trace['source']
            folder = ROOT / 'preview_forces' / task / date
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder / f'{episode}_{side}.npz', force_normal_n=force,
                source_frame=source_indices, review_only=True, valid_rows=rows,
                calibration=RC.CALIBRATION_NAME, noise_area_mm2=noise)
            print(side, json.dumps(metric), 'noise', noise, 'baseline', probe_force, flush=True)
    save_cache(output, out)


def put(im, text, xy, scale=0.43, color=(220, 220, 220)):
    cv2.putText(im, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def append_review(panel, i, cache):
    n, width = cache['job']['n_frames'], panel.shape[1]
    band = np.full((210, width, 3), 24, np.uint8)
    for s, side in enumerate(('left', 'right')):
        d = cache['sides'][side]
        x = s * 320
        peak = d['display_depth_peak']
        heat = cv2.applyColorMap(np.uint8(np.clip(d['depth'][i]/peak, 0, 1)*255), cv2.COLORMAP_VIRIDIS)
        band[29:149, x:x+160] = cv2.cvtColor(d['mask'][i], cv2.COLOR_GRAY2BGR)
        band[29:149, x+160:x+320] = heat
        put(band, f'{side.upper()} contact', (x+5, 20))
        put(band, f'depth 0-{peak:.2f} a.u.', (x+163, 20), .37)
        put(band, f"area {d['area'][i]:.2f} mm2  |  source {int(d['source'][i])}", (x+5, 168), .4)
        put(band, f"dI8 {d['before'][i]:.2f}  /  v7 {d['v7'][i]:.2f}  /  v8 {d['after'][i]:.2f} N",
            (x+5, 188), .38)
        status = ('CALIBRATION CEILING' if d['after'][i] >= cache['calibration']['after_force_ceiling_n']-1e-6
                  else 'below reference noise' if d['area'][i] <= max(d['noise_area_mm2'], 0.052)
                  else 'provisional force')
        put(band, status, (x+5, 205), .34, (120, 180, 240))
        gx, gy, gw, gh = 650 + s*310, 45, 294, 130
        put(band, f'{side.upper()} normal-force estimate', (gx, 20), .42)
        for val in (0, 5, 10, 15):
            yy = gy + gh - round(val/15*gh)
            cv2.line(band, (gx, yy), (gx+gw, yy), (65, 65, 65), 1)
            put(band, str(val), (gx, yy-3), .3)
        for key, col in [('before', (60, 150, 240)), ('v7', (140, 140, 140)),
                         ('after', (170, 220, 70))]:
            xx = np.linspace(gx, gx+gw, n)
            yy = gy+gh - np.clip(d[key]/15, 0, 1)*gh
            cv2.polylines(band, [np.column_stack((xx, yy)).astype(np.int32)], False, col, 1, cv2.LINE_AA)
        cursor = gx+round(i/max(1, n-1)*gw)
        cv2.line(band, (cursor, gy), (cursor, gy+gh), (255, 255, 255), 1)
        put(band, 'dI8 old', (gx, 194), .34, (60, 150, 240))
        put(band, 'v7 (8 N)', (gx+86, 194), .34, (140, 140, 140))
        put(band, 'v8 (15 N)', (gx+174, 194), .34, (170, 220, 70))
    # Canonical renderer's status bar remains intact. The review label is separate.
    warning = cache.get('render_warning')
    label = np.full((46 if warning else 26, width, 3), 238, np.uint8)
    put(label, f'REVIEW v8 | {i/30:.1f}/30.0 s | Measured calibration up to 15 N | Absolute N unverified on React',
        (10, 18), .45, (30, 30, 30))
    if warning:
        put(label, warning, (10, 38), .43, (20, 80, 150))
    return np.vstack((panel, label, band))


def render(job):
    dest = ROOT / 'public' / f"{job['id']}.mp4"
    if dest.exists():
        print('render cached', dest, flush=True)
        return
    cache = joblib.load(ROOT / 'cache' / f"{job['id']}.joblib")
    for d in cache['sides'].values():
        d['display_depth_peak'] = max(float(np.quantile(d['depth'], 0.995)), 1e-6)
    bep = renderer()
    from twm.calib_epoch import world_offset_m
    from twm.force_recovery.dexforce import gel_axis
    task, date, episode = (job[k] for k in ('task', 'date', 'episode'))
    show_targets = True
    try:
        bep._release_poses(task, date, episode)
    except bep.UndeclaredPoseFrame:
        show_targets = False
        cache['render_warning'] = 'Virtual target omitted: release pose frame is undeclared'
        print(cache['render_warning'], flush=True)
    cams, left, right, up = bep._load_proj_calibs(task, date)
    delta = world_offset_m(task, date, episode, up_axis='y')
    axes = {s: gel_axis(task, s) for s in ('left', 'right')}
    temp = ROOT / 'intermediate' / f"{job['id']}.mp4"
    bep.build_one_preview(DATA_ROOT/task/date/f'{episode}.h5', temp, 30, 1,
        cams, left, right, *delta, proj_up_axis=up, press_axes=axes,
        window_start=job['source_start'], force_root=ROOT/'preview_forces',
        show_virtual_targets=show_targets,
        frame_transform=lambda p, frame: append_review(p, frame-job['source_start'], cache))
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(temp), '-c:v', 'libx264',
                    '-threads', '2', '-pix_fmt', 'yuv420p', '-crf', '20', '-movflags', '+faststart', str(dest)], check=True)
    if bep._decoded_frame_count(dest) != job['n_frames']:
        raise RuntimeError(f'Incomplete encoded review: {dest}')
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-ss', '15', '-i', str(dest),
                    '-frames:v', '1', str(dest.with_suffix('.jpg'))], check=True)
    dest.with_suffix('.render.json').write_text(json.dumps(
        {'virtual_targets': show_targets, 'warning': cache.get('render_warning')}, indent=2))
    print('rendered', dest, flush=True)


def report():
    data = []
    calibration = models()[3]
    (ROOT/'public').mkdir(parents=True, exist_ok=True)
    for job in jobs():
        p = ROOT / 'cache' / f"{job['id']}.joblib"
        if not p.exists():
            continue
        cache = joblib.load(p)
        for d in cache['sides'].values():
            d['metric'].pop('max_change_below_old_ceiling_n', None)
            d['metric'].update(extension_metrics(d['v7'], d['after'], d['area'], d['fresh'],
                d['noise_area_mm2'], calibration['before_v7_force_ceiling_n']))
            near_edge = (d['mask'][:, :2].any(axis=(1, 2)) |
                         d['mask'][:, -2:].any(axis=(1, 2)) |
                         d['mask'][:, :, :2].any(axis=(1, 2)) |
                         d['mask'][:, :, -2:].any(axis=(1, 2)))
            d['metric']['near_crop_edge_fraction'] = float(near_edge[d['fresh']].mean())
        columns = [np.arange(job['n_frames'])/30,
                   np.arange(job['row_start'], job['row_start']+job['n_frames'])]
        names = ['clip_time_s', 'release_row']
        for side, d in cache['sides'].items():
            for key in ('source', 'before', 'v7', 'after', 'area', 'intensity', 'fresh'):
                columns.append(d[key])
                names.append(f'{side}_{key}')
        np.savetxt(ROOT/'public'/f"{job['id']}.csv", np.column_stack(columns),
                   delimiter=',', header=','.join(names), comments='', fmt='%.8g')
        data.append({'job': job, 'calibration': calibration,
            'baseline_definition': 'dI8 and v7 use the same aligned images/reference as v8, '
                                   'not the published NPZ values.',
            'sides': {
            s: {k: d[k] for k in ('metric', 'audit_metric', 'noise_area_mm2',
                  'reference_intensity_range', 'reference_rows', 'baseline_probe_rows',
                  'baseline_probe_force_n')} for s, d in cache['sides'].items()}})
    (ROOT/'public'/'metrics.json').write_text(json.dumps(data, indent=2))
    (ROOT/'public'/'calibration.json').write_text(json.dumps(RC.range_report(), indent=2))
    return data


def page():
    data = report()
    base = f'https://huggingface.co/datasets/{REPO}/resolve/main/{DATASET_PATH}/'
    sections = []
    ready_count = 0
    for task in ('pushT', 'motherboard', 'rope', 'toy'):
        clips = []
        for record in data:
            j = record['job']
            if j['task'] != task:
                continue
            if not (ROOT/'public'/f"{j['id']}.jpg").exists():
                clips.append(f'<article><h3>{j["date"]} / {j["episode"]}</h3><p>渲染中</p></article>')
                continue
            ready_count += 1
            rows = []
            for side, d in record['sides'].items():
                m = d['metric']; b = m['bands']['light_proxy_3_6']
                light = ('--' if not b['n'] else
                         f"{100*b['before_nonzero_fraction']:.1f}% / "
                         f"{100*b['v7_nonzero_fraction']:.1f}% / {100*b['after_nonzero_fraction']:.1f}%")
                baseline = d['baseline_probe_force_n']
                baseline_peak = f'{max(baseline):.2f} N' if baseline else '--'
                rows.append(f'<tr><td>{side}</td><td>{m["n_fresh"]}</td><td>{b["n"]}</td>'
                    f'<td>{light}</td><td>{m["candidate_range_n"][1]:.2f} N</td>'
                    f'<td>{100*m["candidate_ceiling_fraction"]:.1f}%</td>'
                    f'<td>{d["noise_area_mm2"]:.2f} mm²</td>'
                    f'<td>{100*m["near_crop_edge_fraction"]:.1f}%</td>'
                    f'<td>{baseline_peak}</td></tr>')
            title = f'{j["date"]} / {j["episode"]} / {j["row_start"]/30:.1f}–{(j["row_start"]+900)/30:.1f}s'
            render_note = ROOT/'public'/f"{j['id']}.render.json"
            warning = ''
            if render_note.exists() and not json.loads(render_note.read_text())['virtual_targets']:
                warning = '<p class="status">Release 位姿坐标系未声明：此段省略虚拟目标点，未猜测坐标变换。</p>'
            clips.append(f'<article><h3>{html.escape(title)}</h3>'
                + warning +
                f'<video controls playsinline preload="none" poster="{base}{j["id"]}.jpg" src="{base}{j["id"]}.mp4"></video>'
                '<div class="scroll"><table><thead><tr><th>传感器</th><th>新采集帧</th><th>轻接触代理帧</th>'
                '<th>非零占比：dI8 / v7 / v8</th><th>最大估计</th><th>达到标定上限</th><th>参考噪声面积</th>'
                '<th>接近视野边缘</th><th>留出低强度帧峰值</th></tr></thead><tbody>'
                + ''.join(rows) + '</tbody></table></div>'
                f'<a href="{base}{j["id"]}.mp4" download>MP4</a> · '
                f'<a href="{base}{j["id"]}.csv" download>力曲线 CSV</a></article>')
        sections.append(f'<section id="{task}"><h2>{task}</h2>{"".join(clips)}</section>')
    calibration = data[0]['calibration'] if data else {}
    text = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    text += '<title>React · Force Calibration Review</title><style>'
    text += ('*{box-sizing:border-box;letter-spacing:0}body{margin:0;background:#f5f6f7;color:#20272b;'
             'font:15px/1.6 system-ui,sans-serif}main{max-width:1320px;margin:auto;padding:22px 20px 60px}'
             'h1{font-size:26px;margin:0 0 8px}h2{font-size:22px;margin:28px 0 8px}h3{font-size:16px;margin:14px 0 8px}'
             'p{max-width:1000px;margin:8px 0}nav{display:flex;gap:24px;flex-wrap:wrap;margin:18px 0}'
             'summary{cursor:pointer;min-height:36px;padding:6px 0;color:#536069}'
             'a{color:#087769;min-height:40px;display:inline-flex;align-items:center}video{display:block;width:100%;'
             'aspect-ratio:auto 1280/1004;background:#171b1d}article{border-top:1px solid #cbd2d5;padding:8px 0 24px}'
             '.scroll{overflow:auto}table{width:100%;border-collapse:collapse;white-space:nowrap;font-size:13px}'
             'th,td{padding:8px 12px;text-align:left;border-bottom:1px solid #d7dcdf}th{font-weight:500;color:#536069}'
             '.status{border-left:3px solid #b26e12;padding-left:12px}footer{color:#536069;font-size:13px}'
             '@media(max-width:600px){main{padding:16px 12px}h1{font-size:22px}nav{gap:18px}}')
    text += '</style><main><h1>React · Force Calibration Review</h1>'
    text += (f'<p class="status">候选 v8 · 0–15 N · 视频 {ready_count}/7 · 未覆盖正式数据 · '
             'React 绝对力未验证</p>')
    text += (f'<p>低力段映射与接触门控保留 v7；高力端加入实测标定。'
             f'位置留出 MAE：≤8 N {calibration.get("low_range_mae_n",0):.3f} N，'
             f'8–15 N {calibration.get("high_range_mae_n",0):.3f} N。'
             f'拟合上限 {calibration.get("after_force_ceiling_n",0):.2f} N。</p>')
    text += (f'<p>相同位置留出样本的全量程 MAE：v7 {calibration.get("v7_full_range_mae_n",0):.3f} N'
             f' → v8 {calibration.get("v8_full_range_mae_n",0):.3f} N（不是 React 误差）。</p>'
             '<details><summary>评估口径与已知限制</summary><p>'
             '轻接触代理为 intensity ∈ [3, 6)，非零力为 &gt; 0.02 N；并非人工接触召回率。'
             '曲线为 dI8 历史基线、v7（8 N）和 v8（15 N）候选。深度单位 a.u.，不是毫米。</p>')
    text += ('<p>高力样本若落在旧映射的低分数区间，仍会被低估；扩展量程不等于消除所有饱和。'
             '标定集没有精确零力标签，近零门控仍是工程规则。</p>')
    text += '<p>历史模型也使用相同对齐帧和参考图重算，曲线不是直接读取旧发布文件。</p>'
    text += ('<p>低强度帧也可能是参考漂移，而非接触。低于参考噪声时显示 0，表示无法分辨力信号，'
             '不等于测得无力；达到标定上限时真实力可能更大。参考图仍依赖低强度帧近似空载。'
             '接触位于触觉图边缘时，重建可能放大微弱信号，这类牛顿值需要特别谨慎核查。'
             '这些检查不证明 0.5 N 精度或 OOD 安全。</p></details>')
    text += '<nav>' + ''.join(f'<a href="#{t}">{t}</a>' for t in ('pushT','motherboard','rope','toy'))
    text += (f'<a href="{base}metrics.json">评估数据 JSON</a>'
             f'<a href="{base}calibration.json">标定误差 JSON</a></nav>') + ''.join(sections)
    text += '<a href="../task-review-2026-09-16/index.html">上一版 v7 预览</a>'
    text += ('<footer>7 个定向选择的窗口，每段 30 秒、1×；按轻接触与切换次数选取，'
             '不代表全部任务分布。JSON 另含每个 episode 双侧各 96 个均匀抽样新帧的检查，'
             '以及未用于参考图的低强度帧检查。轻接触权重是工程规则，不是力真值标定。</footer></main></html>')
    (ROOT/'public'/'index.html').write_text(text)
    (ROOT/'public'/'README.md').write_text('# React force calibration review\n\n'
        f'[Open the review page]({SPACE_HOST}/{SPACE_PATH}/index.html)\n\n'
        'Experimental v8, seven 30-second clips, four tasks. Absolute React force accuracy is unverified.\n'
        'Depth is in arbitrary units. Measured 8-15 N calibration extends the unchanged v7 low-score curve.\n')


def publish():
    from huggingface_hub import HfApi
    verify()
    page()
    for j in jobs():
        p = ROOT/'public'/f"{j['id']}.mp4"
        if not p.exists() or renderer()._decoded_frame_count(p) != 900:
            raise ValueError(f'Missing or incomplete video: {p}')
    api = HfApi()
    result = api.upload_folder(repo_id=REPO, repo_type='dataset', folder_path=ROOT/'public',
        path_in_repo=DATASET_PATH, commit_message='Add 0-15 N v8 force review with unchanged contact sensitivity')
    print('dataset', result, flush=True)
    result = api.upload_file(repo_id=SPACE, repo_type='space', path_or_fileobj=ROOT/'public'/'index.html',
        path_in_repo=f'{SPACE_PATH}/index.html', commit_message='Add four-task force review page without replacing existing results')
    print('space', result, flush=True)


def verify():
    """Re-read actual named sensor frames; no index formula substitutes for pixels."""
    _, _, predict, _ = models()
    checks = []
    for j in jobs():
        cache = joblib.load(ROOT/'cache'/f"{j['id']}.joblib")
        task, date, episode = (j[k] for k in ('task', 'date', 'episode'))
        t = pq.read_table(STAGE_ROOT/task/'meta'/date/f'{episode}.parquet')
        raw = DATA_ROOT/task/date/f'{episode}.h5'
        meta = open_episode(raw, task)
        with h5py.File(raw, 'r') as h5:
            for side, d in cache['sides'].items():
                frames = h5[f'gelsight/{side}/frames']
                stack = reference_stack(frames, meta.align[side].index_map,
                    t[f'tactile_{side}_intensity'].to_numpy(), t[f'tactile_{side}_is_new'].to_numpy())
                cropped = np.stack([crop(im).astype(np.float32) for im in stack])
                ref = np.median(cropped, axis=0)
                noise = reference_noise_area(cropped)
                if abs(noise-d['noise_area_mm2']) > 1e-9:
                    raise AssertionError('Reference noise mismatch')
                stored = np.load(ROOT/'preview_forces'/task/date/f'{episode}_{side}.npz')
                for i in sorted(set([0, 450, 899, int(np.argmax(d['after']))])):
                    source = int(d['source'][i])
                    r = j['row_start']+i
                    if int(stored['source_frame'][r]) != source:
                        raise AssertionError('Preview source mismatch')
                    st = RC.force_stages(crop(frames[source]).astype(np.float32), ref)
                    f = predict(st, noise_area_mm2=noise)
                    if abs(f-d['after'][i]) > 1e-8 or abs(f-float(stored['force_normal_n'][r])) > 1e-6:
                        raise AssertionError(f'Force mismatch {j["id"]}/{side}/{r}')
                    checks.append({'clip': j['id'], 'side': side, 'row': r,
                                   'source_frame': source, 'force_n': f})
        video = ROOT/'public'/f"{j['id']}.mp4"
        p = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(video), '-f', 'null', '-'],
                           capture_output=True, text=True)
        if p.returncode or p.stderr.strip() or renderer()._decoded_frame_count(video) != 900:
            raise AssertionError(f'Video decode failed: {video}: {p.stderr}')
        print('verified', j['id'], flush=True)
    sources = [Path(__file__), Path(RC.__file__), Path(CF.__file__),
               Path(__file__).with_name('run_episode.py'), Path(renderer().__file__)]
    (ROOT/'public'/'verification.json').write_text(json.dumps(
        {'named_frame_checks': checks, 'fully_decoded_videos': len(jobs()),
         'tolerance_n': 1e-6, 'not_a_force_accuracy_test': True,
         'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
         'calibration_cache_sha256': hashlib.sha256(RC.CACHE.read_bytes()).hexdigest(),
         'tail_cache_sha256': hashlib.sha256(RC.TAIL_CACHE.read_bytes()).hexdigest()}, indent=2))
    print(len(checks), 'force/frame checks passed', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['extract', 'render', 'page', 'verify', 'publish'])
    parser.add_argument('--clip')
    args = parser.parse_args()
    if args.command in ('page', 'verify', 'publish'):
        globals()[args.command]()
        return
    for job in jobs():
        if args.clip is None or args.clip == job['id']:
            globals()[args.command](job)
    report()


if __name__ == '__main__':
    main()
