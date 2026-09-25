"""Compare CPU inference with a frozen pre-optimization package; no NPZ writes.

Run from the repository root with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1.
The baseline directory must contain the old force_recovery/__init__.py.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

import h5py
import hdf5plugin  # noqa: F401
import cv2
import numpy as np
import pyarrow.parquet as pq

JOBS = (
    ('pushT', '2026-09-11', 'episode_001'),
    ('motherboard', '2026-09-11', 'episode_000'),
    ('rope', '2026-09-11', 'episode_000'),
    ('toy', '2026-09-14', 'episode_000'),
)


def assert_equivalent(before, after):
    """Exact keys/labels/masks; at most 1e-10 absolute geometry error."""
    if isinstance(before, dict):
        assert before.keys() == after.keys(), (before.keys(), after.keys())
        for key in before:
            assert_equivalent(before[key], after[key])
    elif np.asarray(before).dtype.kind in 'fc':
        np.testing.assert_allclose(after, before, rtol=0, atol=1e-10, equal_nan=False)
    else:
        np.testing.assert_array_equal(after, before)


def load_baseline(directory):
    directory = Path(directory).resolve()
    name = 'force_speed_baseline'
    spec = importlib.util.spec_from_file_location(
        name, directory/'__init__.py', submodule_search_locations=[str(directory)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return name


def _modules(package):
    return {name: importlib.import_module(f'{package}.{name}') for name in
            ('run_episode', 'react_calib', 'debug_gallery', 'calib_free', 'lut_calibration')}


def _evaluate(modules, predict, images, ref, noise):
    out = []
    for img in images:
        geometry = modules['debug_gallery'].stages(img, ref)['feats']
        force = predict(modules['react_calib'].force_stages(img, ref), noise_area_mm2=noise)
        out.append([force, geometry['vol'], geometry['area'], geometry['maxd']])
    return np.asarray(out)


def benchmark(baseline_dir, *, frames=16, repeats=3):
    if frames < 1 or repeats < 2:
        raise ValueError('Need >=1 sample and >=2 alternating repeats')
    packages = {'before': load_baseline(baseline_dir), 'after': 'twm.force_recovery'}
    modules = {key: _modules(pkg) for key, pkg in packages.items()}
    run = modules['after']['run_episode']
    crop = modules['after']['lut_calibration'].crop
    predictors, fit_seconds = {}, {}
    for key, arm in modules.items():
        start = perf_counter()
        predictors[key] = arm['react_calib'].fit(report=False)
        fit_seconds[key] = perf_counter()-start
    records = []
    for task, date, episode in JOBS:
        raw = run.DATA_ROOT/task/date/(episode+'.h5')
        table = pq.read_table(run.STAGE_ROOT/task/'meta'/date/(episode+'.parquet'))
        source = run.open_episode(raw, task)
        assert source.T == len(table)
        np.testing.assert_array_equal(table['source_h5_frame'].to_numpy(),
                                      source.trim+np.arange(source.T))
        np.testing.assert_array_equal(table['timestamp'].to_numpy(), source.trimmed_cam_ts)
        with h5py.File(raw, 'r') as h5:
            for side in ('left', 'right'):
                captures = h5[f'gelsight/{side}/frames']
                fresh = table[f'tactile_{side}_is_new'].to_numpy()
                intensity = table[f'tactile_{side}_intensity'].to_numpy()
                index = source.align[side].index_map
                refs = np.stack([crop(im).astype(np.float32) for im in
                                 run.reference_stack(captures, index, intensity, fresh)])
                ref = np.median(refs, axis=0)
                noise = run.reference_noise_area(refs)
                assert noise == modules['before']['run_episode'].reference_noise_area(refs)
                choices = np.flatnonzero(fresh)
                uniform = choices[np.linspace(0, len(choices)-1, frames).astype(int)]
                ranked = choices[np.argsort(intensity[choices])]
                # Include low/high intensity and intermediate contact levels.
                stratified = ranked[np.linspace(0, len(ranked)-1, 8).astype(int)]
                rows = np.unique(np.r_[uniform, stratified])
                indices = index[rows]
                images = [crop(captures[int(i)]).astype(np.float32) for i in indices]
                max_force_error, max_geometry_error = 0., 0.
                for image in images:
                    force_stages, geometries, forces = {}, {}, {}
                    for key, arm in modules.items():
                        force_stages[key] = arm['react_calib'].force_stages(image, ref)
                        geometries[key] = arm['debug_gallery'].stages(image, ref)
                        forces[key] = predictors[key](force_stages[key], noise_area_mm2=noise)
                    assert_equivalent(force_stages['before'], force_stages['after'])
                    assert_equivalent(geometries['before'], geometries['after'])
                    error = abs(forces['after']-forces['before'])
                    assert error <= 1e-6
                    assert (forces['after'] > .02) == (forces['before'] > .02)
                    max_force_error = max(max_force_error, error)
                    max_geometry_error = max(max_geometry_error, float(np.max(np.abs(
                        geometries['before']['depth']-geometries['after']['depth']))))
                # The diagnostic API keeps normal maps by default.
                assert_equivalent(modules['before']['calib_free'].reconstruct(images[0], ref),
                                  modules['after']['calib_free'].reconstruct(images[0], ref))
                times = {mode: {key: [] for key in packages} for mode in ('compute', 'with_h5')}
                for mode in times:
                    for repeat in range(repeats):
                        order = ['before', 'after']
                        if (repeat+len(records)) % 2:
                            order.reverse()
                        outputs = {}
                        for key in order:
                            start = perf_counter()
                            source_images = (images if mode == 'compute' else
                                (crop(captures[int(i)]).astype(np.float32) for i in indices))
                            outputs[key] = _evaluate(modules[key], predictors[key], source_images, ref, noise)
                            times[mode][key].append(perf_counter()-start)
                        np.testing.assert_allclose(outputs['after'], outputs['before'], rtol=0, atol=1e-6)
                medians = {mode: {key: float(np.median(v)) for key, v in arms.items()}
                           for mode, arms in times.items()}
                record = dict(task=task, date=date, episode=episode, side=side,
                    rows=rows.tolist(), n_frames=len(rows), fresh_rows=int(fresh.sum()),
                    max_force_error_n=max_force_error, max_geometry_error_mm=max_geometry_error,
                    raw_seconds=times, median_seconds=medians,
                    speedup={mode: arms['before']/arms['after'] for mode, arms in medians.items()})
                records.append(record)
                print(json.dumps({k: record[k] for k in ('task', 'side', 'n_frames',
                                  'speedup', 'max_force_error_n')}), flush=True)
    total_frames = sum(r['n_frames'] for r in records)
    totals = {mode: {key: sum(r['median_seconds'][mode][key] for r in records)
                     for key in packages} for mode in ('compute', 'with_h5')}
    files = ('poisson.py', 'calib_free.py', 'debug_gallery.py', 'react_calib.py', 'run_episode.py')
    hashes = {key: {name: hashlib.sha256((Path(arm['run_episode'].__file__).parent/name).read_bytes()).hexdigest()
                    for name in files} for key, arm in modules.items()}
    return dict(n_frames=total_frames, repeats=repeats, model_fit_seconds=fit_seconds,
        environment=dict(python=sys.version, platform=platform.platform(),
            logical_cpus=os.cpu_count(), numpy=np.__version__, opencv=cv2.__version__,
            opencv_threads=cv2.getNumThreads(),
            openblas_threads=os.environ.get('OPENBLAS_NUM_THREADS'),
            omp_threads=os.environ.get('OMP_NUM_THREADS')),
        ms_per_frame={mode: {key: sec/total_frames*1000 for key, sec in arms.items()}
                      for mode, arms in totals.items()},
        speedup={mode: arms['before']/arms['after'] for mode, arms in totals.items()},
        source_sha256=hashes, records=records,
        note='Warm caches; repeated single-process timings under current host load. '
             'Setup, model fitting and NPZ writing excluded; not full-dataset throughput.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=16)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('--out must be a new report path')
    result = benchmark(args.baseline_dir, frames=args.frames, repeats=args.repeats)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ('n_frames', 'ms_per_frame', 'speedup')}, indent=2))
    print(args.out)


if __name__ == '__main__':
    main()
