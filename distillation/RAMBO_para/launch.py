"""Freeze one independent RAMBO* run; prepare only, never submit implicitly."""
import argparse
import json
from pathlib import Path
import shutil

from distillation.experiments.study import source_files
from distillation.training.T1_single_topo.common import ROOT, file_hash, write_json
from .run import configuration, protect_baseline

RUNTIME = Path('/lus/flare/projects/FRAME-IDP/siebenschuh/acopf_datasets/dataset_selection/'
               'runtime/rambo_julia_20261014.tar')


def prepare(root, config, runtime=RUNTIME):
    root, runtime = Path(root).expanduser().resolve(), Path(runtime).expanduser().resolve()
    config = configuration(**config)
    protect_baseline(root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('Use a new empty RAMBO_para campaign root; resubmit its frozen PBS script to resume')
    if not runtime.is_file():
        raise FileNotFoundError(runtime)
    files = source_files()
    for name in ('distillation/RAMBO_para/aux/run.pbs', 'distillation/RAMBO_para/README.md'):
        files[name] = file_hash(ROOT / name)
    root.mkdir(parents=True, exist_ok=True)
    for name in files:
        target = root / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    write_json(root / 'requested_config.json', config)
    manifest = dict(format='lumina-rambo-para-campaign-v1', method='RAMBO_para', display_name='RAMBO*',
                    output=str(root / 'run'), config_sha256=file_hash(root / 'requested_config.json'),
                    sources=files, runtime_archive=str(runtime), runtime_sha256=file_hash(runtime))
    write_json(root / 'manifest.json', manifest)
    return manifest


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['format'] != 'lumina-rambo-para-campaign-v1':
        raise ValueError('Expected a separate RAMBO_para campaign')
    for name, expected in manifest['sources'].items():
        if file_hash(root / 'source' / name) != expected:
            raise ValueError(f'Frozen source changed: {name}')
    for path, expected in ((root / 'requested_config.json', manifest['config_sha256']),
                           (Path(manifest['runtime_archive']), manifest['runtime_sha256'])):
        if file_hash(path) != expected:
            raise ValueError(f'Frozen input changed: {path}')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--case', default='case500')
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--proposal-batch-size', type=int, default=32)
    parser.add_argument('--bootstrap', type=Path)
    parser.add_argument('--runtime-archive', type=Path, default=RUNTIME)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        manifest = verify(args.root)
        print(manifest['runtime_archive'])
    else:
        cfg = configuration(case=args.case, count=args.count, seed=args.seed,
                            proposal_batch_size=args.proposal_batch_size, bootstrap=args.bootstrap)
        manifest = prepare(args.root, cfg, args.runtime_archive)
        print(json.dumps(dict(method=manifest['method'], output=manifest['output'],
                              pbs=str(args.root.resolve() / 'source/distillation/RAMBO_para/aux/run.pbs'))))


if __name__ == '__main__':
    main()
