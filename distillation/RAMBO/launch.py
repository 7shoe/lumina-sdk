"""Freeze a matched RAMBO/uniform operating-point campaign before submission."""
import argparse
from pathlib import Path
import shutil

from distillation.BAS.launch import freeze_source as freeze_python
from distillation.training.T1_single_topo.common import ROOT, file_hash, write_json
from .run import configuration


def freeze_source(destination):
    destination = Path(destination)
    freeze_python(destination)
    # Julia sources, lockfile, and nominal cases are part of the experiment.
    for path in (ROOT / 'distillation/RAMBO').rglob('*'):
        if any(part in {'runs', 'reports', '__pycache__'} for part in path.relative_to(ROOT).parts):
            continue
        if path.is_file() and (path.suffix in ('.jl', '.toml', '.m', '.sh') or path.name == 'LICENSE'):
            relative = path.relative_to(ROOT)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    hashes = {str(p.relative_to(destination)): file_hash(p) for p in sorted(destination.rglob('*')) if p.is_file()}
    write_json(destination.with_name(destination.name + '_hashes.json'), hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--counts', '--count', dest='counts', type=int, nargs='+', default=[100, 800])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--cases', nargs='+', default=['case14', 'case30', 'case57', 'case118', 'case500'])
    parser.add_argument('--samplers', nargs='+', choices=['rambo', 'uniform'], default=['rambo', 'uniform'])
    parser.add_argument('--paper-bounds', action='store_true', help='Use [0.5, 2.0] instead of the OPFData envelope')
    parser.add_argument('--max-attempts', type=int, help='Default: max(1000, 10 * count) per library')
    parser.add_argument('--qualify', action='store_true', help='Three-point generation plus restart and student checks')
    args = parser.parse_args()
    args.root = args.root.expanduser().resolve()
    
    if (args.root / 'manifest.json').exists():
        raise FileExistsError('Use the existing PBS manifest to resume')
    
    experiments = []
    domain, bounds = ('paper', [.5, 2.]) if args.paper_bounds else ('opfdata_envelope', [.8, 1.2])
    for case in args.cases:
        for count in ([3] if args.qualify else args.counts):
            for seed in args.seeds:
                for method in args.samplers:
                    config = configuration(case=case, method=method, seed=seed, load_bounds=bounds,
                                           count=count, max_attempts=args.max_attempts)
                    output = args.root / config['case'] / domain / method / f'M{count}' / f'seed{seed}'
                    experiments.append(dict(config=config, output=str(output.resolve())))
    if len({e['output'] for e in experiments}) != len(experiments):
        raise ValueError('Cases, counts, seeds and samplers must be unique')
    
    freeze_source(args.root / 'source')
    
    # write JSON
    write_json(args.root / 'manifest.json',
               dict(method='RAMBO operating-point generation',
                    qualify=args.qualify,
                    experiments=experiments))
    
    # stdout
    print(f'{len(experiments)} independent CPU chains; source frozen in {args.root / "source"}')

if __name__ == '__main__':
    main()
