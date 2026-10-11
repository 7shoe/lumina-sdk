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
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--cases', nargs='+', default=['case30', 'case57', 'case118'])
    parser.add_argument('--qualify', action='store_true', help='Three-point generation plus restart and student checks')
    args = parser.parse_args()
    
    if (args.root / 'manifest.json').exists():
        raise FileExistsError('Use the existing PBS manifest to resume')
    
    experiments = []
    for case in args.cases:
        for domain, bounds in (('opfdata_envelope', [.8, 1.2]), ('paper', [.5, 2.])):
            for seed in args.seeds:
                for method in ('rambo', 'uniform'):
                    config = configuration(case=case, method=method, seed=seed, load_bounds=bounds,
                                           count=3 if args.qualify else args.count)
                    output = args.root / case / domain / method / f'seed{seed}'
                    experiments.append(dict(config=config, output=str(output.resolve())))
    
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
