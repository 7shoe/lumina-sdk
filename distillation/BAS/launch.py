"""Write matched baseline configurations and freeze source for PBS execution."""
import argparse
from pathlib import Path
import shutil

from distillation.training.T1_single_topo.common import ROOT, file_hash, write_json
from .models import model_options
from .run import configuration


def freeze_source(destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    hashes = {}
    for directory in ('lumina', 'experiments', 'distillation'):
        # sort paths for a reproducible traversal of the source tree
        for path in sorted((ROOT / directory).rglob('*')):
            relative = path.relative_to(ROOT)
            
            # exclude generated outputs/caches and retain only listed file types
            if (not path.is_file() or any(part in {'runs', 'reports', '__pycache__'} for part in relative.parts)
                    or path.suffix not in {'.py', '.yaml', '.pbs', '.md', '.ipynb'}):
                continue
            
            # preserve the repository layout
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            hashes[str(relative)] = file_hash(target)
            
    # store the checksum manifest (alongside src dir)
    write_json(destination.with_name(destination.name + '_hashes.json'), hashes)


def main():
    """Build the campaign's experiment grid, source snapshot, and execution manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--cases', nargs='+', default=['case30', 'case57', 'case118'])
    parser.add_argument('--models', nargs='+', default=['MLP', 'GIN', 'HeteroGNN'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--methods', nargs='+', default=['bas_ig', 'bas_le', 'random', 'stratified', 'inactive', 'mcdue'])
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    
    # do not overwrite an existing campaign;
    if (args.root / 'manifest.json').exists():
        raise FileExistsError('Campaign manifests are immutable; reuse the same PBS command to resume')
    experiments = []
    # cartesian product
    for case in args.cases:
        for model in args.models:
            for seed in args.seeds:
                for method in args.methods:
                    c = configuration(case=case, model=model, model_seed=seed,
                        model_options=model_options(model, width=8 if args.smoke else 64, layers=2 if args.smoke else 3),
                        method='bas' if method.startswith('bas_') else method,
                        metric=method[4:] if method.startswith('bas_') else 'ig')
                    if args.smoke:
                        c.update(buckets=4, panel_size=32, initial_size=16, max_selected=48,
                                 acquisition_size=16, candidate_size=64, batch_size=8, score_batch_size=8,
                                 mc_draws=5, max_rounds=6, validation_size=32,
                                 schedule=dict(minimum=1e-6, maximum=.01, decrease=.5, increase=2., patience_lr=1, patience_acquire=1))
                    output = args.root / case / model / method / f'seed{seed}'
                    experiments.append(dict(config=configuration(**c), output=str(output.resolve())))
    args.root.mkdir(parents=True, exist_ok=True)
    
    # snapshot source before writing the manifest
    freeze_source(args.root / 'source')
    
    # write JSON
    write_json(args.root / 'manifest.json',
               dict(method='BAS finite-pool adaptation',
                    experiments=experiments))
    
    # print status
    print(f'{len(experiments)} independent experiments; {(len(experiments) + 11) // 12} node(s) for one worker per experiment')

if __name__ == '__main__':
    main()
