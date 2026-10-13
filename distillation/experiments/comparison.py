"""Prepare matched DD/DCGM/HMDC manifests; launch with distillation.synthetic.campaign."""
import argparse
import json
from pathlib import Path

from distillation.DD import campaign as dd
from distillation.DC import campaign as dc
from distillation.HMDC import campaign as hm
from distillation.training.T1_single_topo.common import write_json
from experiments.lumina_bench.models import SUPPORTED_MODELS


def manifest(root, references, *, cases=(30,57,118), models=('GAT','GIN','HeteroGNN'),
             pairs=(('GAT','GIN'),('GIN','HeteroGNN')),
             protocols=('conservative','random_fixed','random_size'),
             targets=('solver','frozen','learned'), initialization='random', checkpoints=None,
             training_seconds=2400):
    root, references = Path(root).resolve(), Path(references).resolve()
    experiments = []
    members = set(models) | {k for pair in pairs for k in pair}
    if not members <= set(SUPPORTED_MODELS):
        raise ValueError('Select supported benchmark models')
    if initialization == 'pretrained' and not members <= set(checkpoints or {}):
        raise ValueError('Provide a pretrained checkpoint for every requested model')
    if initialization != 'pretrained' and checkpoints:
        raise ValueError('Checkpoints require initialization=pretrained')
    for case in cases:
        configurations = [('DD', model, dd.default_config(case, model)) for model in models]
        configurations += [('DCGM', model, dc.default_config(case, model)) for model in models]
        configurations += [('HMDC', '+'.join(pair), hm.default_config(case, pair)) for pair in pairs]
        for method, name, base in configurations:
            for protocol in protocols:
                for target in targets:
                    config = base | dict(initialization=initialization, generation_protocol=protocol,
                        target_mode=target, training_seconds=training_seconds,
                        reference_file=str(references/f'case{case}.pt'),
                        initial_cache=str(root/'initializations'/f'case{case}'/protocol))
                    if initialization == 'pretrained':
                        if method == 'HMDC':
                            config['pretrained_checkpoints'] = {k: checkpoints[k] for k in config['evaluation_models']}
                        else:
                            config['pretrained_checkpoint'] = checkpoints[name]
                    (dd.validate_config if method == 'DD' else hm.validate_config if method == 'HMDC'
                     else dc.validate_condensation)(config)
                    output = root/method/name/f'case{case}'/protocol/target/initialization
                    experiments.append(dict(method=method, config=config, output=str(output)))
    return dict(method='DD', experiments=experiments,
        comparison='100 grids; width 128/3 layers unless checkpoint-defined; FP64; 100 ordinary-SGD evaluation updates; '
                   'shared train references, initial grids, validation IDs and seed pools; method-specific synthesis objectives')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--references', type=Path, required=True, help='Directory containing training-only case<N>.pt reference files')
    parser.add_argument('--cases', type=int, nargs='+', default=[30,57,118])
    parser.add_argument('--models', choices=SUPPORTED_MODELS, nargs='+', default=['GAT','GIN','HeteroGNN'])
    parser.add_argument('--pairs', nargs='+', default=['GAT,GIN','GIN,HeteroGNN'])
    parser.add_argument('--protocols', choices=['conservative','random_fixed','random_size'], nargs='+',
                        default=['conservative','random_fixed','random_size'])
    parser.add_argument('--targets', choices=['solver','frozen','learned'], nargs='+', default=['solver','frozen','learned'])
    parser.add_argument('--initialization', choices=['random','fixed','pretrained'], default='random')
    parser.add_argument('--checkpoints', type=Path, help='JSON mapping model names to T1/T2 checkpoint paths')
    parser.add_argument('--training-seconds', type=float, default=2400)
    args = parser.parse_args()
    for case in args.cases:
        if not (args.references/f'case{case}.pt').is_file():
            parser.error(f'Missing training references: {args.references}/case{case}.pt')
    result = manifest(args.root, args.references, cases=args.cases, models=args.models,
        pairs=[p.split(',') for p in args.pairs], protocols=args.protocols, targets=args.targets,
        initialization=args.initialization, checkpoints=json.loads(args.checkpoints.read_text()) if args.checkpoints else None,
        training_seconds=args.training_seconds)
    path = args.root/'manifest.json'
    if path.exists() and json.loads(path.read_text()) != result:
        raise ValueError('Manifest differs; use another experiment directory')
    write_json(path, result)
    print(json.dumps(dict(experiments=len(result['experiments']), manifest=str(path))))


if __name__ == '__main__':
    main()
