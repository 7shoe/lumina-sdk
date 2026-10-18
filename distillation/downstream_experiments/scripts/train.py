"""One resumable student trainer; untouched test splits are evaluated only at the end."""
import argparse
from collections import Counter
from pathlib import Path
import time

import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Batch

from experiments.lumina_bench.runner import build_model, seed_all, select_device
from distillation.experiments.staging import case_dataset, local_path
from distillation.synthetic.learning import PrecisionModel, mse
from distillation.training.T1_single_topo.common import (
    ROOT, claim, digest, file_hash, read_json, source_identity as t1_sources, write_json)
from distillation.training.T1_single_topo.state import Stop, capture_rng, restore_rng, save_torch
from distillation.training.T1_single_topo.train import evaluate
from distillation.training.T2_multi_topo.train import graph_mse
from .data import case_name, materialize


def source_identity():
    files = list(Path(__file__).parent.glob('*.py')) + [Path(__file__).parents[1] / 'dataset_path.py']
    files += [ROOT / p for p in ('distillation/experiments/datasets.py',
        'distillation/experiments/staging.py', 'distillation/synthetic/learning.py',
        'distillation/synthetic/artifacts.py', 'distillation/synthetic/inputs.py',
        'distillation/selection/artifacts.py', 'distillation/training/T2_multi_topo/train.py')]
    return t1_sources() | {str(p.relative_to(ROOT)):file_hash(p) for p in files if p.exists()}


def folder_for(manifest, run):
    return Path(manifest['paths']['models']) / 'runs' / run['specification']['model'] / run['run_id']


def completed(folder, spec):
    marker = Path(folder) / 'complete.json'
    if not marker.exists():
        return False
    meta = read_json(marker)
    if meta['specification'] != digest(spec):
        raise ValueError('Completed run identity changed')
    for name, checksum in meta['files'].items():
        path = Path(folder) / name
        if not path.exists():
            return False
        if file_hash(path) != checksum:
            raise ValueError(f'Completed result changed: {path}')
    return True


def initial_model(spec, sample, device):
    """Load only explicitly requested initial weights; never export trained_state."""
    init, kind = spec['initialization'], spec['model']
    seed_all(init.get('seed', spec['seed']))
    model = build_model({'model_type':kind, 'models':{kind:init['model_options']}}, sample, torch.device('cpu'))
    if init.get('precision', 'float32') == 'float64':
        model = PrecisionModel(model, torch.float64)
    if init['kind'] != 'fresh':
        path = local_path(init['path'])
        if file_hash(path) != init['sha256']:
            raise ValueError('Initialization checkpoint checksum changed')
        payload = torch.load(path, weights_only=False, map_location='cpu')
        if init['kind'] == 'pretrained':
            saved = payload['specification']
            if (saved['format'] != 'lumina-t1-v1' or saved['split_seed'] != 42
                    or saved['model'] != kind or saved['case'] != case_name(init['source_case'])
                    or saved['model_options'] != init['model_options']
                    or saved['data_identity'] != init['data_identity']):
                raise ValueError('Incompatible T1 initialization')
            weights = payload['model']
        elif init['kind'] == 'recorded':
            student = payload.get('model_checkpoints', {}).get(kind, payload.get('model_checkpoint', {}))
            if (payload['config']['initialization'] != 'fixed' or student.get('model', kind) != kind
                    or student['seed'] != init['seed'] or student['model_options'] != init['model_options']):
                raise ValueError('Expected the exact fixed, untrained initialization')
            weights = student['initial_state']
        else:
            raise ValueError(f"Unknown initialization {init['kind']}")
        if any(not torch.isfinite(v).all() for v in weights.values()):
            raise ValueError('Nonfinite initialization')
        model.load_state_dict(weights, strict=True)
    model = model.to(device)
    # Separate weight initialization from shuffle/dropout randomness.
    seed_all(spec['seed'])
    return model


def next_batch(count, size, epoch, offset, seed):
    """Shuffle without replacement per epoch; retain a full batch across boundaries."""
    selected = []
    while len(selected) < size:
        order = np.random.default_rng(seed + epoch).permutation(count)
        take = min(size - len(selected), count - offset)
        selected.extend(order[offset:offset + take].tolist())
        offset += take
        if offset == count:
            epoch, offset = epoch + 1, 0
    return selected, epoch, offset


def train_run(manifest, run, *, device='cpu', stop=None, session_steps=0):
    started_at = time.monotonic()
    spec = run['specification']
    if run['run_id'] != 'RUN-'+digest(spec)[:20] or spec['sources'] != digest(source_identity()):
        raise ValueError('Run identity/source changed; prepare a new campaign')
    folder = folder_for(manifest, run)
    with claim(folder / 'run.lock') as acquired:
        if not acquired:
            return 'busy'
        if completed(folder, spec):
            return 'complete'
        prior_seconds = read_json(folder/'status.json').get('wall_seconds',0) if (folder/'status.json').exists() else 0
        config_path = folder / 'config.json'
        if config_path.exists() and read_json(config_path) != spec:
            raise ValueError('Run specification changed')
        write_json(config_path, spec)
        recipe = manifest['corpora'][spec['corpus_id']]
        payload, corpus_meta = materialize(recipe, manifest['paths']['corpora'], manifest['paths']['data'])
        if (folder/'corpus.json').exists() and read_json(folder/'corpus.json') != corpus_meta:
            raise ValueError('Training corpus changed since the previous attempt')
        write_json(folder / 'corpus.json', corpus_meta)
        graphs, graph_cases = payload['graphs'], payload['cases']
        datasets, splits = {}, {}
        for case in spec['evaluation_cases']:
            ds = case_dataset(manifest['paths']['data'], case_name(case))
            if digest(ds.manifest) != spec['data'][str(case)]['identity']:
                raise ValueError('Evaluation data identity changed')
            datasets[case], splits[case] = ds, ds.splits(42)
        write_json(folder / 'splits.json', {str(c):{k:dict(count=len(v), sha256=digest(v.tolist()))
                   for k,v in values.items()} for c,values in splits.items()})
        stop = stop or Stop()
        device = select_device(device)
        sample_case = spec['sample_case']
        sample = datasets[sample_case][int(splits[sample_case]['train'][0])].clone()
        model = initial_model(spec, sample, device)
        optimizer = torch.optim.AdamW(model.parameters(), **spec['optimizer'])
        training = spec['training']
        budget = training['max_updates']
        milestones = sorted(set(training['validation_updates'] + [budget]))
        versions = dict(torch=str(torch.__version__), pyg=torch_geometric.__version__, numpy=np.__version__)
        state = dict(steps=0, samples=0, epoch=0, offset=0, phase='train', best_score=None,
                     best_step=None, samples_per_case={str(c):0 for c in recipe['cases']})
        last = folder / 'last.pt'
        if last.exists():
            saved = torch.load(last, weights_only=False, map_location='cpu')
            if saved['specification'] != spec or saved['versions'] != versions:
                raise ValueError('Resume configuration/software changed')
            model.load_state_dict(saved['model'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            state = saved['state']
            restore_rng(saved['rng'], device)
        else:
            save_torch(folder / 'initial.pt', dict(specification=spec, model=model.state_dict(), example_graph=sample))

        def save(path=last):
            save_torch(path, dict(specification=spec, versions=versions, model=model.state_dict(),
                optimizer=optimizer.state_dict(), state=state, rng=capture_rng(device), example_graph=sample))

        def status(value):
            write_json(folder / 'status.json', dict(status=value, phase=state['phase'],
                steps=state['steps'], samples=state['samples'], samples_per_case=state['samples_per_case'],
                device=str(device), updated=time.time(),wall_seconds=prior_seconds+time.monotonic()-started_at))

        if not last.exists():
            save()
        started_steps, saved_at = state['steps'], time.monotonic()
        status('running')
        while state['phase'] != 'evaluate':
            if stop() or (session_steps and state['steps'] - started_steps >= session_steps):
                save(); status('paused'); return 'paused'
            if state['phase'] == 'validate':
                reports = {}
                for case in spec['evaluation_cases']:
                    identity = dict(specification=digest(spec), step=state['steps'], case=case, split='val')
                    metrics = evaluate(model, datasets[case], splits[case]['val'], device,
                        training['eval_batch_size'], folder / f'validation-case{case}-progress.json', identity, stop)
                    if metrics is None:
                        save(); status('paused'); return 'paused'
                    reports[str(case)] = metrics
                score = sum(r['score'] for r in reports.values()) / len(reports)
                if not np.isfinite(score):
                    raise FloatingPointError('Nonfinite validation score')
                if state['best_score'] is None or score < state['best_score']:
                    state['best_score'], state['best_step'] = score, state['steps']
                    save_torch(folder / 'best.pt', dict(specification=spec, model=model.state_dict(),
                        steps=state['steps'], samples=state['samples'], validation=reports, example_graph=sample))
                write_json(folder / 'history' / f"{state['steps']:08d}.json",
                    dict(steps=state['steps'], samples=state['samples'], score=score, validation=reports))
                state['phase'] = 'evaluate' if state['steps'] == budget else 'train'
                save(); status('running'); saved_at = time.monotonic()
                continue
            ids, epoch, offset = next_batch(len(graphs), training['batch_size'], state['epoch'],
                                            state['offset'], spec['seed'])
            batch = Batch.from_data_list([graphs[i].clone() for i in ids]).to(device)
            model.train(); optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            loss = (2*mse(prediction, {k:batch[k].y for k in ('bus','generator')}, batch)
                    if spec['initialization'].get('precision') == 'float64' else graph_mse(prediction, batch))
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            state.update(epoch=epoch, offset=offset, steps=state['steps']+1, samples=state['samples']+len(ids))
            for case, n in Counter(graph_cases[i] for i in ids).items():
                state['samples_per_case'][str(case)] += n
            if state['steps'] in milestones:
                state['phase'] = 'validate'
            if state['steps'] % training['log_updates'] == 0:
                write_json(folder / 'training.json', dict(steps=state['steps'], loss=float(loss.detach()),
                    gradient_norm=float(norm), samples=state['samples']))
            if state['phase'] == 'validate' or time.monotonic()-saved_at >= training['checkpoint_seconds']:
                save(); status('running'); saved_at = time.monotonic()

        if not (folder / 'final.pt').exists():
            save(folder / 'final.pt')
        best = torch.load(folder / 'best.pt', weights_only=False, map_location='cpu')
        model.load_state_dict(best['model'], strict=True)
        best_hash = file_hash(folder / 'best.pt')
        result_files = ['initial.pt', 'best.pt', 'final.pt', 'corpus.json', 'splits.json']
        # No test result participates in training, checkpoint selection, or corpus selection.
        for split in ('val', 'test'):
            for case in spec['evaluation_cases']:
                name = f'{split}-case{case}.json'
                result_files.append(name)
                identity = dict(specification=digest(spec), checkpoint_sha256=best_hash, case=case,
                                split=split, indices_sha256=digest(splits[case][split].tolist()))
                if (folder/name).exists() and read_json(folder/name).get('identity') == identity:
                    continue
                # The saved best checkpoint already has a complete validation evaluation.
                metrics = best['validation'][str(case)] if split == 'val' else evaluate(
                    model, datasets[case], splits[case][split], device, training['eval_batch_size'],
                    folder / f'{split}-case{case}-progress.json', identity, stop)
                if metrics is None:
                    status('paused'); return 'paused'
                write_json(folder/name, dict(identity=identity, metrics=metrics, model=spec['model'],
                    case=case, seed=spec['seed'], checkpoint_step=best['steps'], trained_steps=budget))
        if spec['initialization']['kind'] == 'pretrained':
            initial = torch.load(folder/'initial.pt', weights_only=False, map_location='cpu')
            model.load_state_dict(initial['model'], strict=True)
            for case in spec['evaluation_cases']:
                name = f'initial-test-case{case}.json'
                result_files.append(name)
                identity = dict(initialization=spec['initialization'], case=case,
                    data=spec['data'][str(case)], sources=spec['sources'], split='test')
                cached = Path(manifest['paths']['models']) / 'baselines' / (digest(identity)+'.json')
                if cached.exists() and read_json(cached)['identity'] == identity:
                    write_json(folder/name, read_json(cached)); continue
                metrics = evaluate(model, datasets[case], splits[case]['test'], device,
                    training['eval_batch_size'], folder/f'initial-test-case{case}-progress.json', identity, stop)
                if metrics is None:
                    status('paused'); return 'paused'
                result = dict(identity=identity, metrics=metrics)
                write_json(cached, result); write_json(folder/name, result)
        write_json(folder/'complete.json', dict(specification=digest(spec),
            files={name:file_hash(folder/name) for name in result_files}))
        status('complete')
        return 'complete'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seconds', type=float, default=3300)
    parser.add_argument('--session-steps', type=int, default=0)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    run = next(r for r in manifest['runs'] if r['run_id'] == args.run_id)
    folder = folder_for(manifest, run)
    try:
        with Stop(args.seconds) as stop:
            result = train_run(manifest, run, device=args.device, stop=stop, session_steps=args.session_steps)
        print(result, flush=True)
    except BlockingIOError:
        print('busy', flush=True)
        return 75
    except Exception as error:
        write_json(folder/'failure.json', dict(error=repr(error), recover_from='last.pt', time=time.time()))
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
