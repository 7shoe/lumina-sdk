"""Train, acquire and resume one BAS or matched-control experiment."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Batch
import yaml

from distillation.selection import Buckets, TrainingPool, ids_hash, input_view
from distillation.selection.data import load_descriptors
from distillation.training.T1_single_topo.common import (ROOT,
                                                         canonical_case,
                                                         claim,
                                                         digest,
                                                         file_hash,
                                                         source_identity,
                                                         write_json)
from distillation.training.T1_single_topo.data import CaseDataset
from distillation.training.T1_single_topo.state import (Stop,
                                                        capture_rng,
                                                        restore_rng,
                                                        save_torch)

from experiments.lumina_bench.metrics import MetricAccumulator, sample_metrics
from experiments.lumina_bench.runner import select_device

from .models import (SUPPORTED_MODELS,
                     has_dropout,
                     make_model,
                     model_options)
from .schedule import Patience
from .scores import (acquisition_scores,
                     per_graph_loss)


def configuration(**updates):
    """
    Config entry point for a BAS or matched-control experiment
    """
    # load config YAML
    config = yaml.safe_load(Path(__file__).with_name('config.yaml').read_text())
    if 'model' in updates and 'model_options' not in updates:
        updates['model_options'] = model_options(updates['model'])
    unknown = set(updates) - set(config)
    if unknown:
        raise ValueError(f'Unknown configuration fields: {sorted(unknown)}')
    config.update(updates)
    config['case'] = canonical_case(config['case'])
    
    # split unified always: same 240K train samples
    if config['split_seed'] != 42:
        raise ValueError('Preserve the existing split: split_seed must be 42')
    
    # config
    if type(config['deterministic']) is not bool:
        raise ValueError('deterministic must be a boolean')
    if config['model'] not in ('MLP', *SUPPORTED_MODELS):
        raise ValueError('Unsupported model')
    if config['method'] not in ('bas', 'random', 'stratified', 'mcdue', 'inactive'):
        raise ValueError('Unsupported selection method')
    if config['metric'] not in ('le', 'ig', 'lg', 'mcv_p', 'mcv_l') or config['loss'] not in ('l1', 'mse'):
        raise ValueError('Unsupported acquisition metric or loss')
    if config['partition'] not in ('quantile', 'width'):
        raise ValueError('Unsupported partition')
    for key in ('buckets', 'panel_size', 'initial_size', 'max_selected', 'acquisition_size',
                'candidate_size', 'mc_draws', 'batch_size', 'score_batch_size', 'epochs_per_round', 'max_rounds'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('selection_seed', 'model_seed', 'validation_size'):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
    if config['panel_size'] < config['buckets'] or config['initial_size'] > config['max_selected']:
        raise ValueError('Panel needs at least one example per requested bucket; initial <= maximum')
    if config['checkpoint_seconds'] <= 0 or config['mc_draws'] < 2:
        raise ValueError('Positive checkpoint interval and at least two MC draws required')
    
    # check scheduler constuctro
    Patience(lr=config['optimizer']['lr'], **config['schedule'])
    
    return config


def implementation_identity():
    """
    Hash implementation files so resumed runs reject source changes
    """
    
    files = source_identity()
    for folder in ('BAS', 'selection'):
        for path in (ROOT / 'distillation' / folder).glob('*.py'):
            files[str(path.relative_to(ROOT))] = file_hash(path)
    return files


def score_ids(model, pool, ids, metric, config, *, labeled=True, round_index=0):
    device = next(model.parameters()).device
    scores = []
    for offset in range(0, len(ids), config['score_batch_size']):
        part = ids[offset:offset + config['score_batch_size']]
        graphs = pool.labeled(part) if labeled else pool.inputs(part)
        batch = Batch.from_data_list(graphs).to(device)
        scores.extend(acquisition_scores(model, batch, metric, loss=config['loss'],
                      draws=config['mc_draws'], seed=config['selection_seed'] + 100000 * round_index + offset))
    return np.asarray(scores, dtype=np.float64)


def evaluate(model, dataset, ids, batch_size=128):
    """
    Ordinary evaluation only
    (!) acqusition never receives these labels/scores
    """
    device = next(model.parameters()).device
    accumulator = MetricAccumulator()
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for offset in range(0, len(ids), batch_size):
                batch = Batch.from_data_list([dataset[int(i)].clone() for i in ids[offset:offset + batch_size]])
                predictions = model(input_view(batch).to(device))
                predictions = {k: v.detach().cpu() for k, v in predictions.items()}
                metrics = sample_metrics(predictions, batch)
                metrics['l1'] = per_graph_loss(predictions, batch, 'l1').double()
                dimensions = 2 * (batch['bus'].ptr.diff() + batch['generator'].ptr.diff())
                metrics['mae'] = metrics['l1'] / dimensions
                accumulator.update(metrics)
    finally:
        model.train(was_training)
    return accumulator.result()


def acquire(model, pool, partition, memberships, config, round_index):
    remaining = pool.remaining_ids
    budget = min(config['acquisition_size'], config['max_selected'] - len(pool.selected_ids), len(remaining))
    rng = np.random.default_rng(config['selection_seed'] + 1000 + round_index)
    method, details = config['method'], {}
    if method == 'random':
        chosen = rng.choice(remaining, budget, replace=False)
    elif method == 'mcdue':
        candidates = rng.choice(remaining, min(len(remaining), max(budget, config['candidate_size'])), replace=False)
        scores = score_ids(model, pool, candidates, 'mcv_p', config, labeled=False, round_index=round_index)
        chosen = candidates[np.argsort(-scores, kind='stable')[:budget]]
        details = dict(candidate_ids=candidates.tolist(), candidates_scored=len(candidates))
    else:
        weights = np.ones(partition.count)
        if method == 'bas':
            scores = score_ids(model, pool, pool.reference_ids, config['metric'], config, round_index=round_index)
            bins = memberships[pool.reference_ids]
            weights = np.bincount(bins, weights=scores, minlength=partition.count) / np.bincount(bins, minlength=partition.count)
        chosen, quotas = partition.sample(remaining, memberships[remaining], weights, budget, rng)
        details = dict(bucket_scores=weights.tolist(), quotas=quotas.tolist())
    details['round'] = round_index
    pool.acquire(chosen, reason=method, details=details)
    return details


def export_selection(folder, pool, spec, partition, descriptor, memberships, round_index):
    record = dict(format='lumina-selection-v1', method=spec['config']['method'], metric=spec['config']['metric'],
                  data_identity=spec['data_identity'], specification=digest(spec),
                  pool=pool.state_dict(), buckets=partition.state_dict(), descriptor=descriptor,
                  selected_count=len(pool.selected_ids), reference_count=len(pool.reference_ids),
                  labeled_count=len(pool.selected_ids) + len(pool.reference_ids), round=round_index,
                  selected_bucket_counts=np.bincount(memberships[pool.selected_ids], minlength=partition.count).tolist(),
                  test_used=False, solver_calls=0)
    write_json(folder / 'selection.json', record)


def run(config, folder, *, device='cpu', stop=None, session_steps=0, dataset=None):
    """Continue the same experiment after a deadline; stop after max_rounds.

    Checkpoints include in-epoch order/cursor, acquired IDs, both patience
    counters, optimizer, RNG and exact input/split/source identity. A round is
    a configured number of full epochs over the currently selected subset.
    """
    config, folder = configuration(**config), Path(folder)
    with claim(folder / 'run.lock') as acquired:
        if not acquired:
            return dict(status='busy')
        try:
            return _run(config, folder, device, stop or Stop(), session_steps, dataset)
        except Exception as error:
            write_json(folder / 'failure.json', dict(error=repr(error), recover_from='latest.pt'))
            raise


def _run(config, folder, device, stop, session_steps, dataset):
    session_started = time.monotonic()
    device = select_device(device)
    torch.use_deterministic_algorithms(config['deterministic'])
    dataset = dataset or CaseDataset(config['data_root'], config['case'])
    splits = dataset.splits(42)
    if config['panel_size'] + config['max_selected'] > len(splits['train']):
        raise ValueError('Panel + selected budget exceeds the existing training split')
    spec = dict(format='lumina-bas-v1', config=config, data_identity=digest(dataset.manifest),
                split_hashes={k: ids_hash(v) for k, v in splits.items()}, sources=implementation_identity())
    spec_path = folder / 'config.json'
    if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
        raise ValueError('Configuration, source, data or exact split changed; use a new run directory')
    write_json(spec_path, spec)
    if (folder / 'complete.json').exists():
        complete = json.loads((folder / 'complete.json').read_text())
        if any(file_hash(folder / name) != checksum for name, checksum in complete['artifacts'].items()):
            raise ValueError('Completed artifact changed')
        return complete
    
    # fit bucket boundary using training-input descriptors 
    values, descriptor = load_descriptors(dataset, splits['train'], folder.parent / f'{config["case"]}_train_inputs.pt')
    partition = Buckets.fit(values, config['buckets'], config['partition'])
    
    # map dset id to buckets (nontraining get -1)
    memberships = np.full(len(dataset), -1, dtype=np.int64)
    memberships[splits['train']] = partition.assign(values)
    
    # draw a fixed ref panel for round assessment
    panel, _ = partition.sample(splits['train'], memberships[splits['train']], np.ones(partition.count),
                                config['panel_size'], np.random.default_rng(config['selection_seed']))
    pool = TrainingPool(dataset, splits, panel)
    
    # inactive control receives its full training budget
    initial = np.random.default_rng(config['selection_seed'] + 1).permutation(pool.remaining_ids)
    initial_count = config['max_selected'] if config['method'] == 'inactive' else config['initial_size']
    pool.acquire(initial[:initial_count], reason='initial')
    model = make_model(config, pool.inputs(pool.selected_ids[:1])[0], device)
    if (config['method'] == 'mcdue' or config['method'] == 'bas' and config['metric'].startswith('mcv')) and not has_dropout(model):
        raise ValueError('This model has no effective dropout; use LE/IG/LG or a dropout-enabled model')
    optimizer = torch.optim.AdamW(model.parameters(), **config['optimizer'])
    schedule = Patience(lr=config['optimizer']['lr'], **config['schedule'])
    state = dict(round=0, epoch=0, offset=0, phase='train', steps=0, presentations=0,
                 best_validation=None, best_round=None, train_loss_sum=0., train_loss_count=0,
                 seconds=dict(training=0., assessment=0., acquisition=0.))
    versions = dict(torch=str(torch.__version__), pyg=torch_geometric.__version__, numpy=np.__version__,
                    python=platform.python_version(), device_type=device.type,
                    deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    latest = folder / 'latest.pt'
    if latest.exists():
        saved = torch.load(latest, map_location='cpu', weights_only=False)
        if saved['specification'] != digest(spec) or saved['versions'] != versions:
            raise ValueError('Checkpoint identity or execution environment changed')
        if saved['buckets'] != partition.state_dict():
            raise ValueError('Bucket boundaries changed')
        
        # restore selected subset/optimizer/schecduler/etc.
        pool.load_state_dict(saved['pool'])
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        schedule = Patience(**saved['schedule'])
        state = saved['state']
        restore_rng(saved['rng'], device)
    elapsed_before = state.get('wall_seconds', 0.)

    def checkpoint(status):
        state['wall_seconds'] = elapsed_before + time.monotonic() - session_started
        payload = dict(specification=digest(spec), versions=versions, config=config, state=state,
                       pool=pool.state_dict(), buckets=partition.state_dict(), schedule=asdict(schedule),
                       model=model.state_dict(), optimizer=optimizer.state_dict(), rng=capture_rng(device))
        save_torch(latest, payload)
        export_selection(folder, pool, spec, partition, descriptor, memberships, state['round'])
        summary = dict(status=status, round=state['round'], steps=state['steps'], presentations=state['presentations'],
                       selected=len(pool.selected_ids), reference=len(panel), acquisition_events=len(pool.history) - 1,
                       best_validation_l1=state['best_validation'], seconds=state['seconds'],
                       wall_seconds=state['wall_seconds'], test_used=False)
        write_json(folder / 'status.json', summary)
        return summary

    if not latest.exists():
        write_json(folder / 'partition.json', dict(**partition.state_dict(), descriptor=descriptor,
                   training_counts=np.bincount(memberships[splits['train']], minlength=partition.count).tolist(),
                   panel_ids=panel.tolist(), panel_counts=np.bincount(memberships[panel], minlength=partition.count).tolist()))
        checkpoint('running')
    validation_ids = splits['val'][:config['validation_size']] if config['validation_size'] else splits['val']
    saved_at, initial_steps = time.monotonic(), state['steps']
    while state['round'] < config['max_rounds']:
        if stop() or session_steps and state['steps'] - initial_steps >= session_steps:
            return checkpoint('paused')
        if state['phase'] == 'train':
            start = time.monotonic()
            seed = config['selection_seed'] + 10000000 + state['round'] * config['epochs_per_round'] + state['epoch']
            order = np.random.default_rng(seed).permutation(pool.selected_ids)
            ids = order[state['offset']:state['offset'] + config['batch_size']]
            batch = Batch.from_data_list(pool.labeled(ids)).to(device)
            model.train()
            optimizer.zero_grad(set_to_none=True)# reconstruct this epoch's shuffle det. then resume
            loss = per_graph_loss(model(input_view(batch)), batch, config['loss']).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('Nonfinite model gradient')
            optimizer.step()
            
            # count optim update
            state['steps'] += 1
            state['presentations'] += len(ids)
            state['offset'] += len(ids)
            state['train_loss_sum'] += float(loss.detach()) * len(ids)
            state['train_loss_count'] += len(ids)
            if state['offset'] == len(order):
                state['offset'] = 0
                state['epoch'] += 1
                
            # each round trains for fixed number of full epochs
            if state['epoch'] == config['epochs_per_round']:
                state['epoch'], state['phase'] = 0, 'assess'
            state['seconds']['training'] += time.monotonic() - start
        else:
            start = time.monotonic()
            
            # ref-panel loss drives scheduling; val-loss selects best model
            panel_loss = float(score_ids(model, pool, panel, 'le', config, round_index=state['round']).mean())
            metrics = evaluate(model, dataset, validation_ids, config['score_batch_size'])
            if state['best_validation'] is None or metrics['l1'] < state['best_validation']:
                state['best_validation'], state['best_round'] = metrics['l1'], state['round']
                save_torch(folder / 'best.pt', dict(model=model.state_dict(), config=config, specification=digest(spec),
                           data_identity=spec['data_identity'],
                           selected_ids=pool.selected_ids.copy(), reference_ids=panel.tolist(), validation=metrics,
                           validation_ids=validation_ids.tolist(), round=state['round']))
            state['seconds']['assessment'] += time.monotonic() - start
            can_acquire = (config['method'] != 'inactive' and len(pool.selected_ids) < config['max_selected']
                           and state['round'] + 1 < config['max_rounds'])
            event = None
            selected_before = len(pool.selected_ids)
            
            # patience scheduler updates its state
            if schedule.step(panel_loss, can_acquire=can_acquire):
                start = time.monotonic()
                event = acquire(model, pool, partition, memberships, config, state['round'])
                state['seconds']['acquisition'] += time.monotonic() - start
            for group in optimizer.param_groups:
                group['lr'] = schedule.lr
            record = dict(round=state['round'], training_count=selected_before, selected_after=len(pool.selected_ids),
                          training_loss=state['train_loss_sum'] / state['train_loss_count'], panel_loss=panel_loss,
                          validation=metrics, learning_rate=schedule.lr, rho1=schedule.rho1, rho2=schedule.rho2,
                          acquisition=event, steps=state['steps'], presentations=state['presentations'])
            write_json(folder / 'history' / f'{state["round"]:06d}.json', record)
            print(json.dumps({k: record[k] for k in ('round', 'training_count', 'selected_after', 'panel_loss', 'learning_rate')}) , flush=True)
            
            # save the coklpeted round state 
            state['round'] += 1
            state.update(phase='train', train_loss_sum=0., train_loss_count=0)
            checkpoint('running')
            saved_at = time.monotonic()
        if time.monotonic() - saved_at >= config['checkpoint_seconds']:
            checkpoint('running')
            saved_at = time.monotonic()
            
    # record final artifact hash
    summary = checkpoint('complete')
    summary['artifacts'] = {name: file_hash(folder / name) for name in ('latest.pt', 'best.pt', 'selection.json')}
    write_json(folder / 'complete.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.yaml'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seconds', type=float, default=3300, help='Per-invocation deadline; same command resumes')
    args = parser.parse_args()
    torch.set_num_threads(1)
    with Stop(args.seconds) as stop:
        result = run(yaml.safe_load(args.config.read_text()), args.output, device=args.device, stop=stop)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
