"""Resumable paired-model HMDC, independent pair replicas on twelve XPUs.

Algorithm in hmdc.py; generation/feasibility/data layouts remain shared. Only
pair members select the dataset. Other architectures are evaluated afterwards.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from distillation.DC.campaign import calibrate, panel
from distillation.conversion.matrix_form import mat2opf
from distillation.synthetic import describe, teaching_targets
from distillation.synthetic.experiments import initialize
from distillation.synthetic.inputs import SyntheticInputs, feasible_step
from distillation.synthetic.runtime import (averaged_gradients,
                                            collective_audit,
                                            graph_batch,
                                            log,
                                            sampler_ids,
                                            save)
from distillation.training.T1_single_topo.common import write_json
from distillation.training.T1_single_topo.data import CaseDataset
from experiments.lumina_bench.models import SUPPORTED_MODELS
from .hmdc import (alignment_step,
                   data_gradients,
                   model_step)
from .models import (build_pair,
                     model_config)
from .selection import select_real

FORMAT = 'hmdc-opf-v1'


def collectively(comm, function):
    """Every rank reaches the same failure boundary before subsequent collectives."""
    value, error = None, None
    try:
        value = function()
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    
    # share local failures before any rank enters the next collective
    errors = comm.allgather(error)
    if any(errors):
        raise RuntimeError(f'HMDC collective failure: {errors}')
    
    return value


def validate_config(config):
    pair = config['pair']
    if len(pair) != 2 or len(set(pair)) != 2 or any(k not in SUPPORTED_MODELS for k in pair):
        raise ValueError('Select two distinct supported benchmark models')
    if not set(pair) <= set(config['evaluation_models']) <= set(SUPPORTED_MODELS):
        raise ValueError('Evaluation must include both pair members and only benchmark models')
    if len(set(config['evaluation_models'])) != len(config['evaluation_models']):
        raise ValueError('Evaluation models must be unique')
    if config.get('model_precision') != 'float64':
        raise ValueError('HMDC currently qualifies float64 mixed derivatives only')
    if config['synthetic_count'] != config['batches'] * config['batch_size']:
        raise ValueError('Synthetic count and batch schedule differ')
    for key in ('restart_rounds', 'real_steps', 'query_interval', 'candidate_pool', 'real_batch',
                'validation_samples', 'final_samples', 'evaluation_epochs', 'max_updates'):
        if isinstance(config[key], bool) or int(config[key]) != config[key] or config[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if config['candidate_pool'] < config['real_batch']:
        raise ValueError('Candidate pool must contain at least one real matching batch')


def qualify_device(graphs, config, real, comm, device):
    rates = {kind: 1e-9 for kind in config['pair']}
    try:
        results = []
        
        # compare identical seeded pairs on CPU and accelerator, including input grads
        for target in ('cpu', device):
            pair = build_pair(graphs[0], config, rates, 500+comm.rank, target)
            synthetic = SyntheticInputs(graphs[:3], config['strength'])
            info = data_gradients(pair.models, pair.alignment, pair.balance, synthetic, real,
                                  mutual=config.get('mutual', True), balanced=config.get('balanced', True))
            families = {}
            for name, p in synthetic.named_parameters():
                families.setdefault(name.split('.')[-1], []).append(p.grad.double().cpu().flatten())
            results.append((info, {key: torch.cat(values) for key, values in families.items()}))
        (a, cpu), (b, accelerator) = results
        rows = [dict(family=k, cpu_norm=float(v.norm()), xpu_norm=float(accelerator[k].norm()),
                     error=float((v-accelerator[k]).norm()), tolerance=1e-7+.02*float(v.norm()))
                for k,v in cpu.items()]
        loss_ok = all(abs(x-y) <= 1e-7+.001*abs(x) for x,y in zip(a['terms'],b['terms']))
        result = dict(rank=comm.rank, passed=loss_ok and all(r['error']<=r['tolerance'] for r in rows),
                      cpu=a, xpu=b, gradients=rows,
                      criterion='each objective 1e-7 + 0.1%; each normalized input-gradient family L2 1e-7 + 2%')
    except Exception as exc:
        result = dict(rank=comm.rank, passed=False, error=f'{type(exc).__name__}: {exc}')
    return comm.allgather(result)


def selection_panel(synthetic, config, rates, dataset, ids, comm, device):
    results, students = {}, {}
    for kind in config['pair']:
        results[kind], students[kind] = panel(synthetic, model_config(config, kind), rates[kind],
                                             dataset, ids, comm, device)
    return results, students


def selection_score(results, baseline):
    # normalize each model's error by its initial baseline before averaging
    if any(r['mean'] is None for r in results.values()):
        return None
    return float(np.mean([r['mean']/max(baseline[k], 1e-12) for k,r in results.items()]))


def query_schedule(iteration, restart_rounds, interval):
    """Restart alternation with the pair, retaining a globally unique pool cursor."""
    restart, local_round = divmod(iteration, restart_rounds)
    queries_per_restart = (restart_rounds + interval - 1) // interval
    query = local_round // interval
    return restart * queries_per_restart + query, query % 2, local_round % interval == 0


def export(path, synthetic, config, state, physical, checks, students):
    with torch.no_grad():
        batch, labels = synthetic.batch(detached=True)
    save(path, dict(format=FORMAT, config=config, training_state=deepcopy(state), inputs=batch,
                    teaching_targets=labels, physical_solved_graphs=physical, solver_checks=checks,
                    model_checkpoints=students, student_rates=state['rates'],
                    target_semantics='fresh detached AC-OPF labels',
                    generation_statistics=[describe(g) for g in physical]))


def checkpoint(folder, synthetic, optimizer, config, state, physical, checks, pair, comm):
    # preserve every replica's pair state for an exact continuation of the pair training
    pairs = comm.allgather(pair.state_dict() if pair else None)
    
    # primary rank
    if comm.rank == 0:
        save(folder/'latest.pt', dict(format=FORMAT, config=config, state=state,
            anchor_graphs=[mat2opf(m) for m in synthetic.anchors], synthetic_state=synthetic.state_dict(),
            labels=synthetic.labels, optimizer=optimizer.state_dict(), physical_graphs=physical,
            solver_checks=checks, pairs=pairs, workers=comm.size))
        
        # assemble JSON
        public = {k:v for k,v in state.items() if not k.startswith('seen_')}
        public.update({f'distinct_{key[5:]}':len(values) for key,values in state.items() if key.startswith('seen_')})
        public['distinct_real_training_examples'] = len(state['seen_matching_ids'] | state['seen_training_ids'])
        
        # write JSON
        write_json(folder/'status.json', public)


def restore(payload):
    synthetic = SyntheticInputs(payload['anchor_graphs'], payload['config']['strength'])
    synthetic.load_state_dict(payload['synthetic_state'])
    synthetic.labels = deepcopy(payload['labels'])
    optimizer = torch.optim.SGD(synthetic.parameters(), lr=payload['config']['data_rate'], momentum=.5)
    optimizer.load_state_dict(payload['optimizer'])
    
    return synthetic, optimizer


def run(config, folder, comm, device, *, allocation_seconds=3300, qualify=False, stop_after_updates=None):
    """Runner w/ lazy MPI import"""
    from mpi4py import MPI
    
    validate_config(config)
    folder, started = Path(folder), time.monotonic()
    folder.mkdir(parents=True, exist_ok=True)
    def load_existing():
        if (folder/'config.json').exists() and json.loads((folder/'config.json').read_text()) != config:
            raise ValueError('Output directory already belongs to another configuration')
        if not (folder/'latest.pt').exists():
            return None
        value = torch.load(folder/'latest.pt', weights_only=False, map_location='cpu')
        if value['config'] != config or value['workers'] != comm.size:
            raise ValueError('Resume requires identical configuration and replica count')
        return value
    payload = collectively(comm, lambda: load_existing() if comm.rank==0 else None)
    payload = comm.bcast(payload, root=0)
    if payload and payload['state']['status']=='COMPLETE':
        return payload['state']
    
    # primary rank
    if comm.rank == 0:
        write_json(folder/'config.json', config)
    dataset = collectively(comm, lambda: CaseDataset(config['data_root'], f"case{config['case']}"))
    split = dataset.splits(config['split_seed'])
    pair = None
    if payload:
        synthetic, optimizer = restore(payload)
        state, physical, checks = payload['state'], payload['physical_graphs'], payload['solver_checks']
        original = torch.load(folder/'original.pt', weights_only=False, map_location='cpu')['graphs']
        if payload['pairs'][comm.rank]:
            pair = build_pair(original[0], config, state['training_rates'], 1, device)
            pair.load_state_dict(payload['pairs'][comm.rank])
        if (folder/'best.pt').exists():
            best = torch.load(folder/'best.pt', weights_only=False, map_location='cpu')['training_state']
            if best['best_validation'] < state['best_validation']:
                state['best_validation'], state['best_iteration'] = best['best_validation'], best['best_iteration']
    else:
        original, ids, physical, _ = initialize(config, folder, comm)
        synthetic = SyntheticInputs(physical, config['strength'])
        optimizer = torch.optim.SGD(synthetic.parameters(), lr=config['data_rate'], momentum=.5)
        ok, physical, checks = collective_audit(synthetic, comm)
        if not ok:
            raise RuntimeError('Initial HMDC inputs failed AC-OPF feasibility recheck')
        synthetic.labels = teaching_targets(physical)
        state = dict(iteration=0, active_seconds=0., accepted_updates=0, rejected_updates=0,
                     seed_ids=ids, seen_matching_ids=set(), seen_training_ids=set(), seen_candidate_ids=set(),
                     rates={}, training_rates={}, best_validation=None, best_iteration=None,
                     selection_baseline=None, status='TRAINING', resume_count=0)
    
    state['resume_count'] += 1
    state['status'] = 'TRAINING'
    if not set(state['seed_ids']) <= set(split['train'].tolist()):
        raise ValueError('Initialization references must belong to the training split')
    population = split['train'][~np.isin(split['train'], state['seed_ids'])]
    val_ids = split['val'][:config['validation_samples']]
    
    if config['validation_samples']+config['final_samples'] > len(split['val']):
        raise ValueError('Selection and final validation panels do not fit the split')
    if comm.rank == 0:
        write_json(folder/'split.json', dict(split_seed=config['split_seed'], train_pool_size=len(population),
            excluded_seed_ids=state['seed_ids'], validation_ids=val_ids.tolist(), test_accessed=False,
            train_pool_sha256=hashlib.sha256(population.tobytes()).hexdigest()))
    
    if qualify:
        real = graph_batch(dataset, sampler_ids(population, 0, comm.rank, comm.size, 4))
        qualification = qualify_device(physical, config, real, comm, device)
        if comm.rank == 0:
            write_json(folder/'device_checks.json', qualification)
        if not all(row['passed'] for row in qualification):
            raise RuntimeError('HMDC CPU/XPU qualification failed')
        
    # Rates depend only on real reference graphs and training-only calibration.
    # Cache each model separately to survive a pre-training allocation boundary.
    for kind in config['evaluation_models']:
        path = folder/f'calibration_{kind}.json'
        cached = (json.loads(path.read_text()) if path.exists() else None) if comm.rank==0 else None
        cached = comm.bcast(cached,root=0)
        if cached is None:
            rate, cached = calibrate(original, model_config(config, kind), dataset, population, comm, device)
            if comm.rank == 0:
                write_json(path, cached)
        state['rates'][kind] = cached['chosen_rate']
    
    if not state['training_rates']:
        state['training_rates'] = {k:state['rates'][k]*config.get('mutual_rate_factor',1.) for k in config['pair']}
    if state['best_validation'] is None:
        results, students = selection_panel(synthetic,config,state['rates'],dataset,val_ids,comm,device)
        if any(r['mean'] is None for r in results.values()):
            raise FloatingPointError('Initial pair-member evaluation is nonfinite')
        state['selection_baseline'] = {k:r['mean'] for k,r in results.items()}
        state['best_validation'], state['best_iteration'] = 1., 0
        if comm.rank == 0:
            export(folder/'best.pt',synthetic,config,state,physical,checks,students)
            log(folder/'history.jsonl',dict(stage='initial_validation',results=results))
    checkpoint(folder,synthetic,optimizer,config,state,physical,checks,pair,comm)
    
    # save
    last_save = last_eval = time.monotonic()
    this_run = 0
    while state['active_seconds'] < config['training_seconds'] and state['iteration'] < config['max_updates']:
        expired = time.monotonic()-started >= allocation_seconds-config['reserve_seconds']
        if comm.bcast(expired if comm.rank==0 else None,root=0):
            break
        if stop_after_updates is not None and this_run >= stop_after_updates:
            break
        tick, iteration = time.monotonic(), state['iteration']
        if pair is None:
            seed = 1_000_000+(iteration//config['restart_rounds'])*comm.size+comm.rank
            pair = collectively(comm,lambda: build_pair(original[0],config,state['training_rates'],seed,device))
        candidates = []
        query,owner,refresh=query_schedule(iteration,config['restart_rounds'],config['query_interval'])
        if not pair.query_ids or refresh:
            candidates = sampler_ids(population,query,comm.rank,comm.size,
                                     config['candidate_pool'],seed=51_001)
            pair.query_ids = collectively(comm,lambda: select_real(pair.models[owner],dataset,candidates,
                config['real_batch'],batch_size=config['eval_batch'],seed=61_001+iteration*comm.size+comm.rank))
        real = collectively(comm,lambda: graph_batch(dataset,pair.query_ids))
        optimizer.zero_grad(set_to_none=True)
        info = collectively(comm,lambda: data_gradients(pair.models,pair.alignment,pair.balance,synthetic,real,
                              mutual=config.get('mutual',True),balanced=config.get('balanced',True)))
        norm = averaged_gradients(synthetic,comm)
        accepted, proposed, audit, attempts = feasible_step(synthetic,optimizer,
                                                          audit=lambda s:collective_audit(s,comm))
        if accepted:
            physical,checks=proposed,audit
        state['accepted_updates' if accepted else 'rejected_updates'] += 1
        training_ids = []
        def train_pair():
            statistics = []
            for step in range(config['real_steps']):
                ids = sampler_ids(population,iteration*config['real_steps']+step,comm.rank,comm.size,
                                  config['real_batch'],seed=71_001)
                batch = graph_batch(dataset,ids);training_ids.extend(ids)
                statistics.append(model_step(pair.models,pair.alignment,pair.optimizers,batch,
                                              mutual=config.get('mutual',True)))
            if config.get('mutual',True):
                for step in range(config['real_steps']):
                    ids = sampler_ids(population,iteration*config['real_steps']+step,comm.rank,comm.size,
                                      config['real_batch'],seed=81_001)
                    training_ids.extend(ids)
                    alignment_step(pair.models,pair.alignment,pair.alignment_optimizer,graph_batch(dataset,ids))
            return statistics
        train_info = collectively(comm,train_pair)
        observed = comm.allgather(dict(matching=pair.query_ids,training=training_ids,candidate=candidates))
        for record in observed:
            for name in record:
                state[f'seen_{name}_ids'].update(record[name])
        state['iteration'] += 1;this_run += 1
        if state['iteration'] % config['restart_rounds'] == 0:
            pair=None  # Students, SSD, GBM restart together; synthetic momentum persists.
        state['active_seconds'] += comm.allreduce(time.monotonic()-tick,op=MPI.MAX)
        rows = comm.allgather(dict(matching=info,training=train_info))
        if comm.rank==0:
            log(folder/'history.jsonl',dict(stage='matching',iteration=state['iteration'],workers=rows,
                data_gradient_norm=norm,accepted=accepted,attempts=attempts,active_seconds=state['active_seconds']))
        if comm.bcast(time.monotonic()-last_eval >= config['eval_seconds'] if comm.rank==0 else None,root=0):
            results,students=selection_panel(synthetic,config,state['rates'],dataset,val_ids,comm,device)
            score=selection_score(results,state['selection_baseline'])
            if score is not None and score < state['best_validation']:
                state['best_validation'],state['best_iteration']=score,state['iteration']
                if comm.rank==0:export(folder/'best.pt',synthetic,config,state,physical,checks,students)
            if comm.rank==0:log(folder/'history.jsonl',dict(stage='validation',iteration=state['iteration'],score=score,results=results))
            last_eval=time.monotonic()
        if comm.bcast(time.monotonic()-last_save >= config['checkpoint_seconds'] if comm.rank==0 else None,root=0):
            checkpoint(folder,synthetic,optimizer,config,state,physical,checks,pair,comm)
            last_save=time.monotonic()
    complete=state['active_seconds'] >= config['training_seconds'] or state['iteration'] >= config['max_updates']
    state['status']='FINALIZING' if complete else 'RESUMABLE'
    checkpoint(folder,synthetic,optimizer,config,state,physical,checks,pair,comm)
    if not complete:
        return state
    results,students=selection_panel(synthetic,config,state['rates'],dataset,val_ids,comm,device)
    score=selection_score(results,state['selection_baseline'])
    if score is not None and score < state['best_validation']:
        state['best_validation'],state['best_iteration']=score,state['iteration']
        if comm.rank==0:export(folder/'best.pt',synthetic,config,state,physical,checks,students)
    if comm.rank==0:export(folder/'condensed.pt',synthetic,config,state,physical,checks,students)
    comm.Barrier()
    final_evaluation(folder,original,config,state,dataset,split,comm,device)
    state['status']='COMPLETE'
    checkpoint(folder,synthetic,optimizer,config,state,physical,checks,pair,comm)
    return state


def final_evaluation(folder, original, config, state, dataset, split, comm, device):
    # Use a disjoint validation panel for final comparisons, leaving the test split untouched
    ids=split['val'][config['validation_samples']:config['validation_samples']+config['final_samples']]
    best=torch.load(folder/'best.pt',weights_only=False,map_location='cpu')['physical_solved_graphs']
    initial=torch.load(folder/'initial.pt',weights_only=False,map_location='cpu')['graphs']
    variants=dict(best=best,initial_synthetic=initial,random_real=original,untrained=original)
    previous=collectively(comm,lambda: (json.loads((folder/'evaluation.json').read_text())
        if (folder/'evaluation.json').exists() else {}) if comm.rank==0 else None)
    previous=comm.bcast(previous,root=0)
    
    # current results
    results=previous.get('results',{}) if previous.get('best_iteration')==state['best_iteration'] else {}
    for kind in config['evaluation_models']:
        cached=kind in results and (folder/f'student_{kind}.pt').exists() if comm.rank==0 else None
        if comm.bcast(cached,root=0):
            continue
        results[kind]={}
        for name,graphs in variants.items():
            result,student=panel(SyntheticInputs(graphs,config['strength']),model_config(config,kind),
                state['rates'][kind],dataset,ids,comm,device,seed_base=20_000_000,trained=name!='untrained')
            results[kind][name]=result
            if comm.rank==0 and name=='best':save(folder/f'student_{kind}.pt',student)
        if comm.rank==0:
            write_json(folder/'evaluation.json',dict(results=results,validation_ids=ids.tolist(),test_accessed=False,
                pair=config['pair'],held_out_models=[k for k in config['evaluation_models'] if k not in config['pair']],
                best_iteration=state['best_iteration'],
                complete=len(results)==len(config['evaluation_models']),
                note='Fresh ordinary-SGD evaluation; no mutual distillation during downstream training'))
