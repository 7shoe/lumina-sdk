"""RAMBO generation followed by the same student evaluation as DD/DCGM/HMDC.

One sequential CPU/Julia library is shared by all model/initialization evaluations.
Only verified final OPF labels are exported; no neural objective guides generation.
"""
from pathlib import Path
import itertools
import json
import math
import time

import numpy as np
import torch

from distillation.RAMBO import run as generator
from distillation.RAMBO.data import coverage
from distillation.selection import ids_hash
from distillation.selection.generated import GeneratedDataset
from distillation.synthetic import evaluation
from distillation.synthetic.experiments import common_config
from distillation.synthetic.inputs import SyntheticInputs
from distillation.synthetic.learning import model_for, model_options, mse, pretrained_config
from distillation.synthetic.runtime import collectively, save
from distillation.training.T1_single_topo.common import digest, file_hash, write_json
from distillation.experiments.staging import case_dataset as CaseDataset, local_path
from distillation.training.T1_single_topo.state import Stop
from experiments.lumina_bench.models import SUPPORTED_MODELS

EVALUATION_FIELDS = ('evaluation_epochs', 'validation_samples', 'final_samples', 'eval_batch',
    'student_rates', 'calibration_samples', 'calibration_seed', 'calibration_model_seed', 'retention_samples',
    'calibration_panels', 'student_rate_factor', 'trace_every', 'trace_samples', 'evaluation_reserve_seconds')
SHARED_FIELDS = ('synthetic_count', 'batch_size', 'split_seed', 'model_precision', 'reserve_seconds', 'qualify')
GENERATION_FIELDS = ('warm_start', 'epsilon', 'log_delta', 'history_size', 'variable_size',
    'tolerance', 'duplicate_tolerance', 'stage_seconds', 'request_seconds', 'max_attempts')


def expand(spec, root, source_hashes, group, case_info):
    """RAMBO has its own domains/samplers; synthetic topology/target axes do not apply."""
    from .study import keys, retention_inputs
    entry = spec['methods']['RAMBO']
    keys(entry.get('settings', {}), GENERATION_FIELDS, 'RAMBO generation settings')
    keys(spec['evaluation'], EVALUATION_FIELDS, 'RAMBO evaluation')
    domains = group.get('domains', list(entry['domains']))
    samplers = group.get('samplers', entry['samplers'])
    models = group.get('models', entry['models'])
    if not models or not set(models) <= set(SUPPORTED_MODELS):
        raise ValueError('RAMBO evaluation requires supported benchmark models')
    if not domains or not set(domains) <= set(entry['domains']): raise ValueError('Unknown RAMBO domain')
    if not samplers or not set(samplers) <= {'rambo', 'uniform'}: raise ValueError('Unknown RAMBO sampler')
    rows = []
    for case, seed, domain, sampler in itertools.product(group['cases'], group['seeds'], domains, samplers):
        if type(case) is not int or case not in (14, 30, 57, 118, 500):
            raise ValueError('RAMBO supports cases 14, 30, 57, 118 and 500')
        defaults = common_config(case) | {'evaluation_reserve_seconds': 90}
        base = {k: defaults[k] for k in SHARED_FIELDS + EVALUATION_FIELDS}
        base.update({k: v for k, v in spec['shared'].items() if k in SHARED_FIELDS}, **spec['evaluation'])
        if base['split_seed'] != 42: raise ValueError('The benchmark split remains 42')
        for k in ('synthetic_count', 'batch_size', 'evaluation_epochs', 'validation_samples', 'final_samples',
                  'eval_batch', 'calibration_samples', 'calibration_panels'):
            if type(base[k]) is not int or base[k] < 1: raise ValueError(f'Positive integer required: {k}')
        if base['synthetic_count'] % base['batch_size']: raise ValueError('Use complete student batches')
        if base['model_precision'] not in ('float32', 'float64'): raise ValueError('Invalid model precision')
        if not base['student_rates'] or any(not math.isfinite(v) or v <= 0 for v in base['student_rates']):
            raise ValueError('Positive finite student rates required')
        if not 0 < base['student_rate_factor'] <= 1: raise ValueError('Invalid student rate factor')
        info = case_info(case)
        gen = generator.configuration(case=f'case{case}', seed=seed, count=base['synthetic_count'], method=sampler,
            load_bounds=entry['domains'][domain], data_root=str(Path(spec['paths']['data']).expanduser().resolve()),
            raw_root=str(Path(spec['paths']['raw']).expanduser().resolve()), **entry.get('settings', {}))
        generation_id = 'RAMBO-' + digest(dict(config=gen, data_sha256=info['data_sha256'],
            source_sha256=digest(source_hashes)))[:16]
        generation_root = Path(spec['paths'].get('generated', str(root/'generated'))).expanduser().resolve()
        generation_output = generation_root/spec['campaign']/spec['profile']/generation_id
        for model, regime in itertools.product(models, group['initializations']):
            init = spec['initialization'][regime]
            cfg = base | dict(case=case, model=model, model_options=model_options(model, **spec['model_defaults']),
                initialization=init['mode'], model_seed=init.get('model_seed', 1_000_000), data_seed=seed,
                data_root=gen['data_root'], raw_root=gen['raw_root'], strength=.1,  # Fixed inputs: offsets stay zero.
                target_mode='solver', generation_protocol=sampler, generation_domain=domain, bus_range=[case, case],
                batches=base['synthetic_count']//base['batch_size'], reference_file=str(root/'shared/references'/f'case{case}.pt'),
                generator=gen, generation_id=generation_id, generation_output=str(generation_output))
            if init['mode'] == 'pretrained':
                checkpoint = init.get('checkpoints', {}).get(model) or init.get('checkpoint_pattern', '').format(model=model)
                if not checkpoint: raise ValueError('Missing pretrained checkpoint')
                cfg = pretrained_config(cfg | {'pretrained_checkpoint': checkpoint}, refresh=True)
            cfg = retention_inputs(cfg)
            identity = dict(method='RAMBO', config={k: v for k, v in cfg.items() if k not in
                ('generation_output', 'reference_file', 'pretrained_checkpoint')},
                source_sha256=digest(source_hashes), data_sha256=info['data_sha256'],
                reference_sha256=info['reference_sha256'], workers=12)
            run_id = 'RAMBO-eval-' + digest(identity)[:16]
            cfg['artifact_identity'] = dict(dataset_id=run_id, generation_id=generation_id,
                campaign=spec['campaign'], profile=spec['profile'], method='RAMBO', synthesis_seed=seed,
                initialization_name=regime, **{k: identity[k] for k in
                    ('source_sha256', 'data_sha256', 'reference_sha256', 'workers')})
            output = root/'runs/RAMBO'/model/f'case{case}'/domain/sampler/regime/f'seed{seed}-{run_id.split("-")[-1]}'
            rows.append(dict(method='RAMBO', dataset_id=run_id, config=cfg, output=str(output),
                             smoke=spec['profile'].endswith('smoke'),
                             verify_run=spec['execution'].get('verify_run', False)))
    return rows


def qualify_device(graphs, config, comm, device):
    """Compare ordinary student predictions/loss/gradients; RAMBO has no meta-gradient."""
    def check():
        batch, labels = SyntheticInputs(graphs[:2]).batch(detached=True)
        results = []
        for target in ('cpu', device):
            model = model_for(graphs[0], config, 500 + comm.rank, target)
            prediction = model(batch.to(target))
            loss = mse(prediction, labels, batch.to(target)); loss.backward()
            results.append((torch.cat([v.detach().cpu().flatten() for v in prediction.values()]),
                            float(loss.detach()), torch.cat([p.grad.detach().cpu().flatten() for p in model.parameters()
                                                   if p.grad is not None])))
        (p, a, g), (q, b, h) = results
        torch.testing.assert_close(q, p, rtol=2e-3, atol=1e-6)
        error, tolerance = float((g-h).norm()), 1e-6 + .02*float(g.norm())
        if abs(a-b) > 1e-6 + .002*abs(a) or error > tolerance: raise AssertionError('Student gradient mismatch')
        return dict(rank=comm.rank, passed=True, cpu_loss=a, device_loss=b, gradient_error=error,
                    gradient_tolerance=tolerance, scope='ordinary student; no RAMBO neural generation objective')
    return comm.allgather(collectively(comm, check))


def run(config, folder, comm, device, *, allocation_seconds=3300, qualify=False,
        stop_after_arms=None, generation_smoke=False, stop_after_updates=None):
    """Resume generation and evaluation arms independently, preserving physical labels."""
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + allocation_seconds
    config = collectively(comm, lambda: pretrained_config(config, refresh=True))
    identity = dict(config=config, workers=comm.size, device_type=torch.device(device).type,
                    torch=str(torch.__version__))
    def setup():
        path = folder/'run_identity.json'
        if path.exists() and json.loads(path.read_text()) != identity: raise ValueError('RAMBO evaluation identity changed')
        write_json(path, identity); write_json(folder/'config.json', config)
    collectively(comm, lambda: setup() if comm.rank == 0 else None)
    def status(value, **extra):
        result = dict(status=value, generation_id=config['generation_id'], **extra)
        if comm.rank == 0: write_json(folder/'status.json', result)
        return result
    generation_folder = Path(config['generation_output'])
    if (folder/'status.json').exists() and json.loads((folder/'status.json').read_text())['status'] == 'COMPLETE':
        from distillation.synthetic.artifacts import load_dataset
        if (folder/'artifact.json').exists():
            collectively(comm, lambda: load_dataset(folder))
            return json.loads((folder/'status.json').read_text())
    def generate():
        with Stop(max(0, deadline-time.monotonic()-config['reserve_seconds'])) as stop:
            if generation_smoke and not (generation_folder/'state.json').exists():
                first = generator.run(config['generator'], generation_folder, stop=stop, session_stages=2)
                if first['status'] not in ('paused', 'complete', 'busy'): raise RuntimeError(first)
                if first['status'] == 'paused': write_json(generation_folder/'smoke_resume.json', first)
            return generator.run(config['generator'], generation_folder, stop=stop)
    outcome = collectively(comm, lambda: generate() if comm.rank == 0 else None)
    outcome = comm.bcast(outcome, root=0)
    if outcome['status'] != 'complete':
        return status('EXHAUSTED' if outcome['status'] == 'exhausted' else 'RESUMABLE', generation=outcome)
    def inputs():
        generated = GeneratedDataset(generation_folder/'dataset.json')
        source = torch.load(local_path(config['reference_file']), weights_only=False, map_location='cpu')
        dataset = CaseDataset(config['data_root'], f"case{config['case']}"); split = dataset.splits(42)
        origin = generated.manifest['provenance']
        if origin['data_identity'] != digest(dataset.manifest) or origin['split_hashes'] != {k: ids_hash(v) for k, v in split.items()}:
            raise ValueError('Generation data/split changed')
        count = config['synthetic_count']; ids = source['ids'][:count]
        if len(generated) != count or len(ids) != count or len(source['graphs']) < count:
            raise ValueError('Complete generation and real control must meet the same requested count')
        if not set(ids) <= set(split['train'].tolist()): raise ValueError('References must be training-only')
        if config['validation_samples'] + config['final_samples'] > len(split['val']): raise ValueError('Validation panels overlap/exceed split')
        return generated, source['graphs'][:count], ids, dataset, split
    generated, original, seed_ids, dataset, split = collectively(comm, inputs)
    physical = list(generated)
    if any(g['bus'].num_nodes != config['case'] for g in physical): raise ValueError('RAMBO topology size changed')
    synthetic = SyntheticInputs(physical, config['strength'])
    population = split['train'][~np.isin(split['train'], seed_ids)]
    ids = split['val'][config['validation_samples']:config['validation_samples']+config['final_samples']]
    if comm.rank == 0:
        write_json(folder/'split.json', dict(split_seed=42, split_hashes={k: ids_hash(v) for k,v in split.items()},
            excluded_reference_ids=seed_ids, validation_ids=split['val'][:config['validation_samples']].tolist(),
            reporting_ids=ids.tolist(), test_accessed=False, dataset_selection='none: entire completed generated library'))
    if qualify and not (folder/'device_checks.json').exists():
        checks = qualify_device(physical, config, comm, device)
        if comm.rank == 0: write_json(folder/'device_checks.json', checks)
    partial = folder/'evaluation.partial.json'
    report = json.loads(partial.read_text()) if partial.exists() else {}
    sha = file_hash(generation_folder/'dataset.json')
    if report and report['generation_sha256'] != sha: raise ValueError('Generated data changed during evaluation')
    report.update(generation_sha256=sha, generation=outcome, generation_id=config['generation_id'])
    def persist(results=None):
        if results is not None: report['results'] = results
        if comm.rank == 0: write_json(partial, report)
        comm.Barrier()
    if 'calibration' not in report:
        if time.monotonic()+config['evaluation_reserve_seconds'] >= deadline: return status('EVALUATING', generation=outcome)
        _, report['calibration'] = evaluation.calibrate(original, config, dataset, population, comm, device)
        persist()
    real = SyntheticInputs(original, config['strength'], target_mode='frozen')
    baseline = 'pretrained_baseline' if config['initialization'] == 'pretrained' else 'untrained'
    arms = dict(generated=synthetic, random_real=real); arms[baseline] = real
    if config['initialization'] == 'fixed': arms.update(generated_unseen=synthetic, random_real_unseen=real)
    report['results'], complete = evaluation.evaluate_controls(arms, config, report['calibration']['chosen_rate'],
        dataset, ids, comm, device, results=report.get('results'), persist=persist, deadline=deadline,
        student_path=folder/'student.pt', student_arm='generated', stop_after_arms=stop_after_arms)
    if not complete: return status('EVALUATING', generation=outcome)
    def export():
        report.update(complete=True, validation_ids=ids.tolist(), test_accessed=False,
            student_updates=config['batches']*config['evaluation_epochs'],
            note='Whole generated library; no validation-selected data or learned curriculum. Ordinary matched SGD.')
        write_json(folder/'evaluation.json', report)
        student = torch.load(folder/'student.pt', weights_only=False, map_location='cpu')
        checks = [r['checks'] | {'success': bool(r['checks']['feasible'])} for r in generated.records]
        if not all(r['success'] for r in checks): raise ValueError('Unverified generated solution')
        _, labels = synthetic.batch(detached=True)
        save(folder/'dataset.pt', dict(format='rambo-opf-v1', config=config, target_mode='solver',
            training_state=dict(status='COMPLETE'), physical_solved_graphs=physical, solver_checks=checks,
            teaching_targets=labels, model_checkpoint=student, student_rate=report['calibration']['chosen_rate'],
            source_generation=dict(id=config['generation_id'], manifest=str(generation_folder/'dataset.json'),
                                   sha256=sha, statistics=outcome)))
        if not (generation_folder/'coverage.json').exists():
            write_json(generation_folder/'coverage.json', coverage(generation_folder/'dataset.json'))
    collectively(comm, lambda: export() if comm.rank == 0 else None)
    return status('COMPLETE', generation=outcome, examples=len(physical), evaluation_complete=True)
