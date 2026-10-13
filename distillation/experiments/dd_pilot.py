"""Prepare, qualify and describe the DD pilot; optimization stays in DD/campaign.py."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from distillation.DD import campaign as dd
from distillation.DD.data import load_ids
from distillation.DD.mvp import check_derivatives
from distillation.data.split import split_indices
from distillation.synthetic.experiments import initialize
from distillation.training.T1_single_topo.common import canonical_case, write_json
from experiments.lumina_bench.models import SUPPORTED_MODELS

CASES = (30, 57, 118)
MODELS = ('GAT', 'GIN', 'HeteroGNN')
PROTOCOLS = ('conservative', 'random_fixed', 'random_size')


def config(root, case, model, protocol='conservative', target='solver'):
    return dd.default_config(case, model) | dict(generation_protocol=protocol, target_mode=target,
        reference_file=str(root / 'references' / f'case{case}.pt'),
        initial_cache=str(root / 'initializations' / f'case{case}' / protocol / 'initial_samples'),
        training_seconds=800, real_batch=32, validation_samples=128, final_samples=512,
        eval_seconds=300, checkpoint_seconds=60, generation_attempts=128)


def prepare_references(root, comm):
    """Sample 100 IDs uniformly from train, then stream raw groups on 12 CPUs."""
    if comm.rank == 0:
        (root / 'references').mkdir(parents=True, exist_ok=True)
    comm.Barrier()
    for case in CASES:
        path = root / 'references' / f'case{case}.pt'
        if comm.bcast(path.exists() if comm.rank == 0 else None, root=0):
            continue
        ids = np.random.default_rng(42).choice(split_indices(canonical_case(f'case{case}'))['train'], 100, replace=False).tolist()
        raw = dd.default_config(case, 'GAT')['raw_root']
        local = []
        for group in range(comm.rank, 20, comm.size):
            selected = [i for i in ids if i // 15_000 == group]
            local.extend(zip(selected, load_ids(raw, case, selected)))
        graphs = dict(item for part in comm.allgather(local) for item in part)
        if comm.rank == 0:
            dd.save(path, dict(graphs=[graphs[i] for i in ids], ids=ids, split_seed=42,
                sampling='100 without replacement uniformly from the complete train split; raw FP64'))
            print(json.dumps(dict(stage='references', case=case, count=len(ids))), flush=True)
        comm.Barrier()


def qualify(root, comm, device):
    prepare_references(root, comm)
    reports = []
    for case in CASES:
        dataset = dd.CaseDataset(dd.default_config(case, 'GAT')['data_root'], f'case{case}')
        for protocol in PROTOCOLS:
            cfg = config(root, case, 'GAT', protocol)
            folder = root / 'initializations' / f'case{case}' / protocol
            folder.mkdir(parents=True, exist_ok=True)
            _, _, graphs, _ = initialize(cfg, folder, comm)
            if comm.rank == 0:
                print(json.dumps(dict(stage='initialization', case=case, protocol=protocol, graphs=len(graphs))), flush=True)
        graphs = torch.load(root / 'initializations' / f'case{case}' / 'conservative' / 'initial.pt', weights_only=False)['graphs']
        real = dd.graph_batch(dataset, dataset.splits()['train'][100:104])
        # Full-width backbones, short CPU/XPU comparison, then the actual 100-step trajectory.
        for kind in SUPPORTED_MODELS if case == 30 else MODELS:
            cfg = config(root, case, kind, target='learned')
            model = dd.model_for(graphs[0], cfg, 500 + comm.rank, 'cpu')
            derivative = check_derivatives(model, graphs[:3], real, {k: real[k].y for k in ('bus', 'generator')},
                dict(steps=3, target_mode='learned', initial_rate=1e-7, rate_min=1e-10, rate_max=.01), device)
            passed = derivative['loss_passed'] and all(r['error_l2'] <= 1e-8 + .02 * r['cpu_gradient_norm'] for r in derivative['gradients'])
            checks = comm.allgather(derivative | {'rank': comm.rank, 'passed': passed})
            if not all(r['passed'] for r in checks):
                if comm.rank == 0:
                    write_json(root / f'failed_derivatives_{kind}_{case}.json', checks)
                raise RuntimeError(f'CPU/XPU derivative comparison failed: {kind}, case{case}')
            del model
            synthetic = dd.Schedule(graphs, cfg)
            optimizer = dd.optimizer_for(synthetic, cfg)
            real = dd.graph_batch(dataset, dataset.splits()['train'][100 + comm.rank * 32:132 + comm.rank * 32])
            model = dd.model_for(graphs[0], cfg, 1_000_000 + comm.rank, device)
            torch.xpu.reset_peak_memory_stats(device)
            tick = time.monotonic()
            loss, norm, backtracks = dd.training_gradient(model, synthetic, optimizer, real, comm)
            torch.xpu.synchronize(device)
            row = dict(case=case, model=kind, student_updates=100, synthetic_grids=100,
                real_batch=32, seconds=max(comm.allgather(time.monotonic()-tick)),
                peak_bytes=max(comm.allgather(torch.xpu.max_memory_allocated(device))),
                initial_rate=float(synthetic.rates[0].detach()), rate_backtracks=backtracks,
                loss=loss, gradient_norm=norm, device_checks=checks)
            reports.append(row)
            if comm.rank == 0:
                write_json(root / 'qualification.json', reports)
                print(json.dumps({k:v for k,v in row.items() if k != 'device_checks'}), flush=True)
            del model, synthetic, optimizer, real
            torch.xpu.empty_cache()
            real = dd.graph_batch(dataset, dataset.splits()['train'][100:104])
    # Exercise all target policies, atomic resume, refreshed labels and final evaluation.
    for target in ('solver', 'frozen', 'learned'):
        cfg = config(root, 30, 'GIN', target=target) | dict(max_updates=2, real_batch=4,
            validation_samples=16, final_samples=16, real_control_rates=[1e-7, 1e-6],
            trace_samples=4, inner_rate=1e-7, model_options=dd.model_options('GIN', width=16),
            initialization='fixed' if target == 'frozen' else 'random')
        folder = root / 'smoke' / target
        first = dd.run(cfg, folder, comm, device, stop_after_updates=1)
        assert first['status'] == 'RESUMABLE'
        second = dd.run(cfg, folder, comm, device)
        assert second['status'] == 'COMPLETE' and second['resume_count'] == 2
    if comm.rank == 0:
        write_json(root / 'QUALIFIED.json', dict(status='PASS', models=list(SUPPORTED_MODELS),
            cases=list(CASES), targets=['solver', 'frozen', 'learned'], workers=comm.size))


def manifest(root):
    if not (root / 'QUALIFIED.json').exists():
        raise RuntimeError('Finish debug qualification before preparing the pilot')
    calibrated = {(r['case'], r['model']): r['initial_rate']
                  for r in json.loads((root / 'calibration.json').read_text())}
    experiments = []
    for case in CASES:
        for kind in MODELS:
            for protocol in PROTOCOLS:
                for target in ('solver', 'frozen', 'learned'):
                    cfg = config(root, case, kind, protocol, target)
                    cfg['inner_rate'] = calibrated[case, kind]
                    folder = root / 'experiments' / kind / f'case{case}' / protocol / target
                    experiments.append(dict(method='DD', config=cfg, output=str(folder)))
    for kind in MODELS:
        for target in ('solver', 'frozen', 'learned'):
            cfg = config(root, 30, kind, target=target) | dict(initialization='fixed',
                inner_rate=calibrated[30, kind])
            experiments.append(dict(method='DD', config=cfg, output=str(root / 'ablations' / kind / f'fixed_{target}')))
        for strength in (.02, .20):
            cfg = config(root, 30, kind) | dict(strength=strength, inner_rate=calibrated[30, kind])
            experiments.append(dict(method='DD', config=cfg, output=str(root / 'ablations' / kind / f'strength_{strength}')))
    # Place expensive and cheap cases across all nodes instead of concentrating case118.
    rng = np.random.default_rng(1012)
    experiments = [experiments[i] for i in rng.permutation(len(experiments))]
    write_json(root / 'manifest.json', dict(method='DD', experiments=experiments))
    print(json.dumps(dict(experiments=len(experiments), manifest=str(root / 'manifest.json'))))


def calibrate(root, comm, device):
    """One train-only starting-rate choice per model/case, shared across protocols."""
    if not (root / 'QUALIFIED.json').exists():
        raise RuntimeError('Qualification must pass first')
    path = root / 'calibration.json'
    records = json.loads(path.read_text()) if path.exists() else []
    for case in CASES:
        dataset = dd.CaseDataset(dd.default_config(case, 'GAT')['data_root'], f'case{case}')
        ids = dataset.splits()['train'][1000:1128]
        for kind in MODELS:
            if any(r['case'] == case and r['model'] == kind for r in records):
                continue
            cfg = config(root, case, kind) | {'trace_every': 0}
            graphs = {p: torch.load(root / 'initializations' / f'case{case}' / p / 'initial.pt',
                                   weights_only=False)['graphs'] for p in PROTOCOLS}
            candidates = []
            for rate in cfg['real_control_rates']:
                scores = {}
                for protocol in PROTOCOLS:
                    synthetic = dd.Schedule(graphs[protocol], cfg | {'inner_rate': rate})
                    metric, _ = dd.panel(synthetic, cfg, dataset, ids, comm, device, seed_base=4_000_000)
                    scores[protocol] = metric['mean']
                mean = float(np.mean(list(scores.values()))) if all(x is not None for x in scores.values()) else None
                candidates.append(dict(rate=rate, mean=mean, protocols=scores))
            finite = [r for r in candidates if r['mean'] is not None and math.isfinite(r['mean'])]
            if not finite:
                raise RuntimeError(f'No finite initial rate for {case}/{kind}')
            chosen = min(finite, key=lambda r: r['mean'])['rate']
            # Verify second-order stability at the selected rate on every protocol.
            safe = chosen
            for protocol in PROTOCOLS:
                synthetic = dd.Schedule(graphs[protocol], cfg | {'inner_rate': chosen, 'target_mode': 'learned'})
                model = dd.model_for(graphs[protocol][0], cfg, 1_000_000 + comm.rank, device)
                real = dd.graph_batch(dataset, ids[:32])
                _, _, _ = dd.training_gradient(model, synthetic, dd.optimizer_for(synthetic, cfg), real, comm)
                safe = min(safe, float(synthetic.rates.min().detach()))
                del model, synthetic, real
                torch.xpu.empty_cache()
            records.append(dict(case=case, model=kind, initial_rate=safe, selected_rate=chosen,
                candidates=candidates, training_ids=ids.tolist(), model_seeds=list(range(4_000_000, 4_000_000 + comm.size))))
            if comm.rank == 0:
                write_json(path, records)
                print(json.dumps(dict(stage='calibration', case=case, model=kind, initial_rate=safe)), flush=True)


def report(root):
    """A compact index for picking up curricula without reading scheduler output."""
    records = []
    for experiment in json.loads((root / 'manifest.json').read_text())['experiments']:
        folder, cfg = Path(experiment['output']), experiment['config']
        state = json.loads((folder / 'status.json').read_text()) if (folder / 'status.json').exists() else {}
        metrics = json.loads((folder / 'evaluation.json').read_text()) if (folder / 'evaluation.json').exists() else {}
        records.append(dict(case=cfg['case'], model=cfg['model'], protocol=cfg['generation_protocol'],
            target_mode=cfg['target_mode'], initialization=cfg['initialization'], strength=cfg['strength'],
            status=state.get('status', 'NOT_STARTED'), outer_updates=state.get('iteration', 0),
            accepted_updates=state.get('accepted_updates', 0),
            unique_real_examples=state.get('distinct_real_training_examples', 0),
            student_updates=cfg['batches']*cfg['passes'], synthetic_grids=cfg['synthetic_count'],
            best_mse=metrics.get('best_distilled', {}).get('mean'),
            real_control_mse=metrics.get('real_subset_tuned_rates', {}).get('mean'),
            best=str(folder/'best.pt') if (folder/'best.pt').exists() else None,
            last=str(folder/'distilled.pt') if (folder/'distilled.pt').exists() else None,
            resume=str(folder/'latest.pt') if (folder/'latest.pt').exists() else None,
            error=(folder/'failure.txt').read_text()[-1200:] if (folder/'failure.txt').exists() else None))
    write_json(root / 'datasets.json', records)
    print(json.dumps(dict(statuses=dict(Counter(r['status'] for r in records)),
                         exported=sum(r['best'] is not None for r in records), index=str(root/'datasets.json'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('qualify', 'calibrate', 'manifest', 'report'))
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    if args.stage in ('manifest', 'report'):
        (manifest if args.stage == 'manifest' else report)(args.root)
    else:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        torch.set_num_threads(1)
        threadpool_limits(limits=1)
        torch.xpu.set_device(comm.rank)
        uuids = comm.allgather(str(torch.xpu.get_device_properties(comm.rank).uuid))
        if len(set(uuids)) != 12:
            raise RuntimeError('Qualification requires 12 distinct XPUs')
        (qualify if args.stage == 'qualify' else calibrate)(args.root, comm, torch.device(f'xpu:{comm.rank}'))
