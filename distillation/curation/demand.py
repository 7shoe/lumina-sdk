"""Pilot demand matching, with an explicit stationarity retry for new curation runs."""
import time
import numpy as np
import torch

from distillation.conversion.matrix_form import opf2mat
from distillation.experiments.speculative_followup.train import teaching_graph
from distillation.experiments.speculative_followup_two.data import load_reference
from distillation.experiments.speculative_followup_two.physical import native, solve as strict_solve
from distillation.training.T1_single_topo.common import read_json, write_json
from distillation.training.T1_single_topo.state import save_torch
from distillation.experiments.speculative_followup_three.curation import (Kernel, changed_graphs, demand_bounds, descriptors, groups,
                       predictions, project)


ORACLE_POLICY = 'strict_then_stationarity_retry_1e-7'


def solve(matrices, stop, record=None):
    """Retry numerical PIPS failures; retain the same physical feasibility gate."""
    from distillation.solver import ACOPF, PyPowerSolver
    graphs, checks = strict_solve(matrices, stop, record)
    for i, matrix in enumerate(matrices):
        checks[i].update(oracle_policy=ORACLE_POLICY, stationarity_tolerance=1e-8, retry=False)
        if graphs[i] is not None:
            continue
        if stop():
            raise BlockingIOError('Oracle retry paused')
        prior = checks[i].copy()
        if record is not None:
            record()
        started = time.monotonic()
        try:
            model = ACOPF(matrix)
            solution = PyPowerSolver(PDIPM_GRADTOL=1e-7).solve(model)
            output = solution.info.get('output', {})
            checks[i] = dict(solution.checks, success=solution.success, converged=solution.converged,
                stationarity_tolerance=1e-7, retry=True, oracle_policy=ORACLE_POLICY,
                strict_attempt=prior, seconds=time.monotonic()-started,
                iterations=output.get('iterations'), message=output.get('message'))
            if solution.success:
                graphs[i] = solution.to_hetero(model)
        except Exception as error:
            checks[i] = dict(success=False, retry=True, oracle_policy=ORACLE_POLICY,
                             strict_attempt=prior, error=repr(error), seconds=time.monotonic()-started)
    return graphs, checks


def synthesize(cfg, task, source, ids, rows, device, stop, folder):
    settings = cfg['demand']; seed = task['data_seed']; arm = task['arm']
    if settings.get('oracle_policy') != ORACLE_POLICY:
        raise ValueError('Freeze the explicit curation oracle policy')
    anchors = [source['graphs'][i].clone() for i in ids]
    n = len(anchors); parts = groups(source['demand'], cfg['scoring']['strata'])
    membership = np.empty(len(source['ids']), dtype=int)
    for j, part in enumerate(parts):
        membership[part] = j
    local_groups = [np.flatnonzero(membership[ids] == j) for j in range(len(parts))]
    if any(not len(g) for g in local_groups) and arm != 'random_solver':
        raise ValueError('Every fitted stratum needs an anchor')
    loads = torch.stack([g['load'].x for g in anchors]).to(device)
    all_loads = torch.stack([g['load'].x for g in source['graphs']]).to(device)
    lower, upper = demand_bounds(loads, all_loads, settings['radius'])
    lo = torch.tensor([source['demand'][parts[j]].min() for j in membership[ids]], device=device, dtype=loads.dtype)
    hi = torch.tensor([source['demand'][parts[j]].max() for j in membership[ids]], device=device, dtype=loads.dtype)
    rng = np.random.default_rng(seed+71000)
    mutable = torch.zeros(n, dtype=torch.bool, device=device)
    for group in local_groups:
        mutable[rng.permutation(group)[:len(group)//2]] = True
    if arm == 'demand_quantile':
        mutable[:] = True
    factor = torch.nn.Parameter(torch.ones(loads.shape[:2], device=device))
    def constrained(value):
        return project(value, loads, lower, upper, lo, hi, mutable)
    learned = arm in ('demand_input', 'demand_committee')
    path = folder/'demand-latest.pt'; history = []
    before = after = None
    if learned:
        input_kernel = Kernel(source['features'], seed, cfg['rff_features']).to(device)
        kernels, models = [], []
        for j, (name, values) in enumerate(rows.items()):
            kernel = Kernel(values, seed+j+1, cfg['rff_features']).to(device)
            model, _ = load_reference(cfg, name, anchors[0], device, pretrained=True)
            model.eval().requires_grad_(False)
            kernels.append(kernel); models.append(model)
        def mean_by_group(value, which):
            return torch.stack([value[torch.as_tensor(g, device=value.device)].mean(0) for g in which])
        with torch.no_grad():
            expected_input = mean_by_group(input_kernel(torch.as_tensor(source['features'], device=device)), parts)
            expected_outputs = [mean_by_group(k(torch.as_tensor(v, device=device)), parts)
                                for k, v in zip(kernels, rows.values())]
        def objective(backward=False):
            edited = changed_graphs(anchors, factor)
            observed = mean_by_group(input_kernel(descriptors(edited)), local_groups)
            loss = (observed-expected_input).square().sum(1).mean()
            loss = loss*(.5 if models else 1.) + settings['anchor_penalty']*((factor-1)/settings['radius']).square().mean()
            if backward:
                loss.backward()
            value = float(loss.detach())
            # Separate graphs per architecture keep activation memory bounded.
            for model, kernel, target in zip(models, kernels, expected_outputs):
                encoded = kernel(predictions(model, changed_graphs(anchors, factor)))
                loss = .5/len(models)*(mean_by_group(encoded, local_groups)-target).square().sum(1).mean()
                if backward:
                    loss.backward()
                value += float(loss.detach())
            if not np.isfinite(value):
                raise FloatingPointError('Nonfinite demand matching objective')
            return value
        optimizer = torch.optim.Adam([factor], lr=settings['rate'])
        start = 0; best = factor.detach().clone()
        with torch.no_grad():
            before = objective(); best_value = before
        if path.exists():
            state = torch.load(path, weights_only=False, map_location='cpu')
            if state['protocol'] != cfg['identity']:
                raise ValueError('Demand resume protocol changed')
            with torch.no_grad():
                factor.copy_(state['factor'].to(device))
            optimizer.load_state_dict(state['optimizer']); start = state['step']
            best = state['best'].to(device); best_value = state['best_value']
            before = state['before']; history = state['history']
        def save(step):
            save_torch(path, dict(protocol=cfg['identity'], step=step, factor=factor.detach(),
                optimizer=optimizer.state_dict(), best=best, best_value=best_value, before=before, history=history))
            write_json(folder/'demand-progress.json', dict(step=step, before=before, best=best_value,
                max_load_change=float((best-1).abs().max()), updated=time.time()))
        save(start)
        for step in range(start, settings['updates']):
            if stop():
                save(step); raise BlockingIOError('Demand learning paused')
            optimizer.zero_grad(set_to_none=True); objective(backward=True)
            if factor.grad is None or not torch.isfinite(factor.grad).all():
                raise FloatingPointError('Invalid demand gradient')
            optimizer.step()
            with torch.no_grad():
                factor.copy_(constrained(factor)); score = objective()
                if score < best_value:
                    best, best_value = factor.detach().clone(), score
            if (step+1) % settings.get('checkpoint_updates', 10) == 0 or step+1 == settings['updates']:
                history.append(dict(step=step+1, loss=score, best=best_value))
                save(step+1)
        with torch.no_grad():
            factor.copy_(best)
        after = best_value
    elif arm == 'demand_quantile':
        order = np.argsort(source['demand'][ids], kind='stable')
        total = np.quantile(source['demand'], (np.arange(n)+.5)/n)
        desired = np.ones(n); desired[order] = total/source['demand'][ids][order]
        with torch.no_grad():
            factor.copy_(constrained(torch.tensor(desired, device=device, dtype=loads.dtype)[:, None].expand_as(factor)))
    elif arm == 'demand_random':
        proposal = torch.tensor(rng.uniform(1-settings['radius'], 1+settings['radius'], size=factor.shape),
                                device=device, dtype=loads.dtype)
        with torch.no_grad():
            factor.copy_(constrained(proposal))
    elif arm not in ('solver_anchor', 'random_solver'):
        raise ValueError(arm)

    # All 100 published labels are freshly solved, including unchanged anchors.
    # A failed edit shrinks toward its own real anchor; no other example replaces it.
    ledger = folder/'oracle-calls.json'; calls = read_json(ledger)['calls'] if ledger.exists() else 0
    final_path = folder/'certification.pt'
    final = torch.load(final_path, weights_only=False, map_location='cpu') if final_path.exists() else dict(
        protocol=cfg['identity'], graphs=[], checks=[], fractions=[], factors=[])
    if final['protocol'] != cfg['identity']:
        raise ValueError('Certification resume identity changed')
    def record():
        nonlocal calls
        calls += 1; write_json(ledger, dict(calls=calls))
    for i in range(len(final['graphs']), n):
        if stop():
            raise BlockingIOError('Final AC-OPF certification paused')
        for fraction in settings['backtracking']:
            f = 1+fraction*(factor[i].detach().cpu()-1)
            graph = changed_graphs([anchors[i]], [f])[0]
            solved, check = solve([opf2mat(native(graph))], stop, record)
            if solved[0] is not None:
                result = teaching_graph(graph)
                for kind in ('bus', 'generator'):
                    result[kind].y = solved[0][kind].y.detach().cpu().clone()
                final['graphs'].append(result); final['checks'].append(check[0])
                final['fractions'].append(fraction); final['factors'].append(f)
                save_torch(final_path, final)
                break
        else:
            raise RuntimeError(f'AC-OPF failed even at real anchor {source["ids"][ids[i]]}')
    actual = torch.stack(final['factors'])
    final_matching = None
    if learned:
        with torch.no_grad():
            factor.copy_(actual.to(device)); final_matching = objective()
    return final['graphs'], dict(successful_final_solves=n, new_oracle_calls=calls,
        oracle_policy=ORACLE_POLICY, stationarity_retries_accepted=sum(c.get('retry',False) for c in final['checks']),
        physical_feasibility_tolerance=1e-6,
        physical_certificate=str(final_path), backtracking_fractions=final['fractions'],
        modified_graphs=int(((actual-1).abs().amax(1)>1e-6).sum()),
        max_load_scale_change=float((actual-1).abs().max()),
        mean_load_scale_change=float((actual-1).abs().mean()), matching_before=before,
        matching_before_solver_backtracking=after,
        matching_after_solver_backtracking=final_matching,
        final_labels='fresh converged AC-OPF solutions; physical feasibility tolerance 1e-6',
        approximation='Conditional kernel matching is a proxy; teacher predictions never become published labels')

