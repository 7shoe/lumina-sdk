"""Nominal-load anchoring and lossless exchange with the Julia backend."""
from collections import Counter
from dataclasses import replace
from pathlib import Path
import json
import re
import tarfile
import numpy as np

import torch
from torch_geometric.data import HeteroData

from distillation.conversion.matrix_form import (mat2opf,
                                                 opf2mat,
                                                 validate_network)

from distillation.selection import (input_view,
                                    ids_hash)
from distillation.selection.generated import GeneratedDataset

from distillation.solver import ACOPF
from distillation.solver.problem import as_ppc
from distillation.solver.solution import Solution
from distillation.solver.validate import feasibility

from distillation.training.T1_single_topo.common import (digest,
                                                         file_hash)


def read_case(path):
    """Read numeric MATPOWER matrices in the three pinned PGLib cases only.

    This deliberately does not interpret MATLAB code. Julia/PowerModels parses
    the exported case independently before any labels are accepted.
    """
    
    # strip MATLAB comments before extracting the base and numeric tables
    text = re.sub(r'%[^\n]*', '', Path(path).read_text())
    base = float(re.search(r'mpc.baseMVA\s*=\s*([\d.eE+-]+)', text)[1])
    result = dict(baseMVA=base, version='2')
    for key in ('bus', 'gen', 'branch', 'gencost'):
        body = re.search(r'mpc\.' + key + r'\s*=\s*\[(.*?)\];', text, re.S)[1]
        result[key] = np.asarray([[float(x) for x in row.split()] for row in body.split(';') if row.strip()])
    return result


def native_input(grid):
    """Read only physical inputs from OPFData's native Appendix A schema."""
    graph = HeteroData()
    graph.x = torch.tensor(grid['context'], dtype=torch.float64).reshape(-1)

    
    for kind, width in (('bus', 4), ('generator', 11), ('load', 2), ('shunt', 2)):
        graph[kind].x = torch.tensor(grid['nodes'][kind], dtype=torch.float64).reshape(-1, width)
    
    for kind in ('ac_line', 'transformer', 'generator_link', 'load_link', 'shunt_link'):
        edge = grid['edges'][kind]
        # branches connect buses
        source = 'bus' if kind in ('ac_line', 'transformer') else kind.removesuffix('_link')
        store = graph[source, kind, 'bus']
        store.edge_index = torch.tensor([edge['senders'], edge['receivers']], dtype=torch.long).reshape(2, -1)
        if source == 'bus':
            store.edge_attr = torch.tensor(edge['features'], dtype=torch.float64).reshape(-1, 9 if kind == 'ac_line' else 11)
    return graph


def prepare_template(dataset, case_path, raw_root):
    """Use only a training input for schema; anchor loads to the pinned nominal case."""
    splits = dataset.splits(42)
    source_id = int(splits['train'][0])
    case = dataset.manifest['case']
    group = source_id // dataset.manifest['group_size']
    archive_path = Path(raw_root) / f'{case}_{group}.tar.gz'
    
    # verify that the native template comes from the recorded source archive
    raw_hash = file_hash(archive_path)
    if raw_hash != dataset.manifest['groups'][group]['raw_sha256']:
        raise ValueError('Native archive differs from the source of the processed dataset')
    wanted = f'{case}/group_{group}/example_{source_id}.json'
    graph = None
    with tarfile.open(archive_path, 'r|gz') as archive:
        for member in archive:
            if member.isfile() and '/'.join(Path(member.name).parts[-3:]) == wanted:
                graph = native_input(json.load(archive.extractfile(member))['grid'])
                break
    if graph is None:
        raise ValueError(f'Missing training template {wanted}')
    mats = opf2mat(graph)
    nominal, actual = read_case(case_path), as_ppc(graph)[0]
    if nominal['baseMVA'] != actual['baseMVA']:
        raise ValueError('Nominal and OPFData per-unit bases differ')
    # Ignore setpoints, descriptive voltage levels, and demand; compare physics.
    for key, columns in (('bus', [0, 1, 4, 5, 11, 12]),
                         ('gen', [0, 3, 4, 7, 8, 9]), ('gencost', list(range(7)))):
        np.testing.assert_allclose(actual[key][:, columns], nominal[key][:, columns], rtol=2e-6, atol=2e-6,
                                   err_msg=f'Nominal case mismatch: {key}')
    def branches(ppc):
        # normalize unity taps and sort branche
        rows = ppc['branch'].copy()
        rows[rows[:, 8] == 0, 8] = 1
        return rows[np.lexsort((rows[:, 3], rows[:, 2], rows[:, 1], rows[:, 0]))]
    np.testing.assert_allclose(branches(actual), branches(nominal), rtol=2e-6, atol=2e-6,
                               err_msg='Nominal case mismatch: branch parameters')
    if mats.params.load_bus.unique().numel() != mats.params.L:
        raise ValueError('Initial RAMBO adapter requires at most one load per bus')
    buses = mats.params.load_bus.numpy()
    
    if not np.array_equal(nominal['bus'][:, 0], np.arange(1, mats.params.N + 1)):
        raise ValueError('Pinned case bus indexing changed')
    
    # replace perturbed training loads with nominal PGLib loads in per unit
    pd, qd = (torch.from_numpy(nominal['bus'][buses, k] / nominal['baseMVA']) for k in (2, 3))
    p = replace(mats.params, pd=pd, qd=qd)
    validate_network(p)
    graph = mat2opf(replace(mats, params=p, sol=None), include_solution=False)
    provenance = dict(data_identity=digest(dataset.manifest), template_train_id=source_id,
                      split_hashes={k: ids_hash(v) for k, v in splits.items()},
                      raw_sha256=raw_hash, nominal_case_sha256=file_hash(case_path),
                      nominal_case=str(Path(case_path).resolve()))
    return graph, provenance


def write_matpower(graph, path):
    ppc = as_ppc(graph)[0]
    # preserve double-precision values in the textual exchange with Julia
    lines = ["function mpc = case_rambo", "mpc.version = '2';", f"mpc.baseMVA = {ppc['baseMVA']:.17g};"]
    for key in ('bus', 'gen', 'branch', 'gencost'):
        lines += [f'mpc.{key} = [']
        lines += [' '.join(f'{v:.17g}' for v in row) + ';' for row in ppc[key]]
        lines += ['];']
    Path(path).write_text('\n'.join(lines) + '\n')


def network_record(graph, bounds):
    p = opf2mat(graph).params
    fields = ('vmin', 'vmax', 'pmin', 'pmax', 'qmin', 'qmax', 'c2', 'c1', 'c0', 'pd', 'qd',
              'rate_a', 'angmin', 'angmax')
    result = {k: getattr(p, k).double().tolist() for k in fields}
    for key in ('fr', 'to', 'gen_bus', 'load_bus'):
        result[key] = (getattr(p, key) + 1).tolist()  # Julia/PowerModels indices.
    result.update(n=p.N, g=p.G, l=p.L, ref=int(torch.nonzero(p.bus_type == 3)[0, 0]) + 1)
    
    # code complex admittances as separate real and imaginary JSON arrays
    for name, values in zip(('ff', 'ft', 'tf', 'tt'), p.branch_admittances()):
        result[name + '_re'], result[name + '_im'] = values.real.tolist(), values.imag.tolist()
    shunts = p.y_sh
    result['gs'], result['bs'] = shunts.real.tolist(), shunts.imag.tolist()
    
    for key in ('pd', 'qd'):
        # sorting keeps lower/upper bounds correct even for negative nominal loads
        lo, hi = np.sort(np.asarray(result[key])[:, None] * np.asarray(bounds), axis=1).T
        result[key + '_lo'], result[key + '_hi'] = lo.tolist(), hi.tolist()
    
    return result


def attach_solution(template, point, tolerance=1e-6):
    """Preserve the final Ipopt label; check independently without re-solving it."""
    mats = opf2mat(template)
    p = replace(mats.params, pd=torch.tensor(point['pd'], dtype=torch.float64),
                qd=torch.tensor(point['qd'], dtype=torch.float64))
    graph = mat2opf(replace(mats, params=p, sol=None), include_solution=False)
    problem = ACOPF(graph)
    arrays = [np.asarray(point[key], dtype=np.float64) for key in ('theta', 'vm', 'pg', 'qg')]
    solution = Solution(*arrays, float(point['cost']), True, 'rambo_powermodels_ipopt')
    solution.checks = feasibility(problem, solution, tolerance)
    if not solution.success:
        raise ValueError(f'Independent AC-OPF verification failed: {solution.checks}')
    return solution.to_hetero(problem), solution.checks


def coverage(manifest, threshold=.01):
    """Range-relative active bounds; both line ends; ignore fixed bounds.

    This explicitly distinguishes ever-active individual constraints from joint
    active-set patterns. It does not reproduce the upstream zero-bound division.
    """
    if not 0 < threshold < .5:
        raise ValueError('Use a relative activity threshold between zero and 0.5')
    dataset = GeneratedDataset(manifest)
    patterns, counter, values = [], Counter(), {}
    ignored, total = 0, 0
    for graph in dataset:
        p, problem = opf2mat(graph).params, ACOPF(graph)
        solution = Solution.from_label(graph)
        sf, st = problem.branch_flows(solution.x(problem))
        active, constraints, fixed = [], 0, 0
        def interval(key, value, lo, hi):
            nonlocal constraints, fixed
            value, lo, hi = map(lambda v: np.asarray(v, dtype=float), (value, lo, hi))
            width = hi - lo
            mask = width > 0
            fixed += int((~mask).sum())
            constraints += 2 * int(mask.sum())
            
            # measure proximity relative to interval width, not the bound's value
            for side, margin in (('lo', value - lo), ('hi', hi - value)):
                active.extend(f'{key}:{i}:{side}' for i in np.flatnonzero(mask & (margin <= threshold * width)))
        for key, value, lo, hi in (
                ('vm', solution.vm, p.vmin, p.vmax), ('pg', solution.pg, p.pmin, p.pmax),
                ('qg', solution.qg, p.qmin, p.qmax),
                ('angle', problem.angles(solution.x(problem)), problem.angle_lo, problem.angle_hi)):
            interval(key, value, lo, hi)
        
        # count thermal activity separately at each end of positively rated branches
        rates = p.rate_a.numpy()
        for end, flow in (('from', sf), ('to', st)):
            active.extend(f'thermal:{i}:{end}' for i in np.flatnonzero((rates > 0) &
                          (flow.abs().numpy() >= (1 - threshold) * rates)))
            constraints += int((rates > 0).sum())
        for key, value in dict(vm=solution.vm, pg=solution.pg, qg=solution.qg,
                               pd=p.pd.numpy(), qd=p.qd.numpy()).items():
            values.setdefault(key, []).append(np.asarray(value))
        
        # track joint active sets separately from individual constraint frequencies
        patterns.append(tuple(sorted(active)))
        counter.update(active)
        ignored, total = fixed, constraints
    ranges = {}
    for key, samples in values.items():
        a = np.stack(samples)
        ranges[key] = dict(min=a.min(axis=0).tolist(), max=a.max(axis=0).tolist(),
                           mean_coordinate_range=float(np.ptp(a, axis=0).mean()))
    folder = Path(manifest).parent
    
    # summarize attempt records, excluding separate *_rejected files
    attempts = [json.loads(p.read_text())['result'] for p in sorted((folder / 'attempts').glob('*.json'))
                if not p.stem.endswith('_rejected')]
    result = dict(accepted=len(dataset), method=dataset.manifest['method'],
                  threshold=threshold, fixed_bound_pairs_excluded=ignored,
                  candidate_individual_constraints=total, individual_ever_active=len(counter),
                  joint_active_patterns=len(set(patterns)), activity_counts=dict(sorted(counter.items())),
                  marginal_ranges=ranges, final_solver_statuses=dict(Counter(r['solver_status'] for r in dataset.records)),
                  attempted_solver_statuses=dict(Counter(r['status'] for r in attempts)),
                  solver_seconds=sum(r.get('solve_seconds', 0.) for r in attempts), test_used=False)
    
    if dataset.records:
        result['max_physical_residuals'] = {key: max(r['checks'][key] for r in dataset.records)
                     for key in ('balance', 'thermal', 'bounds', 'angle', 'reference', 'cost_relative')}
    relaxed = [r['relaxed_verification'] for r in dataset.records if 'relaxed_verification' in r]
    
    if relaxed:
        result['max_relaxed_to_final_difference'] = {key: max(r[key] for r in relaxed) for key in relaxed[0]}
    audited = [r for r in attempts if r.get('success') and 'stationarity' in r]
    
    if audited:
        result['max_accepted_kkt_residuals'] = {k: max(r[k] for r in audited)
            for k in ('primal_error', 'stationarity', 'dual_violation', 'complementarity')}
    return result
