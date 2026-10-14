"""Real XPU campaign smoke: qualification, interrupted synthesis, resume and replay."""
import time
from pathlib import Path

import torch
from torch_geometric.data import Batch

from distillation.synthetic.artifacts import publish, load_dataset, train_from_artifact
from distillation.synthetic.runtime import collectively
from distillation.training.T1_single_topo.common import write_json


def check_replay(network, expected, graphs):
    """Check exact recipe replay without requiring deterministic accelerator sums.

    Keep the original elementwise check as the fast path. The bounded fallback
    also checks predictions and teaching loss on EVERY exported graph. Repeated
    attention-model XPU training can differ more than the old 1e-7 weight atol;
    this is recorded as numerical replay, never presented as exact reproduction.
    """
    actual = {k:v.detach().cpu().clone() for k,v in network.state_dict().items()}
    if actual.keys() != expected.keys(): raise AssertionError('Replay state keys changed')
    strict, maximum, squared, scale = True, 0., 0., 0.
    for key,value in actual.items():
        reference = expected[key].detach().cpu()
        if value.shape != reference.shape or value.dtype != reference.dtype:
            raise AssertionError(f'Replay shape/precision changed: {key}')
        if not value.is_floating_point():
            torch.testing.assert_close(value,reference,rtol=0,atol=0)
            continue
        delta = value.double()-reference.double()
        if not torch.isfinite(delta).all(): raise FloatingPointError(f'Nonfinite replay state: {key}')
        if delta.numel(): maximum = max(maximum,float(delta.abs().max()))
        squared += float(delta.square().sum()); scale += float(reference.double().square().sum())
        strict = strict and torch.allclose(value,reference,rtol=1e-5,atol=1e-7)
    relative = (squared/max(scale,1e-30))**.5
    report = dict(strict_state_match=strict,max_abs=maximum,relative_l2=relative)
    if strict: return report
    if maximum > 1e-4 or relative > 1e-5:
        raise AssertionError(f'Replay parameter drift exceeds numerical bounds: {report}')

    from distillation.synthetic.learning import mse
    device = next(network.parameters()).device
    def predictions():
        values,loss = {'bus':[], 'generator':[]},0.
        with torch.no_grad():
            for start in range(0,len(graphs),8):
                batch = Batch.from_data_list(graphs[start:start+8]).to(device)
                result = network(batch)
                loss += float(mse(result,{k:batch[k].y for k in values},batch))*batch.num_graphs
                for k in values: values[k].append(result[k].detach().cpu().double())
        return {k:torch.cat(v) for k,v in values.items()},loss/len(graphs)
    replay,loss_replay = predictions()
    try:
        network.load_state_dict(expected,strict=True)
        reference,loss_reference = predictions()
    finally:
        network.load_state_dict(actual,strict=True)
    drift = {k:float(torch.linalg.vector_norm(replay[k]-reference[k])) /
             max(float(torch.linalg.vector_norm(reference[k])),1e-12) for k in replay}
    loss_change = abs(loss_replay-loss_reference)/max(abs(loss_reference),1e-8)
    report.update(prediction_relative_l2=drift,teaching_loss_relative_change=loss_change,
                  teaching_loss_reference=loss_reference,teaching_loss_replay=loss_replay,
                  checked_graphs=len(graphs),
                  tolerances=dict(weight_max_abs=1e-4,weight_relative_l2=1e-5,
                                  prediction_relative_l2=.005,teaching_loss_relative_change=.001))
    numbers = list(drift.values())+[loss_reference,loss_replay,loss_change]
    if not torch.isfinite(torch.tensor(numbers,dtype=torch.float64)).all() or max(drift.values())>.005 or loss_change>.001:
        raise AssertionError(f'Replay functional drift exceeds numerical bounds: {report}')
    return report


def run(runner, experiment, comm, device, remaining):
    folder=Path(experiment['output']);config=experiment['config'];started=time.monotonic()
    if torch.device(device).type == 'xpu': torch.xpu.reset_peak_memory_stats(device)
    if not (folder/'latest.pt').exists():
        state=runner.run(config,folder,comm,device,allocation_seconds=remaining,
                         qualify=config.get('qualify',False),stop_after_updates=1)
        if state['iteration']!=1 or state['status']!='RESUMABLE':
            raise RuntimeError(f'Smoke interruption did not reach one committed update: {state["status"]}')
        if comm.rank==0: write_json(folder/'smoke_resume.json',dict(iteration=1,status=state['status']))
    state=runner.run(config,folder,comm,device,allocation_seconds=remaining-(time.monotonic()-started),
                     qualify=not (folder/'device_checks.json').exists() and config.get('qualify',False))
    if state['status']!='COMPLETE': return state
    finished = state['iteration'] >= config['max_updates'] or state['active_seconds'] >= config['training_seconds']
    if not finished or state['resume_count']<2:
        raise RuntimeError('Smoke did not resume and complete its matching/outer updates')
    collectively(comm,lambda: publish(experiment) if comm.rank==0 else None)
    def replay():
        loaded=load_dataset(folder)
        if len(loaded['graphs'])!=config['synthetic_count']: raise AssertionError('Wrong export count')
        export=loaded['export']
        students=export.get('model_checkpoints') or {config['model']:export['model_checkpoint']}
        errors={}
        for name,student in students.items():
            network,provenance=train_from_artifact(folder,device=device,model=name)
            errors[name]=check_replay(network,student['trained_state'],loaded['graphs'])
            expected=config['batches']*(config['passes'] if experiment['method']=='DD' else config['evaluation_epochs'])
            if provenance['student_updates']!=expected: raise AssertionError('Wrong downstream update count')
        return errors
    replay_errors=collectively(comm,replay)
    if torch.device(device).type == 'xpu': torch.xpu.synchronize(device)
    rows=comm.allgather(dict(rank=comm.rank,seconds=time.monotonic()-started,
        peak_bytes=torch.xpu.max_memory_allocated(device) if torch.device(device).type == 'xpu' else None,
        replay_max_abs_error={k:v['max_abs'] for k,v in replay_errors.items()},replay_checks=replay_errors))
    if comm.rank==0:
        write_json(folder/'smoke.json',dict(status='PASS',workers=comm.size,case=config['case'],
            target_mode=config['target_mode'],synthetic_count=config['synthetic_count'],
            synthesis_updates=state['iteration'],resume_count=state['resume_count'],
            accepted_updates=state['accepted_updates'],replicas=rows,
            note='Execution/derivative/resume/serialization checks; use held-out evaluation and controls to assess usefulness'))
    return state


def run_rambo(runner, experiment, comm, device, remaining):
    """Interrupt generation/evaluation, resume, then reproduce the exported student."""
    folder=Path(experiment['output']);config=experiment['config'];started=time.monotonic()
    if torch.device(device).type == 'xpu': torch.xpu.reset_peak_memory_stats(device)
    if not (folder/'evaluation.partial.json').exists():
        first=runner.run(config,folder,comm,device,allocation_seconds=remaining,qualify=True,
                         generation_smoke=True,stop_after_arms=1)
        if first['status']!='EVALUATING': return first
        if comm.rank==0: write_json(folder/'smoke_resume.json',first)
    state=runner.run(config,folder,comm,device,allocation_seconds=remaining-(time.monotonic()-started),qualify=True)
    if state['status']!='COMPLETE': return state
    collectively(comm,lambda: publish(experiment) if comm.rank==0 else None)
    def check():
        import json
        loaded=load_dataset(folder);export=loaded['export']
        if len(loaded['graphs'])!=config['synthetic_count']: raise AssertionError('Incomplete generated export')
        report=json.loads((folder/'evaluation.json').read_text())
        if any(v['mean'] is None for v in report['results'].values()): raise AssertionError('Nonfinite student evaluation')
        network,provenance=train_from_artifact(folder,device=device)
        errors=[]
        for key,value in network.state_dict().items():
            saved=export['model_checkpoint']['trained_state'][key].cpu()
            torch.testing.assert_close(value.detach().cpu(),saved,rtol=1e-5,atol=1e-7)
            if value.numel(): errors.append(float((value.detach().cpu()-saved).abs().max()))
        if provenance['student_updates']!=config['batches']*config['evaluation_epochs']:
            raise AssertionError('Wrong student schedule')
        for sdk,physical in zip(loaded['graphs'],loaded['physical_solved_graphs']):
            for kind in ('bus','generator'): torch.testing.assert_close(sdk[kind].y,physical[kind].y,rtol=0,atol=0)
        return max(errors)
    error=collectively(comm,check)
    if torch.device(device).type == 'xpu': torch.xpu.synchronize(device)
    rows=comm.allgather(dict(rank=comm.rank,replay_max_abs_error=error,seconds=time.monotonic()-started,
        peak_bytes=torch.xpu.max_memory_allocated(device) if torch.device(device).type=='xpu' else None))
    if comm.rank==0:
        write_json(folder/'smoke.json',dict(status='PASS',method='RAMBO',workers=comm.size,
            generation_id=config['generation_id'],generation=state['generation'],examples=config['synthetic_count'],
            student_updates=config['batches']*config['evaluation_epochs'],replicas=rows,
            evaluation_resume=(folder/'smoke_resume.json').exists(),
            generation_resume=(Path(config['generation_output'])/'smoke_resume.json').exists(),
            note='Generation is CPU/Julia; checks qualify student execution and artifact reuse, not data usefulness'))
    return state
