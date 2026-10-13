"""Real XPU campaign smoke: qualification, interrupted synthesis, resume and replay."""
import time
from pathlib import Path

import torch

from distillation.synthetic.artifacts import publish, load_dataset, train_from_artifact
from distillation.synthetic.runtime import collectively
from distillation.training.T1_single_topo.common import write_json


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
            errors[name]=max(float((v.detach().cpu()-student['trained_state'][k].cpu()).abs().max())
                             for k,v in network.state_dict().items() if v.numel())
            for k,v in network.state_dict().items():
                torch.testing.assert_close(v.detach().cpu(),student['trained_state'][k].cpu(),rtol=1e-5,atol=1e-7)
            expected=config['batches']*(config['passes'] if experiment['method']=='DD' else config['evaluation_epochs'])
            if provenance['student_updates']!=expected: raise AssertionError('Wrong downstream update count')
        return errors
    replay_errors=collectively(comm,replay)
    if torch.device(device).type == 'xpu': torch.xpu.synchronize(device)
    rows=comm.allgather(dict(rank=comm.rank,seconds=time.monotonic()-started,
        peak_bytes=torch.xpu.max_memory_allocated(device) if torch.device(device).type == 'xpu' else None,
        replay_max_abs_error=replay_errors))
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
