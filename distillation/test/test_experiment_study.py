"""Stable experiment identity, strict YAML, and reusable teaching-target artifacts."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
import yaml

from distillation.experiments import study
from distillation.synthetic import artifacts
from distillation.synthetic.inputs import SyntheticInputs
from distillation.synthetic.learning import model_for, model_options, train_student
from distillation.synthetic.evaluation import cpu_state
from distillation.DD import campaign as dd
from distillation.DC import campaign as dc
from distillation.HMDC import campaign as hm
from distillation.data.split import split_indices
from distillation.test import test_dd_demo as reference

cpu_threads,record,graph=reference.cpu_threads,reference.record,reference.graph


def test_explicit_runtime_revision_preserves_original_manifest_and_checkpoint(tmp_path,monkeypatch):
    root=tmp_path/'study';root.mkdir()
    runtime=tmp_path/'working';source=runtime/'distillation/runtime.py'
    source.parent.mkdir(parents=True);source.write_text('new runtime\n')
    original={'source_hashes':{'distillation/runtime.py':'old-hash'},'experiments':[{'dataset_id':'same'}]}
    (root/'manifest.json').write_text(json.dumps(original))
    (root/'latest.pt').write_bytes(b'original checkpoint')
    monkeypatch.setattr(study,'ROOT',runtime)
    monkeypatch.setattr(study,'source_files',lambda: {'distillation/runtime.py':study.file_hash(source)})
    revision=study.refresh_runtime(root)
    assert revision['changed_files']==['distillation/runtime.py']
    assert (Path(revision['source'])/'distillation/runtime.py').read_text()=='new runtime\n'
    assert json.loads((root/'manifest.json').read_text())==original
    assert (root/'latest.pt').read_bytes()==b'original checkpoint'
    assert study.refresh_runtime(root)==revision


@pytest.fixture
def spec(tmp_path,graph):
    cfg=study.load()
    cfg['methods'].pop('RAMBO', None)  # Original condensation-only identity/regression matrix.
    cfg['paths']={k:str(tmp_path/k) for k in ('output','models','data','raw','references')}
    for case in (30,57,118):
        path=Path(cfg['paths']['data'])/dd.canonical_case(f'case{case}')/'manifest.json'
        path.parent.mkdir(parents=True);path.write_text('{}')
        path=Path(cfg['paths']['references'])/f'case{case}.pt';path.parent.mkdir(exist_ok=True)
        ids=split_indices(dd.canonical_case(f'case{case}'))['train'][:100].tolist()
        torch.save(dict(graphs=[graph]*100,ids=ids),path)
    return cfg


def test_matrix_identity_is_independent_of_scheduling_and_shares_initializations(spec):
    first=study.expand(spec,{'runtime.py':'one'})
    assert len(first['experiments'])==216
    assert len({r['dataset_id'] for r in first['experiments']})==216
    spec['execution'].update(nodes=64,layout='separate',staging=False,copper=False)
    assert first['experiments']==study.expand(spec,{'runtime.py':'one'})['experiments']
    groups={}
    for row in first['experiments']:
        c=row['config'];groups.setdefault((c['case'],c['generation_protocol']),set()).add(c['initial_cache'])
        assert c['synthetic_count']==100 and c['batches']==10
    assert all(len(paths)==1 for paths in groups.values())
    changed=study.expand(spec,{'runtime.py':'two'})
    assert not {r['dataset_id'] for r in changed['experiments']} & {r['dataset_id'] for r in first['experiments']}
    spec['matrix']['seeds']=[43]
    changed=study.expand(spec,{'runtime.py':'one'})
    assert changed['experiments'][0]['config']['training_model_seed'] != first['experiments'][0]['config']['training_model_seed']
    assert changed['experiments'][0]['config']['initial_cache'] != first['experiments'][0]['config']['initial_cache']


def test_strict_yaml_and_unknown_method_settings(spec,tmp_path):
    spec['shared']['typo_rate']=.2
    with pytest.raises(ValueError,match='Unknown fields'): study.expand(spec,{})
    source=yaml.safe_load(study.DEFAULT.read_text());source['mystery']=True
    path=tmp_path/'bad.yaml';path.write_text(yaml.safe_dump(source))
    with pytest.raises(ValueError,match='Unknown fields'): study.load(path)


def test_condition_schedule_overrides_have_distinct_identity(spec):
    spec['matrix'].update(cases=[30], protocols=['conservative'], target_modes=['learned'])
    spec['methods'] = {'DD': spec['methods']['DD']}
    spec['methods']['DD']['models'] = ['GIN']
    spec['extra_conditions'] = [dict(spec['matrix'], methods=['DD'],
        settings=dict(passes=30, evaluation_epochs=30))]
    rows = study.expand(spec, {})['experiments']
    assert len(rows) == 2
    assert {r['config']['passes'] for r in rows} == {10, 30}
    assert len({r['config']['initial_cache'] for r in rows}) == 1
    spec['extra_conditions'][0]['settings']['misspelled'] = 1
    with pytest.raises(ValueError, match='Unknown fields'): study.expand(spec, {})


def test_submit_256_and_skip_completed_on_resubmission(spec, monkeypatch):
    from types import SimpleNamespace
    manifest = study.expand(spec, {})
    root = Path(manifest['root']); root.mkdir(parents=True)
    (root/'manifest.json').write_text(json.dumps(manifest))
    calls = []
    monkeypatch.setattr(study.subprocess, 'run', lambda cmd, **kw:
        (calls.append(cmd) or SimpleNamespace(stdout='256.pbs\n')))
    study.submit(root, nodes=256)
    assert f'select={len(manifest["experiments"])}' in calls[0]
    for row in manifest['experiments'][:-1]:
        folder = Path(row['output']); folder.mkdir(parents=True)
        (folder/'artifact.json').write_text('{"complete":true}')
    study.submit(root, nodes=256)
    assert 'select=1' in calls[-1]
    with pytest.raises(ValueError, match='1 to 256'): study.submit(root, nodes=257)


def test_prepare_freezes_source_and_reference_identity(spec,tmp_path,monkeypatch):
    spec['matrix'].update(cases=[30],protocols=['conservative'],target_modes=['solver'])
    spec['methods']={'DCGM':spec['methods']['DCGM']}
    spec['methods']['DCGM']['models']=['GIN']
    monkeypatch.setattr(study,'source_files',lambda: {})
    path=tmp_path/'study.yaml';path.write_text(yaml.safe_dump({k:v for k,v in spec.items() if k!='profile'}))
    result=study.prepare(spec,path);root=Path(result['root'])
    assert (root/'catalog.csv').exists() and (root/'shared/references/case30.pt').exists()
    assert study.prepare(spec,path)==result
    spec['shared']['strength']=.2
    with pytest.raises(ValueError,match='changed'): study.prepare(spec,path)


def test_separate_submissions_keep_the_same_dataset_ids(spec,tmp_path,monkeypatch):
    from types import SimpleNamespace
    manifest=study.expand(spec,{})
    root=Path(manifest['root']);root.mkdir(parents=True)
    (root/'manifest.json').write_text(json.dumps(manifest))
    calls=[]
    def qsub(command,**kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=f'{len(calls)}.pbs\n')
    monkeypatch.setattr(study.subprocess,'run',qsub)
    study.submit(root,layout='separate',nodes=1)
    assert len(calls)==3 and all('select=1' in command for command in calls)
    selected=[]
    for method in ('DD','DCGM','HMDC'):
        paths=list((root/'manifests').glob(f'{method}-*.json'))
        assert len(paths)==1
        rows=json.loads(paths[0].read_text())['experiments']
        assert all(r['method']==method for r in rows)
        selected.extend(rows)
    assert sorted(r['dataset_id'] for r in selected)==sorted(r['dataset_id'] for r in manifest['experiments'])


def test_retry_without_copper_preserves_frozen_science_and_original_manifest(spec, monkeypatch):
    from types import SimpleNamespace
    manifest = study.expand(spec, {'runtime.py': 'frozen'})
    root = Path(manifest['root']); root.mkdir(parents=True)
    original = json.dumps(manifest)
    (root/'manifest.json').write_text(original)
    calls = []
    def qsub(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout='123.pbs\n')
    monkeypatch.setattr(study.subprocess, 'run', qsub)
    study.submit(root, nodes=32, copper=False)
    record = json.loads(next((root/'jobs').glob('*.json')).read_text())
    retry = json.loads(Path(record['manifest']).read_text())
    assert (root/'manifest.json').read_text() == original
    assert retry['source_hashes'] == manifest['source_hashes']
    assert retry['experiments'] == manifest['experiments']
    assert retry['specification']['execution']['copper'] is False
    assert retry['specification']['execution']['staging'] is True
    assert record['execution']['copper'] is False and record['nodes'] == 32
    assert 'STUDY_COPPER=0' in calls[0][calls[0].index('-v')+1]
    assert calls[0][-1] == str(root/'source/distillation/experiments/condensation.pbs')


def test_discovery_pilot_counts_holdout_checkpoints_and_source_retention(spec, graph, tmp_path):
    from collections import Counter
    from distillation.test.test_dd_pretrained import checkpoint
    from distillation.experiments.staging import assigned
    cfg = study.load(study.DEFAULT.with_name('discovery_pilot.yaml'), 'pilot')
    cfg['paths'] = spec['paths']
    sources = {}
    for kind in ('GIN', 'GAT', 'HEAT'):
        path = tmp_path/f'{kind}.pt'
        checkpoint(graph, path, kind)
        sources[kind] = str(path)
    cfg['initialization']['T1_case30'] = dict(mode='pretrained', checkpoints=sources)
    rows = study.expand(cfg, {'runtime': 'test'})['experiments']
    assert Counter(r['method'] for r in rows) == {'DD': 13, 'DCGM': 13, 'HMDC': 4, 'RAMBO': 18}
    assert len({r['config']['generation_id'] for r in rows if r['method'] == 'RAMBO'}) == 4
    assignments = [assigned(rows, node, 32) for node in range(32)]
    groups = [len({r['config'].get('generation_id', r['dataset_id']) for r in group}) for group in assignments]
    assert sorted(groups) == [1]*30 + [2]*2
    for rows_on_node, count in zip(assignments, groups):
        if count > 1:
            assert {r['config']['generation_protocol'] for r in rows_on_node if r['method']=='RAMBO'} == {'uniform'}
    for row in rows:
        c = row['config']
        assert row['verify_run'] and not row['smoke']
        assert c['synthetic_count'] == 100 and c['batches']*c['evaluation_epochs'] == 100
        if row['method'] != 'RAMBO':
            assert c['target_mode'] == 'learned' and c['training_seconds'] == 800 and c['max_updates'] == 1000
        if row['method'] == 'HMDC':
            assert c['pair'] == ['GIN', 'HEAT'] and set(c['models']) == {'GIN', 'GAT', 'HEAT'}
            if c['initialization'] == 'pretrained': assert c['pretrained_checkpoints'] == sources
        if c['initialization'] == 'pretrained':
            assert c['case'] == 57 and c['retention_data'] == {'pglib_opf_case30_ieee': study.file_hash(
                Path(c['data_root'])/'pglib_opf_case30_ieee'/'manifest.json')}
        else:
            assert 'retention_data' not in c
    changed = deepcopy(cfg)
    changed['evaluation']['retention_samples'] = 128
    altered = study.expand(changed, {'runtime': 'test'})['experiments']
    assert {r['config']['generation_id'] for r in altered if r['method'] == 'RAMBO'} == {
        r['config']['generation_id'] for r in rows if r['method'] == 'RAMBO'}


@pytest.mark.parametrize('budget', ['seconds', 'updates'])
def test_pilot_verification_resumes_to_either_budget_then_replays(tmp_path, monkeypatch, budget):
    from types import SimpleNamespace
    from mpi4py import MPI
    from distillation.experiments import smoke
    model = torch.nn.Linear(1, 1)
    cfg = dict(model='GIN', synthetic_count=1, batches=1, evaluation_epochs=2, qualify=False,
        case=30, target_mode='learned', max_updates=1000, training_seconds=800)
    calls = []
    def run(config, folder, comm, device, **kwargs):
        calls.append(kwargs)
        (folder/'latest.pt').touch()
        state = dict(iteration=1, status='RESUMABLE', resume_count=1, accepted_updates=1, active_seconds=2)
        if len(calls) == 2:
            state.update(status='COMPLETE', resume_count=2, iteration=1000 if budget=='updates' else 3,
                         active_seconds=801 if budget=='seconds' else 50)
        return state
    monkeypatch.setattr(smoke, 'publish', lambda *args: None)
    monkeypatch.setattr(smoke, 'load_dataset', lambda *args: dict(graphs=[None], export={
        'model_checkpoint': {'trained_state': model.state_dict()}}))
    monkeypatch.setattr(smoke, 'train_from_artifact', lambda *args, **kwargs: (model, {'student_updates': 2}))
    result = smoke.run(SimpleNamespace(run=run), dict(output=str(tmp_path), config=cfg, method='DCGM'),
                       MPI.COMM_SELF, 'cpu', 3300)
    assert result['status']=='COMPLETE' and calls[0]['stop_after_updates']==1
    assert 'stop_after_updates' not in calls[1] and calls[1]['allocation_seconds'] <= 3300
    assert json.loads((tmp_path/'smoke.json').read_text())['status']=='PASS'


@pytest.mark.parametrize('change,gain,error', [
    (5e-7,1.,None), (.01,1.,'parameter drift'), (5e-7,1e6,'functional drift'),
    (float('nan'),1.,'Nonfinite replay')])
def test_numerical_replay_checks_weights_and_predictions(graph, change, gain, error):
    from distillation.experiments.smoke import check_replay
    from distillation.synthetic.learning import benchmark_graph
    class Student(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.,0.],dtype=torch.float64))
        def forward(self,batch):
            value = self.weight[0]+gain*self.weight[1]
            return {k:value.expand(batch[k].num_nodes,2) for k in ('bus','generator')}
    model = Student()
    expected = {k:v.detach().clone() for k,v in model.state_dict().items()}
    with torch.no_grad(): model.weight[1] = change
    graphs = [benchmark_graph(graph,labels=True)]*9  # Include an uneven final batch.
    if error:
        with pytest.raises((AssertionError,FloatingPointError),match=error):
            check_replay(model,expected,graphs)
    else:
        report = check_replay(model,expected,graphs)
        assert not report['strict_state_match'] and report['checked_graphs']==9
        assert max(report['prediction_relative_l2'].values())<.005
        assert report['teaching_loss_relative_change']<.001
    torch.testing.assert_close(model.weight[1],torch.tensor(change,dtype=torch.float64),equal_nan=True)


@pytest.mark.parametrize('method',['DD','DCGM','HMDC'])
@pytest.mark.parametrize('mode',['solver','frozen','learned'])
def test_artifacts_preserve_targets_and_replay(method,mode,graph,tmp_path):
    cfg=(dd.default_config(30,'GIN') if method=='DD' else dc.default_config(30,'GIN') if method=='DCGM'
         else hm.default_config(30,['GIN','GAT']))
    cfg.update(target_mode=mode,synthetic_count=1,batches=1,batch_size=1,passes=2,evaluation_epochs=2,
               model_options=model_options('GIN',width=8),models={k:model_options(k,width=8) for k in ['GIN','GAT']},
               inner_rate=1e-7,generation_protocol='conservative',initialization='fixed',model_seed=71)
    cfg['artifact_identity']=dict(dataset_id=f'{method}-test',method=method,source_sha256='test-source')
    data=dd.Schedule([graph],cfg) if method=='DD' else SyntheticInputs([graph],target_mode=mode)
    if mode=='learned':
        with torch.no_grad(): data.target_offsets[0]['generator'].fill_(.2)
    students={}
    for kind in cfg.get('pair',['GIN']):
        local=cfg|dict(model=kind,model_options=model_options(kind,width=8))
        model=model_for(graph,local,71,'cpu');initial=cpu_state(model)
        if method=='DD':
            theta,buffers=dd.inner(model,data,create_graph=False);model.load_state_dict(theta|buffers)
        else: train_student(model,torch.optim.SGD(model.parameters(),lr=1e-7),data,epochs=2,batch_size=1,seed=71)
        students[kind]=dict(model=kind,model_options=local['model_options'],seed=71,training_seed=71,
                            initial_state=initial,trained_state=cpu_state(model))
    state=dict(best_iteration=0,status='COMPLETE',student_rate=1e-7,rates={k:1e-7 for k in students})
    checks=[dict(success=True)]
    if method=='DD': dd.export(tmp_path,data,cfg,state,[graph],checks,students['GIN'],name='best.pt')
    elif method=='DCGM': dc.export(tmp_path,'best.pt',data,cfg,state,[graph],checks,students['GIN'])
    else: hm.export(tmp_path/'best.pt',data,cfg,state,[graph],checks,students)
    (tmp_path/'config.json').write_text(json.dumps(cfg));(tmp_path/'status.json').write_text(json.dumps(state))
    row=dict(output=str(tmp_path),config=cfg,method=method)
    metadata=artifacts.publish(row)
    loaded=artifacts.load_dataset(tmp_path)
    teaching=data.batch(0)[1] if method=='DD' else data.batch()[1]
    torch.testing.assert_close(loaded['graphs'][0]['generator'].y,teaching['generator'],rtol=0,atol=0)
    for kind,student in students.items():
        model,provenance=artifacts.train_from_artifact(tmp_path,model=kind)
        torch.testing.assert_close(cpu_state(model),student['trained_state'],rtol=0,atol=0)
        assert provenance['student_updates']==2 and provenance['dataset_sha256']==metadata['sha256']
    scratch,provenance=artifacts.train_from_artifact(tmp_path,model='GIN',initialization='scratch',seed=123)
    assert provenance['initialization']=='scratch' and provenance['seed']==123
    if method=='DCGM' and mode=='learned':
        checkpoint=tmp_path/'pretrained.pt'
        torch.save(dict(specification=dict(format='lumina-t1-v1',split_seed=42,seed=901,model='GIN',
            model_options=cfg['model_options'],case='pglib_opf_case57_ieee'),model=students['GIN']['initial_state']),checkpoint)
        _,provenance=artifacts.train_from_artifact(tmp_path,initialization='checkpoint',checkpoint=checkpoint)
        assert provenance['source_checkpoint']['specification']['case']=='pglib_opf_case57_ieee'
        assert provenance['seed']==901
        with pytest.raises(ValueError,match='recorded'): artifacts.train_from_artifact(tmp_path,checkpoint=checkpoint)
    with (tmp_path/'best.pt').open('ab') as stream: stream.write(b'changed')
    with pytest.raises(ValueError,match='checksum'): artifacts.load_dataset(tmp_path)
