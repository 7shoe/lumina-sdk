"""Protocol tests: split integrity, initialization, all students, and interrupted evaluation."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch_geometric.data import InMemoryDataset

from experiments.lumina_bench.tests.test_protocol import graph
from distillation.downstream_experiments import dataset_path
from distillation.downstream_experiments.scripts import campaign, data, train
from distillation.training.T1_single_topo.common import digest, file_hash, read_json, write_json
from distillation.training.T1_single_topo.data import CaseDataset, processor_identity
from distillation.training.T1_single_topo.tests.test_t1 import assert_same


@pytest.fixture
def config(tmp_path):
    config = campaign.load_config()
    config.update(name='test',cases=[14],methods=['RND777'],students=['GCN'],seeds=[42],
                  creation_initializations=['random'],transfer_pairs=[])
    config['paths'] = {key:str(tmp_path/key) for key in ('models','corpora','data','t1')}
    config['families'] = dict(individual=True,fixed=False,mixed_sizes=[],adaptation_sizes=[],joint_size=0)
    config['training'].update(max_updates=3,batch_size=4,eval_batch_size=8,validation_updates=[1,3],log_updates=1)
    for options in config['model_options'].values():
        options['hidden_dim' if 'hidden_dim' in options else 'hidden_channels']=8
        options['num_layers']=2
    for case in (14,30):
        folder=Path(config['paths']['data'])/data.case_name(case)
        folder.mkdir(parents=True)
        graphs=[]
        for i in range(160):
            g=graph();g.sample_id=torch.tensor([i]);g['generator'].y=g['generator'].y.clone()+i/1000
            graphs.append(g)
        path=folder/'group_0.pt'
        torch.save(InMemoryDataset.collate(graphs),path)
        stat=path.stat()
        write_json(folder/'manifest.json',dict(format='lumina-t1-data-v1',case=data.case_name(case),
            count=160,group_size=160,order='numeric example_ID',groups=[dict(group=0,count=160,
            bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,sha256=file_hash(path),processor=processor_identity())]))
    return config


def prepared(config):
    manifest=read_json(campaign.prepare(config))
    return manifest,manifest['runs'][0]


def load_checkpoint(manifest,run,name='final.pt'):
    return torch.load(train.folder_for(manifest,run)/name,weights_only=False,map_location='cpu')


def test_feeder_is_independent_of_working_directory(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    rows=dataset_path.find_datasets(method='HMDC',model='GAT+RGAT',case=30,size=100)
    assert rows and all(r['reference_model']=='GAT+RGAT' for r in rows)
    assert dataset_path.dataset_record(rows[0]['dataset_id'])==rows[0]
    with pytest.raises(ValueError,match='No ready'):
        dataset_path.dataset_record('missing')


@pytest.mark.parametrize('creators,expected',[(8,[50]*8),(5,[80]*5),(4,[100]*4),(7,[57]*6+[58])])
def test_creator_quotas(creators,expected):
    values=data.quotas(400,[str(i) for i in range(creators)],777)
    assert sorted(values.values())==expected
    assert values==data.quotas(400,[str(i) for i in range(creators)],777)


def test_random_control_is_nested_and_excludes_holdouts(config):
    small=data.make_corpus((),12,data_root=config['paths']['data'],random_cases=(14,))
    big=data.make_corpus((),100,data_root=config['paths']['data'],random_cases=(14,))
    ds=CaseDataset(config['paths']['data'],data.case_name(14))
    ids=big['members'][0]['indices']
    assert small['members'][0]['indices']==ids[:12]
    assert len(ids)==len(set(ids))==100
    assert set(ids)<=set(ds.splits()['train'])
    expected=np.random.default_rng(777).permutation(ds.splits()['train'])[:100].tolist()
    assert ids==expected
    loaded,meta=data.materialize(big,config['paths']['corpora'],config['paths']['data'])
    for i,g in zip(ids,loaded['graphs']):
        assert_same(g['generator'].y,ds[i]['generator'].y)
    assert meta['count']==100
    broken=deepcopy(big);broken['members'][0]['indices'][0]=int(ds.splits()['test'][0])
    identity={k:v for k,v in broken.items() if k!='corpus_id'}
    broken['corpus_id']='CORPUS-'+digest(identity)[:16]
    with pytest.raises(ValueError,match='holdout'):
        data.materialize(broken,config['paths']['corpora'],config['paths']['data'])


def test_two_case_corpus_has_equal_counts(config):
    recipe=data.make_corpus((),100,data_root=config['paths']['data'],random_cases=(14,30))
    assert [(m['case'],m['count']) for m in recipe['members']]==[(14,50),(30,50)]
    payload,_=data.materialize(recipe,config['paths']['corpora'],config['paths']['data'])
    assert payload['cases'].count(14)==payload['cases'].count(30)==50


@pytest.mark.parametrize('model',['GCN','GAT','GIN','Transformer','HGT','RGAT','HEAT','HeteroGNN'])
def test_all_models_resume_exactly_and_evaluate_full_tests(config,model):
    config['students']=[model]
    manifest,run=prepared(config)
    assert train.train_run(manifest,run)=='complete'
    expected=load_checkpoint(manifest,run)
    resumed=deepcopy(manifest);resumed['paths']['models']+='-resumed'
    assert train.train_run(resumed,run,session_steps=2)=='paused'
    assert train.train_run(resumed,run)=='complete'
    actual=load_checkpoint(resumed,run)
    for field in ('model','optimizer','state','rng'):
        assert_same(expected[field],actual[field])
    folder=train.folder_for(resumed,run)
    report=read_json(folder/'test-case14.json')
    assert report['metrics']['n_samples']==16
    assert report['trained_steps']==3
    assert train.completed(folder,run['specification'])


def test_interrupted_validation_and_test_resume_without_leakage(config,monkeypatch):
    manifest,run=prepared(config)
    evaluate=train.evaluate
    observed=[]
    def interrupt(*args):
        identity=args[-2];observed.append(identity['split'])
        # Complete one evaluation batch, then checkpoint its cursor.
        checks=iter([False,True])
        return evaluate(*args[:-1],lambda:next(checks))
    with monkeypatch.context() as patch:
        patch.setattr(train,'evaluate',interrupt)
        assert train.train_run(manifest,run)=='paused'
    assert observed==['val']
    assert load_checkpoint(manifest,run,'last.pt')['state']['phase']=='validate'
    def interrupt_test(*args):
        return interrupt(*args) if args[-2]['split']=='test' else evaluate(*args)
    with monkeypatch.context() as patch:
        patch.setattr(train,'evaluate',interrupt_test)
        assert train.train_run(manifest,run)=='paused'
    final=load_checkpoint(manifest,run)
    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.AdamW,'step',lambda *a,**kw:pytest.fail('Training resumed during final test'))
        assert train.train_run(manifest,run)=='complete'
    assert_same(final['model'],load_checkpoint(manifest,run)['model'])


def test_exact_state_and_weight_only_pretraining(config,tmp_path):
    manifest,run=prepared(config)
    spec=deepcopy(run['specification']);sample=graph()
    spec['initialization']['seed']=1_000_000
    initial=train.initial_model(spec,sample,torch.device('cpu'))
    weights={k:v.detach().clone() for k,v in initial.state_dict().items()}
    changed={k:v+1 if v.is_floating_point() else v for k,v in weights.items()}
    path=tmp_path/'fixed.pt'
    torch.save(dict(config=dict(initialization='fixed'),model_checkpoint=dict(initial_state=weights,
        trained_state=changed,seed=1_000_000,model='GCN',model_options=spec['initialization']['model_options'])),path)
    spec['initialization'].update(kind='recorded',path=str(path),sha256=file_hash(path))
    actual=train.initial_model(spec,sample,torch.device('cpu'))
    assert_same(weights,actual.state_dict())
    spec['initialization']['sha256']='wrong'
    with pytest.raises(ValueError,match='checksum'):
        train.initial_model(spec,sample,torch.device('cpu'))
    path=tmp_path/'pretrained.pt'
    options=spec['initialization']['model_options']
    identity=data.data_identity(config['paths']['data'],14)['identity']
    torch.save(dict(specification=dict(format='lumina-t1-v1',model='GCN',case=data.case_name(14),
        split_seed=42,model_options=options,data_identity=identity),model=weights,optimizer={'must_not_load':True}),path)
    spec['initialization'].update(kind='pretrained',path=str(path),sha256=file_hash(path),source_case=14,data_identity=identity)
    actual=train.initial_model(spec,sample,torch.device('cpu'))
    assert_same(weights,actual.state_dict())
    spec['initialization']['source_case']=30
    with pytest.raises(ValueError,match='Incompatible'):
        train.initial_model(spec,sample,torch.device('cpu'))


def test_changed_identity_and_concurrent_run_rejected(config):
    manifest,run=prepared(config)
    with train.claim(train.folder_for(manifest,run)/'run.lock'):
        assert train.train_run(manifest,run)=='busy'
    run['specification']['seed']+=1
    with pytest.raises(ValueError,match='identity'):
        train.train_run(manifest,run)


def test_controls_are_not_duplicated_across_creation_regimes(config):
    config['creation_initializations']=['random','pretrained']
    config['target_modes']=['learned','solver']
    manifest,_=prepared(config)
    assert len(manifest['runs'])==1
    assert len(manifest['corpora'])==1


def test_campaign_rejects_ambiguous_creator(config,tmp_path):
    rows=[]
    for i in range(2):
        rows.append(dict(dataset_id=f'DD-{i}',method='DD',case=14,count=100,models=['GCN'],
            generation='conservative',target_mode='learned',initialization='random',initialization_name='random',
            seed=42,data_path='unused',artifact_path='unused',sha256='unused'))
    path=tmp_path/'catalog.yaml';path.write_text(yaml.safe_dump(dict(datasets=rows)))
    config.update(catalog=str(path),methods=['DD'])
    with pytest.raises(ValueError,match='Ambiguous'):
        campaign.prepare(config)


def test_one_failed_student_does_not_cancel_next(config,monkeypatch):
    config['seeds']=[42,43]
    manifest,run=prepared(config)
    path=Path(manifest['paths']['models'])/'manifest.json'
    calls=[]
    def fake_child(command,log,stop):
        identifier=command[command.index('--run-id')+1];calls.append(identifier)
        return 1 if len(calls)==1 else 0
    monkeypatch.setattr(campaign,'run_child',fake_child)
    bundle=dict(run_ids=[r['run_id'] for r in manifest['runs']],devices=[0],output=str(path.parent))
    result=campaign.worker(path,bundle,0,'cpu',60)
    assert len(calls)==2
    assert result[calls[0]]=='failed' and result[calls[1]]=='paused'


def test_pretrained_pipeline_uses_both_full_tests_and_fresh_optimizer(config):
    base,run=prepared(config)
    model=train.initial_model(run['specification'],graph(),torch.device('cpu'))
    path=Path(config['paths']['t1'])/'checkpoints/GCN'/data.case_name(14)/'seed42/best.pt'
    path.parent.mkdir(parents=True)
    spec=dict(format='lumina-t1-v1',model='GCN',case=data.case_name(14),seed=42,split_seed=42,
        model_options=config['model_options']['GCN'],data_identity=data.data_identity(config['paths']['data'],14)['identity'])
    torch.save(dict(specification=spec,model=model.state_dict(),optimizer={'invalid_pretraining_optimizer':True}),path)
    config['name']='adapt';config['cases']=[30];config['families']['adaptation_sizes']=[100]
    manifest,_=prepared(config)
    run=next(r for r in manifest['runs'] if r['specification']['initialization']['kind']=='pretrained')
    assert train.train_run(manifest,run)=='complete'
    final=load_checkpoint(manifest,run)
    assert all(int(v['step'])==3 for v in final['optimizer']['state'].values())
    for case in (14,30):
        folder=train.folder_for(manifest,run)
        assert read_json(folder/f'test-case{case}.json')['metrics']['n_samples']==16
        assert read_json(folder/f'initial-test-case{case}.json')['metrics']['n_samples']==16


def test_fp64_recorded_profile_keeps_identical_initial_values(config,tmp_path):
    manifest,run=prepared(config)
    spec=deepcopy(run['specification']);spec['initialization']['precision']='float64'
    initial=train.initial_model(spec,graph(),torch.device('cpu'))
    path=tmp_path/'double.pt'
    torch.save(dict(config=dict(initialization='fixed'),model_checkpoint=dict(model='GCN',
        seed=42,model_options=config['model_options']['GCN'],initial_state=initial.state_dict(),
        trained_state={})),path)
    spec['initialization'].update(kind='recorded',path=str(path),sha256=file_hash(path))
    replay=train.initial_model(spec,graph(),torch.device('cpu'))
    assert_same(initial.state_dict(),replay.state_dict())
    assert next(replay.parameters()).dtype==torch.float64
