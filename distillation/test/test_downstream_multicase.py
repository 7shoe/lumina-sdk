"""Scientific grouping, paired changes, and committed identities for notebooks 3/4."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from distillation.data.split import split_indices
from distillation.evaluation.downstream.utils.multicase import BEFORE, load_multicase, select_runs, summarize_multicase
from distillation.evaluation.downstream.utils.multicase_tables import make_tables
from distillation.training.T1_single_topo.common import canonical_case, digest


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(value):
    return dict(mse=value, sdk_mse=value, opf_solution_error=value, va_mse=2*value,
                vm_mse=0., pg_mse=0., qg_mse=0., balance_l2=0., thermal_l2=0., bounds_l2=0.,
                violation=0., constraint_violation=0., reference_cost=1., cost_difference_pct=0., n_samples=100, score=value)


@pytest.fixture
def campaign(tmp_path):
    cases, data, indices, splits = [14, 30], {}, {}, {}
    for case in cases:
        path = tmp_path / 'data' / str(case) / 'manifest.json'
        release = dict(count=1000, case=case)
        data[str(case)] = dict(manifest=str(path), sha256=write(path, release), count=1000, identity=digest(release))
        indices[case] = split_indices(canonical_case(f'case{case}'), 1000, 42)
        splits[str(case)] = {k:dict(count=len(v), sha256=digest(v.tolist())) for k,v in indices[case].items()}
    options = dict(hidden_dim=128, num_layers=3, dropout=0.1)
    config = dict(cases=cases, students=['GCN'], seeds=[42,43,44], split_seed=42, subset_seed=777,
        pretrained_seed=42, model_options={'GCN':options}, optimizer={'lr':.001},
        training=dict(max_updates=100, batch_size=25, validation_updates=[10,100]))
    manifest = dict(format='lumina-downstream-campaign-v1', config=config, sources={},
        paths=dict(models=str(tmp_path), corpora=str(tmp_path/'corpora'), t1=str(tmp_path/'t1')), corpora={}, runs=[])
    corpora = []
    for creation in ('random', 'pretrained14', 'pretrained30', 'control'):
        members = []
        for case in cases:
            if creation == 'control':
                members.append(dict(kind='real', case=case, creator='none', count=200,
                                    indices=indices[case]['train'][:200].tolist()))
            else:
                source = int(creation.removeprefix('pretrained')) if creation.startswith('pretrained') else None
                pretrained = source is not None and case != source
                for creator in ('GCN', 'GAT'):
                    members.append(dict(kind='artifact', case=case, creator=creator, count=100, indices=list(range(100)),
                        generation='conservative', target_mode='learned', initialization='pretrained' if pretrained else 'random',
                        initialization_name=f'T1_case{source}' if pretrained else 'random'))
        corpus = dict(method='RND777' if creation == 'control' else 'DD', count=400, cases=cases,
                      members=members, data=data, split_seed=42, sample_seed=777)
        corpus['corpus_id'] = 'CORPUS-' + digest(corpus)[:16]
        manifest['corpora'][corpus['corpus_id']] = corpus
        corpora.append((creation, corpus))
    for family, source in [('joint-fresh',None), ('adaptation-replay',14), ('adaptation-replay',30)]:
        for creation, corpus in corpora:
            if source and creation not in ('random','control',f'pretrained{source}'):
                continue
            for seed in config['seeds']:
                init = dict(kind='fresh', seed=seed, model_options=options, precision='float32')
                if source:
                    init.update(kind='pretrained', seed=42, source_case=source, sha256=f'T1-{source}',
                        path=str(tmp_path/'t1'/'checkpoints'/'GCN'/canonical_case(f'case{source}')/'seed42'/'best.pt'),
                        data_identity=data[str(source)]['identity'])
                spec = dict(corpus_id=corpus['corpus_id'], model='GCN', seed=seed, initialization=init,
                    evaluation_cases=cases, split_seed=42, data=data, training=config['training'], optimizer=config['optimizer'],
                    sources=digest({}), selection='equal mean of per-case validation sdk_mse + violation')
                run = dict(run_id='RUN-'+digest(spec)[:20], specification=spec, families=[family])
                manifest['runs'].append(run)
                folder = tmp_path/'runs'/'GCN'/run['run_id']
                write(folder/'config.json', spec)
                files = {'best.pt':'best-sha', 'initial.pt':'initial-sha'}
                files['corpus.json'] = write(folder/'corpus.json', dict(recipe=corpus, count=400,
                    cases={'14':200,'30':200}, sha256='dataset-sha', unique_inputs=400, unique_teaching_pairs=400))
                files['splits.json'] = write(folder/'splits.json', splits)
                for case in cases:
                    value = (2. if case == source else 4.) if source else case/100.
                    name = f'test-case{case}.json'
                    files[name] = write(folder/name, dict(model='GCN', case=case, seed=seed, metrics=metrics(value),
                        trained_steps=100, checkpoint_step=10, identity=dict(specification=digest(spec),
                        checkpoint_sha256='best-sha', case=case, split='test', indices_sha256=splits[str(case)]['test']['sha256'])))
                    if source:
                        name = f'initial-test-case{case}.json'
                        files[name] = write(folder/name, dict(identity=dict(initialization=init, case=case,
                            data=data[str(case)], sources=digest({}), split='test'), metrics=metrics(1. if case == source else 5.)))
                write(folder/'complete.json', dict(specification=digest(spec), files=files))
    path = tmp_path/'manifest.json'
    write(path, manifest)
    return path


def first_folder(path, family):
    manifest = json.loads(path.read_text())
    run = next(r for r in manifest['runs'] if family in r['families'])
    return path.parent/'runs'/'GCN'/run['run_id']


def rehash(folder, name, result):
    marker = json.loads((folder/'complete.json').read_text())
    marker['files'][name] = write(folder/name, result)
    write(folder/'complete.json', marker)


def test_joint_directions_separate_and_controls_reused(campaign):
    report = load_multicase('joint-fresh', campaign)
    assert len(report['rows']) == 24
    assert [p['regime'] for p in report['panels']] == ['random','pretrained:14:30','pretrained:30:14']
    summary = summarize_multicase(report)
    assert len({r['corpus_id'] for r in summary if r['column']=='RND777'}) == 1
    assert len({r['corpus_id'] for r in summary if r['column']=='DD'}) == 3
    assert len(make_tables(report, summary)) == 6


def test_replay_paired_changes_and_source_direction(campaign):
    report = load_multicase('adaptation-replay', campaign)
    summary = summarize_multicase(report)
    assert len(report['rows']) == 36
    for row in summary:
        if row['column'] == BEFORE:
            assert row['mean'] == (1. if row['case']==row['group'][0] else 5.)
            assert row['delta'] == 0.
        elif row['column'] in ('DD','RND777'):
            assert row['delta'] == (1. if row['case']==row['group'][0] else -1.)
    assert len(make_tables(report, summary)) == 16


def test_initial_checksum_corruption(campaign):
    folder = first_folder(campaign, 'adaptation-replay')
    path = folder/'initial-test-case14.json'
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError, match='Checksum mismatch'):
        load_multicase('adaptation-replay', campaign)


def test_wrong_initial_source_even_with_valid_checksum(campaign):
    folder = first_folder(campaign, 'adaptation-replay')
    name = 'initial-test-case14.json'
    result = json.loads((folder/name).read_text())
    result['identity']['initialization']['source_case'] = 30
    rehash(folder, name, result)
    with pytest.raises(ValueError, match='Wrong initial test identity'):
        load_multicase('adaptation-replay', campaign)


def test_two_cases_cannot_use_different_checkpoints(campaign):
    folder = first_folder(campaign, 'joint-fresh')
    name = 'test-case30.json'
    result = json.loads((folder/name).read_text())
    result['checkpoint_step'] = 100
    rehash(folder, name, result)
    with pytest.raises(ValueError, match='different checkpoints'):
        load_multicase('joint-fresh', campaign)


def test_corpus_must_have_200_examples_per_case(campaign):
    folder = first_folder(campaign, 'joint-fresh')
    result = json.loads((folder/'corpus.json').read_text())
    result['cases'] = {'14':199,'30':201}
    rehash(folder, 'corpus.json', result)
    with pytest.raises(ValueError, match='Wrong training corpus'):
        load_multicase('joint-fresh', campaign)


def test_missing_seed_is_not_averaged(campaign):
    (first_folder(campaign, 'joint-fresh')/'complete.json').unlink()
    summary = summarize_multicase(load_multicase('joint-fresh', campaign))
    partial = [r for r in summary if r['column']=='DD' and r['regime']=='random']
    assert all(r['mean'] is None and r['seeds']==2 and r['status']=='incomplete' for r in partial)


def test_duplicate_seed_is_rejected(campaign):
    report = load_multicase('joint-fresh', campaign)
    report['rows'].append(deepcopy(report['rows'][0]))
    with pytest.raises(ValueError, match='Duplicate/unexpected seeds'):
        summarize_multicase(report)


def test_replay_direction_must_match_synthesis(campaign):
    manifest = json.loads(campaign.read_text())
    run = next(r for r in manifest['runs'] if 'adaptation-replay' in r['families']
               and any(m.get('initialization')=='pretrained' for m in manifest['corpora'][r['specification']['corpus_id']]['members']))
    run['specification']['initialization']['source_case'] = 30
    with pytest.raises(ValueError, match='directions disagree'):
        select_runs(manifest, 'adaptation-replay')


def test_initial_baselines_must_agree_across_methods(campaign):
    folder = first_folder(campaign, 'adaptation-replay')
    name = 'initial-test-case14.json'
    result = json.loads((folder/name).read_text())
    result['metrics'] = metrics(8.)
    rehash(folder, name, result)
    with pytest.raises(ValueError, match='Inconsistent starting-model evaluations'):
        load_multicase('adaptation-replay', campaign)
