"""Checks for the scientific comparisons and committed-report provenance in 5–9."""
from copy import deepcopy
import json

import pytest

from distillation.evaluation.downstream.utils.comparisons import load_comparison, seed_cell, difference
from distillation.evaluation.downstream.utils.comparison_tables import make_pages
from distillation.test.test_downstream_multicase import campaign, write, first_folder, rehash
from distillation.training.T1_single_topo.common import digest


def observations(values=(2.,4.,6.)):
    return [dict(seed=seed, corpus_id='one', model='GCN', case=30, run_id=str(seed), status='complete',
                 metrics={'sdk_mse':v, 'balance_l2':v*2}, before_metrics={'sdk_mse':1.})
            for seed,v in zip((42,43,44),values)]


def test_mean_and_paired_sign():
    a = seed_cell(observations(), [42,43,44])
    b = seed_cell(observations((3.,6.,9.)), [42,43,44])
    assert a['mean'] == 4.
    assert difference(a,b)['mean'] == -2.
    assert difference(a,b)['seed_values'] == {42:-1.,43:-2.,44:-3.}
    assert seed_cell(observations(), [42,43,44], view='delta')['mean'] == 3.


def test_partial_seed_coverage_is_missing():
    complete = seed_cell(observations(), [42,43,44])
    partial = seed_cell(observations()[:-1], [42,43,44])
    assert partial['mean'] is None
    assert difference(complete,partial)['mean'] is None


def test_duplicate_seeds_and_mixed_corpora_rejected():
    rows = observations()
    with pytest.raises(ValueError, match='Duplicate'):
        seed_cell(rows + [deepcopy(rows[0])], [42,43,44])
    rows[0]['corpus_id'] = 'different'
    with pytest.raises(ValueError, match='distinct datasets'):
        seed_cell(rows, [42,43,44])


def test_differences_must_match_seeds_and_metrics():
    a = seed_cell(observations(), [42,43,44])
    b = deepcopy(a)
    b['seed_values'][45] = b['seed_values'].pop(44)
    with pytest.raises(ValueError, match='Unpaired'):
        difference(a,b)
    b = seed_cell(observations(), [42,43,44], metric='balance_l2')
    with pytest.raises(ValueError, match='different metrics'):
        difference(a,b)


def test_replay_reader_retains_directions_and_before_metrics(campaign):
    report = load_comparison('replay', campaign)
    assert len(report['rows']) == 36
    assert {(r['source'],r['target']) for r in report['rows']} == {(14,30),(30,14)}
    assert all(r['before_metrics'] and r['arm']=='replay' for r in report['rows'])


def test_corruption_is_not_silently_missing(campaign):
    folder = first_folder(campaign,'adaptation-replay')
    path = folder/'test-case14.json'
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError, match='Checksum mismatch'):
        load_comparison('replay', campaign)


def test_wrong_holdout_identity_with_valid_hash_rejected(campaign):
    folder = first_folder(campaign,'adaptation-replay')
    name = 'test-case14.json'
    result = json.loads((folder/name).read_text())
    result['identity']['indices_sha256'] = 'different-test'
    rehash(folder,name,result)
    with pytest.raises(ValueError, match='Wrong test identity'):
        load_comparison('replay', campaign)


def test_target_only_uses_target_corpus_but_both_test_cases(campaign):
    manifest = json.loads(campaign.read_text())
    runs = [r for r in manifest['runs'] if 'adaptation-replay' in r['families']]
    for run in runs:
        spec = deepcopy(run['specification'])
        source = spec['initialization']['source_case']
        target = next(c for c in spec['evaluation_cases'] if c != source)
        corpus = deepcopy(manifest['corpora'][spec['corpus_id']])
        corpus['members'] = [m for m in corpus['members'] if m['case']==target]
        for m in corpus['members']:
            m['count'] *= 2
            # Real controls use canonical train-only IDs from the original recipe.
            if m['kind'] == 'real':
                from distillation.data.split import split_indices
                from distillation.training.T1_single_topo.common import canonical_case
                m['indices'] = split_indices(canonical_case(f'case{target}'),1000,42)['train'][:400].tolist()
            else:
                m['indices'] = list(range(m['count']))
        corpus['cases'] = [target]
        corpus['data'] = {str(target):corpus['data'][str(target)]}
        corpus.pop('corpus_id')
        corpus['corpus_id'] = 'CORPUS-'+digest(corpus)[:16]
        manifest['corpora'][corpus['corpus_id']] = corpus
        spec['corpus_id'] = corpus['corpus_id']
        new = dict(run_id='RUN-'+digest(spec)[:20], specification=spec, families=['adaptation-M400'])
        manifest['runs'].append(new)
        origin = campaign.parent/'runs'/'GCN'/run['run_id']
        folder = campaign.parent/'runs'/'GCN'/new['run_id']
        marker = json.loads((origin/'complete.json').read_text())
        files = {}
        for name,checksum in marker['files'].items():
            if not name.endswith('.json'):
                files[name] = checksum
                continue
            value = json.loads((origin/name).read_text())
            if name == 'corpus.json':
                value.update(recipe=corpus, cases={str(target):400})
            elif name.startswith('test-case'):
                value['identity']['specification'] = digest(spec)
            files[name] = write(folder/name,value)
        write(folder/'config.json', spec)
        write(folder/'complete.json',dict(specification=digest(spec),files=files))
    write(campaign,manifest)
    report = load_comparison('replay', campaign)
    target_rows = [r for r in report['rows'] if r['arm']=='target']
    assert len(target_rows) == 36
    assert all(r['train_cases']==[r['target']] for r in target_rows)
    assert len(make_pages(report)) == 4


def test_missing_marker_leaves_replay_cells_unavailable(campaign):
    (first_folder(campaign,'adaptation-replay')/'complete.json').unlink()
    report = load_comparison('replay',campaign)
    pages = make_pages(report)
    assert sum(r['status']=='incomplete' for r in report['rows']) == 2
    cells = [cell for p in pages for t in p['tables'] for row in t['cells'] for cell in row]
    assert any(cell['status']=='incomplete' and cell['mean'] is None for cell in cells)
