"""Guard the scientific grouping and committed test identities used by the notebook."""
from copy import deepcopy
import hashlib
import json
import re

import pytest

from distillation.data.split import split_indices
from distillation.evaluation.downstream.utils.paths import load_results, own_data_runs
from distillation.evaluation.downstream.utils.tables import render_table, summarize
from distillation.evaluation.downstream.utils.transfer import (
    METHODS, render_transfer_table, summarize_transfer, transfer_runs,
)
from distillation.training.T1_single_topo.common import canonical_case, digest


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def campaign(tmp_path):
    release = tmp_path / 'data' / 'manifest.json'
    sha = write(release, dict(count=1000))
    info = dict(manifest=str(release), count=1000, sha256=sha, identity=digest(dict(count=1000)))
    data = {'30': info}
    member = dict(kind='artifact', creator='GAT', case=30, count=100, indices=list(range(100)),
                  dataset_id='DD-source', data_path='/source/best.pt', sha256='source-sha',
                  initialization='random', generation='conservative', target_mode='learned')
    corpus = dict(method='DD', count=100, cases=[30], members=[member], data=data,
                  sample_seed=777, split_seed=42)
    corpus['corpus_id'] = 'CORPUS-' + digest(corpus)[:16]
    training = dict(max_updates=10000, batch_size=25, validation_updates=[100, 1000, 10000])
    options = dict(hidden_dim=128, num_layers=3, dropout=0.1)
    config = dict(cases=[30], students=['GAT'], seeds=[42, 43, 44], split_seed=42,
                  model_options={'GAT': options}, training=training, optimizer={'lr': 0.001})
    manifest = dict(format='lumina-downstream-campaign-v1', config=config, sources={},
                    paths=dict(models=str(tmp_path), corpora=str(tmp_path / 'corpora')),
                    corpora={corpus['corpus_id']: corpus}, runs=[])
    splits = {k:dict(count=len(v), sha256=digest(v.tolist()))
              for k,v in split_indices(canonical_case('case30'), 1000, 42).items()}
    for seed in config['seeds']:
        spec = dict(corpus_id=corpus['corpus_id'], model='GAT', seed=seed, split_seed=42,
                    initialization=dict(kind='fresh', seed=seed, model_options=options, precision='float32'),
                    evaluation_cases=[30], data=data, training=training, optimizer=config['optimizer'],
                    sources=digest({}), selection='equal mean of per-case validation sdk_mse + violation')
        run = dict(run_id='RUN-' + digest(spec)[:20], specification=spec, families=['individual'])
        manifest['runs'].append(run)
        folder = tmp_path / 'runs' / 'GAT' / run['run_id']
        write(folder / 'config.json', spec)
        value = float(seed - 41)
        metrics = dict(mse=value, sdk_mse=value, opf_solution_error=value, va_mse=2*value,
                       vm_mse=0., pg_mse=0., qg_mse=0., balance_l2=0., thermal_l2=0.,
                       bounds_l2=0., violation=0., constraint_violation=0., reference_cost=1.,
                       cost_difference_pct=0., n_samples=100, score=value)
        report = dict(model='GAT', case=30, seed=seed, metrics=metrics, checkpoint_step=100,
                      trained_steps=10000, identity=dict(specification=digest(spec),
                      checkpoint_sha256='best-sha', case=30, split='test', indices_sha256=splits['test']['sha256']))
        files = {'best.pt': 'best-sha'}
        files['corpus.json'] = write(folder / 'corpus.json',
            dict(recipe=corpus, count=100, cases={'30': 100}, sha256='dataset-sha'))
        files['splits.json'] = write(folder / 'splits.json', {'30': splits})
        files['test-case30.json'] = write(folder / 'test-case30.json', report)
        write(folder / 'complete.json', dict(specification=digest(spec), files=files))
    path = tmp_path / 'manifest.json'
    write(path, manifest)
    return path


def first_folder(path):
    manifest = json.loads(path.read_text())
    return path.parent / 'runs' / 'GAT' / manifest['runs'][0]['run_id']


def test_verified_mean_sd_and_provenance(campaign):
    report = load_results(campaign)
    cell = next(r for r in summarize(report) if r['panel'] == 'random' and r['column'] == 'DD')
    assert (cell['mean'], cell['sd'], cell['seeds']) == (2., 1., 3)
    assert all(r['status'] == 'complete' and r['result_sha256'] for r in report['rows'])
    assert all('/corpora/CORPUS-' in r['dataset_path'] for r in report['rows'])


def test_missing_seed_is_not_averaged(campaign):
    (first_folder(campaign) / 'complete.json').unlink()
    summary = summarize(load_results(campaign))
    cell = next(r for r in summary if r['panel'] == 'random' and r['column'] == 'DD')
    assert cell['mean'] is None and cell['seeds'] == 2 and cell['status'] == 'incomplete'
    assert re.findall(r'<td[^>]*>(.*?)</td>', render_table(summary, 30, 'random', ['GAT']))[0] == '-'


def test_checksum_corruption_raises(campaign):
    path = first_folder(campaign) / 'test-case30.json'
    path.write_text(path.read_text() + ' ')
    with pytest.raises(ValueError, match='Checksum mismatch'):
        load_results(campaign)


@pytest.mark.parametrize('field,value', [('split', 'val'), ('indices_sha256', 'other-split'),
                                        ('checkpoint_sha256', 'wrong-checkpoint'), ('case', 57)])
def test_rehashed_wrong_identity_raises(campaign, field, value):
    folder = first_folder(campaign)
    report = json.loads((folder / 'test-case30.json').read_text())
    report['identity'][field] = value
    marker = json.loads((folder / 'complete.json').read_text())
    marker['files']['test-case30.json'] = write(folder / 'test-case30.json', report)
    write(folder / 'complete.json', marker)
    with pytest.raises(ValueError, match='Wrong test identity'):
        load_results(campaign)


def test_duplicate_comparison_raises(campaign):
    manifest = json.loads(campaign.read_text())
    manifest['runs'].append(deepcopy(manifest['runs'][0]))
    write(campaign, manifest)
    with pytest.raises(ValueError, match='Ambiguous comparison'):
        load_results(campaign)


def test_own_creator_and_regime_filter(campaign):
    manifest = json.loads(campaign.read_text())
    template = manifest['runs'][0]
    original = next(iter(manifest['corpora'].values()))
    manifest['runs'] = []
    variants = [('DD', 'GAT', 'random', 'fresh', 'individual', 'random'),
                ('DD', 'GCN', 'random', 'fresh', 'individual', None),
                ('DD', 'GAT', 'pretrained', 'fresh', 'individual', None),
                ('DD', 'GAT', 'fixed', 'recorded', 'fixed-matched', 'matched'),
                ('DD', 'GAT', 'fixed', 'fresh', 'fixed-fresh', None),
                ('DD', 'GAT', 'random', 'recorded', 'fixed-control-matched', None),
                ('BAS', 'MLP', None, 'recorded', 'fixed-control-matched', 'matched'),
                ('US', 'none', None, 'fresh', 'individual', 'random')]
    expected = []
    for i, (method, creator, creation, init, family, panel) in enumerate(variants):
        corpus, run = deepcopy(original), deepcopy(template)
        corpus['method'] = method
        corpus['members'][0].update(creator=creator, initialization=creation)
        run['specification']['corpus_id'] = str(i)
        run['specification']['initialization']['kind'] = init
        run['families'] = [family]
        manifest['corpora'][str(i)] = corpus
        manifest['runs'].append(run)
        if panel:
            expected.append((panel, str(i)))
    assert [(p,r['specification']['corpus_id']) for p,r,c in own_data_runs(manifest)] == expected
    manifest['corpora']['0']['members'] *= 2  # An M100 mixture is still excluded.
    assert all(r['specification']['corpus_id'] != '0' for _,r,_ in own_data_runs(manifest))


def test_hmdc_pairs_are_separate_own_creator_comparisons(campaign):
    manifest = json.loads(campaign.read_text())
    original = next(iter(manifest['corpora'].values()))
    template = manifest['runs'][0]
    manifest['runs'] = []
    for pair in ('GCN+GIN', 'GIN+HEAT', 'GAT+RGAT'):
        corpus, run = deepcopy(original), deepcopy(template)
        corpus['method'] = 'HMDC'
        corpus['members'][0]['creator'] = pair
        run['specification'].update(model='GIN', corpus_id=pair)
        manifest['corpora'][pair] = corpus
        manifest['runs'].append(run)
    assert [c['members'][0]['creator'] for _,_,c in own_data_runs(manifest)] == ['GCN+GIN', 'GIN+HEAT']
    report = load_results(campaign)
    originals = report['rows']
    report['rows'] = []
    for pair, value in [('GCN+GIN', 1.), ('GIN+HEAT', 9.)]:
        for row in deepcopy(originals):
            row.update(column=f'HMDC ({pair})', corpus_id=pair, model='GIN')
            row['metrics']['sdk_mse'] = value
            report['rows'].append(row)
    report['config']['students'] = ['GIN']
    summary = summarize(report)
    cells = [r for r in summary if r['status'] == 'complete']
    assert [r['mean'] for r in cells] == [1., 9.]
    assert all(r['status'] == 'not applicable' for r in summary if r['column'] == 'HMDC (GAT+RGAT)')


def test_duplicate_seed_or_changed_corpus_never_averaged(campaign):
    report = load_results(campaign)
    report['rows'][0]['corpus_id'] = 'other-corpus'
    with pytest.raises(ValueError, match='different creator datasets'):
        summarize(report)
    report['rows'][0]['corpus_id'] = report['rows'][1]['corpus_id']
    report['rows'].append(report['rows'][0])
    with pytest.raises(ValueError, match='Duplicate/unexpected seeds'):
        summarize(report)


def test_bold_uses_unrounded_mean_and_preserves_ties(campaign):
    summary = summarize(load_results(campaign))
    for row in summary:
        if row['panel'] == 'random' and row['column'] in ('DD', 'DCGM'):
            row.update(mean=1.00000004 if row['column'] == 'DD' else 1.00000003, sd=0.)
    text = render_table(summary, 30, 'random', ['GAT'])
    assert re.findall(r'<td[^>]*>(.*?)</td>', text)[:2] == ['1.0000', '<strong><u>1.0000</u></strong>']
    assert 'HMDC' not in text and '±' not in text
    for row in summary:
        if row['panel'] == 'random' and row['column'] == 'DD':
            row['mean'] = 1.00000003
    text = render_table(summary, 30, 'random', ['GAT'])
    assert re.findall(r'<td[^>]*>(.*?)</td>', text)[:2] == ['<strong><u>1.0000</u></strong>'] * 2


def test_transfer_selector_excludes_other_regimes_and_mixtures(campaign):
    manifest = json.loads(campaign.read_text())
    manifest['config']['students'].append('GCN')
    template = manifest['runs'][0]
    original = next(iter(manifest['corpora'].values()))
    manifest['runs'] = []
    variants = [('DD', 'GCN', 'random', 'fresh', 'individual', True),
                ('DD', 'GAT', 'random', 'fresh', 'individual', True),
                ('DD', 'GAT', 'pretrained', 'fresh', 'individual', False),
                ('DD', 'GAT', 'fixed', 'recorded', 'fixed-matched', False),
                ('DD', 'GCN', 'random', 'pretrained', 'adaptation-individual', False),
                ('DD', 'GCN', 'random', 'fresh', 'mixed-M100', False),
                ('HMDC', 'GCN+GIN', 'random', 'fresh', 'individual', True),
                ('US', 'none', None, 'fresh', 'individual', True),
                ('BAS', 'MLP', None, 'fresh', 'individual', False)]
    expected = []
    for i, (method, creator, creation, init, family, wanted) in enumerate(variants):
        corpus, run = deepcopy(original), deepcopy(template)
        corpus['method'] = method
        corpus['members'][0].update(creator=creator, initialization=creation)
        run['specification']['corpus_id'] = str(i)
        run['specification']['initialization']['kind'] = init
        run['families'] = [family]
        manifest['corpora'][str(i)] = corpus
        manifest['runs'].append(run)
        if wanted:
            expected.append(str(i))
    assert [r['specification']['corpus_id'] for _, r, _ in transfer_runs(manifest)] == expected
    manifest['corpora']['0']['members'] *= 2
    manifest['corpora']['1']['members'][0]['target_mode'] = 'solver'
    assert [r['specification']['corpus_id'] for _, r, _ in transfer_runs(manifest)] == ['6', '7']


@pytest.fixture
def transfer_report(campaign):
    report = load_results(campaign, select=transfer_runs)
    originals = deepcopy(report['rows'])
    report['config']['students'] = ['GAT', 'GCN']
    report['rows'] = []
    for method, creator, value in [('DD', 'GAT', 2.), ('DD', 'GCN', 9.),
                                   ('DCGM', 'GCN', 5.), ('HMDC', 'GCN+GIN', 3.), ('US', 'none', 7.)]:
        for row in deepcopy(originals):
            row.update(method=method, creator=creator, corpus_id=method + creator)
            row['metrics']['sdk_mse'] = value
            report['rows'].append(row)
    return report


def test_transfer_separates_creators_and_reuses_controls(transfer_report):
    summary = summarize_transfer(transfer_report)
    cells = {(r['method'], r['column']): r for r in summary if r['model'] == 'GAT'}
    assert cells['DD', 'GAT']['mean'] == 2.
    assert cells['DD', 'GCN']['mean'] == 9.
    assert cells['DCGM', 'GCN']['mean'] == 5.
    assert cells['HMDC', 'GCN+GIN']['mean'] == 3.  # GAT was not a creator-pair member.
    assert not cells['HMDC', 'GCN+GIN']['is_creator']
    assert cells['DD', 'GAT']['is_creator']
    assert cells['DCGM', 'GAT']['mean'] is None and cells['DCGM', 'GAT']['status'] == 'not run'
    assert {cells[m, 'US']['mean'] for m in METHODS} == {7.}
    assert {cells[m, 'US']['corpus_id'] for m in METHODS} == {'USnone'}
    assert re.findall(r'<td[^>]*>(.*?)</td>', render_transfer_table(summary, 30, 'DD', ['GAT', 'GCN']))[:2] == [
        '<strong><u>2.0000</u></strong>', '9.0000']


def test_transfer_partial_seed_coverage_stays_missing(transfer_report):
    row = next(r for r in transfer_report['rows'] if r['method'] == 'DD' and r['creator'] == 'GCN')
    row['status'] = 'incomplete'
    summary = summarize_transfer(transfer_report)
    foreign = next(r for r in summary if r['method'] == 'DD' and r['model'] == 'GAT' and r['column'] == 'GCN')
    assert (foreign['mean'], foreign['seeds'], foreign['status']) == (None, 2, 'incomplete')


@pytest.mark.parametrize('change', ['duplicate seed', 'changed corpus'])
def test_transfer_ambiguous_group_raises(transfer_report, change):
    if change == 'duplicate seed':
        transfer_report['rows'].append(deepcopy(transfer_report['rows'][0]))
    else:
        transfer_report['rows'][0]['corpus_id'] = 'other'
    with pytest.raises(ValueError, match='Duplicate/unexpected seeds|different creator datasets'):
        summarize_transfer(transfer_report)
