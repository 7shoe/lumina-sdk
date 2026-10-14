"""Selection publication, honest controls, resume, and read-only dataset unions."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from distillation.test.test_bas import tiny_case
from distillation.selection.artifacts import publish, load_dataset, subset_order
from distillation.selection.pool import ids_hash
from distillation.experiments import selection as sidecar
from distillation.experiments import datasets
from distillation.training.T1_single_topo.common import digest, file_hash, write_json
from distillation.training.T1_single_topo.data import CaseDataset
from experiments.lumina_bench.tests.test_protocol import graph


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def row(config, folder, method='uniform', counts=(4, 8)):
    dataset = CaseDataset(config['data_root'], config['case'])
    return dict(config=config, output=str(folder), method=method, counts=list(counts), export_seed=4242,
                experiment_id=method+'-test', data_sha256=file_hash(dataset.folder/'manifest.json'))


@pytest.mark.parametrize('method', ['uniform', 'demand_stratified'])
def test_controls_export_without_model_or_panel_labels(tiny_case, tmp_path, monkeypatch, method):
    entry = row(tiny_case, tmp_path/method, method)
    dataset = CaseDataset(tiny_case['data_root'], tiny_case['case'])
    train = set(dataset.splits(42)['train'])
    def forbidden(*args, **kwargs):
        raise AssertionError('Control must not train a selector')
    monkeypatch.setattr(sidecar.bas, 'run', forbidden)
    original = dataset.__class__.__getitem__
    accessed = []
    def guarded(self, index):
        assert index in train
        accessed.append(index)
        return original(self, index)
    monkeypatch.setattr(dataset.__class__, '__getitem__', guarded)
    outcome = sidecar.run_experiment(entry, dataset=dataset)
    assert outcome['status'] == 'COMPLETE'
    folder = Path(entry['output'])/'exports'
    frozen = json.loads((folder/'pool.json').read_text())
    panel = frozen['selection']['reference_ids']
    assert not set(accessed) & set(panel)
    a, b = [load_dataset(folder/f'M{n}') for n in (4, 8)]
    assert a['metadata']['selected_ids'] == b['metadata']['selected_ids'][:4]
    assert b['metadata']['label_access'] == dict(scoring=0, acquired=0, validation=0, test=0)
    assert b['metadata']['selector'] is None
    for actual, index in zip(b['graphs'], b['metadata']['selected_ids']):
        expected = original(dataset, index)
        for kind in actual.node_types:
            torch.testing.assert_close(actual[kind].x, expected[kind].x, rtol=0, atol=0)
        for kind in ('bus', 'generator'):
            torch.testing.assert_close(actual[kind].y, expected[kind].y, rtol=0, atol=0)
        assert int(actual.sample_id) == index
    before = file_hash(folder/'M8/dataset.pt')
    assert sidecar.run_experiment(entry, dataset=dataset) == outcome
    assert file_hash(folder/'M8/dataset.pt') == before
    if method == 'demand_stratified':
        from distillation.selection.buckets import Buckets
        boundary = frozen['selection']['provenance']['buckets']
        partition = Buckets(np.asarray(boundary['boundaries']), boundary['strategy'])
        for item in (a, b):
            memberships = partition.assign([float(g['load'].x[:,0].sum()) for g in item['graphs']])
            counts = np.bincount(memberships, minlength=partition.count)
            assert counts.max() - counts.min() <= 1


def test_bas_resume_and_underfilled_exports_are_explicit(tiny_case, tmp_path, monkeypatch):
    original = sidecar.bas.score_ids
    def plateau(model, pool, ids, metric, config, **kwargs):
        return np.ones(len(ids)) if metric == 'le' else original(model, pool, ids, metric, config, **kwargs)
    monkeypatch.setattr(sidecar.bas, 'score_ids', plateau)
    entry = row(tiny_case, tmp_path/'bas', 'bas')
    assert sidecar.run_experiment(entry, session_steps=1)['status'] == 'paused'
    assert not (Path(entry['output'])/'exports').exists()
    assert sidecar.run_experiment(entry)['status'] == 'COMPLETE'
    meta = load_dataset(Path(entry['output'])/'exports/M8')['metadata']
    assert meta['acquisition_events'] == 2 and meta['selected_pool_count'] == 12
    assert meta['label_access']['scoring'] == 8 and meta['label_access']['acquired'] == 12
    assert meta['selector']['model'] == 'MLP'
    entry = row(tiny_case | {'max_rounds':1}, tmp_path/'no_acquisition', 'bas')
    result = sidecar.run_experiment(entry)
    assert result['status'] == 'UNDERFILLED'
    assert result['datasets'][1] == dict(count=8, status='UNDERFILLED', available=4, ready=False)
    meta = load_dataset(Path(entry['output'])/'exports/M4')['metadata']
    assert meta['acquisition_events'] == 0 and meta['initial_examples'] == 4
    assert meta['provenance']['selection_outcome'] == 'initial_random_only'


class MemoryData:
    def __init__(self):
        self.manifest = dict(case='pglib_opf_case14_ieee', count=1200, source='test')
    def __len__(self):
        return 1200
    def splits(self, seed=42):
        assert seed == 42
        return dict(train=np.arange(1000), val=np.arange(1000,1100), test=np.arange(1100,1200))
    def __getitem__(self, index):
        value = graph()
        value.sample_id = torch.tensor([index])
        return value


def memory_selection(dataset):
    return dict(complete=True, method='uniform', selection_seed=42, selected_ids=list(range(100,1000)),
        reference_ids=list(range(100)), data_identity=digest(dataset.manifest),
        split_hashes={k: ids_hash(v) for k,v in dataset.splits().items()},
        label_access=dict(scoring=0, acquired=0, validation=0, test=0))


def test_actual_100_800_portable_exports_and_corruption(tmp_path):
    data = MemoryData(); selection = memory_selection(data)
    result = publish(data, selection, tmp_path/'portable', counts=[100,800])
    assert result['status'] == 'COMPLETE'
    small, large = [load_dataset(tmp_path/f'portable/M{n}') for n in (100,800)]
    assert len(large['graphs']) == 800
    assert small['metadata']['selected_ids'] == large['metadata']['selected_ids'][:100]
    assert len(set(large['metadata']['selected_ids'])) == 800
    with pytest.raises(ValueError, match='Frozen'):
        publish(data, selection, tmp_path/'portable', seed=43)
    path = tmp_path/'portable/M100/dataset.pt'
    path.write_bytes(path.read_bytes()+b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        load_dataset(path.parent)


@pytest.mark.parametrize('change', ['holdout', 'panel', 'duplicate', 'unfinished', 'split', 'source'])
def test_reject_invalid_publication(tmp_path, change):
    data = MemoryData(); value = memory_selection(data)
    if change == 'holdout': value['selected_ids'][0] = 1001
    if change == 'panel': value['selected_ids'][0] = 0
    if change == 'duplicate': value['selected_ids'][0] = value['selected_ids'][1]
    if change == 'unfinished': value['complete'] = False
    if change == 'split': value['split_hashes']['train'] = 'changed'
    if change == 'source': value['data_identity'] = 'changed'
    with pytest.raises(ValueError):
        publish(data, value, tmp_path/'invalid')
    assert not (tmp_path/'invalid/M100/artifact.json').exists()


def test_stratified_order_balances_prefixes_and_honors_capacity():
    ids = np.arange(1000)
    labels = ids % 30
    order = subset_order(ids, 42, labels)
    assert len(set(order)) == 1000
    for count in (100,800):
        counts = np.bincount(labels[order[:count]], minlength=30)
        assert counts.max()-counts.min() <= 1
    assert subset_order([0,1,2,3], 42, [0,0,0,1]) == subset_order([0,1,2,3], 42, [0,0,0,1])


def test_union_exact_count_duplicates_regimes_and_teaching_targets(tmp_path):
    data = MemoryData(); value = memory_selection(data)
    members = []
    for seed in range(8):
        publish(data, value, tmp_path/f's{seed}', counts=[100], seed=seed)
        members.append(tmp_path/f's{seed}/M100')
    before = [file_hash(p/'dataset.pt') for p in members]
    out = tmp_path/'union.json'
    metadata = datasets.write_union(out, members)
    assert metadata['count'] == 800
    assert metadata['unique_input_graphs'] == metadata['unique_teaching_pairs'] == 1
    assert len(datasets.load_dataset(out)['graphs']) == 800
    assert before == [file_hash(p/'dataset.pt') for p in members]
    with pytest.raises(ValueError, match='same dataset'):
        datasets.write_union(tmp_path/'duplicate.json', [members[0],members[0]], expected_count=200)
    with pytest.raises(ValueError, match='received 500'):
        datasets.write_union(tmp_path/'short.json', members[:5])
    path = members[0]/'artifact.json'
    meta = json.loads(path.read_text()); meta['method'] = 'another'
    write_json(path, meta)
    with pytest.raises(ValueError, match='share method'):
        datasets.write_union(tmp_path/'mixed.json', members)
    with pytest.raises(ValueError, match='Union member changed'):
        datasets.load_dataset(out)
    changed = deepcopy(metadata); changed['count'] = 801
    write_json(tmp_path/'corrupt_union.json', changed)
    with pytest.raises(ValueError, match='Union identity changed'):
        datasets.load_dataset(tmp_path/'corrupt_union.json')


def test_legacy_loader_is_read_only_and_ignores_embedded_weights(tmp_path, monkeypatch):
    from distillation.synthetic import artifacts
    g = graph(); g['bus'].y.fill_(12.34)
    record = dict(format='lumina-synthetic-dataset-v1', method='DD')
    path = tmp_path/'artifact.json'; write_json(path, record)
    before = path.read_bytes()
    def legacy(path):
        return dict(graphs=[g], metadata=record, export=dict(model_checkpoint='unused', schedule='unused'))
    monkeypatch.setattr(artifacts, 'load_dataset', legacy)
    result = datasets.load_dataset(path)
    assert set(result) == {'graphs','metadata'}
    torch.testing.assert_close(result['graphs'][0]['bus'].y, g['bus'].y)
    assert path.read_bytes() == before


def test_prepare_preserves_historical_campaign_and_is_immutable(tiny_case, tmp_path):
    protected = tmp_path/'wave1'
    write_json(protected/'manifest.json', dict(schema_version=1, method='DD', experiments=[]))
    before = (protected/'manifest.json').read_bytes()
    with pytest.raises(ValueError, match='read-only'):
        sidecar.output_guard(protected/'new')
    with pytest.raises(ValueError, match='read-only'):
        datasets.write_union(protected/'union.json', [])
    with pytest.raises(ValueError, match='read-only'):
        publish(MemoryData(), memory_selection(MemoryData()), protected/'exports')
    assert not (protected/'new').exists() and (protected/'manifest.json').read_bytes() == before
    spec = yaml.safe_load(sidecar.DEFAULT.read_text())
    spec.update(output=str(tmp_path/'campaigns'), campaign='test', data_root=tiny_case['data_root'],
                cases=[tiny_case['case']], counts=[4,8], pool_size=12, panel_size=8, buckets=2,
                bas={k: tiny_case[k] for k in ('model','model_options','initial_size','max_rounds','validation_size')})
    path = tmp_path/'selection.yaml'; path.write_text(yaml.safe_dump(spec))
    first = sidecar.prepare(path)
    assert len(first['experiments']) == 3 and sidecar.prepare(path) == first
    frozen = Path(first['root'])/'source/distillation/BAS/config.yaml'
    assert frozen.exists()
    spec['export_seed'] += 1; path.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError, match='campaign changed'):
        sidecar.prepare(path)
