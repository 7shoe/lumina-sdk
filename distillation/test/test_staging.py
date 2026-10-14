"""Local input copies preserve data/splits and never replace durable outputs."""
from copy import deepcopy
import json
import os
from pathlib import Path

import pytest
import torch
from torch_geometric.data import InMemoryDataset

from distillation.experiments import staging
from distillation.synthetic.learning import benchmark_graph
from distillation.test.test_dd_demo import cpu_threads, record, graph
from distillation.training.T1_single_topo.common import file_hash, write_json
from distillation.training.T1_single_topo.data import CaseDataset, processor_identity


@pytest.fixture
def inputs(tmp_path, graph):
    folder = tmp_path/'source'/'pglib_opf_case30_ieee'
    folder.mkdir(parents=True)
    records = []
    for group in range(2):
        graphs = []
        for index in range(group*3, group*3+3):
            value = benchmark_graph(graph, labels=True)
            value.sample_id = torch.tensor([index])
            graphs.append(value)
        path = folder/f'group_{group}.pt'
        torch.save(InMemoryDataset.collate(graphs), path)
        records.append(dict(group=group, count=3, processor=processor_identity(),
            sha256=file_hash(path), bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns))
    write_json(folder/'manifest.json', dict(format='lumina-t1-data-v1', case=folder.name,
        count=6, group_size=3, order='numeric example_ID', groups=records))
    reference = tmp_path/'reference.pt'; torch.save(dict(graphs=[graph], ids=[0]), reference)
    cfg = dict(case=30, data_root=str(folder.parent), reference_file=str(reference),
        artifact_identity=dict(data_sha256=file_hash(folder/'manifest.json'), reference_sha256=file_hash(reference)))
    return dict(method='DD', dataset_id='first', config=cfg)


def test_staged_data_is_exact_shared_and_keeps_canonical_identity(inputs, tmp_path, monkeypatch):
    before = deepcopy(inputs)
    direct = CaseDataset(inputs['config']['data_root'], 'case30')
    second = inputs | dict(method='DCGM', dataset_id='second')
    cache = tmp_path/'local'
    report = staging.stage([inputs, second], cache)
    assert report['files'] == 4  # Two shards, one manifest, one reference, each copied once.
    monkeypatch.setenv('STUDY_INPUT_MAP', str(cache/'inputs.json'))
    cached = staging.case_dataset(inputs['config']['data_root'], 'case30')
    assert staging.case_dataset(inputs['config']['data_root'], 'case30') is cached
    for index in range(len(direct)):
        expected, actual = direct[index], cached[index]
        for kind in expected.node_types:
            torch.testing.assert_close(actual[kind].x, expected[kind].x, rtol=0, atol=0)
            if 'y' in expected[kind]: torch.testing.assert_close(actual[kind].y, expected[kind].y, rtol=0, atol=0)
        assert actual.sample_id.item() == index
    for split, ids in direct.splits().items():
        assert (ids == cached.splits()[split]).all()
    assert inputs == before
    assert staging.local_path(tmp_path/'output') == tmp_path/'output'
    assert staging.local_path(inputs['config']['reference_file']).is_relative_to(cache)
    # Worker reuse must still notice accidental cache changes.
    shard = cached.folder/'group_0.pt'; shard.write_bytes(b'broken')
    with pytest.raises(ValueError, match='Cache changed'):
        staging.case_dataset(inputs['config']['data_root'], 'case30')


def test_staging_rejects_corruption_even_with_original_size_and_mtime(inputs, tmp_path):
    folder = Path(inputs['config']['data_root'])/'pglib_opf_case30_ieee'
    path = folder/'group_0.pt'; stat = path.stat()
    content = bytearray(path.read_bytes()); content[-1] ^= 1; path.write_bytes(content)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    cache = tmp_path/'local'
    with pytest.raises(ValueError, match='checksum'):
        staging.stage([inputs], cache)
    assert not (cache/'inputs.json').exists()


def test_retention_stages_source_data_and_pins_its_identity(inputs, tmp_path, monkeypatch):
    import shutil
    cfg = inputs['config']
    source = Path(cfg['data_root'])/'pglib_opf_case57_ieee'
    shutil.copytree(Path(cfg['data_root'])/'pglib_opf_case30_ieee', source)
    manifest = json.loads((source/'manifest.json').read_text()) | {'case': source.name}
    write_json(source/'manifest.json', manifest)
    cfg['retention_data'] = {source.name: file_hash(source/'manifest.json')}
    cache = tmp_path/'local'
    report = staging.stage([inputs, deepcopy(inputs)], cache)
    assert report['files'] == 7  # Both three-file datasets and one common reference.
    monkeypatch.setenv('STUDY_INPUT_MAP', str(cache/'inputs.json'))
    dataset = staging.case_dataset(cfg['data_root'], source.name)
    assert dataset.folder.is_relative_to(cache) and dataset.manifest == manifest
    write_json(source/'manifest.json', manifest | {'changed': True})
    with pytest.raises(ValueError, match='manifest changed'):
        staging.stage([inputs], tmp_path/'changed')


def test_rambo_template_archive_and_pair_weights_are_staged_once(inputs, tmp_path, monkeypatch):
    cfg = inputs['config']
    folder = Path(cfg['data_root'])/'pglib_opf_case30_ieee'
    manifest = json.loads((folder/'manifest.json').read_text())
    for group in manifest['groups']:
        path = tmp_path/f"{folder.name}_{group['group']}.tar.gz"
        path.write_bytes(b'raw archive fixture')
        group['raw_sha256'] = file_hash(path)
    write_json(folder/'manifest.json', manifest)
    cfg['artifact_identity']['data_sha256'] = file_hash(folder/'manifest.json')
    cfg['pretrained_checkpoints'], cfg['pretrained'] = {}, {}
    for model in ['GIN', 'GAT']:
        path = tmp_path/f'{model}.pt'; torch.save({'fixture': model}, path)
        cfg['pretrained_checkpoints'][model] = str(path)
        cfg['pretrained'][model] = {'sha256': file_hash(path)}
    cfg['raw_root'] = str(tmp_path)
    rows = [inputs | dict(method='HMDC'), inputs | dict(method='RAMBO', dataset_id='rambo')]
    cache = tmp_path/'local'
    report = staging.stage(rows, cache)
    assert report['files'] == 7
    monkeypatch.setenv('STUDY_INPUT_MAP', str(cache/'inputs.json'))
    for model, path in cfg['pretrained_checkpoints'].items():
        assert file_hash(staging.local_path(path)) == cfg['pretrained'][model]['sha256']
    archives = [p for p in report['paths'] if p.endswith('.tar.gz')]
    assert len(archives) == 1 and staging.local_path(archives[0]).read_bytes() == b'raw archive fixture'


@pytest.mark.parametrize('nodes', [1, 2, 16, 32, 64])
def test_every_condition_assigned_once_and_shared_generation_stays_on_one_node(nodes):
    rows = [dict(method='DD', dataset_id=f'dd-{i}', config={}) for i in range(80)]
    rows += [dict(method='RAMBO', dataset_id=f'rambo-{i}', config={'generation_id': str(i//3)}) for i in range(18)]
    assignments = [staging.assigned(rows, node, nodes) for node in range(nodes)]
    assert sorted(r['dataset_id'] for group in assignments for r in group) == sorted(r['dataset_id'] for r in rows)
    for library in range(6):
        assert sum(any(r['config'].get('generation_id') == str(library) for r in group) for group in assignments) == 1
