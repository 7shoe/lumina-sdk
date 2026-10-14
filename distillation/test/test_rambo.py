"""RAMBO boundaries, physical export, acceptance state machine and restart tests."""
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import pytest
import torch

from distillation.RAMBO import data, run as runner
from distillation.RAMBO.objective import distance, sample_subsets
from distillation.RAMBO.aux.train import model_graph
from distillation.conversion.matrix_form import mat2opf, opf2mat
from distillation.selection import input_view
from distillation.selection.generated import GeneratedDataset
from distillation.synthetic import artifacts
from distillation.synthetic.learning import benchmark_graph
from distillation.solver import ACOPF, PyPowerSolver, Solution
from distillation.test.test_dd_demo import cpu_threads, record, graph
from distillation.training.T1_single_topo.common import digest, file_hash, write_json
from distillation.training.T1_single_topo.state import save_torch
from distillation.training.T1_single_topo.tests.test_t1 import assert_same


@pytest.mark.parametrize('case', [14, 30, 57, 118, 500])
@pytest.mark.parametrize('count', [100, 800])
def test_pinned_cases_and_requested_cardinalities(case, count):
    config = runner.configuration(case=f'case{case}', count=count)
    nominal = data.read_case(Path(runner.__file__).with_name('cases') / (config['case'] + '.m'))
    assert nominal['bus'].shape[0] == case
    assert config['count'] == count and config['max_attempts'] == 10 * count
    assert config['load_bounds'] == [.8, 1.2]
    if case == 500:
        assert config['case'] == 'pglib_opf_case500_goc'


def test_cpu_manifest_separates_counts_and_keeps_opfdata_bounds(tmp_path, monkeypatch):
    from distillation.RAMBO import launch
    monkeypatch.setattr(launch, 'freeze_source', lambda path: None)
    monkeypatch.setattr('sys.argv', ['launch', str(tmp_path), '--cases', 'case14', 'case500'])
    launch.main()
    rows = json.loads((tmp_path / 'manifest.json').read_text())['experiments']
    assert len(rows) == 8 and len({r['output'] for r in rows}) == 8
    assert {r['config']['count'] for r in rows} == {100, 800}
    assert {r['config']['method'] for r in rows} == {'rambo', 'uniform'}
    for row in rows:
        assert row['config']['load_bounds'] == [.8, 1.2]
        assert f"M{row['config']['count']}" in Path(row['output']).parts


def test_distance_matches_log_l1_and_independent_subsets():
    library = [dict(pg=[1., 2.], pd=[3.], vm=[1., 1.])] * 4
    point = dict(pg=[2., 4.], pd=[4.], vm=[1.1, 1.2])
    sets = sample_subsets(library, dict(g=2, l=1, n=2), np.random.default_rng(42), 3, 30)
    assert {len(s) for s in sets.values()} == {3}
    assert len({tuple(t['sample'] for t in s) for s in sets.values()}) > 1
    expected = 3 * sum(np.log(1e-8 + v) for v in (3., 1., .3))
    assert distance(point, library, sets) == pytest.approx(expected)
    assert np.isfinite(distance(library[0], library, sets))
    small = sample_subsets(library, dict(g=2, l=1, n=2), np.random.default_rng(42), 2, 1)
    assert all(len(t['variables']) == 1 for s in small.values() for t in s)


def test_native_input_has_no_labels_and_sign_aware_bounds(record):
    graph = data.native_input(record['grid'])
    assert all('y' not in s and 'edge_label' not in s and 'objective' not in s for s in graph.stores)
    graph['load'].x[0] = torch.tensor([-1., 0.])
    net = data.network_record(graph, [.5, 2.])
    assert (net['pd_lo'][0], net['pd_hi'][0]) == (-2., -.5)
    assert net['qd_lo'][0] == net['qd_hi'][0] == 0.
    point = dict(pd=net['pd'], qd=net['qd'])
    almost = deepcopy(point)
    almost['pd'][0] += 1e-10
    assert runner.duplicate(almost, [point], net, 1e-7)
    almost['pd'][0] += .01
    assert not runner.duplicate(almost, [point], net, 1e-7)


@pytest.mark.parametrize('nominal_reference', [0, 1, 3])
@pytest.mark.parametrize('offline', [False, True])
def test_template_reads_exact_train_id_and_checks_archive(tmp_path, record, monkeypatch, nominal_reference, offline):
    case, index = 'pglib_opf_case30_ieee', 7
    graph = data.native_input(record['grid'])
    casefile = tmp_path / 'case.m'
    nominal = graph.clone()
    nominal['bus'].x[0, 1] = 2
    nominal['bus'].x[nominal_reference, 1] = 3
    data.write_matpower(nominal, casefile)
    if offline:
        tables = data.read_case(casefile)
        text = casefile.read_text()
        for key, status in (('gen', 7), ('gencost', None), ('branch', 10)):
            row = tables[key][0].copy()
            if status is not None:
                row[status] = 0
            text = text.replace(f'mpc.{key} = [\n', f'mpc.{key} = [\n' +
                                ' '.join(f'{v:.17g}' for v in row) + ';\n')
        casefile.write_text(text)
    archive_path = tmp_path / f'{case}_0.tar.gz'
    with tarfile.open(archive_path, 'w:gz') as archive:
        # Holdout records need not even contain valid JSON: they are not read.
        for i in (2, index, 9):
            payload = json.dumps(record).encode() if i == index else b'not opened'
            info = tarfile.TarInfo(f'release/{case}/group_0/example_{i}.json')
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    class Dataset:
        manifest = dict(case=case, group_size=20, groups=[dict(raw_sha256=file_hash(archive_path))])
        def splits(self, seed):
            assert seed == 42
            return dict(train=[index], val=[2], test=[9])
        def __getitem__(self, i):
            raise AssertionError('No cached labels needed')
    prepared, provenance = data.prepare_template(Dataset(), casefile, tmp_path)
    assert provenance['template_train_id'] == index
    assert provenance['reference_buses'] == dict(opfdata=1, nominal=nominal_reference + 1)
    assert provenance['omitted_offline'] == dict(generators=int(offline), branches=int(offline))
    assert prepared['bus'].x[0, 1] == 3  # Preserve the benchmark's angle-label gauge.
    assert 'objective' not in prepared
    # Completed-library evaluations reuse the verified template without rescanning gzip.
    def no_read(*args, **kwargs):
        raise AssertionError('Repeated raw archive extraction')
    with monkeypatch.context() as patch:
        patch.setattr(data.tarfile, 'open', no_read)
        again, origin = data.prepare_template(Dataset(), casefile, tmp_path)
        torch.testing.assert_close(again['load'].x, prepared['load'].x)
        assert origin == provenance
        again['load'].x.zero_()
        unchanged, _ = data.prepare_template(Dataset(), casefile, tmp_path)
        torch.testing.assert_close(unchanged['load'].x, prepared['load'].x)
    # A non-reference type change is still a mismatch, not a gauge relocation.
    bad = nominal.clone()
    bad['bus'].x[5, 1] = 2
    data.write_matpower(bad, casefile)
    with pytest.raises(AssertionError, match='non-reference bus types'):
        data.prepare_template(Dataset(), casefile, tmp_path)
    Dataset.manifest['groups'][0]['raw_sha256'] = 'changed'
    with pytest.raises(ValueError, match='archive differs'):
        data.prepare_template(Dataset(), casefile, tmp_path)


def point_of(graph):
    s, p = Solution.from_label(graph), opf2mat(graph).params
    return dict(theta=s.theta.tolist(), vm=s.vm.tolist(), pg=s.pg.tolist(), qg=s.qg.tolist(),
                pd=p.pd.tolist(), qd=p.qd.tolist(), cost=s.objective)


def test_final_labels_preserved_and_bad_solutions_rejected(graph):
    point = point_of(graph)
    exported, checks = data.attach_solution(input_view(graph), point)
    assert checks['feasible']
    torch.testing.assert_close(exported['bus'].y, graph['bus'].y, atol=0, rtol=0)
    assert model_graph(exported)['bus'].x.shape[1] == 7
    point['vm'][0] += .1
    with pytest.raises(ValueError, match='verification failed'):
        data.attach_solution(input_view(graph), point)


@pytest.fixture
def fake_backend(monkeypatch, graph):
    # A real small physical solve for final-label checks, with scripted search
    # stages to force interruption and rejected candidates deterministically.
    template = input_view(graph)
    monkeypatch.setattr(runner, 'prepare_template', lambda *args: (template, dict(data_identity='test')))
    class Backend:
        versions = {'julia': 'test'}
        requests = []
        fail_stage = None
        def __init__(self, *args):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def request(self, r, timeout):
            self.requests.append(deepcopy(r))
            stage = r['stage']
            if stage == self.fail_stage:
                return dict(success=False, status='ITERATION_LIMIT')
            if stage == 'opf':
                p = replace(opf2mat(template).params, pd=torch.tensor(r['pd']), qd=torch.tensor(r['qd']))
                problem = ACOPF(mat2opf(replace(opf2mat(template), params=p), include_solution=False))
                solved = PyPowerSolver().solve(problem)
                assert solved.success
                point = point_of(solved.to_hetero(problem))
            else:
                point = deepcopy(r['history'][-1])
                point['pd'] = (np.array(point['pd']) * .99).tolist()
                point['qg'] = [123.] * len(point['qg'])  # Deliberately untrusted relaxed labels.
            return dict(success=True, status='LOCALLY_SOLVED', point=point, objective=1.)
    return Backend


@pytest.mark.parametrize('warm_start', ['opf', 'feasible'])
def test_stage_resume_uses_final_labels_only(tmp_path, fake_backend, warm_start):
    config = runner.configuration(count=2, warm_start=warm_start)
    first = runner.run(config, tmp_path, backend_factory=fake_backend, dataset=object(), session_stages=2)
    assert first['status'] == 'paused' and first['accepted'] == 1
    assert not (tmp_path / 'artifact.json').exists()
    stages_before = len(fake_backend.requests)
    result = runner.run(config, tmp_path, backend_factory=fake_backend, dataset=object())
    assert result['status'] == 'complete'
    assert len(fake_backend.requests) == 5 if warm_start == 'opf' else len(fake_backend.requests) == 4
    state = json.loads((tmp_path / 'state.json').read_text())
    assert all(max(np.abs(p['qg'])) < 10 for p in state['library'])
    assert len(fake_backend.requests) > stages_before
    assert all(len(r['history']) == 1 for r in fake_backend.requests[1:])
    generated = GeneratedDataset(tmp_path / 'dataset.json')
    assert len(generated) == 2
    assert all('sample_id' not in g for g in generated)
    # Idempotent completion neither re-solves nor silently accepts tampering.
    before = len(fake_backend.requests)
    assert runner.run(config, tmp_path, backend_factory=fake_backend, dataset=object()) == result
    assert len(fake_backend.requests) == before
    path = tmp_path / generated.records[0]['file']
    path.write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='changed'):
        GeneratedDataset(tmp_path / 'dataset.json')


def test_failed_search_never_fills_budget_or_poison_library(tmp_path, fake_backend):
    fake_backend.fail_stage = 'bilevel'
    result = runner.run(runner.configuration(count=2, max_attempts=3), tmp_path,
                        backend_factory=fake_backend, dataset=object())
    assert result['status'] == 'exhausted' and result['accepted'] == 1
    assert result['rejections'] == 2
    assert len(GeneratedDataset(tmp_path / 'dataset.json')) == 1
    assert not (tmp_path / 'artifact.json').exists()
    with pytest.raises(ValueError, match='entire completed'):
        data.export_dataset(tmp_path)


def test_data_only_export_preserves_targets_and_loads_without_generation_files(tmp_path, fake_backend):
    config = runner.configuration(count=2)
    assert runner.run(config, tmp_path, backend_factory=fake_backend, dataset=object())['status'] == 'complete'
    loaded = artifacts.load_dataset(tmp_path)
    meta = loaded['metadata']
    assert meta['complete'] and not meta['evaluation_complete'] and meta['count'] == 2
    assert meta['synthesis_models'] == meta['evaluation_models'] == []
    assert not meta['split']['test_accessed'] and not meta['split']['validation_accessed']
    for actual, physical in zip(loaded['graphs'], loaded['physical_solved_graphs']):
        expected = benchmark_graph(physical, labels=True)
        for kind in ('bus', 'generator'):
            torch.testing.assert_close(actual[kind].y, physical[kind].y, rtol=0, atol=0)
        for kind in actual.node_types:
            torch.testing.assert_close(actual[kind].x, expected[kind].x, rtol=0, atol=0)
        for edge in actual.edge_types:
            torch.testing.assert_close(actual[edge].edge_index, expected[edge].edge_index, rtol=0, atol=0)
            if 'edge_attr' in actual[edge]:
                torch.testing.assert_close(actual[edge].edge_attr, expected[edge].edge_attr, rtol=0, atol=0)
    assert len(set(loaded['export']['generated_ids'])) == 2
    assert 'model_checkpoint' not in loaded['export']
    (tmp_path / 'samples').rename(tmp_path / 'samples_moved')
    (tmp_path / 'dataset.json').unlink()
    assert len(artifacts.load_dataset(tmp_path)['graphs']) == 2
    with pytest.raises(ValueError, match='Data-only artifact'):
        artifacts.train_from_artifact(tmp_path)
    with (tmp_path / 'dataset.pt').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        artifacts.load_dataset(tmp_path)


def test_uniform_control_uses_only_ordinary_opf(tmp_path, fake_backend):
    result = runner.run(runner.configuration(count=2, method='uniform'), tmp_path,
                        backend_factory=fake_backend, dataset=object())
    assert result['status'] == 'complete' and result['solver_calls'] == 2
    assert [r['stage'] for r in fake_backend.requests] == ['opf', 'opf']


def test_timeout_is_counted_and_partial_dataset_is_resumable(tmp_path, fake_backend):
    original = fake_backend.request
    def timeout(self, r, timeout):
        if r['stage'] == 'feasible':
            raise TimeoutError('test timeout')
        return original(self, r, timeout)
    fake_backend.request = timeout
    with pytest.raises(TimeoutError):
        runner.run(runner.configuration(count=2), tmp_path, backend_factory=fake_backend, dataset=object())
    state = json.loads((tmp_path / 'state.json').read_text())
    assert state['solver_calls'] == 2 and state['rejections'] == 1 and state['pending'] is None
    fake_backend.request = original
    result = runner.run(runner.configuration(count=2), tmp_path, backend_factory=fake_backend, dataset=object())
    assert result['status'] == 'complete' and result['accepted'] == 2


@pytest.mark.parametrize('kind', ['MLP', 'HGT'])
def test_fresh_student_resume_and_original_control_preserve_split(tmp_path, monkeypatch, graph, kind):
    from distillation.RAMBO.aux import train as training
    from distillation.selection import ids_hash
    from distillation.data.split import split_indices
    splits = split_indices('pglib_opf_case30_ieee', 30, 42)
    class Dataset:
        manifest = dict(case='pglib_opf_case30_ieee', fixture=True)
        reads = []
        def splits(self, seed):
            assert seed == 42
            return splits
        def __getitem__(self, index):
            assert index not in splits['test'], 'Test labels must stay untouched'
            self.reads.append(index)
            return model_graph(graph)
    dataset = Dataset()
    monkeypatch.setattr(training, 'CaseDataset', lambda *args: dataset)
    folder = tmp_path / 'generated'
    records = []
    for i in range(3):
        path = folder / 'samples' / f'{i:06d}.pt'
        identity = f'generated:test:{i}'
        save_torch(path, dict(id=identity, graph=graph))
        records.append(dict(id=identity, file=str(path.relative_to(folder)), sha256=file_hash(path)))
    manifest = folder / 'dataset.json'
    write_json(manifest, dict(format='lumina-generated-v1', role='training', method='rambo', records=records,
               config=dict(data_root='unused', case='case30', seed=42),
               provenance=dict(data_identity=digest(dataset.manifest), split_hashes={k: ids_hash(v) for k, v in splits.items()})))
    options = dict(model=kind, epochs=2, batch_size=2, validation_size=2)
    assert training.train(manifest, tmp_path / 'live', **options)['status'] == 'complete'
    assert training.train(manifest, tmp_path / 'resumed', session_steps=1, **options)['status'] == 'paused'
    assert training.train(manifest, tmp_path / 'resumed', **options)['status'] == 'complete'
    snapshots = [torch.load(tmp_path / key / 'latest.pt', weights_only=False) for key in ('live', 'resumed')]
    for key in ('model', 'optimizer', 'state', 'rng'):
        assert_same(snapshots[0][key], snapshots[1][key])
    assert set(dataset.reads) <= set(splits['val'])
    assert training.train(manifest, tmp_path / 'original', random_control=True, **options)['status'] == 'complete'
    config = json.loads((tmp_path / 'original/config.json').read_text())
    assert set(config['identities']) <= set(splits['train'])
