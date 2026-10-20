"""Resumable case pools, teacher predictions, and M100 publication."""
from pathlib import Path
import time

import numpy as np
import torch

from distillation.experiments.datasets import graph_digest, load_dataset
from distillation.experiments.speculative_followup.train import teaching_graph
from distillation.experiments.speculative_followup_two.data import load_reference
from distillation.experiments.speculative_followup_three.curation import descriptors, predictions
from .demand import synthesize
from distillation.selection.artifacts import publish as publish_selection
from distillation.selection.pool import ids_hash
from distillation.training.T1_single_topo.common import claim, digest, file_hash, read_json, write_json
from distillation.training.T1_single_topo.state import save_torch
from . import METHODS, SELECTORS
from .artifacts import publish
from .methods import choose, correct_labels


def cached(folder, protocol):
    marker = Path(folder)/'cache.json'
    if not marker.exists():
        return None
    info = read_json(marker)
    if info['protocol'] != protocol or file_hash(Path(folder)/'cache.pt') != info['sha256']:
        raise ValueError('Creation cache identity changed')
    return torch.load(Path(folder)/'cache.pt', weights_only=False, map_location='cpu')


def store(folder, value, protocol):
    folder = Path(folder)
    save_torch(folder/'cache.pt', value)
    write_json(folder/'cache.json', dict(protocol=protocol, sha256=file_hash(folder/'cache.pt')))


def pool(config, dataset):
    folder = Path(config['paths']['datasets'])/'pool'
    value = cached(folder, config['identity'])
    if value is not None:
        return value
    with claim(folder/'create.lock') as acquired:
        if not acquired:
            raise BlockingIOError('Candidate pool is being prepared')
        train = dataset.splits(42)['train']
        if not config['count'] <= config['pool_count'] <= len(train):
            raise ValueError('Invalid candidate-pool size')
        ids = np.random.default_rng(config['data_seed']).permutation(train)[:config['pool_count']]
        graphs = [teaching_graph(dataset[int(i)]) for i in ids]
        ref = graphs[0]
        if any(set(g.node_types) != set(ref.node_types) or set(g.edge_types) != set(ref.edge_types) or
               any(g[k].x.shape != ref[k].x.shape for k in ref.node_types) or
               any(not torch.equal(g[e].edge_index, ref[e].edge_index) for e in ref.edge_types) for g in graphs):
            raise ValueError('Fixed ordered case topology required')
        value = dict(graphs=graphs, ids=ids.tolist(), features=descriptors(graphs).double().numpy(),
            demand=np.array([float(g['load'].x[:,0].sum()) for g in graphs]))
        store(folder, value, config['identity'])
        return value


def teacher_folder(config, name):
    return Path(config['paths']['datasets'])/'teachers'/name


def teacher(config, dataset, name, device, stop):
    folder = teacher_folder(config, name)
    if cached(folder, config['identity']) is not None:
        return
    source = pool(config, dataset)
    model, _ = load_reference(config, name, source['graphs'][0], device, pretrained=True)
    model.eval().requires_grad_(False)
    path = folder/'latest.pt'
    state = torch.load(path, weights_only=False, map_location='cpu') if path.exists() else dict(
        protocol=config['identity'], cursor=0, rows=[])
    if state['protocol'] != config['identity']:
        raise ValueError('Teacher prediction resume changed')
    size = config['scoring']['eval_batch_size']
    with torch.no_grad():
        while state['cursor'] < len(source['graphs']):
            if stop():
                save_torch(path, state)
                raise BlockingIOError('Teacher predictions paused')
            end = min(len(source['graphs']), state['cursor']+size)
            values = predictions(model, source['graphs'][state['cursor']:end]).cpu().numpy()
            if not np.isfinite(values).all():
                raise FloatingPointError('Nonfinite T1 teacher prediction')
            state['rows'].append(values); state['cursor'] = end
            if end % max(512,size) == 0 or end == len(source['graphs']):
                save_torch(path, state)
            write_json(folder/'progress.json', dict(cursor=end, updated=time.time()))
    store(folder, dict(prediction=np.concatenate(state['rows']), source_ids_sha256=digest(source['ids']),
        teacher=config['teachers'][name]), config['identity'])


def destination(config, method, student=None):
    return Path(config['paths']['datasets'])/'datasets'/method/(student or 'shared')


def artifact_path(config, method, student=None):
    folder = destination(config, method, student)
    return folder/f"M{config['count']}"/'artifact.json'


def generate(config, dataset, method, device, stop, student=None):
    folder = destination(config, method, student); path = artifact_path(config, method, student)
    if path.exists():
        meta = load_dataset(path)['metadata']
        if meta['provenance']['protocol'] != config['identity'] or meta['count'] != config['count']:
            raise ValueError('Published creation protocol changed')
        return meta
    if (method == 'ALC') != (student is not None):
        raise ValueError('Only ALC has an eligible student architecture')
    source = pool(config, dataset); ids = choose(method, source, config)
    anchor_ids = [source['ids'][int(i)] for i in ids]
    graphs = [source['graphs'][int(i)].clone() for i in ids]
    info = dict(target_mode='original_opfdata')
    if method == 'LDM':
        graphs, info = synthesize(config, dict(arm='demand_input',data_seed=config['data_seed']),
            source, ids, {}, device, stop, folder)
        info.update(target_mode='solver')
    if method == 'ALC':
        rows = {}
        for name in config['committee']:
            if name == student:
                continue
            value = cached(teacher_folder(config, name), config['identity'])
            if value is None:
                raise BlockingIOError(f'Waiting for teacher {name}')
            if value['source_ids_sha256'] != digest(source['ids']):
                raise ValueError('Teacher and candidate pool disagree')
            rows[name] = value['prediction']
        graphs, info = correct_labels(config, source, ids, rows, student)
        if any(graph_digest(a,targets=False) != graph_digest(b,targets=False)
               for a,b in zip(graphs,[source['graphs'][int(i)] for i in ids])):
            raise ValueError('Label correction changed inputs')
    name = next(k for k,v in METHODS.items() if v == method)
    teachers = {k:config['teachers'][k] for k in config['committee'] if k != student} if method == 'ALC' else {}
    provenance = dict(protocol=config['identity'], data_seed=config['data_seed'],
        candidate_count=len(source['ids']), candidate_ids=source['ids'],
        candidate_ids_sha256=ids_hash(source['ids']), creation_split='train',
        teachers=teachers, excluded_student=student, eligible_students=[student] if student else config['committee'],
        parameters={k:config[k] for k in ('count','rff_features','scoring','novel','demand')}) | info
    label_access = dict(stored_pool_targets=len(source['ids']),
        targets_used_for_curation=len(source['ids']) if method in ('ITH','PIH','ALC') else 0,
        pretrained_teachers=len(teachers), pretrained_train_examples_per_teacher=len(dataset.splits(42)['train']) if teachers else 0,
        new_solver_calls=info.get('new_oracle_calls',0))
    if method in SELECTORS:
        if len({graph_digest(g,targets=False) for g in graphs}) != config['count']:
            raise ValueError('Selected inputs are not unique')
        selection = dict(complete=True,method=name,reference_ids=[], selected_ids=anchor_ids,
            data_identity=digest(dataset.manifest), split_hashes={k:ids_hash(v) for k,v in dataset.splits(42).items()},
            selection_seed=config['data_seed'], label_access=label_access, provenance=provenance)
        result = publish_selection(dataset, selection, folder, counts=(config['count'],), seed=config['data_seed'])
        if result['status'] != 'COMPLETE':
            raise BlockingIOError('Selection publication incomplete')
        return load_dataset(path)['metadata']
    metadata = dict(method=name,case=config['case'],initialization=None, initialization_name=None,
        generation_protocol='bounded_demand' if method == 'LDM' else 'fixed_real_inputs',
        target_mode=info['target_mode'], target_semantics=(
            'Fresh checked local AC-OPF solutions' if method == 'LDM' else
            'Bounded analytical teaching labels; AC-OPF feasibility is not guaranteed'),
        models=list(teachers), model_role='leave-one-architecture-out teachers' if teachers else 'none',
        provenance=provenance, label_access=label_access,
        data_sha256=config['data_sha256'], selection_seed=config['data_seed'])
    return publish(graphs, dataset, anchor_ids, path.parent, metadata)
