"""Portable edited teaching pairs, with the same identity contract as other exports."""
from pathlib import Path

import torch

from distillation.experiments.datasets import graph_digest
from distillation.selection.artifacts import output_guard
from distillation.selection.pool import ids_hash
from distillation.training.T1_single_topo.common import claim, digest, file_hash, read_json, write_json
from distillation.training.T1_single_topo.state import save_torch

FORMAT = 'lumina-curated-dataset-v1'


def publish(graphs, dataset, ids, folder, metadata):
    folder = Path(folder); output_guard(folder)
    ids = list(map(int, ids)); splits = dataset.splits(42)
    if len(graphs) != len(ids) or len(set(ids)) != len(ids) or not set(ids) <= set(splits['train']):
        raise ValueError('Unique training anchors and one graph per anchor required')
    graphs = [g.clone().detach().cpu() for g in graphs]
    for graph in graphs:
        for kind in ('bus', 'generator'):
            if graph[kind].y.shape != (graph[kind].num_nodes, 2) or not torch.isfinite(graph[kind].y).all():
                raise ValueError('Invalid teaching targets')
        for store in graph.stores:
            for name in ('x', 'edge_attr'):
                if name in store and not torch.isfinite(store[name]).all():
                    raise ValueError('Nonfinite teaching inputs')
    if len({graph_digest(g, targets=False) for g in graphs}) != len(graphs):
        raise ValueError('Repeated inputs do not count toward M100')
    identity = dict(metadata, anchor_ids=ids, data_identity=digest(dataset.manifest), count=len(graphs),
        split=dict(split_seed=42, split_hashes={k:ids_hash(v) for k,v in splits.items()}, test_accessed=False),
        graph_hashes=[graph_digest(g) for g in graphs])
    meta = dict(identity, format=FORMAT, identity=identity, dataset_id='CURATE-'+digest(identity)[:16],
        complete=True, status='COMPLETE', dataset='dataset.pt',
        targets={'bus':['va (radians)', 'vm (per unit)'], 'generator':['pg (per unit)', 'qg (per unit)']},
        feature_encoding='SDK processed HeteroData',
        training_recipe='Independent downstream training; no creation weights or optimizer state')
    with claim(folder/'publish.lock') as acquired:
        if not acquired:
            raise BlockingIOError('Artifact publication in progress')
        if (folder/'artifact.json').exists():
            actual = load_dataset(folder)['metadata']
            if actual['identity'] != identity:
                raise ValueError('Frozen curated artifact changed')
            return actual
        save_torch(folder/'dataset.pt', dict(format=FORMAT, graphs=graphs, identity=identity))
        meta['sha256'] = file_hash(folder/'dataset.pt')
        write_json(folder/'artifact.json', meta)
    return meta


def load_dataset(path):
    path = Path(path); path = path/'artifact.json' if path.is_dir() else path
    meta = read_json(path); identity = meta['identity']
    if (meta.get('format') != FORMAT or not meta.get('complete') or
            meta['dataset_id'] != 'CURATE-'+digest(identity)[:16] or
            any(meta.get(k) != v for k,v in identity.items())):
        raise ValueError('Curated artifact identity changed')
    payload_path = path.parent/meta['dataset']
    if file_hash(payload_path) != meta['sha256']:
        raise ValueError('Curated dataset checksum changed')
    value = torch.load(payload_path, map_location='cpu', weights_only=False)
    if (value['format'] != FORMAT or value['identity'] != identity or len(value['graphs']) != meta['count'] or
            [graph_digest(g) for g in value['graphs']] != identity['graph_hashes']):
        raise ValueError('Curated graph/target identity changed')
    return dict(graphs=value['graphs'], metadata=meta)
