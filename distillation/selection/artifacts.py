"""Portable, immutable subsets for any train-only selection policy."""
import json
from pathlib import Path

import numpy as np
import torch

from .pool import TrainingPool, ids_hash
from distillation.training.T1_single_topo.common import claim, digest, file_hash, write_json
from distillation.training.T1_single_topo.state import save_torch

FORMAT = 'lumina-selected-dataset-v1'


def output_guard(root):
    """Keep new selection/union writes outside historical synthesis campaigns."""
    root = Path(root).resolve()
    for parent in (root, *root.parents):
        path = parent/'manifest.json'
        if path.exists():
            value = json.loads(path.read_text())
            if 'experiments' in value and value.get('format') != 'lumina-selection-campaign-v1':
                raise ValueError('Use a separate selection directory; existing campaign is read-only')


def subset_order(ids, seed, strata=None):
    """Nested random subsets, or balanced prefixes of shuffled demand buckets."""
    ids = np.asarray(ids, dtype=np.int64)
    rng = np.random.default_rng(seed)
    if strata is None:
        return rng.permutation(ids).tolist()
    strata = np.asarray(strata)
    if strata.shape != ids.shape:
        raise ValueError('One stratum is required per selected ID')
    buckets = [rng.permutation(ids[strata == k]).tolist() for k in np.unique(strata)]
    order = []
    while buckets:
        for index in rng.permutation(len(buckets)):
            order.append(int(buckets[index].pop()))
        buckets = [b for b in buckets if b]
    return order


def publish(dataset, selection, folder, *, counts=(100, 800), seed=42, strata=None):
    """Freeze one completed selection pool and publish independently usable subsets.

    Policies supply IDs/provenance; this function never trains, scores or solves.
    Underfilled sizes are reported without silently adding or duplicating examples.
    """
    folder = Path(folder)
    output_guard(folder)
    if not selection.get('complete'):
        raise ValueError('Finish selection before publishing its subsets')
    if not counts or any(type(n) is not int or n < 1 for n in counts) or len(set(counts)) != len(counts):
        raise ValueError('Counts must be unique positive integers')
    splits = dataset.splits(42)
    pool = TrainingPool(dataset, splits, selection['reference_ids'])
    pool.acquire(selection['selected_ids'], reason='export')
    if selection['data_identity'] != digest(dataset.manifest):
        raise ValueError('Selection source data changed')
    split_hashes = {k: ids_hash(v) for k, v in splits.items()}
    if selection['split_hashes'] != split_hashes:
        raise ValueError('Selection splits changed')
    order = subset_order(pool.selected_ids, seed, strata)
    rule = 'balanced demand buckets, nested prefixes' if strata is not None else 'seeded permutation of complete selected pool'
    frozen = dict(format='lumina-selection-export-v1', selection=selection,
                  counts=sorted(counts), export_seed=seed, subset_rule=rule, ordered_ids=order)
    with claim(folder/'export.lock') as acquired:
        if not acquired:
            return dict(status='BUSY', datasets=[])
        path = folder/'pool.json'
        if path.exists() and json.loads(path.read_text()) != frozen:
            raise ValueError('Frozen selection/export settings changed; use a new directory')
        if not path.exists():
            write_json(path, frozen)
        records = []
        for count in sorted(counts):
            destination = folder/f'M{count}'
            if count > len(order):
                records.append(dict(count=count, status='UNDERFILLED', available=len(order), ready=False))
                continue
            ids = order[:count]
            identity = dict(pool_sha256=file_hash(path), ids=ids, count=count)
            metadata_path = destination/'artifact.json'
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text())
                if metadata.get('identity') != identity:
                    raise ValueError('Published selection identity changed')
                load_dataset(destination)
            else:
                graphs = [dataset[i].clone().detach().cpu() for i in ids]
                for graph in graphs:
                    for kind in ('bus', 'generator'):
                        if not torch.isfinite(graph[kind].y).all():
                            raise ValueError('Nonfinite source targets')
                save_torch(destination/'dataset.pt', dict(format=FORMAT, graphs=graphs,
                    selected_ids=ids, data_identity=selection['data_identity'], split_hashes=split_hashes))
                initial = set(selection.get('initial_ids', []))
                metadata = dict(format=FORMAT, dataset_id='SELECT-'+digest(identity)[:16],
                    identity=identity, method=selection['method'], kind='selected_real',
                    case=int(dataset.manifest['case'].split('case')[1].split('_')[0]),
                    count=count, status='COMPLETE', complete=True, dataset='dataset.pt',
                    sha256=file_hash(destination/'dataset.pt'), selected_ids=ids,
                    selected_ids_sha256=ids_hash(ids), data_identity=selection['data_identity'],
                    split=dict(split_seed=42, split_hashes=split_hashes, test_accessed=False),
                    target_mode='original_opfdata', target_semantics='unchanged OPFData bus and generator targets',
                    feature_encoding='original SDK processed HeteroData; no conversion or re-solving',
                    targets={'bus':['va (radians)', 'vm (per unit)'],
                             'generator':['pg (per unit)', 'qg (per unit)']},
                    selector=selection.get('selector'), selection_seed=selection['selection_seed'],
                    export_seed=seed, subset_rule=rule, selected_pool_count=len(order),
                    reserved_panel_count=len(pool.reference_ids), label_access=selection['label_access'],
                    export_label_count=count, acquisition_events=selection.get('acquisition_events', 0),
                    initial_examples=sum(i in initial for i in ids),
                    source_selection=dict(path=str(path.resolve()), sha256=identity['pool_sha256']),
                    provenance=selection.get('provenance', {}), solver_calls=0,
                    training_recipe='Separate downstream experiment; initialize fresh student weights')
                write_json(metadata_path, metadata)
            records.append(dict(count=count, status='COMPLETE', ready=True,
                dataset_id=metadata['dataset_id'], artifact=str(metadata_path.resolve()), sha256=metadata['sha256']))
        summary = dict(status='COMPLETE' if all(r['ready'] for r in records) else 'UNDERFILLED',
                       selected_pool_count=len(order), datasets=records)
        write_json(folder/'catalog.json', summary)
        return summary


def load_dataset(path):
    """Read a portable selected dataset without the original OPFData directory."""
    path = Path(path)
    metadata_path = path/'artifact.json' if path.is_dir() else path
    metadata = json.loads(metadata_path.read_text())
    if metadata.get('format') != FORMAT or not metadata.get('complete'):
        raise ValueError('Expected a completed portable selection artifact')
    source = metadata_path.parent/metadata['dataset']
    if file_hash(source) != metadata['sha256']:
        raise ValueError('Selected dataset checksum changed')
    payload = torch.load(source, map_location='cpu', weights_only=False)
    ids = payload['selected_ids']
    if (payload['format'] != FORMAT or len(payload['graphs']) != metadata['count'] or
            len(ids) != len(set(ids)) or ids != metadata['selected_ids'] or
            ids_hash(ids) != metadata['selected_ids_sha256'] or
            payload['data_identity'] != metadata['data_identity'] or
            payload['split_hashes'] != metadata['split']['split_hashes']):
        raise ValueError('Selected dataset identity/count changed')
    return dict(graphs=payload['graphs'], metadata=metadata)
