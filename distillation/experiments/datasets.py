"""Read-only dataset adapter and explicit unions; existing synthesis exports stay intact."""
import argparse
import hashlib
import json
from pathlib import Path

from distillation.training.T1_single_topo.common import digest, write_json

UNION = 'lumina-dataset-union-v1'


def cohort(metadata):
    return {k: metadata.get(k) for k in ('method', 'case', 'generation_protocol', 'target_mode',
        'initialization', 'initialization_name', 'data_identity', 'data_sha256')}


def load_dataset(path):
    path = Path(path)
    path = path/'artifact.json' if path.is_dir() else path
    metadata = json.loads(path.read_text())
    if metadata.get('format') == UNION:
        identity = metadata['identity']
        if (digest(identity) != metadata['sha256'] or identity['members'] != metadata['members'] or
                identity['count'] != metadata['count'] or identity['cohort'] != cohort(metadata)):
            raise ValueError('Union identity changed')
        graphs = []
        for member in metadata['members']:
            source = Path(member['artifact'])
            item = load_dataset(source if source.is_absolute() else path.parent/source)
            if (item['metadata']['sha256'] != member['sha256'] or len(item['graphs']) != member['count'] or
                    item['metadata']['dataset_id'] != member['dataset_id'] or cohort(item['metadata']) != identity['cohort']):
                raise ValueError('Union member changed')
            graphs.extend(item['graphs'])
        if len(graphs) != metadata['count']:
            raise ValueError('Union count changed')
        return dict(graphs=graphs, metadata=metadata)
    if metadata.get('format') == 'lumina-selected-dataset-v1':
        from distillation.selection.artifacts import load_dataset as reader
    elif metadata.get('format') == 'lumina-synthetic-dataset-v1':
        from distillation.synthetic.artifacts import load_dataset as reader
    elif metadata.get('format') == 'lumina-curated-dataset-v1':
        from distillation.curation.artifacts import load_dataset as reader
    else:
        raise ValueError('Unknown dataset artifact format')
    item = reader(path)
    # Embedded checkpoints and native curricula never become downstream weights/settings.
    return dict(graphs=item['graphs'], metadata=item['metadata'])


def graph_digest(graph, *, targets=True):
    """Exact SDK input/teaching-pair identity, excluding IDs and solver metadata."""
    h = hashlib.sha256()
    h.update(str(float(graph.baseMVA)).encode())
    for key in sorted(graph.node_types) + sorted(graph.edge_types):
        for name in ('x', 'edge_index', 'edge_attr', 'y'):
            if name == 'y' and not targets or name not in graph[key]:
                continue
            value = graph[key][name].detach().cpu().contiguous()
            h.update(str((key, name, tuple(value.shape), str(value.dtype))).encode())
            h.update(value.numpy().tobytes())
    return h.hexdigest()


def write_union(output, members, *, expected_count=800):
    """Reference explicit datasets from one method/case/regime; never rewrite them."""
    output = Path(output)
    from distillation.selection.artifacts import output_guard
    output_guard(output.parent)
    if not members:
        raise ValueError('Choose explicit member artifacts')
    if type(expected_count) is not int or expected_count < 1:
        raise ValueError('Expected count must be a positive integer')
    records, graphs, group, seen = [], [], None, set()
    for member in members:
        path = Path(member).resolve()
        path = path/'artifact.json' if path.is_dir() else path
        item = load_dataset(path); meta = item['metadata']
        if meta['format'] == UNION:
            raise ValueError('Use original datasets as union members')
        current = cohort(meta)
        if group is not None and current != group:
            raise ValueError('Union members must share method, case, target and initialization regime')
        group = current
        source = meta.get('source_generation', {}).get('sha256', meta['sha256'])
        if source in seen:
            raise ValueError('The same dataset/library cannot count twice')
        seen.add(source)
        records.append(dict(artifact=str(path), dataset_id=meta['dataset_id'], sha256=meta['sha256'],
                            count=len(item['graphs']), synthesis_models=meta.get('synthesis_models', [])))
        graphs.extend(item['graphs'])
    if len(graphs) != expected_count:
        raise ValueError(f'Expected {expected_count} examples, received {len(graphs)}; choose explicit additional datasets')
    identity = dict(cohort=group, members=records, count=len(graphs))
    metadata = dict(format=UNION, dataset_id='UNION-'+digest(identity)[:16], sha256=digest(identity),
        identity=identity, **group, members=records, count=len(graphs), complete=True,
        unique_input_graphs=len({graph_digest(g, targets=False) for g in graphs}),
        unique_teaching_pairs=len({graph_digest(g) for g in graphs}),
        policy='concatenate all examples without deduplication; load teaching targets unchanged')
    if output.exists():
        if json.loads(output.read_text()) != metadata:
            raise ValueError('Union manifest exists with a different identity')
    else:
        write_json(output, metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('members', nargs='+', type=Path)
    parser.add_argument('--count', type=int, default=800)
    args = parser.parse_args()
    record = write_union(args.output, args.members, expected_count=args.count)
    print(json.dumps({k: record[k] for k in ('dataset_id', 'count', 'unique_input_graphs', 'unique_teaching_pairs')}))


if __name__ == '__main__':
    main()
