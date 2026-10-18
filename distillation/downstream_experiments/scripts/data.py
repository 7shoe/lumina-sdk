"""Frozen corpus recipes; all published dataset paths come from dataset_path.py."""
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from distillation.data.split import split_indices
from distillation.downstream_experiments.dataset_path import CATALOG, dataset_record
from distillation.experiments.datasets import graph_digest, load_dataset
from distillation.experiments.staging import case_dataset
from distillation.training.T1_single_topo.common import (
    canonical_case, claim, digest, file_hash, read_json, write_json)
from distillation.training.T1_single_topo.state import save_torch


def case_name(case):
    return canonical_case(f'case{case}' if isinstance(case, int) else case)


@lru_cache(maxsize=64)
def data_identity(root, case):
    path = Path(root) / case_name(case) / 'manifest.json'
    manifest = read_json(path)
    return dict(manifest=str(path.resolve()), sha256=file_hash(path),
                identity=digest(manifest), count=manifest['count'])


def quotas(total, keys, seed):
    """Uniform creator/case allocation, with a reproducible remainder."""
    keys = sorted(keys)
    if not keys or total < len(keys):
        raise ValueError('Need at least one example per contributor')
    values = dict.fromkeys(keys, total // len(keys))
    for key in np.random.default_rng(seed).permutation(keys)[:total % len(keys)]:
        values[key] += 1
    return values


def artifact_member(record, count, seed):
    path = Path(record['artifact_path'])
    meta = read_json(path)
    if (not meta.get('complete') or meta.get('dataset_id') != record['dataset_id']
            or meta['sha256'] != record['sha256'] or meta['count'] != record['count']):
        raise ValueError(f'Catalog/artifact mismatch: {path}')
    if (meta.get('case') != record['case'] or meta.get('target_mode') != record['target_mode']
            or meta.get('method','').lower() != record['method'].lower()
            or meta.get('initialization') != record['initialization']):
        raise ValueError(f'Catalog scientific metadata mismatch: {path}')
    payload = (path.parent / meta['dataset']).resolve()
    if payload != Path(record['data_path']).resolve():
        raise ValueError(f'Catalog payload path mismatch: {path}')
    if count > record['count']:
        raise ValueError(f"Need {count}, only {record['count']} available: {record['dataset_id']}")
    order = np.random.default_rng(seed).permutation(record['count'])[:count].tolist()
    return dict(kind='artifact', dataset_id=record['dataset_id'], artifact_path=str(path.resolve()),
                artifact_sha256=file_hash(path), data_path=str(payload), sha256=record['sha256'],
                case=record['case'], creator=record['reference_model'], indices=order,
                count=count, target_mode=record['target_mode'], initialization=record['initialization'],
                generation=record['generation'], initialization_name=record['initialization_name'])


def make_corpus(ids, count, *, data_root, seed=777, catalog=CATALOG, random_cases=()):
    """Balance cases, then creators. RND samples the entire canonical train split."""
    records = [dataset_record(i, catalog=catalog) for i in ids]
    if bool(records) == bool(random_cases):
        raise ValueError('Choose published dataset IDs or real random controls')
    members = []
    if random_cases:
        for case, n in quotas(count, random_cases, seed).items():
            info = data_identity(str(data_root), int(case))
            train = split_indices(case_name(int(case)), info['count'], 42)['train']
            if n > len(train):
                raise ValueError('Random subset exceeds the training population')
            ids = np.random.default_rng(seed).permutation(train)[:n].tolist()
            members.append(dict(kind='real', case=int(case), indices=ids, count=n,
                                creator='none', data=info, target_mode='original_opfdata'))
        method = 'RND777'
    else:
        if len({r['acronym'] for r in records}) != 1 or len({r['dataset_id'] for r in records}) != len(records):
            raise ValueError('Use unique datasets from one method')
        method = records[0]['acronym']
        groups = defaultdict(list)
        for row in records:
            groups[row['case']].append(row)
        for case, n in quotas(count, groups, seed).items():
            rows = groups[case]
            if len({r['reference_model'] for r in rows}) != len(rows):
                raise ValueError('Choose exactly one artifact per creator and case')
            regimes = {(r['generation'], r['target_mode'], r['initialization'], r['initialization_name']) for r in rows}
            if len(regimes) != 1:
                raise ValueError('Do not mix synthesis regimes within a case')
            allocation = quotas(n, [r['reference_model'] for r in rows], seed)
            for row in sorted(rows, key=lambda r:r['reference_model']):
                members.append(artifact_member(row, allocation[row['reference_model']], seed))
    cases = sorted({m['case'] for m in members})
    identity = dict(format='lumina-downstream-corpus-v1', method=method, count=count, cases=cases,
                    sample_seed=seed, members=members, split_seed=42,
                    data={str(c):data_identity(str(data_root), c) for c in cases})
    return identity | {'corpus_id':'CORPUS-'+digest(identity)[:16]}


def check_corpus(recipe):
    identity = {k:v for k,v in recipe.items() if k != 'corpus_id'}
    if recipe['corpus_id'] != 'CORPUS-'+digest(identity)[:16]:
        raise ValueError('Corpus recipe identity changed')
    for info in recipe['data'].values():
        if file_hash(info['manifest']) != info['sha256']:
            raise ValueError('Real data manifest changed')
    for member in recipe['members']:
        if len(member['indices']) != len(set(member['indices'])) or len(member['indices']) != member['count']:
            raise ValueError('Repeated or inconsistent member indices')
        if member['kind'] == 'artifact' and file_hash(member['artifact_path']) != member['artifact_sha256']:
            raise ValueError('Source artifact metadata changed')


def materialize(recipe, root, data_root):
    """Publish a small portable graph list once; never modify source exports."""
    check_corpus(recipe)
    folder = Path(root) / recipe['corpus_id']
    with claim(folder / 'corpus.lock') as acquired:
        if not acquired:
            raise BlockingIOError(f'Another worker is assembling {folder}')
        manifest = folder / 'manifest.json'
        if manifest.exists():
            meta = read_json(manifest)
            if meta['recipe'] != recipe or file_hash(folder / 'dataset.pt') != meta['sha256']:
                raise ValueError('Published downstream corpus changed')
            return torch.load(folder / 'dataset.pt', weights_only=False, map_location='cpu'), meta
        graphs, cases = [], []
        for member in recipe['members']:
            case = member['case']
            train = set(split_indices(case_name(case), recipe['data'][str(case)]['count'], 42)['train'].tolist())
            if member['kind'] == 'real':
                if not set(member['indices']) <= train:
                    raise ValueError('Random control contains a holdout ID')
                source = case_dataset(data_root, case_name(case))
            else:
                loaded = load_dataset(member['artifact_path'])
                if loaded['metadata']['sha256'] != member['sha256']:
                    raise ValueError('Source dataset checksum changed')
                original = loaded['metadata']
                info = recipe['data'][str(case)]
                if ((original.get('data_identity') and original['data_identity'] != info['identity'])
                        or (original.get('data_sha256') and original['data_sha256'] != info['sha256'])):
                    raise ValueError('Dataset was produced from a different real-data release')
                # Selection artifacts carry exact OPFData IDs; audit them against the full train split.
                selected = loaded['metadata'].get('selected_ids')
                if selected is not None and not set(selected) <= train:
                    raise ValueError('Selected dataset contains a holdout ID')
                source = loaded['graphs']
            for i in member['indices']:
                graph = source[i].clone().detach().cpu()
                for kind in ('bus', 'generator'):
                    if not torch.isfinite(graph[kind].y).all():
                        raise ValueError('Nonfinite teaching targets')
                graphs.append(graph); cases.append(case)
        if len(graphs) != recipe['count']:
            raise ValueError('Corpus count mismatch')
        payload = dict(graphs=graphs, cases=cases)
        save_torch(folder / 'dataset.pt', payload)
        meta = dict(recipe=recipe, sha256=file_hash(folder / 'dataset.pt'),
                    count=len(graphs), cases=dict(Counter(cases)),
                    unique_inputs=len({graph_digest(g, targets=False) for g in graphs}),
                    unique_teaching_pairs=len({graph_digest(g) for g in graphs}))
        write_json(manifest, meta)
        return payload, meta
