"""Disposable, verified node-local inputs; canonical paths remain in run identities."""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


def assigned(experiments, node, nodes):
    """Keep all evaluations of one RAMBO library together; other runs stay independent."""
    groups = {}
    for index, row in enumerate(experiments):
        key = ('RAMBO', row['config']['generation_id']) if row.get('method') == 'RAMBO' else ('run', index)
        groups.setdefault(key, []).append(row)
    # When a second assignment is necessary, prefer uniform generation over RAMBO's bilevel solves.
    ordered = sorted(groups.values(), key=lambda group:
        group[0].get('method') == 'RAMBO' and group[0]['config'].get('generation_protocol') == 'uniform')
    return [row for group in ordered[node::nodes] for row in group]


def signature(path):
    stat = Path(path).stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@lru_cache(maxsize=4)
def _mapping(path, stamp):
    return json.loads(Path(path).read_text())['paths']


def local_path(path):
    """Resolve only explicitly staged immutable inputs, never output directories."""
    mapping = os.environ.get('STUDY_INPUT_MAP')
    original = Path(path).expanduser().resolve()
    return Path(_mapping(mapping, signature(mapping)).get(str(original), str(original))) if mapping else original


@lru_cache(maxsize=6)
def _dataset(folder, stamp):
    from distillation.training.T1_single_topo.data import CaseDataset
    path = Path(folder)
    return CaseDataset(path.parent, path.name)


def case_dataset(root, case):
    """Reuse mmap metadata within a worker; changed files still invalidate the cache."""
    from distillation.training.T1_single_topo.common import canonical_case
    folder = local_path(Path(root) / canonical_case(case))
    manifest = folder / 'manifest.json'
    records = json.loads(manifest.read_text())['groups']
    stamp = (signature(manifest), *(signature(folder / f"group_{r['group']}.pt") for r in records))
    return _dataset(str(folder), stamp)


def stage(experiments, destination):
    """Called by one rank per node before readers start. Copy each input only once."""
    from distillation.training.T1_single_topo.common import canonical_case, file_hash, write_json
    from distillation.data.split import split_indices
    started = time.monotonic()
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    paths, files = {}, {}

    def add(source, target, expected, size=None, mtime=None):
        source = Path(source).expanduser().resolve()
        value = (target, expected, size, mtime)
        if source in files and files[source] != value:
            raise ValueError(f'Conflicting staged input: {source}')
        files[source] = value

    def add_case(cfg, case, expected):
        case = canonical_case(case)
        source = Path(cfg['data_root']).expanduser().resolve() / case
        token = hashlib.sha256(str(source.parent).encode()).hexdigest()[:16]
        target = destination / 'data' / token / case
        manifest = source / 'manifest.json'
        if file_hash(manifest) != expected:
            raise ValueError(f'Data manifest changed: {manifest}')
        records = json.loads(manifest.read_text())
        paths[str(source)] = str(target)
        add(manifest, target / manifest.name, expected)
        for record in records['groups']:
            name = f"group_{record['group']}.pt"
            add(source / name, target / name, record['sha256'], record['bytes'], record['mtime_ns'])
        return records

    for row in experiments:
        cfg = row['config']
        case = canonical_case(f"case{cfg['case']}")
        records = add_case(cfg, case, cfg['artifact_identity']['data_sha256'])
        for source_case, expected in cfg.get('retention_data', {}).items():
            add_case(cfg, source_case, expected)
        inputs = [(cfg['reference_file'], cfg['artifact_identity']['reference_sha256'])]
        if cfg.get('pretrained_checkpoint'):
            inputs.append((cfg['pretrained_checkpoint'], cfg['pretrained']['sha256']))
        for kind, path in cfg.get('pretrained_checkpoints', {}).items():
            inputs.append((path, cfg['pretrained'][kind]['sha256']))
        if row['method'] == 'RAMBO':
            index = int(split_indices(case, records['count'], 42)['train'][0])
            group = index // records['group_size']
            inputs.append((str(Path(cfg['raw_root']) / f'{case}_{group}.tar.gz'),
                           records['groups'][group]['raw_sha256']))
        for path, checksum in inputs:
            path = Path(path).expanduser().resolve()
            target = destination / 'files' / checksum / path.name
            add(path, target, checksum)
            paths[str(path)] = str(target)

    required = sum(source.stat().st_size for source in files)
    if required > shutil.disk_usage(destination).free:
        raise OSError(f'Insufficient node-local space for {required} input bytes')
    for source, (target, expected, size, mtime) in files.items():
        before = signature(source)
        if size is not None and before[:2] != (size, mtime):
            raise ValueError(f'Data shard changed: {source}')
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(target.name + '.tmp')
        try:
            shutil.copy2(source, temp)  # Preserve the timestamps pinned by CaseDataset.
            if file_hash(temp) != expected or signature(source) != before:
                raise ValueError(f'Staged input checksum changed: {source}')
            temp.replace(target)
        finally:
            temp.unlink(missing_ok=True)
    report = dict(paths=paths, files=len(files), bytes=required, seconds=time.monotonic()-started,
                  datasets=[r['dataset_id'] for r in experiments])
    write_json(destination / 'inputs.json', report)  # Publish only after every copy is verified.
    return report
