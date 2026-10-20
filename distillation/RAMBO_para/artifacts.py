"""Portable RAMBO* datasets, with a separate identity from sequential RAMBO."""
import json
from pathlib import Path
import re

import torch

from distillation.selection.generated import GeneratedDataset
from distillation.training.T1_single_topo.common import file_hash, write_json
from distillation.training.T1_single_topo.state import save_torch

METHOD = 'RAMBO_para'
DISPLAY = 'RAMBO*'
PROTOCOL = 'frozen_history_batches_attempt_order_v1'


def export_dataset(folder):
    folder = Path(folder)
    generated = GeneratedDataset(folder / 'dataset.json')
    manifest, cfg = generated.manifest, generated.manifest['config']
    status = json.loads((folder / 'status.json').read_text())
    if manifest['method'] != METHOD or status['status'] != 'complete' or len(generated) != cfg['count']:
        raise ValueError('Only a complete RAMBO_para library can be published')
    metadata_path, manifest_sha = folder / 'artifact.json', file_hash(folder / 'dataset.json')
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if (metadata['method'] != METHOD or metadata['source_generation']['sha256'] != manifest_sha or
                metadata['sha256'] != file_hash(folder / 'dataset.pt') or
                metadata['recipe_sha256'] != file_hash(folder / 'recipe.json')):
            raise ValueError('Published RAMBO_para artifact changed')
        return metadata
    physical = list(generated)
    checks = [r['checks'] | {'success': bool(r['checks']['feasible'])} for r in generated.records]
    if not all(r['success'] for r in checks):
        raise ValueError('Cannot publish unverified physical labels')
    identity = METHOD + '-' + manifest['specification'][:16]
    source = dict(id=identity, manifest=str((folder / 'dataset.json').resolve()), sha256=manifest_sha,
                  statistics=status, bootstrap=manifest.get('bootstrap'))
    config = cfg | dict(strength=.1, target_mode='solver', split_seed=42,
                        synthetic_count=len(generated), model_precision='float64')
    save_torch(folder / 'dataset.pt', dict(format='rambo-para-opf-v1', method=METHOD, config=config,
        target_mode='solver', training_state=dict(status='COMPLETE'), physical_solved_graphs=physical,
        solver_checks=checks, teaching_targets={k: torch.cat([g[k].y for g in physical])
                                               for k in ('bus', 'generator')},
        generated_ids=[r['id'] for r in generated.records], source_generation=source))
    sha = file_hash(folder / 'dataset.pt')
    write_json(folder / 'recipe.json', dict(format='lumina-synthetic-recipe-v1', dataset_sha256=sha,
        models={}, ordinary=None, native=None, note='Data only; use a separate matched downstream training protocol.'))
    counts = [int(g['bus'].num_nodes) for g in physical]
    metadata = dict(format='lumina-synthetic-dataset-v1', dataset_id=identity, generation_id=identity,
        method=METHOD, display_name=DISPLAY, kind='generated_operating_points', status='COMPLETE', complete=True,
        evaluation_complete=False, case=int(re.search(r'case(\d+)', cfg['case'])[1]), count=len(generated),
        synthesis_models=[], evaluation_models=[], initialization=None, generation_protocol=PROTOCOL,
        proposal_batch_size=cfg['proposal_batch_size'], load_bounds=cfg['load_bounds'], synthesis_seed=cfg['seed'],
        target_mode='solver', target_semantics='verified final Julia/Ipopt AC-OPF solutions, preserved unchanged',
        dataset='dataset.pt', sha256=sha, export_format='rambo-para-opf-v1', config='config.json',
        recipe='recipe.json', recipe_sha256=file_hash(folder / 'recipe.json'), resume='state.json',
        source_generation=source, dataset_selection='verified candidates in attempt order; no validation selection',
        bus_counts=counts, bus_limits=[min(counts), max(counts)], physical_audit_passed=True,
        feature_encoding='synthetic.learning.benchmark_graph; SDK OPF feature ordering',
        targets={'bus': ['va (radians)', 'vm (per unit)'], 'generator': ['pg (per unit)', 'qg (per unit)']},
        split=dict(split_seed=42, split_hashes=manifest['provenance'].get('split_hashes', {}),
                   data_identity=manifest['provenance']['data_identity'], validation_accessed=False, test_accessed=False),
        provenance=manifest['provenance'])
    write_json(metadata_path, metadata)
    return metadata
