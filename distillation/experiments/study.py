"""Resolve DD/DCGM/HMDC/RAMBO studies into frozen configurations, jobs and catalogs."""
import argparse
from collections import Counter
from copy import deepcopy
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import yaml

from distillation.experiments.comparison import dd, dc, hm
from distillation.synthetic.learning import model_options, pretrained_config
from distillation.HMDC.models import pretrained_pair_config
from distillation.training.T1_single_topo.common import atomic_file, claim, digest, file_hash, write_json
from experiments.lumina_bench.models import SUPPORTED_MODELS

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = Path(__file__).with_name('configs')/'condensation.yaml'
FACTORIES = {'DD': dd.default_config, 'DCGM': dc.default_config, 'HMDC': hm.default_config}
METHODS = (*FACTORIES, 'RAMBO')


def merge(base, extra):
    result = deepcopy(base)
    for key, value in extra.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
    return result


def keys(value, allowed, where):
    if not isinstance(value, dict) or set(value)-set(allowed):
        raise ValueError(f'Unknown fields or invalid mapping in {where}: {value}')


def load(path=DEFAULT, profile='reference'):
    value = yaml.safe_load(Path(path).read_text())
    keys(value, ['schema_version','campaign','paths','matrix','shared','model_defaults','bus_limits','initialization',
                 'methods','evaluation','execution','profiles'], 'study')
    if value['schema_version'] != 1 or not re.fullmatch(r'[A-Za-z0-9_-]+', value['campaign']):
        raise ValueError('Use schema_version 1 and a simple campaign name')
    profiles = value.pop('profiles', {'reference': {}})
    if profile not in profiles: raise ValueError(f'Unknown profile: {profile}')
    keys(profiles[profile], ['matrix','shared','model_defaults','bus_limits','initialization','methods','evaluation',
                            'execution','extra_conditions'], 'profile')
    value = merge(value, profiles[profile])
    keys(value['paths'], ['output','models','data','raw','references','initializations','generated'], 'paths')
    keys(value['matrix'], ['cases','protocols','target_modes','initializations','seeds'], 'matrix')
    keys(value['model_defaults'], ['width','layers'], 'model_defaults')
    keys(value['execution'], ['layout','nodes','ranks_per_node','account','queue','walltime',
                              'allocation_seconds','activate','staging','copper','verify_run',
                              'max_retries','stall_seconds','minimum_attempt_seconds','launch_interval',
                              'hmdc_microbatch'], 'execution')
    for name in ('staging', 'copper', 'verify_run'):
        if type(value['execution'].get(name, False)) is not bool: raise ValueError(f'execution.{name} must be boolean')
    if value['execution']['ranks_per_node'] != 12: raise ValueError('Use exactly 12 ranks per node')
    if value['execution']['layout'] not in ('combined','separate'): raise ValueError('Unknown execution layout')
    for name, minimum in [('max_retries',0),('hmdc_microbatch',1)]:
        item = value['execution'].get(name, 1)
        if type(item) is not int or item < minimum: raise ValueError(f'Invalid execution.{name}')
    for name in ('stall_seconds','minimum_attempt_seconds','launch_interval'):
        item = value['execution'].get(name, 1.)
        if type(item) not in (int,float) or not 0 <= item < float('inf'):
            raise ValueError(f'Invalid execution.{name}')
    for method, settings in value['methods'].items():
        if method not in METHODS: raise ValueError(f'Unknown method: {method}; DC is named DCGM in manifests')
        keys(settings, ['models','domains','samplers','settings'] if method == 'RAMBO' else
             ['pairs','settings','evaluators'] if method == 'HMDC' else ['models','settings'], method)
        if method == 'RAMBO':
            if not isinstance(settings.get('domains'), dict) or not settings['domains']:
                raise ValueError('RAMBO needs named load domains')
            if any(not re.fullmatch(r'[A-Za-z0-9_-]+', name) for name in settings['domains']):
                raise ValueError('Invalid RAMBO domain name')
    for name, spec in value['initialization'].items():
        if not re.fullmatch(r'[A-Za-z0-9_-]+',name): raise ValueError('Invalid initialization name')
        keys(spec, ['mode','model_seed','checkpoints','checkpoint_pattern'], f'initialization.{name}')
        if spec['mode'] not in ('random','fixed','pretrained'): raise ValueError('Invalid initialization mode')
        if spec['mode'] != 'pretrained' and (spec.get('checkpoints') or spec.get('checkpoint_pattern')):
            raise ValueError('Checkpoint sources require pretrained initialization')
    value['profile'] = profile
    return value


def source_files(root=ROOT):
    """Include runtime dependencies and launcher; omit results, references and auxiliary programs."""
    result = {}
    for parent in ('lumina','experiments/lumina_bench','distillation'):
        for folder, directories, files in os.walk(root/parent):
            directories[:] = sorted(d for d in directories if d not in {
                'aux','test','tests','runs','reports','__pycache__','.git','mnist_mvp','configs'})
            for name in sorted(files):
                path = Path(folder)/name
                if path.suffix in ('.py','.pbs') or (
                        'RAMBO' in path.parts and path.suffix in ('.yaml','.jl','.toml','.m')):
                    result[str(path.relative_to(root))] = file_hash(path)
    path = root/'experiments/__init__.py'
    if path.exists(): result[str(path.relative_to(root))] = file_hash(path)
    return result


def retention_inputs(config):
    """Pin source-case data used only for final pretrained-retention reporting."""
    count = config.get('retention_samples', 0)
    if type(count) is not int or count < 0: raise ValueError('retention_samples must be a nonnegative integer')
    if not count or config.get('initialization') != 'pretrained': return config
    sources = [config['pretrained']] if 'model' in config else list(config['pretrained'].values())
    cases = set()
    for source in sources:
        spec = source['specification']
        cases.update(spec.get('cases', [spec['case']] if 'case' in spec else []))
    if not cases: raise ValueError('Pretrained retention requires recorded source cases')
    return config | {'retention_data': {dd.canonical_case(case): file_hash(
        Path(config['data_root'])/dd.canonical_case(case)/'manifest.json') for case in sorted(cases)}}


def expand(spec, source_hashes=None):
    """Pure expansion apart from reading reference/data/checkpoint identities. Never writes."""
    source_hashes = source_files() if source_hashes is None else source_hashes
    root = Path(spec['paths']['output']).expanduser().resolve()/spec['campaign']/spec['profile']
    cases = {}
    def case_info(case):
        if case not in cases:
            reference = Path(spec['paths']['references']).expanduser().resolve()/f'case{case}.pt'
            data = Path(spec['paths']['data']).expanduser().resolve()/dd.canonical_case(f'case{case}')/'manifest.json'
            cases[case] = dict(reference_source=str(reference), reference_sha256=file_hash(reference),
                               data_manifest=str(data), data_sha256=file_hash(data))
        return cases[case]
    groups = [spec['matrix'] | {'methods': list(spec['methods'])}] + spec.get('extra_conditions', [])
    rows = {}
    allowed = set(dd.default_config(30,'GIN')) | set(dc.default_config(30,'GIN')) | set(hm.default_config(30,['GIN','GAT']))
    allowed |= {'generation_strength','generation_max_edits','min_topologies',
                'training_model_seed','profile_real_batches','query_seed','query_selection_seed','real_training_seed','alignment_seed'}
    keys(spec['shared'], allowed, 'shared')
    keys(spec['evaluation'], allowed, 'evaluation')
    for group in groups:
        keys(group, ['cases','protocols','target_modes','initializations','seeds','methods','models','pairs','domains','samplers','settings'], 'condition matrix')
        dimensions = [group[k] for k in ('methods','cases','protocols','target_modes','initializations','seeds')]
        if any(not isinstance(v,list) or not v or len({str(x) for x in v}) != len(v) for v in dimensions):
            raise ValueError('Matrix dimensions must be nonempty unique lists')
        if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in group['seeds']):
            raise ValueError('Invalid dataset seed')
        if not set(group['methods']) <= set(spec['methods']): raise ValueError('Missing method settings')
        if 'RAMBO' in group['methods']:
            from .rambo import expand as expand_rambo
            rows.update({row['dataset_id']: row for row in expand_rambo(spec, root, source_hashes, group, case_info)})
        dimensions[0] = [method for method in group['methods'] if method != 'RAMBO']
        for method, case, protocol, target, regime, seed in itertools.product(*dimensions):
            if method not in spec['methods']: raise ValueError(f'Missing method settings: {method}')
            if type(case) is not int or case not in (14,30,57,118,500): raise ValueError('Unsupported case')
            if protocol not in ('conservative','random_fixed','random_size'): raise ValueError('Unknown generation protocol')
            if target not in ('solver','frozen','learned'): raise ValueError('Unknown target mode')
            if type(seed) is not int or not 0 <= seed < 2**32: raise ValueError('Invalid synthesis seed')
            case_info(case)
            entry = spec['methods'][method]
            choices = group.get('pairs' if method == 'HMDC' else 'models', entry['pairs' if method == 'HMDC' else 'models'])
            if not choices: raise ValueError('Each method needs a model or pair')
            init = spec['initialization'][regime]
            for choice in choices:
                members = list(choice) if method == 'HMDC' else [choice]
                if not set(members) <= set(SUPPORTED_MODELS): raise ValueError('Unknown benchmark model')
                cfg = FACTORIES[method](case, choice)
                if method == 'HMDC' and entry.get('evaluators'):
                    cfg['evaluation_models'] = entry['evaluators'].get('+'.join(members), members)
                allowed = set(cfg) | {'generation_strength','generation_max_edits','min_topologies',
                    'training_model_seed','profile_real_batches','query_seed','query_selection_seed','real_training_seed','alignment_seed'}
                for label, overrides in [('shared',spec['shared']),('evaluation',spec['evaluation']),
                                         (method,entry.get('settings',{})), ('condition settings', group.get('settings',{}))]:
                    keys(overrides, allowed, label)
                    cfg.update(overrides)
                if cfg['synthetic_count'] % cfg['batch_size']: raise ValueError('synthetic_count must divide into complete batches')
                cfg['batches'] = cfg['synthetic_count']//cfg['batch_size']
                cfg['bus_range'] = spec.get('bus_limits',{}).get(case,cfg['bus_range'])
                limits=cfg['bus_range']
                if len(limits)!=2 or any(type(v) is not int or v<1 for v in limits) or not limits[0]<=case<=limits[1]:
                    raise ValueError('Bus limits must be positive integers containing the base case')
                cfg.update(generation_protocol=protocol, target_mode=target, initialization=init['mode'],
                    model_seed=init.get('model_seed',1_000_000), data_seed=seed,
                    training_model_seed=1_000_000+seed*10_000, sampler_seed=20261010+seed,
                    query_seed=51_001+seed, query_selection_seed=61_001+seed,
                    real_training_seed=71_001+seed, alignment_seed=81_001+seed,
                    reference_file=str(root/'shared/references'/f'case{case}.pt'),
                    data_root=str(Path(spec['paths']['data']).expanduser().resolve()),
                    raw_root=str(Path(spec['paths']['raw']).expanduser().resolve()),
                    generation_strength=cfg.get('generation_strength',.05 if protocol=='conservative' else .5),
                    generation_max_edits=cfg.get('generation_max_edits',2), min_topologies=cfg.get('min_topologies',3))
                evaluated = cfg['evaluation_models'] if method == 'HMDC' else members
                options = {k:model_options(k,**spec['model_defaults']) for k in evaluated}
                cfg.update({'models': options} if method == 'HMDC' else {'model_options':options[choice]})
                if init['mode'] == 'pretrained':
                    checkpoints = {k: init.get('checkpoints',{}).get(k) or init.get('checkpoint_pattern','').format(model=k) for k in evaluated}
                    if not all(checkpoints.values()): raise ValueError('Missing pretrained checkpoint')
                    cfg.update({'pretrained_checkpoints':checkpoints} if method=='HMDC' else {'pretrained_checkpoint':checkpoints[choice]})
                    cfg = (pretrained_pair_config if method=='HMDC' else pretrained_config)(cfg, refresh=True)
                cfg = retention_inputs(cfg)
                (dd.validate_config if method=='DD' else hm.validate_config if method=='HMDC' else dc.validate_condensation)(cfg)
                cache_identity = {k:cfg[k] for k in ('case','data_seed','synthetic_count','batches','batch_size','bus_range',
                    'generation_protocol','generation_strength','generation_max_edits','generation_attempts')}
                cache_id = digest(cache_identity | {'reference_sha256':cases[case]['reference_sha256']})[:16]
                cfg['initial_cache'] = str(root/'shared/initializations'/f'case{case}'/protocol/cache_id)
                identity = dict(method=method, config={k:v for k,v in cfg.items() if k not in
                    ('reference_file','initial_cache','data_root','raw_root','pretrained_checkpoint','pretrained_checkpoints')},
                    source_sha256=digest(source_hashes), data_sha256=cases[case]['data_sha256'],
                    reference_sha256=cases[case]['reference_sha256'], workers=12)
                run_id = f'{method}-{digest(identity)[:16]}'
                name = '+'.join(members)
                output = root/'runs'/method/name/f'case{case}'/protocol/target/regime/f'seed{seed}-{run_id.split("-")[-1]}'
                cfg['artifact_identity'] = dict(dataset_id=run_id, campaign=spec['campaign'],profile=spec['profile'],
                    method=method, synthesis_seed=seed, initialization_name=regime, **{k:identity[k] for k in
                    ('source_sha256','data_sha256','reference_sha256','workers')})
                rows[run_id] = dict(method=method, dataset_id=run_id, config=cfg, output=str(output),
                                    smoke=spec['profile'].endswith('smoke'),
                                    verify_run=spec['execution'].get('verify_run', False))
    # Round-robin methods/models rather than concentrating expensive cases on a node.
    experiments = sorted(rows.values(),key=lambda r: (r['config']['case'],r['config']['target_mode'],r['dataset_id']))
    return dict(schema_version=1, method='DD', root=str(root), specification=spec,
                source_hashes=source_hashes, references=cases, experiments=experiments)


def prepare(spec, path):
    from distillation.synthetic.artifacts import catalog
    from distillation.synthetic.experiments import generation_settings
    from distillation.data.split import split_indices
    import torch
    manifest = expand(spec)
    root = Path(manifest['root'])
    with claim(root/'prepare.lock') as acquired:
        if not acquired: raise RuntimeError('Another preparation is active')
        target = root/'manifest.json'
        normalized = json.loads(json.dumps(manifest))
        if target.exists():
            if json.loads(target.read_text()) != normalized: raise ValueError('Study/source/data changed; choose a new campaign')
            return manifest
        for relative, expected in manifest['source_hashes'].items():
            source, destination = ROOT/relative, root/'source'/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,destination)
            if file_hash(destination) != expected: raise ValueError('Source changed during snapshot')
        write_json(root/'source_hashes.json',manifest['source_hashes'])
        for case, info in manifest['references'].items():
            content = torch.load(info['reference_source'],weights_only=False,map_location='cpu')
            train = set(split_indices(dd.canonical_case(f'case{case}'))['train'].tolist())
            required = max(r['config']['synthetic_count'] for r in manifest['experiments'] if r['config']['case']==case)
            if len(content['graphs'])<required or len(content['ids'])<required or not set(content['ids']) <= train:
                raise ValueError('References must contain enough training-only examples')
            reference = root/'shared/references'/f'case{case}.pt'
            reference.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(info['reference_source'],reference)
            if file_hash(reference) != info['reference_sha256']: raise ValueError('Reference changed during preparation')
        prepared = set()
        for row in manifest['experiments']:
            cfg = row['config']; folder = Path(row['output']); folder.mkdir(parents=True,exist_ok=True)
            write_json(folder/'requested_config.json',cfg)
            if row['method'] == 'RAMBO': continue
            if cfg['initial_cache'] in prepared: continue
            prepared.add(cfg['initial_cache'])
            if spec['paths'].get('initializations'):
                source = Path(spec['paths']['initializations'])/f"case{cfg['case']}"/cfg['generation_protocol']/'initial_samples'
                refs = torch.load(cfg['reference_file'],weights_only=False,map_location='cpu')
                expected = digest(generation_settings(cfg,refs['ids'][:cfg['synthetic_count']]))
                cache = Path(cfg['initial_cache']); cache.mkdir(parents=True,exist_ok=True)
                for i in range(cfg['synthetic_count']):
                    sample = source/f'{i:04d}.pt'
                    if sample.exists() and torch.load(sample,weights_only=False,map_location='cpu').get('generation_fingerprint') == expected:
                        shutil.copy2(sample,cache/sample.name)
        with atomic_file(root/'campaign.yaml','w') as stream: stream.write(Path(path).read_text())
        write_json(root/'resolved.json',spec)
        write_json(target,manifest)
        for method in ['all']+list(spec['methods']):
            write_json(root/'manifests'/f'{method}.json',manifest | {'experiments':[
                r for r in manifest['experiments'] if method=='all' or r['method']==method]})
        (root/'README.md').write_text(
            f"# {spec['campaign']} / {spec['profile']}\n\n"
            'Start with catalog.csv (one row per dataset/evaluation recipe). artifact.json explains best.pt '
            '(condensation) or dataset.pt (complete RAMBO library); recipe.json describes its use. '
            'RAMBO generation_id identifies a library reused across evaluations. latest.pt resumes condensation; '
            'RAMBO resumes generation/state.json and evaluation.partial.json. '
            'Physical solutions and teaching targets are distinct. A smoke PASS establishes operability, not usefulness.\n\n'
            'campaign.yaml is the submitted YAML; resolved.json records the selected profile; manifest.json pins all '
            'conditions, source hashes and input identities. source/ is the frozen runtime. '
            'Re-submit with the same manifest to resume. COMPLETE datasets have checksums. '
            'Downstream training is a separate experiment; record dataset checksum, initialization and recipe.\n')
        catalog(manifest)
    return manifest


def refresh_runtime(root):
    """Freeze an explicitly requested runtime revision without rewriting checkpoints.

    Scientific configuration remains identical and is still checked by each
    runner. Preserve both source snapshots and record every changed file.
    """
    root = Path(root).resolve()
    original = json.loads((root/'manifest.json').read_text())['source_hashes']
    hashes = source_files()
    destination = root/'source_revisions'/digest(hashes)[:16]/'source'
    for relative, expected in hashes.items():
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists(): shutil.copy2(ROOT/relative, target)
        if file_hash(target) != expected: raise ValueError(f'Runtime snapshot changed: {relative}')
    record = dict(source=str(destination), source_hashes=hashes,
                  original_source_sha256=digest(original), runtime_source_sha256=digest(hashes),
                  changed_files=[k for k in sorted(original.keys()|hashes.keys()) if original.get(k)!=hashes.get(k)],
                  note='Runtime revision; original experiment configurations and checkpoints preserved.')
    write_json(destination.parent/'source_hashes.json', hashes)
    write_json(destination.parent/'revision.json', record)
    write_json(root/'runtime.json', record)
    return record


def submit(root, *, methods=None, layout=None, nodes=None, copper=None, update_runtime=False):
    root = Path(root).resolve()
    manifest = json.loads((root/'manifest.json').read_text())
    execution = dict(manifest['specification']['execution'])
    revision = (refresh_runtime(root) if update_runtime else
                json.loads((root/'runtime.json').read_text()) if (root/'runtime.json').exists() else None)
    source = Path(revision['source']) if revision else root/'source'
    if revision:
        manifest = manifest | {'source_hashes':revision['source_hashes'], 'runtime_revision':revision}
    if copper is not None:
        if type(copper) is not bool: raise ValueError('copper override must be boolean')
        execution['copper'] = copper
    methods = methods or list(manifest['specification']['methods'])
    if not set(methods) <= set(METHODS): raise ValueError('Unknown method')
    groups = [[m] for m in methods] if (layout or execution['layout'])=='separate' else [methods]
    for group in groups:
        selected = [r for r in manifest['experiments'] if r['method'] in group]
        if not selected: raise ValueError('Empty submission')
        def finished(row):
            folder = Path(row['output'])
            artifact = folder/'artifact.json'
            if not artifact.exists() or not json.loads(artifact.read_text()).get('complete'):
                return False
            if row.get('smoke') or row.get('verify_run'):
                check = folder/'smoke.json'
                return check.exists() and json.loads(check.read_text()).get('status') == 'PASS'
            return True
        selected = [row for row in selected if not finished(row)]
        if not selected:
            print(json.dumps(dict(methods=group, status='already_complete')), flush=True)
            continue
        label = '+'.join(group)
        stamp = str(time.time_ns())
        path = root/'manifests'/f'{label}-{stamp}.json'
        write_json(path,manifest | {'experiments':selected,
            'specification':manifest['specification'] | {'execution':execution}})
        job_nodes = nodes or execution['nodes']
        if type(job_nodes) is not int or not 1 <= job_nodes <= 256: raise ValueError('Use 1 to 256 nodes per job')
        job_nodes = min(job_nodes, len(selected))
        env = dict(STUDY_SOURCE=str(source),STUDY_MANIFEST=str(path),
                   STUDY_ACTIVATE=execution['activate'],STUDY_BUDGET=str(execution['allocation_seconds']),
                   STUDY_COPPER=str(int(execution.get('copper', False))))
        if any(',' in v or '\n' in v for v in env.values()): raise ValueError('Invalid PBS environment value')
        log = root/'jobs'/f'{label}-{stamp}.log'; log.parent.mkdir(parents=True,exist_ok=True)
        command = ['qsub','-A',execution['account'],'-q',execution['queue'],'-l',f'select={job_nodes}',
            '-l',f"walltime={execution['walltime']}",'-o',str(log),'-v',','.join(f'{k}={v}' for k,v in env.items()),
            str(source/'distillation/experiments/condensation.pbs')]
        result = subprocess.run(command,check=True,text=True,capture_output=True)
        record = dict(job_id=result.stdout.strip(),methods=group,nodes=job_nodes,manifest=str(path),log=str(log),
                      execution=execution,command=command,runtime_revision=revision)
        write_json(root/'jobs'/f'{stamp}.json',record)
        print(json.dumps(record),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['preview','prepare','submit','catalog'])
    parser.add_argument('--config',type=Path,default=DEFAULT)
    parser.add_argument('--profile',default='reference')
    parser.add_argument('--root',type=Path)
    parser.add_argument('--methods',nargs='+',choices=METHODS)
    parser.add_argument('--layout',choices=['combined','separate'])
    parser.add_argument('--nodes',type=int,help='Nodes PER job; separate submits one job per method')
    parser.add_argument('--refresh-runtime',action='store_true',
                        help='Freeze current runtime as an explicit, recorded revision for existing checkpoints')
    parser.add_argument('--copper',action=argparse.BooleanOptionalAction,default=None,
                        help='Override optional package caching for this submission; preserve frozen experiment identities')
    args=parser.parse_args()
    if args.action in ('preview','prepare'):
        spec=load(args.config,args.profile)
        result=prepare(spec,args.config) if args.action=='prepare' else expand(spec)
        print(json.dumps(dict(root=result['root'],conditions=len(result['experiments']),
            by_method=dict(Counter(r['method'] for r in result['experiments'])),
            paths=[r['output'] for r in result['experiments']]),indent=2))
    else:
        if args.root is None: parser.error('--root is required')
        if args.action=='submit': submit(args.root,methods=args.methods,layout=args.layout,nodes=args.nodes,
                                        copper=args.copper,update_runtime=args.refresh_runtime)
        else:
            from distillation.synthetic.artifacts import catalog
            catalog(json.loads((args.root/'manifest.json').read_text()))


if __name__=='__main__': main()
