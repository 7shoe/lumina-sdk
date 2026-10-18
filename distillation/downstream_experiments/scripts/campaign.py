"""Prepare explicit downstream comparisons; launch independent students on each XPU."""
import argparse
import asyncio
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, wait
from copy import deepcopy
import csv
from functools import lru_cache
import json
import os
from pathlib import Path
import signal
import socket
import sys
import tempfile
import time

import torch
import yaml

from distillation.downstream_experiments.dataset_path import CATALOG, find_datasets
from distillation.experiments.launch import supervise
from distillation.experiments.staging import stage
from distillation.training.T1_single_topo.campaign import run_child
from distillation.training.T1_single_topo.common import (
    ROOT, atomic_file, claim, digest, file_hash, read_json, write_json)
from distillation.training.T1_single_topo.state import Stop
from .data import case_name, data_identity, make_corpus
from .train import completed, folder_for, source_identity

DEFAULT_CONFIG = Path(__file__).parents[1] / 'config.yaml'
SYNTHETIC = {'DD', 'DCGM', 'HMDC'}


@lru_cache(maxsize=512)
def artifact_metadata(path):
    return read_json(path)


def load_config(path=DEFAULT_CONFIG):
    config = yaml.safe_load(Path(path).read_text())
    if config.get('split_seed') != 42 or config.get('subset_seed') != 777:
        raise ValueError('Use canonical split_seed=42 and random-subset seed=777')
    if not config['name'] or Path(config['name']).name != config['name']:
        raise ValueError('Campaign name must be one directory component')
    defaults = yaml.safe_load((ROOT/'distillation/training/T1_single_topo/config.yaml').read_text())
    config['model_options'] = defaults['models'] | config.get('model_options', {})
    if not set(config['students']) <= set(defaults['models']):
        raise ValueError('Unknown student architecture')
    if not set(config['methods']) <= (SYNTHETIC | {'BAS','RAMBO','US','DSS','RND777'}):
        raise ValueError('Unknown dataset method')
    for key in ('students','methods','cases','seeds'):
        if not config[key] or len(set(config[key])) != len(config[key]):
            raise ValueError(f'{key} must be nonempty and unique')
    if any(type(s) is not int or not 0 <= s < 2**32 for s in config['seeds']):
        raise ValueError('Seeds must be integers in [0, 2**32)')
    catalog = Path(config.get('catalog') or CATALOG).expanduser()
    config['catalog'] = str((catalog if catalog.is_absolute() else Path(path).resolve().parent/catalog).resolve())
    config['optimizer'] = defaults['optimizer'] | config.get('optimizer', {})
    training = config['training']
    for key in ('max_updates','batch_size','eval_batch_size','log_updates'):
        if type(training[key]) is not int or training[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if (training['checkpoint_seconds'] <= 0 or training['gradient_clip'] <= 0
            or any(type(n) is not int or not 0 < n <= training['max_updates'] for n in training['validation_updates'])):
        raise ValueError('Invalid training/validation schedule')
    devices = config['execution']['devices']
    if not devices or len(set(devices)) != len(devices) or any(type(d) is not int or not 0 <= d < 12 for d in devices):
        raise ValueError('Choose distinct XPU indices in [0, 12)')
    return config


def prepare(config):
    """Resolve paths only through the feeder; freeze all conditions before training."""
    records = find_datasets(catalog=config['catalog'])
    artifact_metadata.cache_clear()
    data_identity.cache_clear()
    if config.get('dataset_ids'):
        wanted = set(config['dataset_ids'])
        if wanted - {r['dataset_id'] for r in records}:
            raise ValueError('dataset_ids contains an unavailable/excluded dataset')
        records = [r for r in records if r['dataset_id'] in wanted]
    paths = {k:str(Path(v).expanduser().resolve()) for k,v in config['paths'].items()}
    paths['models'] = str(Path(paths['models'])/config['name'])
    paths['corpora'] = str(Path(paths['corpora'])/config['name'])
    sources = source_identity()
    corpora, runs, missing = {}, {}, []
    preferred = set(config.get('canonical_ids', []))

    @lru_cache(maxsize=None)
    def choose(method, case, creation='random', target='learned', size=100):
        rows = [r for r in records if r['acronym'] == method and r['case'] == case]
        if method in SYNTHETIC:
            rows = [r for r in rows if r['count'] == 100 and r['initialization'] == creation
                    and r['target_mode'] == target and r['generation'] == config['generation']]
        else:
            rows = [r for r in rows if r['count'] >= size]
            if rows:
                rows = [r for r in rows if r['count'] == min(x['count'] for x in rows)]
        grouped = defaultdict(list)
        for row in rows:
            grouped[row['reference_model']].append(row)
        selected = []
        for creator, choices in sorted(grouped.items()):
            if len(choices) > 1:
                pinned = [r for r in choices if r['dataset_id'] in preferred]
                if len(pinned) != 1:
                    raise ValueError(f'Ambiguous {method}/{case}/{creator}/{creation}/{target}; pin one canonical_id')
                choices = pinned
            selected.append(choices[0])
        if not selected:
            missing.append(dict(method=method, case=case, creation=creation, target=target,
                                requested_count=size, reason='No published compatible dataset'))
        return selected

    @lru_cache(maxsize=None)
    def corpus(ids, size, random_cases=()):
        value = make_corpus(ids, size, data_root=paths['data'], seed=config['subset_seed'],
                            catalog=config['catalog'], random_cases=random_cases)
        corpora[value['corpus_id']] = value
        return value

    def pool(method, case, creation, target, size):
        if method == 'RND777':
            return corpus((), size, (case,))
        rows = choose(method, case, creation, target, size)
        if not rows:
            return None
        if sum(r['count'] for r in rows) < size:
            missing.append(dict(method=method, case=case, creation=creation, requested_count=size,
                                reason='Insufficient examples; never duplicate to fill a corpus'))
            return None
        return corpus(tuple(r['dataset_id'] for r in rows), size)

    @lru_cache(maxsize=None)
    def pretrained(model, source):
        path = Path(paths['t1'])/'checkpoints'/model/case_name(source)/f"seed{config['pretrained_seed']}"/'best.pt'
        if not path.exists():
            missing.append(dict(model=model, source_case=source, reason='Missing T1 checkpoint', path=str(path)))
            return None
        payload = torch.load(path, weights_only=False, map_location='cpu')
        spec = payload['specification']
        data = data_identity(paths['data'], source)
        if (spec['format'] != 'lumina-t1-v1' or spec['model'] != model or spec['split_seed'] != 42
                or spec['case'] != case_name(source) or spec['data_identity'] != data['identity']):
            raise ValueError(f'Incompatible source checkpoint: {path}')
        return dict(kind='pretrained', path=str(path), sha256=file_hash(path), source_case=source,
                    data_identity=data['identity'], seed=spec['seed'], model_options=spec['model_options'], precision='float32')

    def fresh(model, seed, profile=None):
        return dict(kind='fresh', seed=seed, model_options=(profile or {}).get('model_options', config['model_options'][model]),
                    precision=(profile or {}).get('precision', 'float32'))

    def add(value, model, seed, init, family, evaluation_cases=None):
        if value is None or init is None:
            return
        cases = sorted(set(evaluation_cases or value['cases']))
        if init['kind']=='pretrained':
            for member in value['members']:
                if member.get('initialization') != 'pretrained':
                    continue
                if member['initialization_name'] != f"T1_case{init['source_case']}":
                    raise ValueError('Pretrained-created data and student source cases disagree')
                if model in member['creator'].split('+'):
                    identity = artifact_metadata(member['artifact_path']).get('source_checkpoint_identity', {})
                    identity = identity if 'sha256' in identity else identity.get(model, {})
                    if identity.get('sha256') != init['sha256']:
                        raise ValueError('Student source checkpoint differs from the synthesis checkpoint')
        spec = dict(format='lumina-downstream-v1', corpus_id=value['corpus_id'], model=model, seed=seed,
            initialization=init, evaluation_cases=cases, sample_case=min(value['cases']), split_seed=42,
            data={str(c):data_identity(paths['data'], c) for c in cases}, training=config['training'],
            optimizer=config['optimizer'], sources=digest(sources), selection='equal mean of per-case validation sdk_mse + violation')
        identifier = 'RUN-'+digest(spec)[:20]
        if identifier not in runs:
            runs[identifier] = dict(run_id=identifier, specification=spec, families=[])
        if family not in runs[identifier]['families']:
            runs[identifier]['families'].append(family)

    def students(value, family, source=None):
        for model in config['students']:
            for seed in config['seeds']:
                init = pretrained(model, source) if source is not None else fresh(model, seed)
                add(value, model, seed, init, family,
                    sorted(set(value['cases']+[source])) if value is not None and source is not None else None)

    families = config['families']
    for case in config['cases']:
        for target in config['target_modes']:
            for creation in config['creation_initializations']:
                for method in config['methods']:
                    if families['individual'] or 100 in families['adaptation_sizes']:
                        individual = ([corpus((),100,(case,))] if method == 'RND777' else
                            [corpus((r['dataset_id'],),100) for r in choose(method,case,creation,target)])
                        for value in individual:
                            if families['individual']:
                                students(value, 'individual')
                            if 100 in families['adaptation_sizes']:
                                students(value, 'adaptation-individual', config['source_cases'][case])
                    for size in families['mixed_sizes']:
                        value = pool(method,case,creation,target,size)
                        if value:
                            students(value, f'mixed-M{size}')
                    for size in families['adaptation_sizes']:
                        if size == 100:
                            continue
                        value = pool(method,case,creation,target,size)
                        if value:
                            students(value, f'adaptation-M{size}', config['source_cases'][case])

    if families['fixed']:
        fixed = [r for r in records if r['acronym'] in config['methods'] and r['initialization']=='fixed'
                 and r['count']==100 and r['generation']==config['generation']
                 and r['case'] in config['cases'] and r['target_mode'] in config['target_modes']]
        profiles = {}
        for row in fixed:
            for model in row['models']:
                if model not in config['students']:
                    continue
                payload = torch.load(row['data_path'], weights_only=False, map_location='cpu')
                student = payload.get('model_checkpoints', {}).get(model, payload.get('model_checkpoint'))
                init = dict(kind='recorded',path=row['data_path'],sha256=row['sha256'],seed=student['seed'],
                    model_options=student['model_options'],precision=payload['config'].get('model_precision','float32'))
                # Same profile across controls; own fixed artifacts always retain their actual initial state.
                value = corpus((row['dataset_id'],),100)
                for seed in config['seeds']:
                    add(value,model,seed,init,'fixed-matched')
                    add(value,model,seed,fresh(model,seed,init),'fixed-fresh')
                profiles.setdefault((row['case'],model),init)
        for (case,model), init in profiles.items():
            for method in config['methods']:
                controls = ([corpus((),100,(case,))] if method=='RND777' else
                    [corpus((r['dataset_id'],),100) for r in choose(method,case)
                     if method not in SYNTHETIC or model in r['models']])
                for value in controls:
                    for seed in config['seeds']:
                        add(value,model,seed,init,'fixed-control-matched')
                        add(value,model,seed,fresh(model,seed,init),'fixed-control-fresh')

    size = families['joint_size']
    if size:
        if size % 2:
            raise ValueError('Joint corpus size must be even')
        for source, target_case in config['transfer_pairs']:
            if not {source,target_case} <= set(config['cases']):
                continue
            for target in config['target_modes']:
                for creation in config['creation_initializations']:
                    for method in config['methods']:
                        if method=='RND777':
                            value = corpus((),size,tuple(sorted((source,target_case))))
                        else:
                            a = choose(method,source,'random',target,size//2)
                            b = choose(method,target_case,creation,target,size//2)
                            if not a or not b or min(sum(r['count'] for r in a),sum(r['count'] for r in b)) < size//2:
                                continue
                            if creation=='pretrained' and method in SYNTHETIC and any(
                                    r['initialization_name'] != f'T1_case{source}' for r in b):
                                raise ValueError('Target dataset was synthesized for a different source case')
                            value = corpus(tuple(r['dataset_id'] for r in sorted(a+b,key=lambda r:r['dataset_id'])),size)
                        students(value,'joint-fresh')
                        students(value,'adaptation-replay',source)

    manifest = dict(format='lumina-downstream-campaign-v1', config=config, paths=paths, sources=sources,
        catalog_sha256=file_hash(config['catalog']), corpora=corpora, runs=list(runs.values()), unavailable=missing)
    root = Path(paths['models'])
    root.mkdir(parents=True,exist_ok=True)
    path = root/'manifest.json'
    if path.exists() and read_json(path) != manifest:
        raise ValueError('Campaign/configuration/source changed; use a new campaign name')
    write_json(path,manifest)
    write_json(root/'coverage.json',dict(runs=len(runs),corpora=len(corpora),unavailable=missing,
        by_family=dict(Counter(f for r in runs.values() for f in r['families']))))
    print(json.dumps(dict(manifest=str(path),runs=len(runs),corpora=len(corpora),unavailable=len(missing))),flush=True)
    return path


def worker(manifest_path, bundle, slot, device, seconds):
    manifest = read_json(manifest_path)
    by_id = {r['run_id']:r for r in manifest['runs']}
    tasks = bundle['run_ids'][slot::len(bundle['devices'])]
    outcomes = {}
    with Stop(seconds) as stop:
        for identifier in tasks:
            if stop(): break
            run = by_id[identifier]; folder = folder_for(manifest,run)
            with claim(folder/'worker.lock') as acquired:
                if not acquired: continue
                try:
                    if completed(folder,run['specification']):
                        outcomes[identifier]='complete'; continue
                    command = [sys.executable,'-u','-m',__package__+'.train','--manifest',str(manifest_path),
                        '--run-id',identifier,'--device',device,'--seconds',str(max(1,stop.deadline-time.monotonic()-15))]
                    with (folder/'run.log').open('a',buffering=1) as log:
                        log.write(f'\nJob {os.environ.get("PBS_JOBID","local")} host={socket.gethostname()} {device}\n')
                        code = run_child(command,log,stop)
                        while code==75 and not stop():
                            time.sleep(2); code=run_child(command,log,stop)
                    outcomes[identifier] = 'complete' if code==0 and completed(folder,run['specification']) else (
                        'paused' if code==0 else 'failed')
                except Exception as error:
                    outcomes[identifier]='failed'
                    write_json(folder/'failure.json',dict(error=repr(error),time=time.time()))
            print(json.dumps(dict(run_id=identifier,device=device,status=outcomes[identifier])),flush=True)
    write_json(Path(bundle['output'])/f'worker-{slot}.json',outcomes)
    return outcomes


def node(manifest_path, bundle, seconds, result_path):
    started = time.monotonic()
    manifest = read_json(manifest_path)
    runs = [r for r in manifest['runs'] if r['run_id'] in set(bundle['run_ids'])]
    output = Path(bundle['output']); output.mkdir(parents=True,exist_ok=True)
    if manifest['config']['execution'].get('stage_inputs',False):
        cases = sorted({c for r in runs for c in r['specification']['evaluation_cases']})
        checkpoints = {r['run_id']:r['specification']['initialization'] for r in runs
                       if r['specification']['initialization']['kind'] != 'fresh'}
        rows = [dict(method='downstream',dataset_id=f'case{c}',config=dict(case=c,data_root=manifest['paths']['data'],
            artifact_identity=dict(data_sha256=data_identity(manifest['paths']['data'],c)['sha256'],
                reference_sha256=file_hash(manifest_path)),reference_file=str(manifest_path),
            pretrained_checkpoints={k:v['path'] for k,v in checkpoints.items()},
            pretrained={k:dict(sha256=v['sha256']) for k,v in checkpoints.items()})) for c in cases]
        with tempfile.TemporaryDirectory(prefix='lumina-downstream-',dir=os.environ.get('TMPDIR','/tmp')) as temporary:
            stage(rows,temporary)
            os.environ['STUDY_INPUT_MAP']=str(Path(temporary)/'inputs.json')
            return node_workers(manifest_path,bundle,max(1,seconds-(time.monotonic()-started)),result_path)
    return node_workers(manifest_path,bundle,seconds,result_path)


def node_workers(manifest_path,bundle,seconds,result_path):
    output = Path(bundle['output'])
    with Stop(seconds) as stop, ThreadPoolExecutor(max_workers=len(bundle['devices'])) as executor:
        def launch(slot,device):
            command = [sys.executable,'-u','-m',__package__+'.campaign','worker','--manifest',str(manifest_path),
                '--bundle',str(output/'bundle.json'),'--slot',str(slot),'--device',f'xpu:{device}',
                '--seconds',str(max(1,stop.deadline-time.monotonic()-20))]
            with (output/f'worker-{slot}.log').open('a',buffering=1) as log:
                return run_child(command,log,stop)
        futures = [executor.submit(launch,slot,device) for slot,device in enumerate(bundle['devices'])]
        while not all(f.done() for f in futures):
            write_json(output/'status.json',dict(status='running',updated=time.time(),
                       workers=sum(not f.done() for f in futures)))
            wait(futures,timeout=10)
        codes = [f.result() for f in futures]
    manifest = read_json(manifest_path)
    selected = [r for r in manifest['runs'] if r['run_id'] in set(bundle['run_ids'])]
    done = all(completed(folder_for(manifest,r),r['specification']) for r in selected)
    reports = [read_json(p) for p in output.glob('worker-*.json')]
    failed = any(codes) or any(v=='failed' for r in reports for v in r.values())
    status = 'COMPLETE' if done else 'FAILED' if failed else 'RESUMABLE'
    write_json(output/'artifact.json',dict(complete=done,status=status))
    write_json(result_path,dict(status=status,run_ids=bundle['run_ids'],worker_codes=codes))
    return int(failed)


def launch(manifest_path,seconds):
    manifest = read_json(manifest_path)
    hosts = list(dict.fromkeys(Path(os.environ['PBS_NODEFILE']).read_text().split()))
    pending = [r for r in manifest['runs'] if not completed(folder_for(manifest,r),r['specification'])]
    pending.sort(key=lambda r:(r['specification']['evaluation_cases'],r['specification']['corpus_id']))
    directory = Path(manifest['paths']['models'])/'jobs'/f"{os.environ.get('PBS_JOBID',time.time_ns())}"
    bundles = []
    count = max(1,(len(pending)+len(hosts)-1)//len(hosts))
    for index,start in enumerate(range(0,len(pending),count)):
        folder = directory/f'node-{index}'
        bundle = dict(dataset_id=f'node-{index}',method='downstream',config={},output=str(folder),
            run_ids=[r['run_id'] for r in pending[start:start+count]],devices=manifest['config']['execution']['devices'])
        write_json(folder/'bundle.json',bundle); bundles.append(bundle)
    wrapper = dict(root=str(directory),experiments=bundles,specification=dict(execution=manifest['config']['execution']))
    def command(hostfile,subset,remaining,result):
        bundle = read_json(subset)['experiments'][0]
        return ['mpiexec','--pmi=pmix','--envall','--hostfile',str(hostfile),'-n','1','--ppn','1',
            '--cpu-bind','none','/usr/bin/env',sys.executable,'-u','-m',__package__+'.campaign','node',
            '--manifest',str(manifest_path),'--bundle',str(Path(bundle['output'])/'bundle.json'),
            '--seconds',str(remaining),'--result',str(result)]
    async def supervise_nodes():
        stop = asyncio.Event()
        for sig in (signal.SIGTERM,signal.SIGINT):
            asyncio.get_running_loop().add_signal_handler(sig,stop.set)
        return await supervise(wrapper,hosts,directory,time.time()+seconds,command=command,stop=stop)
    summary = asyncio.run(supervise_nodes())
    collect(manifest_path)
    return int(any(summary['counts'].get(k,0) for k in ('FAILED','TIMED_OUT')))


def collect(manifest_path):
    manifest = read_json(manifest_path)
    rows = []
    for run in manifest['runs']:
        spec = run['specification']; folder = folder_for(manifest,run)
        corpus = manifest['corpora'][spec['corpus_id']]
        progress = read_json(folder/'status.json') if (folder/'status.json').exists() else {}
        status = progress.get('status','pending')
        if (folder/'failure.json').exists() and status != 'complete': status='failed'
        for case in spec['evaluation_cases']:
            path = folder/f'test-case{case}.json'
            report = read_json(path) if path.exists() else {}
            initial = folder/f'initial-test-case{case}.json'
            before = read_json(initial).get('metrics',{}) if initial.exists() else {}
            row = dict(run_id=run['run_id'],families=';'.join(run['families']),status=status,
                method=corpus['method'],creators=';'.join(sorted({m['creator'] for m in corpus['members']})),
                training_cases=','.join(map(str,corpus['cases'])),test_case=case,model=spec['model'],seed=spec['seed'],
                count=corpus['count'],student_initialization=spec['initialization']['kind'],
                target_modes=','.join(sorted({m['target_mode'] for m in corpus['members']})),
                creation_initializations=','.join(sorted({m.get('initialization') or 'none' for m in corpus['members']})),
                checkpoint=str(folder/'best.pt'),result=str(path),
                initial_test_mse=before.get('sdk_mse'),checkpoint_step=report.get('checkpoint_step'),
                wall_seconds=progress.get('wall_seconds'),steps=progress.get('steps'))
            row.update({k:report.get('metrics',{}).get(k) for k in ('sdk_mse','violation','cost_difference','n_samples')})
            rows.append(row)
    root = Path(manifest['paths']['models'])
    write_json(root/'results.json',rows)
    if rows:
        with atomic_file(root/'results.csv','w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','launch','worker','node','collect'))
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--bundle',type=Path)
    parser.add_argument('--slot',type=int,default=0)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--seconds',type=float,default=3300)
    parser.add_argument('--result',type=Path)
    args = parser.parse_args()
    if args.action=='prepare': prepare(load_config(args.config)); return 0
    if not args.manifest: parser.error('--manifest is required')
    args.manifest=args.manifest.resolve()
    if args.action=='collect': collect(args.manifest); return 0
    if args.action=='launch': return launch(args.manifest,args.seconds)
    if args.action=='node': return node(args.manifest,read_json(args.bundle),args.seconds,args.result)
    if args.action=='run':
        manifest=read_json(args.manifest)
        bundle=dict(run_ids=[r['run_id'] for r in manifest['runs']],devices=[0],output=manifest['paths']['models'])
    else: bundle=read_json(args.bundle)
    result=worker(args.manifest,bundle,args.slot,args.device,args.seconds)
    if args.action=='run': collect(args.manifest)
    return int(any(v=='failed' for v in result.values()))


if __name__=='__main__':
    raise SystemExit(main())
