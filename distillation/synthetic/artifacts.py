"""Dataset identities, catalogs and target-preserving reuse for condensation and RAMBO."""
import csv
import json
from pathlib import Path

import torch

from .inputs import SyntheticInputs, TARGET_SEMANTICS
from .learning import model_for, model_from_export, pretrained_config, train_student, initialization_seed
from distillation.training.T1_single_topo.common import atomic_file, claim, file_hash, write_json


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def publish(experiment, *, failure=None):
    """Publish only metadata; preserve native exports and resumable filenames."""
    folder = Path(experiment['output'])
    cfg = read(folder/'config.json',experiment['config'])
    status = read(folder/'status.json',{}).get('status','PENDING')
    if failure: status='FAILED'
    generated = experiment['method'] == 'RAMBO'
    best = folder/('dataset.pt' if generated else 'best.pt')
    record = dict(cfg.get('artifact_identity',{}), format='lumina-synthetic-dataset-v1',
        method=experiment['method'], status=status, complete=status=='COMPLETE',
        purpose='smoke validation only' if experiment.get('smoke') else 'synthesis experiment',
        kind='generated_operating_points' if generated else 'distilled_grids',
        case=cfg['case'], synthesis_models=[] if generated else cfg.get('pair',[cfg.get('model')]),
        evaluation_models=cfg.get('evaluation_models', cfg.get('pair',[cfg.get('model')])),
        generation_protocol=cfg['generation_protocol'], target_mode=cfg['target_mode'],
        target_semantics=('verified final Julia/Ipopt AC-OPF solutions, preserved unchanged' if generated else
                          TARGET_SEMANTICS[cfg['target_mode']]), initialization=cfg['initialization'],
        pretrained=cfg.get('pretrained',cfg.get('pretrained_metadata')),
        count=cfg['synthetic_count'], bus_limits=cfg['bus_range'],
        config='config.json' if (folder/'config.json').exists() else 'requested_config.json',
        dataset=best.name if best.exists() else None,
        resume='evaluation.partial.json' if generated else 'latest.pt',
        final='dataset.pt' if generated else 'distilled.pt' if experiment['method']=='DD' else 'condensed.pt',
        failure=failure, feature_encoding='synthetic.learning.benchmark_graph; SDK OPF feature ordering',
        targets={'bus':['va (radians)','vm (per unit)'], 'generator':['pg (per unit)','qg (per unit)']},
        physical_labels='physical_solved_graphs: separate AC-OPF solutions; never replace teaching targets',
        split=read(folder/'split.json',{}))
    if generated:
        record.update(generation_id=cfg['generation_id'], generation_domain=cfg['generation_domain'],
            generation_manifest=str(Path(cfg['generation_output'])/'dataset.json'),
            generation_resume=str(Path(cfg['generation_output'])/'state.json'),
            generation=read(Path(cfg['generation_output'])/'status.json',{}),
            dataset_selection='entire completed generated library; no validation selection')
    if best.exists():
        artifact=torch.load(best,weights_only=False,map_location='cpu')
        students=artifact.get('model_checkpoints') or {cfg['model']:artifact['model_checkpoint']}
        ordinary=read(folder/'common_evaluation.json',{}) if experiment['method']=='DD' else read(folder/'evaluation.json',{})
        rates=artifact.get('student_rates') or ({cfg['model']:artifact['student_rate']} if 'student_rate' in artifact else
            {cfg['model']:ordinary.get('calibration',{}).get('chosen_rate')})
        results=ordinary.get('results',{})
        selected={k:v.get('best',{}) for k,v in results.items()} if experiment['method']=='HMDC' else {
            cfg['model']:results.get('generated' if generated else 'best',{})}
        summary={k:{field:value.get(field) for field in ('mean','std','student_updates','unique_initializations')}
                 for k,value in selected.items()}
        record.update(export_format=artifact['format'], sha256=file_hash(best),
            best_iteration=artifact['training_state'].get('best_iteration'),
            bus_counts=[int(g['bus'].num_nodes) for g in artifact['physical_solved_graphs']],
            physical_audit_passed=len(artifact['solver_checks'])==cfg['synthetic_count'] and all(r['success'] for r in artifact['solver_checks']),
            source_checkpoint_identity=cfg.get('pretrained'),
            evaluation=summary, evaluation_file='common_evaluation.json' if experiment['method']=='DD' else 'evaluation.json')
        if generated: record['source_generation'] = artifact['source_generation']
        recipe=dict(format='lumina-synthetic-recipe-v1', dataset_sha256=record['sha256'],
            loss='graph_balanced_mse: equal graph/type weights; mean over two channels',
            optimizer='SGD',momentum=0.,weight_decay=0.,dropout=False,batch_norm='frozen',
            recorded_initialization=cfg['initialization'],
            initialization_note='recorded reuses embedded starting weights for pretrained runs; scratch is a separate transfer experiment',
            models={name:dict(options=s['model_options'],seed=s['seed'],training_seed=s.get('training_seed',s['seed']),
                              ordinary_rate=rates.get(name)) for name,s in students.items()},
            ordinary=dict(batch_size=cfg['batch_size'],epochs=cfg['evaluation_epochs'],
                updates=cfg['batches']*cfg['evaluation_epochs'],order='seeded shuffle each epoch'),
            native=dict(kind='learned curriculum' if experiment['method']=='DD' else 'ordinary SGD',
                updates=len(artifact['schedule']) if 'schedule' in artifact else cfg['batches']*cfg['evaluation_epochs']))
        if 'schedule' in artifact:
            recipe['native'].update(batch_order=[s['batch'] for s in artifact['schedule']],
                                    rates=[s['rate'] for s in artifact['schedule']])
        write_json(folder/'recipe.json',recipe)
        record['recipe']='recipe.json'
    write_json(folder/'artifact.json',record)
    return record


def catalog(manifest):
    """Atomic campaign index; concurrent method jobs always index the full study."""
    root=Path(manifest['root'])
    manifest=read(root/'manifest.json',manifest)
    with claim(root/'catalog.lock') as acquired:
        if not acquired: return
        rows=[]
        for experiment in manifest['experiments']:
            folder=Path(experiment['output']);cfg=experiment['config']
            artifact=read(folder/'artifact.json',{})
            status=read(folder/'status.json',{}).get('status','PENDING')
            if artifact.get('status')=='FAILED': status='FAILED'
            rows.append(dict(dataset_id=experiment['dataset_id'],method=experiment['method'],
                kind=artifact.get('kind','generated_operating_points' if experiment['method']=='RAMBO' else 'distilled_grids'),
                generation_id=cfg.get('generation_id'), generation_domain=cfg.get('generation_domain'),
                models='+'.join(cfg.get('pair',[cfg.get('model')])),case=cfg['case'],
                generation=cfg['generation_protocol'],target_mode=cfg['target_mode'],
                initialization=cfg['initialization'],initialization_name=cfg['artifact_identity']['initialization_name'],
                synthesis_seed=cfg['data_seed'],synthetic_count=cfg['synthetic_count'],
                purpose='smoke' if experiment.get('smoke') else 'experiment',status=status,
                ready=bool(status=='COMPLETE' and artifact.get('complete') and artifact.get('sha256')),
                dataset=str(folder/artifact['dataset']) if artifact.get('dataset') else None,
                sha256=artifact.get('sha256'),artifact=str(folder/'artifact.json') if artifact else None,
                recipe=str(folder/'recipe.json') if (folder/'recipe.json').exists() else None,
                ordinary_results=artifact.get('evaluation'),
                downstream_output=str(Path(manifest['specification']['paths']['models'])/
                    manifest['specification']['campaign']/experiment['dataset_id'])))
        write_json(root/'catalog.json',rows)
        if rows:
            with atomic_file(root/'catalog.csv','w') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader()
                writer.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)
        print(json.dumps(dict(catalog=str(root/'catalog.csv'),conditions=len(rows),ready=sum(r['ready'] for r in rows))),flush=True)


def load_dataset(path, *, require_complete=True):
    """Return SDK graphs with saved teaching/physical targets, plus provenance.

    Use only trusted .pt exports. Loading needs PyTorch/PyG and this source, but no
    raw OPFData, original pretrained files, MPI, or new OPF solve.
    """
    path=Path(path)
    metadata=read(path/'artifact.json' if path.is_dir() else path,{})
    if metadata.get('format')!='lumina-synthetic-dataset-v1': raise ValueError('Pass a run directory or artifact.json')
    folder=path if path.is_dir() else path.parent
    if require_complete and not metadata['complete']: raise ValueError('Dataset is provisional; synthesis/evaluation is incomplete')
    if not metadata.get('dataset'): raise ValueError('No dataset exported yet')
    source=folder/metadata['dataset']
    if file_hash(source)!=metadata['sha256']: raise ValueError('Dataset checksum changed')
    artifact=torch.load(source,weights_only=False,map_location='cpu')
    data=SyntheticInputs.from_export(artifact)
    batch,labels=data.batch(detached=True)
    # Split labels explicitly: input batches intentionally carry no supervision.
    graphs=[];offset={k:0 for k in labels}
    for graph in batch.to_data_list():
        for name,value in labels.items():
            n=graph[name].num_nodes
            graph[name].y=value[offset[name]:offset[name]+n].clone();offset[name]+=n
        graphs.append(graph)
    return dict(graphs=graphs,physical_solved_graphs=artifact['physical_solved_graphs'],
                metadata=metadata,recipe=read(folder/'recipe.json'),export=artifact)


def train_from_artifact(path, *, device='cpu',model=None,initialization='recorded',checkpoint=None,
                        seed=None,recipe='native',rate=None,epochs=None):
    """Train from recorded weights, fresh random weights, or an explicit checkpoint.

    Returns (model, provenance). Saving/evaluating this downstream model is a
    separate experiment. Hyperparameter overrides are recorded in provenance.
    """
    loaded=load_dataset(path);artifact=loaded['export'];config=artifact['config'].copy()
    students=artifact.get('model_checkpoints') or {config['model']:artifact['model_checkpoint']}
    kind=model or (next(iter(students)) if len(students)==1 else None)
    if kind not in students: raise ValueError('Choose a model recorded in recipe.json; use load_dataset for other architectures')
    student=students[kind];config.update(model=kind,model_options=student['model_options'])
    actual_seed=student['seed'] if seed is None else seed
    if initialization=='recorded':
        if checkpoint: raise ValueError('recorded initialization uses embedded weights')
        network=model_from_export(artifact,device,seed=actual_seed,model=kind)
        source=config.get('pretrained')
    elif initialization in ('scratch','checkpoint'):
        for key in ('pretrained','pretrained_checkpoint','pretrained_checkpoints'): config.pop(key,None)
        if initialization=='checkpoint':
            if not checkpoint: raise ValueError('checkpoint initialization needs a checkpoint path')
            config.update(initialization='pretrained',pretrained_checkpoint=str(checkpoint))
            config=pretrained_config(config,refresh=True)
        else:
            if checkpoint: raise ValueError('scratch initialization cannot use a checkpoint')
            config['initialization']='random'
        network=model_for(artifact['physical_solved_graphs'][0],config,actual_seed,device)
        actual_seed=initialization_seed(config,actual_seed)
        source=config.get('pretrained')
    else: raise ValueError('initialization must be recorded, scratch or checkpoint')
    if recipe not in ('native','ordinary'): raise ValueError('recipe must be native or ordinary')
    if recipe=='native' and 'schedule' in artifact:
        if rate is not None or epochs is not None: raise ValueError('Native DD uses its saved rates and ordered steps')
        from distillation.DD.campaign import fit_export
        theta,buffers=fit_export(network,artifact);network.load_state_dict(theta|buffers)
        updates=len(artifact['schedule']);training_seed=None
    else:
        rate=rate if rate is not None else loaded['recipe']['models'][kind]['ordinary_rate']
        if rate is None: raise ValueError('No calibrated ordinary-SGD rate; supply rate explicitly')
        epochs=config['evaluation_epochs'] if epochs is None else epochs
        if rate<=0 or type(epochs) is not int or epochs<1: raise ValueError('Use positive rate and integer epochs')
        training_seed=student.get('training_seed',student['seed']) if seed is None else seed
        train_student(network,torch.optim.SGD(network.parameters(),lr=rate),SyntheticInputs.from_export(artifact),
                      epochs=epochs,batch_size=config['batch_size'],seed=training_seed)
        updates=config['batches']*epochs
    return network,dict(dataset_id=loaded['metadata'].get('dataset_id'),dataset_sha256=loaded['metadata']['sha256'],
        model=kind,model_options=config['model_options'],precision=config['model_precision'],
        initialization=initialization,source_checkpoint=source,seed=actual_seed,training_seed=training_seed,
        recipe=recipe,rate=rate,epochs=epochs,student_updates=updates)
