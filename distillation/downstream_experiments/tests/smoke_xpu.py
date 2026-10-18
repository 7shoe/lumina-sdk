"""Prepare/verify a two-node XPU qualification on real case14 DD and RND777 data.

All eight normal benchmark models, two optimizer updates, full 30k validation/test.
Submit the printed manifest using run_downstream.pbs; this script does not submit.
"""
import argparse
from pathlib import Path

from distillation.downstream_experiments.dataset_path import find_datasets
from distillation.downstream_experiments.scripts.campaign import collect, load_config, prepare
from distillation.downstream_experiments.scripts.train import completed, folder_for, source_identity
from distillation.training.T1_single_topo.common import digest, read_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',help='Defaults to a name derived from the current source identity')
    parser.add_argument('--verify',type=Path)
    args=parser.parse_args()
    if args.verify:
        manifest=read_json(args.verify)
        assert len(manifest['runs'])==16
        for run in manifest['runs']:
            folder=folder_for(manifest,run)
            assert completed(folder,run['specification']),folder
            for split in ('val','test'):
                report=read_json(folder/f'{split}-case14.json')
                assert report['metrics']['n_samples']==30_000,(folder,split)
            assert read_json(folder/'status.json')['device'].startswith('xpu:')
        collect(args.verify)
        print('PASS: 16 independent XPU runs; all eight models; full real validation/test splits')
        return
    config=load_config()
    rows=find_datasets(method='DD',model='GAT',case=14,size=100,initialization='random',target_mode='learned')
    assert len(rows)==1
    config.update(name=args.name or 'qualification_xpu_'+digest(source_identity())[:12],cases=[14],seeds=[42],methods=['DD','RND777'],
        dataset_ids=[rows[0]['dataset_id']],creation_initializations=['random'],transfer_pairs=[])
    config['families']=dict(individual=True,fixed=False,mixed_sizes=[],adaptation_sizes=[],joint_size=0)
    config['training'].update(max_updates=2,validation_updates=[2],log_updates=1,checkpoint_seconds=30)
    prepare(config)


if __name__=='__main__':
    main()
