"""Real subprocess failures, timeouts and resubmission without MPI or accelerators."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

from distillation.experiments import launch


WORKER = r'''
import json, os, sys, time
from pathlib import Path
manifest=json.loads(Path(sys.argv[1]).read_text());row=manifest['experiments'][0]
folder=Path(row['output']);folder.mkdir(parents=True,exist_ok=True)
attempt=int(os.environ['STUDY_ATTEMPT'])
if row['dataset_id']=='crash' and attempt==1:
    (folder/'latest.pt').write_text('saved before failure')
    sys.exit(17)
if row['dataset_id']=='hang':
    time.sleep(60)
if row['dataset_id']=='crash':
    assert (folder/'latest.pt').read_text()=='saved before failure'
time.sleep(.15)
(folder/'artifact.json').write_text(json.dumps({'complete':True}))
(folder/'smoke.json').write_text(json.dumps({'status':'PASS'}))
Path(sys.argv[2]).write_text(json.dumps({'status':'COMPLETE'}))
'''


def manifest(tmp_path, names):
    return dict(root=str(tmp_path), specification={'execution':dict(max_retries=1,
        stall_seconds=2,minimum_attempt_seconds=0,launch_interval=0)}, experiments=[
        dict(dataset_id=name,output=str(tmp_path/name),method='DD',config={'case':30},verify_run=True)
        for name in names])


def command(tmp_path):
    script=tmp_path/'worker.py';script.write_text(WORKER)
    return lambda hostfile, manifest, seconds, result: [sys.executable,str(script),str(manifest),str(result)]


def test_failed_worker_does_not_cancel_sibling_and_retries_checkpoint(tmp_path):
    spec=manifest(tmp_path,['crash','healthy'])
    report=asyncio.run(launch.supervise(spec,['node-a','node-b'],tmp_path/'job1',time.time()+90,
        command=command(tmp_path),poll_seconds=.02))
    assert report['counts']=={'COMPLETE':2}
    assert json.loads((tmp_path/'job1/crash/attempt1/status.json').read_text())['returncode']==17
    assert json.loads((tmp_path/'crash/launch_status.json').read_text())['attempt']==2
    assert json.loads((tmp_path/'crash/launch_status.json').read_text())['host']=='node-b'
    assert json.loads((tmp_path/'healthy/launch_status.json').read_text())['attempt']==1
    again=asyncio.run(launch.supervise(spec,['node-c'],tmp_path/'job2',time.time()+90,
        command=lambda *args: (_ for _ in ()).throw(AssertionError('Completed run relaunched'))))
    assert again['records']==[] and not again['pending']


def test_timeout_quarantines_only_its_host_and_preserves_sibling(tmp_path):
    spec=manifest(tmp_path,['hang','healthy']);spec['specification']['execution'].update(stall_seconds=.4,max_retries=0)
    report=asyncio.run(launch.supervise(spec,['node-a','node-b'],tmp_path/'job',time.time()+90,
        command=command(tmp_path),poll_seconds=.02))
    assert report['counts']=={'TIMED_OUT':1,'COMPLETE':1}
    assert report['quarantined']==['node-a']


def test_resubmission_retries_failed_run_and_finished_requires_replay(tmp_path):
    spec=manifest(tmp_path,['crash']);spec['specification']['execution']['max_retries']=0
    worker=command(tmp_path)
    first=asyncio.run(launch.supervise(spec,['node-a'],tmp_path/'job1',time.time()+90,
        command=worker,poll_seconds=.02))
    assert first['counts']=={'FAILED':1}
    # A new PBS attempt resumes the same output directory; no dataset identity changes.
    script=tmp_path/'worker.py';script.write_text(WORKER.replace("and attempt==1", "and not (folder/'latest.pt').exists()"))
    second=asyncio.run(launch.supervise(spec,['node-b'],tmp_path/'job2',time.time()+90,
        command=worker,poll_seconds=.02))
    assert second['counts']=={'COMPLETE':1}
    (tmp_path/'crash/smoke.json').unlink()
    assert not launch.complete(spec['experiments'][0])


def test_command_binds_exactly_one_host_and_twelve_ranks(tmp_path):
    argv=launch.mpi_command(tmp_path/'hosts',tmp_path/'manifest.json',30,tmp_path/'result.json')
    assert argv[argv.index('-n')+1]=='12'
    assert argv[argv.index('--ppn')+1]=='12'
    assert argv[argv.index('--hostfile')+1]==str(tmp_path/'hosts')
    assert argv[argv.index('/usr/bin/env')+1]==sys.executable


def test_bad_output_directory_does_not_cancel_other_experiment(tmp_path):
    spec=manifest(tmp_path,['broken','healthy']);spec['specification']['execution']['max_retries']=0
    (tmp_path/'broken').write_text('A file, not a run directory')
    report=asyncio.run(launch.supervise(spec,['node-a','node-b'],tmp_path/'job',time.time()+90,
        command=command(tmp_path),poll_seconds=.02))
    assert report['counts']=={'FAILED':1,'COMPLETE':1}


def test_rambo_library_evaluations_remain_serial_on_one_node(tmp_path):
    spec=manifest(tmp_path,['first','second','other'])
    for row in spec['experiments'][:2]:row.update(method='RAMBO',config={'case':30,'generation_id':'library'})
    report=asyncio.run(launch.supervise(spec,['node-a','node-b'],tmp_path/'job',time.time()+90,
        command=command(tmp_path),poll_seconds=.02))
    records={r['dataset_id']:r for r in report['records']}
    assert report['counts']=={'COMPLETE':3}
    assert records['first']['host']==records['second']['host']
    assert records['second']['started']>=records['first']['finished']


def test_heartbeat_content_keeps_worker_alive_with_old_fixed_mtime(tmp_path):
    spec=manifest(tmp_path,['heartbeat']);spec['specification']['execution'].update(stall_seconds=.3,max_retries=0)
    script=tmp_path/'worker.py'
    script.write_text('''
import json, os, sys, time
from pathlib import Path
row=json.loads(Path(sys.argv[1]).read_text())['experiments'][0]
folder=Path(row['output']); folder.mkdir(parents=True,exist_ok=True)
for i in range(8):
    (folder/'status.json').write_text(json.dumps({'updated':i}))
    os.utime(folder/'status.json',(1,1))
    time.sleep(.1)
(folder/'artifact.json').write_text(json.dumps({'complete':True}))
(folder/'smoke.json').write_text(json.dumps({'status':'PASS'}))
Path(sys.argv[2]).write_text(json.dumps({'status':'COMPLETE'}))
''')
    argv=lambda hostfile,manifest,seconds,result:[sys.executable,str(script),str(manifest),str(result)]
    report=asyncio.run(launch.supervise(spec,['node-a'],tmp_path/'job',time.time()+90,command=argv,poll_seconds=.02))
    assert report['counts']=={'COMPLETE':1}


def test_cooperative_deadline_is_resumable_not_failed(tmp_path):
    spec=manifest(tmp_path,['pause']);spec['specification']['execution'].update(stall_seconds=0)
    script=tmp_path/'worker.py'
    script.write_text('''
import json, signal, sys, time
from pathlib import Path
def stop(*args):
    Path(sys.argv[2]).write_text(json.dumps({'status':'RESUMABLE'}))
    sys.exit(0)
signal.signal(signal.SIGTERM,stop)
time.sleep(60)
''')
    argv=lambda hostfile,manifest,seconds,result:[sys.executable,str(script),str(manifest),str(result)]
    report=asyncio.run(launch.supervise(spec,['node-a'],tmp_path/'job',time.time()+30.6,command=argv,poll_seconds=.02))
    assert report['counts']=={'RESUMABLE':1}
