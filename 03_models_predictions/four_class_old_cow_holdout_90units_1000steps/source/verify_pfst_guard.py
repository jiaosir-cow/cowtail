"""Small launch/limit/kill-on-close checks; no stress test or business query."""
import argparse
import ctypes as C
from ctypes import wintypes as W
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil

from safe_server_queue import Job, problems, ready_jobs
from windows_launcher import SuspendedChild, verify_worker_membership
from resource_policy import CPU_PERCENT,MEMORY_GIB

def child(path):
    if sys.stdin.readline().strip()!='GO':raise RuntimeError('No release token')
    membership=verify_worker_membership(os.environ.get('DATASET_X'))
    assert membership['cpu_hard_cap_percent']==CPU_PERCENT
    assert membership['job_memory_gib']==MEMORY_GIB
    path.write_text(json.dumps(membership),encoding='utf-8')
    while True:time.sleep(.2)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--child',action='store_true')
    args=parser.parse_args()
    args.out=args.out.resolve()
    if args.child:return child(args.out)
    args.out.mkdir(parents=True,exist_ok=True)
    child_path=args.out/'worker_membership.json'
    if child_path.exists():raise RuntimeError('Use a fresh guard-test output directory')
    with (args.out/'worker.log').open('ab',buffering=0) as log:
        job=Job();env=os.environ.copy();env['DATASET_X']=job.name
        process=SuspendedChild([sys.executable,'-B',str(Path(__file__).resolve()),'--child','--out',str(child_path)],output=log,cwd=str(Path(__file__).parent),environment=env)
        try:
            job.assign(process)
            assert not child_path.exists() and process.poll() is None
            process.resume();process.stdin.write(b'GO\n');process.stdin.flush();process.stdin.close()
            deadline=time.monotonic()+10
            while not child_path.exists() and time.monotonic()<deadline:time.sleep(.1)
            assert child_path.exists(),'Actual worker never verified membership'
            membership=json.loads(child_path.read_text())
            assert process.poll() is None,'Worker exited before kill-on-close verification'
            # Releasing the last supervisor handle must affect only its own child.
            job.close();code=process.wait(10)
            assert process.poll() is not None,'Job close did not terminate the launcher'
            assert not psutil.pid_exists(membership['actual_worker_pid']),'Actual worker survived job close'
        finally:
            job.close()
            if process.poll() is None:process.terminate();process.wait(10)
            process.close()
    good={'available_ram_gib':64,'free_disk_gib':200,'cpu_percent':10,
          'services':{'8030':{'ok':True,'seconds':.1},'8031':{'ok':True,'seconds':.1},'mysql_listener':{'ok':True}},
          'gpu':{'free_mib':22000,'temperature_c':60}}
    assert not problems(good)
    cases=[]
    for field,value in [('available_ram_gib',20),('free_disk_gib',50),('cpu_percent',90)]:
        s=json.loads(json.dumps(good));s[field]=value;assert problems(s);cases.append(field)
    for field,value in [('free_mib',10000),('temperature_c',79)]:
        s=json.loads(json.dumps(good));s['gpu'][field]=value;assert problems(s);cases.append(field)
    s=json.loads(json.dumps(good));s['services']['8030']['seconds']=1.1;assert problems(s);cases.append('http_latency')
    assert ready_jobs([],set(),{'active':{}})==[]
    report={'status':'PASS','membership':membership,'suspended_before_assignment':True,
            'kill_on_guard_close':True,'observed_exit_code':code,'synthetic_threshold_cases':cases,
            'service_queries_performed':False,'stress_allocation_performed':False}
    (args.out/'GUARD_TEST_PASS.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
