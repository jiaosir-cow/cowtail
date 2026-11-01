"""Fail-closed Windows training queue. It can terminate only its own OS job.

Job CPU hard cap + committed-memory hard cap + low priority are enforced before
the child is released. A lost supervisor closes the job and kills its children.
CUDA allocator caps are not GPU compute caps; GPU jobs need paced training loops.
"""
import argparse
import ctypes as C
from ctypes import wintypes as W
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import urllib.error
import uuid
import psutil
from windows_launcher import SuspendedChild
from resource_policy import CPU_PERCENT,MEMORY_GIB,RAM_RESERVE_GIB,SEVERE_RAM_GIB,GPU_RESERVE_MIB,SYSTEM_CPU_STOP,POLICY_ID,GPU_SLOTS

GIB=1024**3
PROBE_8030='/__DATASET_X'


def now():return datetime.now(timezone.utc).isoformat()


def dump(path,obj):
    tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(obj,indent=2),encoding='utf-8')
    for attempt in range(20):
        try:tmp.replace(path);return
        except PermissionError:
            if attempt==19:raise
            time.sleep(.05)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()


class Basic(C.Structure):
    _fields_=[('ProcessTime',C.c_int64),('JobTime',C.c_int64),('Flags',W.DWORD),
        ('MinWorkingSet',C.c_size_t),('MaxWorkingSet',C.c_size_t),('ActiveProcesses',W.DWORD),
        ('Affinity',C.c_size_t),('Priority',W.DWORD),('Scheduling',W.DWORD)]


class IO(C.Structure):
    _fields_=[(name,C.c_uint64) for name in ['ReadOps','WriteOps','OtherOps','ReadBytes','WriteBytes','OtherBytes']]


class Extended(C.Structure):
    _fields_=[('Basic',Basic),('IO',IO),('ProcessMemory',C.c_size_t),('JobMemory',C.c_size_t),
        ('PeakProcessMemory',C.c_size_t),('PeakJobMemory',C.c_size_t)]


class Rate(C.Structure):
    _fields_=[('Flags',W.DWORD),('CpuRate',W.DWORD)]


class Job:
    def __init__(self,memory_gib=MEMORY_GIB,cpu_percent=CPU_PERCENT):
        if os.name!='nt':raise RuntimeError('Windows job objects required')
        k=self.k=C.WinDLL('kernel32',use_last_error=True)
        k.CreateJobObjectW.argtypes=[C.c_void_p,W.LPCWSTR];k.CreateJobObjectW.restype=W.HANDLE
        k.SetInformationJobObject.argtypes=[W.HANDLE,C.c_int,C.c_void_p,W.DWORD];k.SetInformationJobObject.restype=W.BOOL
        k.QueryInformationJobObject.argtypes=[W.HANDLE,C.c_int,C.c_void_p,W.DWORD,C.c_void_p];k.QueryInformationJobObject.restype=W.BOOL
        k.AssignProcessToJobObject.argtypes=[W.HANDLE,W.HANDLE];k.AssignProcessToJobObject.restype=W.BOOL
        k.TerminateJobObject.argtypes=[W.HANDLE,W.UINT];k.TerminateJobObject.restype=W.BOOL
        k.CloseHandle.argtypes=[W.HANDLE];k.CloseHandle.restype=W.BOOL
        self.name='Local\\DATASET_X'+str(os.getpid())+'_'+uuid.uuid4().hex
        self.h=k.CreateJobObjectW(None,self.name)
        if not self.h:raise C.WinError(C.get_last_error())
        try:
            limits=Extended();limits.Basic.Flags=0x2000|0x200|0x20|0x8 # kill-on-close/job-memory/priority/process-count
            limits.JobMemory=int(memory_gib*GIB);limits.Basic.Priority=0x4000;limits.Basic.ActiveProcesses=16
            self._set(9,limits)
            rate=Rate(1|4,round(cpu_percent*100));self._set(15,rate)
            check=Extended();r=Rate()
            if not k.QueryInformationJobObject(self.h,9,C.byref(check),C.sizeof(check),None):raise C.WinError(C.get_last_error())
            if not k.QueryInformationJobObject(self.h,15,C.byref(r),C.sizeof(r),None):raise C.WinError(C.get_last_error())
            if check.JobMemory!=limits.JobMemory or r.CpuRate!=rate.CpuRate or r.Flags!=5:
                raise RuntimeError('OS limit readback mismatch')
            self.applied={'job_memory_gib':check.JobMemory/GIB,'cpu_hard_cap_percent':r.CpuRate/100,'priority':'BELOW_NORMAL','kill_on_guard_exit':True}
        except BaseException:
            k.CloseHandle(self.h);self.h=None;raise

    def _set(self,kind,value):
        if not self.k.SetInformationJobObject(self.h,kind,C.byref(value),C.sizeof(value)):raise C.WinError(C.get_last_error())

    def assign(self,process):
        if not self.k.AssignProcessToJobObject(self.h,int(process._handle)):raise C.WinError(C.get_last_error())

    def close(self):
        if self.h:self.k.CloseHandle(self.h);self.h=None

    def stop(self):
        if self.h and not self.k.TerminateJobObject(self.h,99):raise C.WinError(C.get_last_error())


def http_status(opener,port):
    start=time.perf_counter()
    if port==8030:
        # A route that does no business work. Expected404 exercises HTTP dispatch
        # without the expensive full-table aggregation performed by /health.
        try:
            with opener.open(f'httPATH_X',timeout=3) as response:
                return {'ok':False,'http':response.status,'seconds':round(time.perf_counter()-start,3),'probe':'http_dispatch_expected_404'}
        except urllib.error.HTTPError as response:
            try:
                return {'ok':response.code==404 and response.headers.get('Server','').lower()=='waitress',
                    'http':response.code,'seconds':round(time.perf_counter()-start,3),'probe':'http_dispatch_expected_404'}
            finally:response.close()
        except Exception as e:return {'ok':False,'error':type(e).__name__,'probe':'http_dispatch_expected_404'}
    try:
        with opener.open('httPATH_X',timeout=3) as response:
            obj=json.load(response)
            return {'ok':response.status==200 and obj.get('data',{}).get('ok') is True,
                'seconds':round(time.perf_counter()-start,3),'probe':'read_only_downloader_health'}
    except Exception as e:return {'ok':False,'error':type(e).__name__,'probe':'read_only_downloader_health'}


def snapshot(root):
    mem=psutil.virtual_memory()
    result={'at':now(),'cpu_percent':psutil.cpu_percent(interval=None),'available_ram_gib':round(mem.available/GIB,2),
        'free_disk_gib':round(shutil.disk_usage(root).free/GIB,2),'services':{},
        'monitor_policy':POLICY_ID,'business_functional_health':'not_exercised'}
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for port in [8030,8031]:
        result['services'][str(port)]=http_status(opener,port)
    # Inspect local listeners, never log into or run queries against MySQL.
    listeners={p.laddr.port:p.pid for p in psutil.net_connections(kind='tcp') if p.status==psutil.CONN_LISTEN}
    result['services']['mysql_listener']={'ok':3306 in listeners,'probe':'passive_listener_no_query','pid':listeners.get(3306)}
    result['business_listener_pids']={str(port):listeners.get(port) for port in (3306,8030,8031)}
    try:
        proc=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.free,temperature.gpu,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=4,creationflags=subprocess.CREATE_NO_WINDOW)
        used,free,temp,util=map(float,proc.stdout.strip().splitlines()[0].split(','))
        result['gpu']={'used_mib':used,'free_mib':free,'temperature_c':temp,'utilization':util}
    except Exception as e:result['gpu']={'error':type(e).__name__}
    return result


def problems(s):
    reasons=[]
    if s['available_ram_gib']<RAM_RESERVE_GIB:reasons.append(f'system RAM reserve below {RAM_RESERVE_GIB} GiB')
    if s['free_disk_gib']<100:reasons.append('disk reserve below 100 GiB')
    if s['cpu_percent']>SYSTEM_CPU_STOP:reasons.append(f'system CPU above {SYSTEM_CPU_STOP} percent')
    for name,status in s['services'].items():
        if not status['ok']:reasons.append('service unhealthy: '+name)
        elif status.get('seconds',0)>1:reasons.append('service latency above 1 second: '+name)
    gpu=s['gpu']
    if 'error' in gpu:reasons.append('GPU monitor unavailable')
    elif gpu['free_mib']<GPU_RESERVE_MIB or gpu['temperature_c']>=78:reasons.append('GPU reserve/temperature threshold')
    return reasons


def ready_jobs(jobs,completed,active):
    """Exactly one PFST worker at a time, including engineering checks."""
    if active:return []
    from collections import Counter
    taken=Counter(r['config']['lane'] for r in active.values())
    if taken['gate']:return []
    result=[]
    for config in jobs:
        if config['label'] in completed or config['label'] in active:continue
        if not set(config.get('requires',[])).issubset(completed):continue
        lane=config['lane']
        if lane=='gate':
            return [] if active or result else [config]
        capacity=GPU_SLOTS if lane=='gpu' else 1
        if taken[lane]<capacity:
            result.append(config);taken[lane]+=1
    return result[:1]


def validate_queue(jobs):
    labels={j['label'] for j in jobs}
    if len(labels)!=len(jobs):raise ValueError('Duplicate labels')
    for j in jobs:
        if not j['label'].replace('_','').replace('-','').isalnum():raise ValueError('Unsafe label')
        if j['lane'] not in ['gate','cpu','gpu']:raise ValueError('Unknown lane')
        if bool(j['gpu'])!=(j['lane']=='gpu'):raise ValueError('GPU lane mismatch')
        if not set(j.get('requires',[])).issubset(labels):raise ValueError('Missing dependency')


def run(args):
    root=Path(args.root).resolve();root.mkdir(exist_ok=True,parents=True)
    if os.name!='nt':raise RuntimeError('Windows only')
    from background_mode import enter_background_mode
    from phase2_common import check_complete
    supervisor_priority=enter_background_mode()
    import msvcrt
    lock=(root/'queue.lock').open('a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    me=psutil.Process();ancestors={p.pid for p in me.parents()}|{me.pid}
    project=str(Path(__file__).parent.parent).lower()
    for p in psutil.process_iter(['pid','name','cmdline']):
        if p.info['pid'] in ancestors or str(p.info['name'] or '').lower()!='python.exe':continue
        cmd=' '.join(p.info['cmdline'] or []).lower().replace('/','\\')
        if project in cmd and ('safe_server_queue.py' in cmd or 'guard_child.py' in cmd):
            raise RuntimeError('Another project training supervisor or child is active')
    jobs=json.loads(Path(args.queue).read_text(encoding='utf-8'));validate_queue(jobs)
    status={'started':now(),'guard_pid':os.getpid(),'queue_sha256':sha(args.queue),'state':'preflight',
        'completed':[],'active':[],'monitor_policy':POLICY_ID,'supervisor_priority':supervisor_priority,
        'resource_scope':'one shared OS Job for all CPU and GPU workers, not per-worker allowances'}
    status_path=root/'STATUS.json';dump(status_path,status)
    source_manifest=Path(__file__).with_name('SOURCE_SHA256SUMS.json')
    for name,digest in json.loads(source_manifest.read_text()).items():
        if sha(Path(__file__).parent/name)!=digest:raise RuntimeError('source hash mismatch: '+name)
    source_hash=sha(source_manifest);status['source_manifest_sha256']=source_hash
    psutil.cpu_percent(interval=None);time.sleep(1);baseline=snapshot(root);dump(root/'baseline.json',baseline)
    if problems(baseline):
        status.update(state='BLOCKED_PREFLIGHT',reason=problems(baseline));dump(status_path,status);return 2
    bundle=Path(args.inputs)/'BUNDLE_SHA256SUMS.json'
    for name,digest in json.loads(bundle.read_text()).items():
        if sha(Path(args.inputs)/name)!=digest:raise RuntimeError('input hash mismatch: '+name)
    input_hash=sha(bundle);status['input_bundle_sha256']=input_hash
    for config in jobs:
        receipt=root/(config['label']+'.done.json')
        if not receipt.exists():continue
        saved=json.loads(receipt.read_text())
        if saved.get('config')!=config or saved.get('source_manifest_sha256')!=source_hash or saved.get('input_bundle_sha256')!=input_hash:
            raise RuntimeError('Existing receipt does not match frozen configuration')
        if not config.get('complete_path') or not check_complete(Path(config['complete_path']).parent):
            raise RuntimeError('Completed job artifacts missing')
        status['completed'].append(config['label'])
    active={};job=Job();status['limits']=job.applied;status['shared_job_name']=job.name;bad_count=0
    def sync_active():
        status['active']=[{'label':label,'lane':r['config']['lane'],'child_pid':r['process'].pid} for label,r in active.items()]
        status['current']=[r['label'] for r in status['active']]
    try:
        while len(status['completed'])<len(jobs):
            if (root/'STOP').exists():
                status.update(state='STOPPED_BY_REQUEST',reason=['STOP requested']);dump(status_path,status);return 2
            sample=snapshot(root);issues=problems(sample)
            if sample['business_listener_pids']!=baseline['business_listener_pids']:
                issues.append('business listener PID changed from the preflight baseline')
            bad_count=bad_count+1 if issues else 0
            limits=Extended()
            if not job.k.QueryInformationJobObject(job.h,9,C.byref(limits),C.sizeof(limits),None):raise C.WinError(C.get_last_error())
            sample['training_peak_committed_gib']=round(limits.PeakJobMemory/GIB,3)
            status.update(last_snapshot=sample,resource_warnings=issues)
            with (root/'resources.jsonl').open('a',encoding='utf-8') as log:
                log.write(json.dumps({'jobs':list(active),**sample})+'\n')
            severe=sample['available_ram_gib']<SEVERE_RAM_GIB or sample['gpu'].get('temperature_c',0)>=82
            if severe or bad_count>=2:
                status.update(state='STOPPED_FOR_SAFETY',reason=issues);dump(status_path,status);return 99
            for label,r in list(active.items()):
                process=r['process'];config=r['config']
                if process.poll() is None:continue
                if process.returncode!=0:
                    status.update(state='JOB_FAILED',failed_job=label,returncode=process.returncode)
                    dump(status_path,status);return process.returncode
                if not check_complete(Path(config['complete_path']).parent):
                    raise RuntimeError('Child exited without complete artifacts: '+label)
                # Do not approve completion while the current service/resource sample is bad.
                if issues:continue
                dump(root/(label+'.done.json'),{'config':config,'finished':now(),'limits':job.applied,
                    'shared_job_name':job.name,'input_bundle_sha256':input_hash,'source_manifest_sha256':source_hash,
                    'health_after':sample})
                status['completed'].append(label);process.close();r['log'].close();del active[label]
            draining=(root/'DRAIN').exists()
            if draining and not active:
                status.update(state='STOPPED_AFTER_JOBS',reason=['Drain requested; no running task interrupted'])
                sync_active();dump(status_path,status);return 0
            if not issues and not draining:
                for config in ready_jobs(jobs,set(status['completed']),active):
                    label=config['label'];config_path=root/(label+'.config.json');dump(config_path,config)
                    output=(root/(label+'.log')).open('ab',buffering=0)
                    env=os.environ.copy();env['DATASET_X']=job.name
                    process=None
                    try:
                        process=SuspendedChild([sys.executable,'-B',str(Path(__file__).with_name('guard_child.py')),str(config_path)],
                            output=output,cwd=config.get('cwd',str(Path(__file__).parent)),environment=env)
                        job.assign(process);process.resume()
                        active[label]={'config':config,'process':process,'log':output}
                        process.stdin.write(b'GO\n');process.stdin.flush();process.stdin.close()
                    except BaseException:
                        if process is not None:
                            process.terminate();process.wait(10);process.close()
                        output.close();raise
            sync_active();status['state']='DRAINING' if draining else 'RUNNING'
            dump(status_path,status)
            if not active and not issues and len(status['completed'])<len(jobs):
                raise RuntimeError('No schedulable task: dependency deadlock')
            if active or issues:time.sleep(5)
        status.update(state='COMPLETE',finished=now());sync_active();dump(status_path,status);return 0
    finally:
        # One shared handle: closing it kills every remaining training descendant,
        # including the other lane on an error, without touching production processes.
        job.close()
        for r in active.values():
            try:r['process'].wait(15)
            finally:r['process'].close();r['log'].close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--queue',required=True);parser.add_argument('--inputs',required=True)
    args=parser.parse_args()
    try:raise SystemExit(run(args))
    except Exception as e:
        target=Path(args.root);target.mkdir(exist_ok=True,parents=True)
        dump(target/'GUARD_ERROR.json',{'at':now(),'error':repr(e)});raise
