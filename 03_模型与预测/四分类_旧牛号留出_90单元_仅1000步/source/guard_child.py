"""Wait for OS job assignment before importing any scientific libraries."""
import json
import os
from pathlib import Path
import runpy
import sys

config=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
if sys.stdin.readline().strip()!='GO':raise SystemExit('No guard authorization')
from windows_launcher import verify_worker_membership
from background_mode import enter_background_mode
from resource_policy import CPU_PERCENT,MEMORY_GIB,GPU_ALLOCATOR_GIB,GPU_DUTY,CPU_THREADS
membership=verify_worker_membership(os.environ.get('DATASET_X'))
if membership['cpu_hard_cap_percent']!=CPU_PERCENT or membership['job_memory_gib']!=MEMORY_GIB:
    raise RuntimeError('Actual worker quota differs from frozen limits')
membership.update(enter_background_mode())
Path(str(sys.argv[1])+'.worker.json').write_text(json.dumps(membership,indent=2),encoding='utf-8')
print(json.dumps({'guard_membership':membership}),flush=True)
requested_threads=int(config.get('threads',CPU_THREADS))
if requested_threads<1 or requested_threads>CPU_THREADS:
    raise RuntimeError('Worker thread request exceeds the PFST resource policy')
threads=str(requested_threads)
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ[key]=threads
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['PYTHONUNBUFFERED']='1'
cache=Path(__file__).parent/'runtime_cache'/'numba';cache.mkdir(parents=True,exist_ok=True)
os.environ['NUMBA_CACHE_DIR']=str(cache)
if not config.get('gpu',False):os.environ['CUDA_VISIBLE_DEVICES']=''
if config.get('gpu',False):
    if config.get('gpu_pacing_duty')!=GPU_DUTY or config.get('gpu_allocator_gib')!=GPU_ALLOCATOR_GIB:
        raise RuntimeError('GPU job lacks checked throughput policy')
    import torch
    if not torch.cuda.is_available():raise RuntimeError('GPU requested but CUDA unavailable')
    torch.set_num_threads(int(threads));torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(GPU_ALLOCATOR_GIB*1024**3/torch.cuda.get_device_properties(0).total_memory)
script=Path(config['script']).resolve()
sys.path.insert(0,str(script.parent))
sys.argv=[str(script),*config.get('arguments',[])]
runpy.run_path(str(script),run_name='__main__')
