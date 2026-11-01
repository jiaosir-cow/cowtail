"""Do not import scientific libraries until the parent assigns its Windows Job."""
import json
import os
from pathlib import Path
import runpy
import sys

from resource_policy import CPU_PERCENT, MEMORY_GIB, CPU_THREADS, GPU_ALLOCATOR_GIB, GPU_DUTY
from windows_launcher import verify_worker_membership


def main():
    config_path = Path(sys.argv[1])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if sys.stdin.readline().strip() != "GO":
        raise RuntimeError("No resource-guard authorization")
    membership = verify_worker_membership(os.environ.get("DATASET_X"))
    if membership["cpu_hard_cap_percent"] != CPU_PERCENT or membership["job_memory_gib"] != MEMORY_GIB:
        raise RuntimeError("Actual OS Job quotas differ from frozen resource limits")
    Path(str(config_path) + ".worker.json").write_text(json.dumps(membership, indent=2), encoding="utf-8")
    root = Path(config["root"]).resolve()
    for variable, relative in {
        "TEMP": "runtime/temp", "TMP": "runtime/temp", "TORCH_HOME": "runtime/cache/torch",
        "XDG_CACHE_HOME": "runtime/cache", "HF_HOME": "runtime/cache/huggingface",
        "NUMBA_CACHE_DIR": "runtime/cache/numba", "MPLCONFIGDIR": "runtime/cache/matplotlib",
        "CUDA_CACHE_PATH": "runtime/cache/cuda", "TORCHINDUCTOR_CACHE_DIR": "runtime/cache/inductor",
        "PYTHONPYCACHEPREFIX": "runtime/cache/pycache",
    }.items():
        target = root / relative
        target.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(target)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = str(CPU_THREADS)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["DATASET_X"] = str(GPU_DUTY)
    os.environ["DATASET_X"] = str(GPU_ALLOCATOR_GIB)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU fallback is forbidden")
    if "3090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("The selected CUDA device is not the authorized RTX 3090")
    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(1)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(GPU_ALLOCATOR_GIB * 1024 ** 3 / total, 0)
    script = root / "source/device_reference_worker.py"
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script), "--root", str(root), "--unit", config["unit_id"], "--mode", config["mode"]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
