"""Fail-closed, single-worker Windows queue. Stops only its own named OS Job.

The CPU quota and aggregate committed-memory quota are Windows hard limits.
CUDA's allocator cap excludes driver/context allocations and is not a GPU compute
quota. The scientific worker must synchronize and pace every training batch.
"""
import argparse
import ctypes as C
from ctypes import wintypes as W
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
import uuid

import psutil
from resource_policy import (
    CPU_PERCENT, MEMORY_GIB, CPU_THREADS, GPU_ALLOCATOR_GIB, GPU_DUTY,
    RAM_RESERVE_GIB, SEVERE_RAM_GIB, GPU_RESERVE_MIB, GPU_TEMPERATURE_STOP,
    GPU_TEMPERATURE_SEVERE, SYSTEM_CPU_STOP, SYSTEM_GPU_STOP, DISK_RESERVE_GIB,
    SAMPLE_SECONDS, PREFLIGHT_SAMPLES, UNIT_TIMEOUT_SECONDS, QUEUE_TIMEOUT_SECONDS,
    POLICY_ID, EXPECTED_HOST, EXPECTED_ROOT, PROTECTED_PORTS,
)
from windows_launcher import SuspendedChild

GIB = 1024 ** 3


def now():
    return datetime.now(timezone.utc).isoformat()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    for retry in range(20):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if retry == 19:
                raise
            time.sleep(0.05)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def contained(root, relative):
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise RuntimeError("Frozen path escapes study root: " + str(relative))
    return path


class Basic(C.Structure):
    _fields_ = [("ProcessTime", C.c_int64), ("JobTime", C.c_int64), ("Flags", W.DWORD),
                ("MinWorkingSet", C.c_size_t), ("MaxWorkingSet", C.c_size_t), ("ActiveProcesses", W.DWORD),
                ("Affinity", C.c_size_t), ("Priority", W.DWORD), ("Scheduling", W.DWORD)]


class IO(C.Structure):
    _fields_ = [(name, C.c_uint64) for name in ("ReadOps", "WriteOps", "OtherOps", "ReadBytes", "WriteBytes", "OtherBytes")]


class Extended(C.Structure):
    _fields_ = [("Basic", Basic), ("IO", IO), ("ProcessMemory", C.c_size_t), ("JobMemory", C.c_size_t),
                ("PeakProcessMemory", C.c_size_t), ("PeakJobMemory", C.c_size_t)]


class Rate(C.Structure):
    _fields_ = [("Flags", W.DWORD), ("CpuRate", W.DWORD)]


class Job:
    def __init__(self):
        if os.name != "nt":
            raise RuntimeError("Windows Job objects are required")
        self.k = k = C.WinDLL("kernel32", use_last_error=True)
        k.CreateJobObjectW.argtypes = [C.c_void_p, W.LPCWSTR]
        k.CreateJobObjectW.restype = W.HANDLE
        for name, args in {
            "SetInformationJobObject": [W.HANDLE, C.c_int, C.c_void_p, W.DWORD],
            "QueryInformationJobObject": [W.HANDLE, C.c_int, C.c_void_p, W.DWORD, C.c_void_p],
            "AssignProcessToJobObject": [W.HANDLE, W.HANDLE],
            "TerminateJobObject": [W.HANDLE, W.UINT], "CloseHandle": [W.HANDLE],
        }.items():
            function = getattr(k, name)
            function.argtypes = args
            function.restype = W.BOOL
        self.name = "Local\\DATASET_X" + str(os.getpid()) + "_" + uuid.uuid4().hex
        self.h = k.CreateJobObjectW(None, self.name)
        if not self.h:
            raise C.WinError(C.get_last_error())
        try:
            limits = Extended()
            # KILL_ON_JOB_CLOSE | JOB_MEMORY | PRIORITY_CLASS | ACTIVE_PROCESS.
            limits.Basic.Flags = 0x2000 | 0x200 | 0x20 | 0x8
            limits.JobMemory = MEMORY_GIB * GIB
            limits.Basic.Priority = 0x4000
            limits.Basic.ActiveProcesses = 16
            self._set(9, limits)
            self._set(15, Rate(5, round(CPU_PERCENT * 100)))
            actual, rate = Extended(), Rate()
            if not k.QueryInformationJobObject(self.h, 9, C.byref(actual), C.sizeof(actual), None):
                raise C.WinError(C.get_last_error())
            if not k.QueryInformationJobObject(self.h, 15, C.byref(rate), C.sizeof(rate), None):
                raise C.WinError(C.get_last_error())
            if actual.JobMemory != MEMORY_GIB * GIB or rate.CpuRate != round(CPU_PERCENT * 100) or rate.Flags != 5:
                raise RuntimeError("Windows hard-limit readback failed")
            self.applied = {"cpu_hard_cap_percent": rate.CpuRate / 100,
                            "job_memory_gib": actual.JobMemory / GIB, "priority": "BELOW_NORMAL",
                            "kill_on_last_handle_close": True, "single_worker": True}
        except BaseException:
            self.close()
            raise

    def _set(self, information_class, value):
        if not self.k.SetInformationJobObject(self.h, information_class, C.byref(value), C.sizeof(value)):
            raise C.WinError(C.get_last_error())

    def assign(self, process):
        if not self.k.AssignProcessToJobObject(self.h, int(process._handle)):
            raise C.WinError(C.get_last_error())

    def peak_gib(self):
        actual = Extended()
        if not self.k.QueryInformationJobObject(self.h, 9, C.byref(actual), C.sizeof(actual), None):
            raise C.WinError(C.get_last_error())
        return actual.PeakJobMemory / GIB

    def close(self):
        if getattr(self, "h", None):
            self.k.CloseHandle(self.h)
            self.h = None


def snapshot(root):
    sample = {"at": now(), "cpu_percent": psutil.cpu_percent(interval=None),
              "available_ram_gib": psutil.virtual_memory().available / GIB,
              "free_disk_gib": shutil.disk_usage(root).free / GIB, "listener_pids": {},
              "listener_created": {}, "policy_id": POLICY_ID,
              "business_health_probe": "passive_listener_pid_only_no_queries"}
    listeners = {}
    for connection in psutil.net_connections(kind="tcp"):
        if connection.status == psutil.CONN_LISTEN and connection.laddr.port in PROTECTED_PORTS:
            listeners.setdefault(connection.laddr.port, set()).add(connection.pid)
    for port in PROTECTED_PORTS:
        pids = sorted(pid for pid in listeners.get(port, set()) if pid is not None)
        sample["listener_pids"][str(port)] = pids
        sample["listener_created"][str(port)] = {str(pid): psutil.Process(pid).create_time() for pid in pids}
    try:
        result = subprocess.run([
            "nvidia-smi", "--id=0", "--query-gpu=name,memory.used,memory.free,temperature.gpu,utilization.gpu",
            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=4,
            creationflags=subprocess.CREATE_NO_WINDOW)
        if result.returncode:
            raise RuntimeError("nvidia-smi failed")
        name, used, free, temperature, utilization = [part.strip() for part in result.stdout.strip().split(",")]
        sample["gpu"] = {"name": name, "used_mib": float(used), "free_mib": float(free),
                         "temperature_c": float(temperature), "utilization": float(utilization)}
    except Exception as error:
        sample["gpu"] = {"error": type(error).__name__}
    return sample


def problems(sample, baseline=None):
    issues = []
    if sample["available_ram_gib"] < RAM_RESERVE_GIB:
        issues.append("RAM reserve below 24 GiB")
    if sample["free_disk_gib"] < DISK_RESERVE_GIB:
        issues.append("C disk reserve below 100 GiB")
    if sample["cpu_percent"] > SYSTEM_CPU_STOP:
        issues.append("system CPU above 80 percent")
    for port, pids in sample["listener_pids"].items():
        if not pids:
            issues.append("business port listener absent: " + port)
    if baseline is not None and (sample["listener_pids"] != baseline["listener_pids"] or
                                 sample["listener_created"] != baseline["listener_created"]):
        issues.append("business listener PID or creation time changed")
    gpu = sample["gpu"]
    if "error" in gpu:
        issues.append("GPU telemetry unavailable")
    else:
        if "3090" not in gpu["name"]:
            issues.append("GPU 0 is not the authorized RTX 3090")
        if gpu["free_mib"] < GPU_RESERVE_MIB:
            issues.append("GPU free memory below 14 GiB")
        if gpu["temperature_c"] >= GPU_TEMPERATURE_STOP:
            issues.append("GPU temperature at least 75 C")
        if gpu["utilization"] > SYSTEM_GPU_STOP:
            issues.append("GPU utilization above 90 percent")
    return issues


def verify_frozen(root):
    manifest_path = root / "configs/FROZEN_PROTOCOL.json"
    expected = os.environ.get("DATASET_X", "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected) or sha(manifest_path) != expected:
        raise RuntimeError("Canonical frozen manifest SHA256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    artifacts = manifest["artifact_sha256"]
    required = {"source/safe_server_queue.py", "source/guard_child.py", "source/windows_launcher.py",
                "source/resource_policy.py", "source/device_reference_worker.py", "configs/queue.json"}
    if not required.issubset(artifacts):
        raise RuntimeError("Frozen manifest does not cover guard, worker and queue")
    for relative, digest in artifacts.items():
        if sha(contained(root, relative)) != digest.lower():
            raise RuntimeError("Frozen artifact mismatch: " + relative)
    if sha(root / "PROTOCOL.md") != manifest["protocol_text_sha256"].lower():
        raise RuntimeError("Frozen protocol text mismatch")
    return expected


def load_queue(path, mode):
    queue = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if queue.get("schema_version") != 1:
        raise RuntimeError("Unsupported queue schema")
    def ids(items):
        return [item if isinstance(item, str) else item["unit_id"] for item in items]
    formal, smoke = ids(queue["formal"]), ids(queue["smoke"])
    if len(formal) != 96 or len(set(formal)) != 96 or len(smoke) != 1 or smoke[0] not in formal:
        raise RuntimeError("Expected 96 unique formal units and one smoke unit")
    for unit in formal:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", unit):
            raise RuntimeError("Unsafe unit identifier")
    return smoke if mode == "smoke" else formal


def check_complete(root, unit, mode, frozen_sha):
    directory = root / ("results" if mode == "formal" else "smoke") / unit
    path = directory / "COMPLETE.json"
    if not path.is_file():
        raise RuntimeError("Worker exited without COMPLETE.json: " + unit)
    receipt = json.loads(path.read_text(encoding="utf-8-sig"))
    if (receipt.get("status") != "COMPLETE" or receipt.get("unit_id") != unit or
            receipt.get("mode") != mode or receipt.get("frozen_sha256") != frozen_sha):
        raise RuntimeError("Worker completion provenance mismatch: " + unit)
    for relative, digest in receipt.get("output_sha256", {}).items():
        if sha(contained(directory.resolve(), relative)) != digest:
            raise RuntimeError("Completed output hash mismatch: " + unit + "/" + relative)
    return {"complete_sha256": sha(path), "path": str(path)}


def run(args):
    if os.name != "nt" or socket.gethostname().upper() != EXPECTED_HOST:
        raise RuntimeError("Unexpected execution host")
    root = Path(args.root).resolve()
    if str(root).replace("\\", "/").lower() != EXPECTED_ROOT.lower():
        raise RuntimeError("Runtime must stay in the authorized C-drive study folder")
    if Path(args.queue).resolve() != root / "configs/queue.json":
        raise RuntimeError("Only the canonical frozen queue is accepted")
    frozen = verify_frozen(root)
    units = load_queue(args.queue, args.mode)
    supervisor = root / "supervisor"
    directory = supervisor / args.mode
    directory.mkdir(parents=True, exist_ok=True)
    import msvcrt
    lock = (supervisor / "queue.lock").open("a+b")
    lock.seek(0)
    lock.write(b"0")
    lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    me = psutil.Process()
    me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    status = {"state": "PREFLIGHT", "started": now(), "mode": args.mode,
              "guard_pid": os.getpid(), "completed": [], "total": len(units),
              "frozen_sha256": frozen, "queue_sha256": sha(args.queue), "policy_id": POLICY_ID,
              "hidden": False, "audit_records_retained": True}
    def save():
        dump(directory / "STATUS.json", status)
        dump(supervisor / "STATUS.json", status)
    save()
    if args.mode == "formal":
        smoke_status = json.loads((supervisor / "smoke/STATUS.json").read_text(encoding="utf-8"))
        if smoke_status.get("state") != "COMPLETE" or smoke_status.get("frozen_sha256") != frozen:
            raise RuntimeError("A complete guarded GPU smoke with the same frozen inputs is required")
        check_complete(root, load_queue(args.queue, "smoke")[0], "smoke", frozen)
    baseline = None
    psutil.cpu_percent(interval=None)
    time.sleep(1)
    for index in range(PREFLIGHT_SAMPLES):
        sample = snapshot(root)
        issues = problems(sample, baseline)
        dump(directory / ("preflight_%02d.json" % index), sample)
        if issues:
            status.update(state="BLOCKED_PREFLIGHT", reason=issues, last_snapshot=sample)
            save()
            return 2
        if baseline is None:
            baseline = sample
        if index + 1 < PREFLIGHT_SAMPLES:
            time.sleep(SAMPLE_SECONDS)
    status["baseline_listener_pids"] = baseline["listener_pids"]
    for unit in units:
        done = directory / (unit + ".done.json")
        if done.exists():
            previous = json.loads(done.read_text(encoding="utf-8"))
            if previous.get("frozen_sha256") != frozen or previous.get("mode") != args.mode:
                raise RuntimeError("Existing guard receipt differs from frozen run")
            complete = check_complete(root, unit, args.mode, frozen)
            if previous.get("completion") != complete:
                raise RuntimeError("Completed unit changed after guard verification")
            status["completed"].append(unit)
    job = Job()
    status["limits"] = job.applied
    status["shared_job_name"] = job.name
    started = time.monotonic()
    active = None
    bad_count = 0
    try:
        while len(status["completed"]) < len(units):
            if (root / "STOP").exists():
                status.update(state="STOPPED_BY_REQUEST", reason=["study STOP file"])
                save()
                return 2
            if time.monotonic() - started >= QUEUE_TIMEOUT_SECONDS:
                status.update(state="STOPPED_QUEUE_TIMEOUT", reason=["12-hour queue limit"])
                save()
                return 2
            sample = snapshot(root)
            issues = problems(sample, baseline)
            sample["training_peak_committed_gib"] = job.peak_gib()
            bad_count = bad_count + 1 if issues else 0
            with (directory / "resources.jsonl").open("a", encoding="utf-8") as output:
                output.write(json.dumps({**sample, "active_unit": active["unit"] if active else None,
                                         "resource_warnings": issues}) + "\n")
            status.update(last_snapshot=sample, resource_warnings=issues)
            severe = (sample["available_ram_gib"] < SEVERE_RAM_GIB or
                      sample["gpu"].get("temperature_c", 0) >= GPU_TEMPERATURE_SEVERE)
            if severe or bad_count >= 2:
                status.update(state="STOPPED_FOR_SAFETY", reason=issues)
                save()
                return 99
            if active is not None:
                process, unit = active["process"], active["unit"]
                if time.monotonic() - active["started"] >= UNIT_TIMEOUT_SECONDS:
                    status.update(state="STOPPED_UNIT_TIMEOUT", failed_unit=unit, reason=["20-minute unit limit"])
                    save()
                    return 2
                if process.poll() is not None:
                    if process.returncode != 0:
                        status.update(state="UNIT_FAILED", failed_unit=unit, returncode=process.returncode)
                        save()
                        return process.returncode
                    complete = check_complete(root, unit, args.mode, frozen)
                    if not issues:
                        dump(directory / (unit + ".done.json"), {"unit_id": unit, "mode": args.mode,
                             "finished": now(), "frozen_sha256": frozen, "completion": complete,
                             "limits": job.applied, "health_after": sample})
                        status["completed"].append(unit)
                        process.close()
                        active["output"].close()
                        active = None
            if active is None and not issues and len(status["completed"]) < len(units):
                if (root / "DRAIN").exists():
                    status.update(state="STOPPED_AFTER_UNIT", reason=["study DRAIN file"])
                    save()
                    return 0
                verify_frozen(root)
                unit = next(unit for unit in units if unit not in status["completed"])
                config_path = directory / (unit + ".config.json")
                dump(config_path, {"root": str(root), "unit_id": unit, "mode": args.mode,
                                   "gpu_allocator_gib": GPU_ALLOCATOR_GIB, "gpu_pacing_duty": GPU_DUTY})
                output = (directory / (unit + ".log")).open("ab", buffering=0)
                env = os.environ.copy()
                env["DATASET_X"] = job.name
                process = None
                try:
                    process = SuspendedChild([sys.executable, "-B", str(root / "source/guard_child.py"), str(config_path)],
                                             output=output, cwd=str(root / "source"), environment=env)
                    job.assign(process)
                    process.resume()
                    active = {"process": process, "unit": unit, "output": output, "started": time.monotonic()}
                    process.stdin.write(b"GO\n")
                    process.stdin.flush()
                    process.stdin.close()
                except BaseException:
                    if process is not None:
                        process.terminate()
                        process.wait(10)
                        process.close()
                    output.close()
                    raise
            status.update(state="RUNNING", active_unit=active["unit"] if active else None,
                          worker_pid=active["process"].pid if active else None)
            save()
            if active is not None or issues:
                time.sleep(SAMPLE_SECONDS)
        status.update(state="COMPLETE", finished=now(), active_unit=None, worker_pid=None)
        save()
        return 0
    except BaseException as error:
        status.update(state="GUARD_ERROR", error=repr(error))
        save()
        raise
    finally:
        # No scan/kill of unrelated processes. Closing this unique Job kills only
        # descendants previously assigned by this supervisor.
        job.close()
        if active is not None:
            try:
                active["process"].wait(15)
            finally:
                active["process"].close()
                active["output"].close()
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    arguments = parser.parse_args()
    try:
        raise SystemExit(run(arguments))
    except Exception as error:
        destination = Path(arguments.root) / "supervisor" / arguments.mode / "GUARD_ERROR.json"
        dump(destination, {"at": now(), "error": repr(error)})
        raise
