"""Guarded fixed-budget hierarchy worker: no held-out labels or score selection."""
from __future__ import annotations

import argparse
import copy
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
ENDPOINTS = (1000, 3000)
METHODS = {"scratch", "act", "ufro"}
# Numerical libraries are imported only after the formal Job membership gate.
np = torch = hm = BaseModel = None


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def canonical_hash(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False,
                     separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def relative_path(name):
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("Expected an experiment-relative path")
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT.resolve()) or path == ROOT.resolve():
        raise ValueError(f"Path outside isolated root: {name}")
    return path


def verify_guard():
    name = os.environ.get("DATASET_X", "")
    if not name or os.name != "nt":
        raise RuntimeError("Formal training requires named Windows Job membership")
    from ctypes import wintypes as w
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenJobObjectW.argtypes = [w.DWORD, w.BOOL, w.LPCWSTR]
    kernel.OpenJobObjectW.restype = w.HANDLE
    kernel.GetCurrentProcess.restype = w.HANDLE
    kernel.IsProcessInJob.argtypes = [w.HANDLE, w.HANDLE, ctypes.POINTER(w.BOOL)]
    kernel.IsProcessInJob.restype = w.BOOL
    kernel.CloseHandle.argtypes = [w.HANDLE]
    handle = kernel.OpenJobObjectW(0x0004, False, name)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        member = w.BOOL(False)
        if not kernel.IsProcessInJob(kernel.GetCurrentProcess(), handle, ctypes.byref(member)):
            raise ctypes.WinError(ctypes.get_last_error())
        if not member.value:
            raise RuntimeError("Worker is not in the configured Windows Job")
    finally:
        kernel.CloseHandle(handle)
    return {"name": name, "membership_verified": True, "pid": os.getpid()}


def initialize_runtime(device_name, formal=False):
    global np, torch, hm, BaseModel
    import numpy as numpy_module
    import torch as torch_module
    import hierarchy_model as hierarchy_module
    from model_paper_kernel8 import PaperKernel8CCNN
    np, torch, hm, BaseModel = numpy_module, torch_module, hierarchy_module, PaperKernel8CCNN
    torch.set_num_threads(2)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if formal and device_name != "cuda":
        raise RuntimeError("Formal execution requires CUDA; no CPU fallback")
    device = torch.device("cuda:0" if device_name == "cuda" else "cpu")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but unavailable")
        torch.cuda.set_device(device)
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, 1 * 2**30 / total), device)
    return device


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_unit(unit_id):
    if not unit_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in unit_id):
        raise ValueError("Invalid unit identifier")
    units = read_json(ROOT / "configs/units.json")
    if "units" in units:
        units = units["units"]
    if isinstance(units, list):
        units = {unit["unit_id"]: unit for unit in units}
    if unit_id not in units:
        raise ValueError(f"Unknown unit: {unit_id}")
    config = units[unit_id]
    required = {"unit_id", "fold_id", "test_cow", "replicate", "percent", "method",
                "budget_dir", "fold_dir", "target_seed"}
    if not required.issubset(config) or config["unit_id"] != unit_id:
        raise ValueError("Incomplete unit contract")
    if config["method"] not in METHODS or int(config["percent"]) not in {5, 10, 25, 50, 100}:
        raise ValueError("Unexpected method or label budget")
    if int(config["replicate"]) not in {0, 1, 2}:
        raise ValueError("Unexpected replicate")
    folds = read_json(ROOT / "configs/folds.json")
    if config["fold_id"] not in folds:
        raise ValueError("Unexpected fold")
    fold_number = int(config["fold_id"][1:])
    expected_seed = 20260909 + 10000 * fold_number + 100 * int(config["replicate"])
    if int(config["target_seed"]) != expected_seed:
        raise ValueError("Target initialization seed contract changed")
    expected_cow = folds[config["fold_id"]]["test_cow"]
    if config["test_cow"] != expected_cow:
        raise ValueError("Test cow/fold mismatch")
    for key in ("budget_dir", "fold_dir"):
        relative_path(config[key])
    if config["fold_dir"] != f"inputs/folds/{config['fold_id']}":
        raise ValueError("Own-fold label directory mismatch")
    expected_budget = f"inputs/budgets/{config['fold_id']}/rep{int(config['replicate']):02d}/p{int(config['percent']):03d}"
    if config["budget_dir"] != expected_budget:
        raise ValueError("Budget directory does not match fold/replicate/percentage")
    if config["method"] != "scratch":
        expected = f"inputs/checkpoints/{config['method']}_seed{config['replicate']}.pt"
        if config.get("checkpoint") != expected:
            raise ValueError("Source checkpoint must match condition and archive replicate")
    elif config.get("checkpoint") not in (None, ""):
        raise ValueError("Scratch cannot read a source checkpoint")
    return config


def relevant_paths(config):
    names = ["configs/units.json", "configs/folds.json", "source/budget_runner.py", "source/hierarchy_model.py",
             "source/model_paper_kernel8.py", "inputs/acceleration.npy", "inputs/windows.csv"]
    names += [f"{config['fold_dir']}/{item}" for item in ("train_labels.npy", "test_indices.npy")]
    names += [f"{config['budget_dir']}/{item}" for item in
              ("train_indices.npy", "batches.npy", "selected_event_ids.json", "summary.json")]
    if config["method"] != "scratch":
        names.append(config["checkpoint"])
    return names


def check_frozen(config):
    path = ROOT / "configs/FROZEN_PROTOCOL.json"
    protocol = read_json(path)
    declared = {}
    for key in ("artifact_sha256", "source_sha256", "input_sha256"):
        declared.update(protocol.get(key, {}))
    bound = {}
    # Do not open another fold's labels, even to hash them. Its complete manifest
    # remains externally bound, while this worker verifies only its own inputs.
    for name in relevant_paths(config):
        if name not in declared or not isinstance(declared[name], str):
            raise ValueError(f"Required frozen hash missing: {name}")
        digest = sha(relative_path(name))
        if digest != declared[name].lower():
            raise ValueError(f"Frozen artifact changed: {name}")
        bound[name] = digest
    return {"protocol_sha256": sha(path), "artifact_sha256": bound,
            "unit_config_sha256": canonical_hash(config)}


def load_inputs(config):
    fold = relative_path(config["fold_dir"])
    budget = relative_path(config["budget_dir"])
    X = np.load(ROOT / "inputs/acceleration.npy", mmap_mode="r", allow_pickle=False)
    y = np.load(fold / "train_labels.npy", mmap_mode="r", allow_pickle=False)
    test = np.load(fold / "test_indices.npy", allow_pickle=False)
    train = np.load(budget / "train_indices.npy", allow_pickle=False)
    batches = np.load(budget / "batches.npy", allow_pickle=False)
    summary = read_json(budget / "summary.json")
    events = read_json(budget / "selected_event_ids.json")
    if X.dtype != np.float32 or X.shape != (28911, 3, 256):
        raise ValueError("Acceleration tensor contract mismatch")
    if y.dtype != np.int64 or y.shape != (len(X),):
        raise ValueError("Own-fold label vector contract mismatch")
    for name, indices in (("train", train), ("test", test)):
        if indices.ndim != 1 or indices.dtype.kind not in "iu" or not len(indices):
            raise ValueError(f"Invalid {name} indices")
        if indices.min() < 0 or indices.max() >= len(X) or len(np.unique(indices)) != len(indices):
            raise ValueError(f"Out of bounds or duplicate {name} indices")
    if np.intersect1d(train, test).size or not np.all(y[test] == -1):
        raise ValueError("Held-out rows are exposed or intersect training")
    if np.any((y[train] < 0) | (y[train] > 3)):
        raise ValueError("Selected training rows lack known labels")
    if batches.shape != (3000, 32) or batches.dtype.kind not in "iu":
        raise ValueError("Expected the frozen 3000 by 32 batch matrix")
    if not np.isin(batches, train).all():
        raise ValueError("Batch references an unselected or held-out row")
    if int(summary["selected_window_count"]) != len(train):
        raise ValueError("Selected-window count mismatch")
    for key, value in (("fold_id", config["fold_id"]), ("test_cow", config["test_cow"]),
                       ("replicate", int(config["replicate"]))):
        if summary.get(key) != value:
            raise ValueError(f"Budget summary mismatch: {key}")
    if int(summary["nominal_percent"]) != int(config["percent"]):
        raise ValueError("Budget percentage mismatch")
    if isinstance(events, dict):
        events = events.get("selected_event_ids", events.get("event_ids"))
    if not isinstance(events, list) or len(set(events)) != len(events):
        raise ValueError("Selected event list invalid")
    if len(events) != int(summary["selected_event_count"]):
        raise ValueError("Selected-event count mismatch")
    # Deliberately only train support: test labels are unavailable to this worker.
    support = {hm.STATE_NAMES[i]: int(np.count_nonzero(y[train] == i)) for i in range(4)}
    return X, y, train, test, batches, summary, support


def source_state(config):
    if config["method"] == "scratch":
        return None, None
    path = relative_path(config["checkpoint"])
    before = sha(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if set(payload) != {"model_state", "metadata_json"}:
        raise ValueError("Source checkpoint format changed")
    metadata = json.loads(payload["metadata_json"])
    if metadata.get("source") != config["method"] or int(metadata.get("seed", -1)) != int(config["replicate"]):
        raise ValueError("Source checkpoint metadata mismatch")
    if int(metadata.get("matched_optimizer_steps", -1)) != 1600:
        raise ValueError("Source pretraining-step contract mismatch")
    if sha(path) != before:
        raise ValueError("Source checkpoint changed during load")
    return payload["model_state"], {"path": config["checkpoint"], "sha256": before, "metadata": metadata}


class Pacer:
    def __init__(self, device, duty=0.25, sleep_fn=time.sleep, clock_fn=time.perf_counter):
        if not 0 < duty <= 0.5:
            raise ValueError("Duty must be in (0, 0.5]")
        self.device, self.duty = device, duty
        self.sleep_fn, self.clock_fn = sleep_fn, clock_fn
        self.busy = self.slept = 0.0
        self.count = 0

    def start(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return self.clock_fn()

    def finish(self, start):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = max(0.0, self.clock_fn() - start)
        idle = elapsed * (1 / self.duty - 1)
        if idle:
            self.sleep_fn(idle)
        self.busy += elapsed
        self.slept += idle
        self.count += 1


class NoPacer:
    def start(self):
        return None

    def finish(self, start):
        return None


def update(model, optimizer, inputs, labels, pacer):
    start = pacer.start()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = hm.masked_losses(model(inputs), labels)
    if not bool(torch.stack([torch.isfinite(value) for value in losses.values()]).all()):
        raise FloatingPointError("Nonfinite training loss")
    losses["total"].backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    model.project_constraints()
    if not bool(torch.stack([torch.isfinite(value).all() for value in model.state_dict().values()]).all()):
        raise FloatingPointError("Nonfinite model state")
    values = {key: float(value.detach().cpu()) for key, value in losses.items()}
    values["gradient_norm_before_clip"] = float(norm.detach().cpu())
    pacer.finish(start)
    return values


def predict(model, X, indices, device, pacer):
    # No target-label argument exists in the inference interface.
    previous_mode = model.training
    model.eval()
    before = hm.state_hash(model.state_dict())
    state_chunks, head_chunks = [], []
    with torch.inference_mode():
        for offset in range(0, len(indices), 256):
            start = pacer.start()
            values = np.array(X[indices[offset:offset + 256]], dtype=np.float32, copy=True)
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite inference input")
            probabilities = model.compose_probabilities(model(torch.from_numpy(values).to(device)))
            state_chunks.append(probabilities["state"].cpu().numpy())
            head_chunks.append(torch.stack([probabilities[key] for key in hm.HEAD_NAMES], dim=1).cpu().numpy())
            pacer.finish(start)
    states, heads = np.concatenate(state_chunks), np.concatenate(head_chunks)
    if not np.isfinite(states).all() or not np.isfinite(heads).all():
        raise FloatingPointError("Nonfinite held-out predictions")
    if not np.allclose(states.sum(axis=1), 1.0, atol=5e-7, rtol=0):
        raise ValueError("Four-state probabilities do not sum to one")
    if before != hm.state_hash(model.state_dict()):
        raise RuntimeError("Inference changed model state")
    model.train(previous_mode)
    return states.astype(np.float32), heads.astype(np.float32)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    return copy.deepcopy(value)


def rng_state():
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy": {"algorithm": numpy_state[0], "state": numpy_state[1].tolist(),
                      "position": numpy_state[2], "has_gauss": numpy_state[3],
                      "cached_gaussian": numpy_state[4]}}


def new_attempt(run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    for number in range(1, 10000):
        attempt = run_dir / f"attempt_{number:03d}"
        try:
            attempt.mkdir(exist_ok=False)
            return attempt
        except FileExistsError:
            continue
    raise RuntimeError("Attempt limit reached")


def verified_complete(run_dir, binding):
    receipt_path = run_dir / "COMPLETE.json"
    if not receipt_path.is_file():
        return False
    receipt = read_json(receipt_path)
    if receipt.get("status") != "complete" or receipt.get("binding") != binding:
        raise ValueError("Existing completion has another protocol/input binding")
    hashes = receipt.get("artifact_sha256", {})
    if not hashes:
        raise ValueError("Completion has no artifact hashes")
    for name, digest in hashes.items():
        path = (run_dir / name).resolve()
        if not path.is_relative_to(run_dir.resolve()) or not path.is_file() or sha(path) != digest:
            raise ValueError(f"Completed artifact changed: {name}")
    return True


def save_endpoint(attempt, step, model, optimizer, X, test, config, binding, device, pacer):
    folder = attempt / f"step{step}"
    folder.mkdir(exist_ok=False)
    model_hash = hm.state_hash(model.state_dict())
    payload = {"model_state": cpu_tree(model.state_dict()),
               "optimizer_state": cpu_tree(optimizer.state_dict()), "step": step,
               "rng_state": rng_state(), "metadata": {"unit": config, "binding": binding,
                   "state_sha256": model_hash, "endpoint": "PRIMARY" if step == 1000 else "SENSITIVITY"}}
    torch.save(payload, folder / "checkpoint.pt")
    restored = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
    if hm.state_hash(restored["model_state"]) != model_hash or restored["step"] != step:
        raise RuntimeError("Safe checkpoint readback failed")
    states, heads = predict(model, X, test, device, pacer)
    np.save(folder / "test_state_probabilities.npy", states, allow_pickle=False)
    np.save(folder / "test_head_probabilities.npy", heads, allow_pickle=False)
    np.save(folder / "test_indices.npy", test, allow_pickle=False)
    write_json(folder / "prediction_metadata.json", {
        "step": step, "endpoint": "PRIMARY" if step == 1000 else "SENSITIVITY",
        "state_order": list(hm.STATE_NAMES), "head_order": list(hm.HEAD_NAMES),
        "head_class_order": [["upright", "lying"], ["non_walking", "walking"], ["standing", "feeding"]],
        "test_window_count": len(test), "state_sha256": model_hash,
        "checkpoint_sha256": sha(folder / "checkpoint.pt"),
        "test_indices_sha256": sha(folder / "test_indices.npy"),
        "held_out_label_access": False, "held_out_scoring": False})


def run_unit(config, guard, binding, device):
    run_dir = ROOT / "results" / config["unit_id"]
    if verified_complete(run_dir, binding):
        print(f"VERIFIED_SKIP {config['unit_id']}", flush=True)
        return
    attempt = new_attempt(run_dir)
    try:
        X, y, train, test, batches, summary, support = load_inputs(config)
        set_seed(int(config["target_seed"]))
        source, source_audit = source_state(config)
        model, init_audit = hm.initialise(BaseModel, config["method"], config["target_seed"], source)
        model.to(device)
        write_json(attempt / "initialization_audit.json", {**init_audit, "source_checkpoint": source_audit})
        write_json(attempt / "input_binding.json", binding)
        write_json(attempt / "train_support.json", {"window_support": support, "budget": summary,
                   "train_window_count": len(train), "test_window_count": len(test)})
        environment = {"python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__,
            "cuda_runtime": torch.version.cuda, "device": str(device),
            "device_name": torch.cuda.get_device_name(device), "guard": guard,
            "precision": "FP32; no AMP/TF32/compile", "intraop_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(), "cuda_allocator_limit_gib": 1,
            "gpu_duty": 0.25, "started_utc": datetime.now(timezone.utc).isoformat()}
        write_json(attempt / "environment.json", environment)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        pacer = Pacer(device, duty=0.25)
        history, totals = [], {}
        for step, indices in enumerate(batches, 1):
            values = np.array(X[indices], dtype=np.float32, copy=True)
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite training input")
            labels = np.array(y[indices], dtype=np.int64, copy=True)
            row = update(model, optimizer, torch.from_numpy(values).to(device),
                         torch.from_numpy(labels).to(device), pacer)
            for key, value in row.items():
                totals[key] = totals.get(key, 0.0) + value
            if step % 100 == 0:
                history.append({"step": step, "mean_over_last_updates": 100,
                                **{key: value / 100 for key, value in totals.items()}})
                totals = {}
                write_json(attempt / "training_history.json", history)
                write_json(attempt / "progress.json", {"status": "running", "unit_id": config["unit_id"],
                    "step": step, "total_updates": 3000, "utc": datetime.now(timezone.utc).isoformat()})
                print(f"PROGRESS {config['unit_id']} step={step}/3000", flush=True)
            if step in ENDPOINTS:
                save_endpoint(attempt, step, model, optimizer, X, test, config, binding, device, pacer)
        after = check_frozen(config)
        if after != binding:
            raise RuntimeError("Input/protocol binding changed during execution")
        write_json(attempt / "completion_metadata.json", {"status": "complete", "unit": config,
            "binding": binding, "parameter_count": 828, "train_interval_count": summary["selected_event_count"],
            "train_window_count": len(train), "test_window_count": len(test), "endpoint_steps": list(ENDPOINTS),
            "selection_rule": "none; both predetermined fixed endpoints retained",
            "held_out_scoring": False, "source_checkpoint_unchanged": True,
            "pacing": {"batches": pacer.count, "busy_seconds": pacer.busy, "idle_seconds": pacer.slept},
            "finished_utc": datetime.now(timezone.utc).isoformat()})
        write_json(attempt / "progress.json", {"status": "complete", "unit_id": config["unit_id"], "step": 3000})
        artifacts = {p.relative_to(run_dir).as_posix(): sha(p) for p in sorted(attempt.rglob("*")) if p.is_file()}
        write_json(run_dir / "COMPLETE.json", {"status": "complete", "unit_id": config["unit_id"],
            "attempt": attempt.name, "binding": binding, "artifact_sha256": artifacts,
            "parameter_count": 828, "train_interval_count": summary["selected_event_count"],
            "train_window_count": len(train), "test_window_count": len(test), "endpoint_steps": list(ENDPOINTS),
            "precision": "FP32", "guard": guard})
        print(f"COMPLETE {config['unit_id']}", flush=True)
    except Exception as exc:
        write_json(attempt / "FAILED.json", {"status": "failed", "error": str(exc), "traceback": traceback.format_exc()})
        raise


def gpu_smoke():
    """Exercise the actual guarded hierarchy/update/pacing path, synthetic only."""
    guard = verify_guard()
    device = initialize_runtime("cuda", formal=True)
    run_dir = ROOT / "engineering/gpu_smoke"
    names = ["source/budget_runner.py", "source/hierarchy_model.py", "source/model_paper_kernel8.py",
             "inputs/checkpoints/act_seed0.pt", "inputs/checkpoints/ufro_seed0.pt"]
    binding = {"kind": "synthetic_hierarchy_gpu_smoke", "artifact_sha256":
               {name: sha(relative_path(name)) for name in names}}
    if verified_complete(run_dir, binding):
        print("VERIFIED_SKIP synthetic_hierarchy_gpu_smoke", flush=True)
        return
    attempt = new_attempt(run_dir)
    try:
        set_seed(20260909)
        generator = torch.Generator().manual_seed(20260909)
        values = torch.randn(32, 3, 256, generator=generator).to(device)
        labels = (torch.arange(32) % 4).long().to(device)
        torch.cuda.reset_peak_memory_stats(device)
        conditions = []
        for method in ("scratch", "act", "ufro"):
            config = {"method": method, "replicate": 0,
                      "checkpoint": f"inputs/checkpoints/{method}_seed0.pt"}
            state, source_audit = source_state(config)
            model, audit = hm.initialise(BaseModel, method, 20270909, state)
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999),
                                         eps=1e-8, weight_decay=0)
            pacer = Pacer(device, duty=0.25)
            started = time.perf_counter()
            losses = []
            for _ in range(30):
                losses.append(update(model, optimizer, values, labels, pacer))
            wall = time.perf_counter() - started
            synthetic_array = values.detach().cpu().numpy()
            states, heads = predict(model, synthetic_array, np.arange(32), device, pacer)
            conditions.append({"method": method, "updates": 30, "wall_seconds": wall,
                "wall_seconds_per_update": wall / 30, "parameter_count": 828,
                "initialization": audit, "source_checkpoint": source_audit,
                "final_state_sha256": hm.state_hash(model.state_dict()),
                "loss_first": losses[0], "loss_last": losses[-1],
                "predictions_shape": list(states.shape), "heads_shape": list(heads.shape),
                "pacing_busy_seconds": pacer.busy, "pacing_idle_seconds": pacer.slept})
            print(f"SMOKE {method} updates=30 wall_seconds={wall:.3f}", flush=True)
            del model, optimizer
        if {name: sha(relative_path(name)) for name in names} != binding["artifact_sha256"]:
            raise RuntimeError("Smoke source/checkpoint files changed")
        write_json(attempt / "SMOKE.json", {"status": "pass", "engineering_only": True,
            "actual_target_data_read": False, "conditions": conditions, "guard": guard,
            "device": str(device), "device_name": torch.cuda.get_device_name(device),
            "torch": str(torch.__version__), "numpy": np.__version__,
            "precision": "FP32; no AMP/TF32/compile", "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device), "cuda_allocator_gib": 1})
        artifacts = {p.relative_to(run_dir).as_posix(): sha(p) for p in sorted(attempt.rglob("*")) if p.is_file()}
        write_json(run_dir / "COMPLETE.json", {"status": "complete", "engineering_only": True,
            "binding": binding, "attempt": attempt.name, "artifact_sha256": artifacts})
    except Exception as exc:
        write_json(attempt / "FAILED.json", {"status": "failed", "error": str(exc), "traceback": traceback.format_exc()})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--gpu-smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.gpu_smoke:
        if args.unit or args.self_test or args.device != "cuda":
            parser.error("GPU smoke requires --device cuda without unit/self-test")
        gpu_smoke()
        return 0
    if args.self_test:
        if args.unit or args.device != "cpu":
            parser.error("Engineering self-test requires --device cpu and no --unit")
        import unittest
        suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_runner.py")
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    if not args.unit:
        parser.error("Formal execution requires --unit")
    guard = verify_guard()  # Must precede torch import and any formal artifacts.
    config = load_unit(args.unit)
    binding = check_frozen(config)
    device = initialize_runtime(args.device, formal=True)
    run_unit(config, guard, binding, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
