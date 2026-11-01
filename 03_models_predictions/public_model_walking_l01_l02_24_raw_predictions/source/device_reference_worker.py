"""Label-free device-reference adaptation; evaluation truth is never loaded here."""
from __future__ import annotations

import argparse
import copy
import csv
import ctypes
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np


SOURCES = ("jbb", "act", "ufro", "multi")
TRAIN_FAMILIES = ("gaussian", "spikes", "packetloss", "burst", "clipping")
CONDITIONS = [("clean", 0)] + [(f, level) for f in (*TRAIN_FAMILIES, "rotation") for level in (1, 2)]
ALLOWED_METADATA = {"row_index", "continuous_row_index", "window_id", "cow_alias", "cow_id",
                    "legacy_cow_id", "device_mac", "session_id", "cache_key", "scale_g", "motion_rms_g"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def relative(root, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("Expected experiment-relative path")
    root = Path(root).resolve()
    result = (root / name).resolve()
    if not result.is_relative_to(root) or result == root:
        raise ValueError("Path escapes isolated experiment directory")
    return result


def verify_guard():
    name = os.environ.get("DATASET_X", "")
    if not name or os.name != "nt":
        raise RuntimeError("Execution requires named Windows Job membership")
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
            raise RuntimeError("Worker is outside configured Job")
    finally:
        kernel.CloseHandle(handle)
    return {"name": name, "membership_verified": True, "pid": os.getpid()}


def verify_manifest(root):
    path = relative(root, "configs/FROZEN_PROTOCOL.json")
    expected = os.environ.get("DATASET_X", "").lower()
    if len(expected) != 64 or sha(path) != expected:
        raise RuntimeError("Frozen manifest does not match supervisor pin")
    manifest = read_json(path)
    artifacts = manifest.get("artifact_sha256")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("Frozen manifest lacks artifact hashes")
    for name, digest in artifacts.items():
        if sha(relative(root, name)) != str(digest).lower():
            raise ValueError("Frozen artifact changed: " + name)
    protocol_text = manifest.get("protocol_text_sha256")
    if protocol_text and sha(relative(root, "PROTOCOL.md")) != protocol_text:
        raise ValueError("Protocol prose changed")
    for name in ("configs/protocol.json", "configs/units.json", "configs/folds.json",
                 "inputs/acceleration.npy", "inputs/metadata.csv", "source/model_paper_kernel8.py",
                 "source/device_reference_worker.py"):
        if name not in artifacts:
            raise ValueError("Required artifact not bound: " + name)
    return expected, artifacts


def keyed_records(obj, key, field):
    if isinstance(obj, dict) and key in obj:
        obj = obj[key]
    if isinstance(obj, list):
        return {str(item[field]): item for item in obj}
    if isinstance(obj, dict):
        return obj
    raise ValueError("Invalid config record collection")


def load_metadata(path, count):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or set(reader.fieldnames) - ALLOWED_METADATA:
            raise ValueError("Metadata has unknown or label-bearing columns")
        if not {"row_index", "cow_id", "legacy_cow_id", "session_id", "scale_g"}.issubset(reader.fieldnames):
            raise ValueError("Metadata missing identity, session or scale")
        records = list(reader)
    if len(records) != count or any(int(r["row_index"]) != i for i, r in enumerate(records)):
        raise ValueError("Metadata row order differs from acceleration")
    scales = np.asarray([float(r["scale_g"]) for r in records], dtype=np.float32)
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Invalid physical scales")
    return records, scales


def verify_indices(train, test, count, metadata):
    for values in (train, test):
        if values.ndim != 1 or values.dtype.kind not in "iu" or not len(values):
            raise ValueError("Indices must be nonempty integer vectors")
        if len(np.unique(values)) != len(values) or values.min() < 0 or values.max() >= count:
            raise ValueError("Repeated or out-of-range row index")
    if np.intersect1d(train, test).size:
        raise ValueError("Training and held-out rows overlap")
    for key in ("session_id", "window_id"):
        if key in metadata[0]:
            train_keys = {metadata[int(i)][key] for i in train}
            test_keys = {metadata[int(i)][key] for i in test}
            if train_keys & test_keys:
                raise ValueError("Training/held-out overlap in " + key)
    disjoint_id_keys = []
    for key in ("cow_id", "legacy_cow_id"):
        a = {metadata[int(i)][key] for i in train}
        b = {metadata[int(i)][key] for i in test}
        if not a & b:
            disjoint_id_keys.append(key)
    if not disjoint_id_keys:
        raise ValueError("Neither user-declared nor legacy animal identity held out")
    return {"train_windows": len(train), "test_windows": len(test),
            "disjoint_identity_keys": disjoint_id_keys,
            "train_sessions": len({metadata[int(i)]["session_id"] for i in train}),
            "test_sessions": len({metadata[int(i)]["session_id"] for i in test})}


def normalize(raw):
    scale = np.median(np.sqrt(np.sum(np.square(raw, dtype=np.float32), axis=1)), axis=1)
    scale = np.maximum(scale, np.float32(1e-6))
    return np.asarray(raw / scale[:, None, None], dtype=np.float32)


def median3(raw):
    padded = np.pad(raw, ((0, 0), (0, 0), (1, 1)), mode="reflect")
    return np.median(np.stack((padded[..., :-2], padded[..., 1:-1], padded[..., 2:]), axis=-1), axis=-1).astype(np.float32)


def condition_name(family, level):
    return family if level == 0 else family + ("_low" if level == 1 else "_high")


def corrupt_raw(raw, rows, family, level, realization_seed):
    """Per-row seeds make corruption independent of evaluation batch partition."""
    if family == "clean":
        return np.array(raw, dtype=np.float32, copy=True)
    if family not in (*TRAIN_FAMILIES, "rotation") or level not in (1, 2):
        raise ValueError("Unknown corruption condition")
    result = np.array(raw, dtype=np.float32, copy=True)
    code = (*TRAIN_FAMILIES, "rotation").index(family) * 2 + level
    length = raw.shape[-1]
    for j, row in enumerate(rows):
        rng = np.random.default_rng(np.random.SeedSequence([int(realization_seed), int(row), code, 90217]))
        item = result[j]
        if family == "gaussian":
            item += rng.normal(0, (0.01, 0.05)[level - 1], item.shape).astype(np.float32)
        elif family == "spikes":
            fraction, amplitude = ((0.005, 0.5), (0.02, 2.0))[level - 1]
            mask = rng.random(item.shape) < fraction
            item += (mask * rng.choice((-amplitude, amplitude), item.shape)).astype(np.float32)
        elif family in ("packetloss", "burst"):
            keep = np.ones(length, dtype=bool)
            if family == "packetloss":
                keep = rng.random(length) >= (0.01, 0.10)[level - 1]
            else:
                width = (8, 32)[level - 1]
                start = int(rng.integers(0, length - width + 1))
                keep[start:start + width] = False
            # Endpoint preservation makes interpolation well defined and recorded.
            keep[0] = keep[-1] = True
            grid = np.arange(length)
            for axis in range(3):
                item[axis] = np.interp(grid, grid[keep], item[axis, keep])
        elif family == "clipping":
            bound = (1.5, 0.75)[level - 1]
            np.clip(item, -bound, bound, out=item)
        elif family == "rotation":
            angle = math.radians((15, 45)[level - 1])
            c, s = math.cos(angle), math.sin(angle)
            first, third = item[0].copy(), item[2].copy()
            item[0], item[2] = c * first + s * third, -s * first + c * third
    return result


def make_inputs(x, scales, rows, family="clean", level=0, seed=0, median=False):
    if family == "clean" and not median:
        # Preserve exact archived normalization for the clean reference.
        return np.array(x[rows], dtype=np.float32, copy=True), np.asarray(x[rows] * scales[rows, None, None], dtype=np.float32)
    raw = np.asarray(x[rows] * scales[rows, None, None], dtype=np.float32)
    damaged = corrupt_raw(raw, rows, family, level, seed)
    if median:
        damaged = median3(damaged)
    return normalize(damaged), damaged


def state_hash(state, excluded=()):
    digest = hashlib.sha256()
    for name in sorted(state):
        if name in excluded:
            continue
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_model_type(root):
    spec = importlib.util.spec_from_file_location("device_reference_architecture", relative(root, "source/model_paper_kernel8.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PaperKernel8CCNN


def new_binary_model(model_type, torch):
    model = model_type(in_channels=3, n_classes=3)
    model.classifier[-1] = torch.nn.Linear(6, 2)
    model.n_classes = 2
    return model


def load_source(root, source, replicate, model_type, torch):
    path = relative(root, f"inputs/checkpoints/{source}_seed{replicate}.pt")
    before = sha(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if set(payload) != {"model_state", "metadata_json"}:
        raise ValueError("Source checkpoint schema mismatch")
    metadata = json.loads(payload["metadata_json"])
    if metadata.get("source") != source or int(metadata.get("seed", -1)) != replicate:
        raise ValueError("Source checkpoint provenance mismatch")
    if int(metadata.get("matched_optimizer_steps", -1)) != 1600:
        raise ValueError("Source checkpoint step mismatch")
    # The archived binary contract is 0=nonwalking, 1=walking; prevent accidental multiclass loading.
    if tuple(payload["model_state"]["classifier.2.weight"].shape) != (2, 6):
        raise ValueError("Expected binary source classifier with walking at index 1")
    for key in ("positive_class_index", "walking_class_index"):
        if key in metadata and int(metadata[key]) != 1:
            raise ValueError("Source metadata contradicts walking positive class index 1")
    model = new_binary_model(model_type, torch)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    if sha(path) != before:
        raise ValueError("Source checkpoint changed during loading")
    return model, {"path": path.relative_to(root).as_posix(), "sha256": before,
                   "metadata": metadata, "positive_class_index": 1,
                   "positive_class_basis": "archived binary walking source model; margin=logit[1]-logit[0]"}


def configure_affine(model):
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in ("feature_bn.weight", "feature_bn.bias"))
    model.eval()
    selected = [p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in selected) != 16:
        raise ValueError("Adaptation must update exactly sixteen BN affine parameters")
    return selected


class Pacer:
    def __init__(self, root, device, torch, duty=0.2):
        if not 0 < duty <= 0.2:
            raise ValueError("Duty cycle exceeds frozen maximum 20 percent")
        self.root, self.device, self.torch, self.duty = Path(root), device, torch, duty
        self.compute_seconds = 0.0
        self.sleep_seconds = 0.0

    def check_stop(self):
        paths = [self.root / "STOP_REQUESTED", self.root / "execution/STOP_REQUESTED"]
        if os.environ.get("DATASET_X"):
            paths.append(Path(os.environ["DATASET_X"]))
        if any(p.exists() for p in paths):
            raise RuntimeError("Supervisor requested resource stop")

    def begin(self):
        self.check_stop()
        self.torch.cuda.synchronize(self.device)
        self.started = time.perf_counter()

    def end(self):
        self.torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - self.started
        self.compute_seconds += elapsed
        remaining = elapsed * (1.0 / self.duty - 1.0)
        self.sleep_seconds += remaining
        while remaining > 0:
            self.check_stop()
            interval = min(remaining, 0.2)
            time.sleep(interval)
            remaining -= interval


def pooled_bn(model, x, scales, indices, batch_size, device, torch, pacer):
    result = copy.deepcopy(model).eval()
    total = 0
    sums = np.zeros(8, dtype=np.float64)
    squares = np.zeros(8, dtype=np.float64)
    feature_cache = np.empty((len(x), 8), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            rows = indices[start:start + batch_size]
            batch, _ = make_inputs(x, scales, rows)
            pacer.begin()
            features = result.forward_features(torch.from_numpy(batch).to(device)).double().cpu().numpy()
            pacer.end()
            feature_cache[rows] = features.astype(np.float32)
            sums += features.sum(axis=0)
            squares += np.square(features).sum(axis=0)
            total += len(rows)
    mean = sums / total
    variance = np.maximum(squares / total - np.square(mean), 0)
    result.feature_bn.running_mean.copy_(torch.tensor(mean, dtype=result.feature_bn.running_mean.dtype, device=device))
    result.feature_bn.running_var.copy_(torch.tensor(variance, dtype=result.feature_bn.running_var.dtype, device=device))
    return result, {"windows": total, "population_mean": mean.tolist(), "population_variance": variance.tolist()}, feature_cache


def raw_rms(raw):
    centered = raw - raw.mean(axis=-1, keepdims=True)
    return np.sqrt(np.mean(np.sum(np.square(centered, dtype=np.float32), axis=1), axis=1)).astype(np.float32)


def evaluate(model, x, scales, indices, out, method, endpoint, batch_size, seed, device, torch, pacer, conditions=None, cache=None):
    files = []
    model.eval()
    with torch.no_grad():
        for family, level in (CONDITIONS if conditions is None else conditions):
            margins, probabilities, rms = [], [], []
            cache_key = (family, level, method == "median3")
            cached = None if cache is None else cache.get(cache_key)
            feature_parts = []
            for start in range(0, len(indices), batch_size):
                rows = indices[start:start + batch_size]
                if cached is None:
                    batch, raw = make_inputs(x, scales, rows, family, level, seed, median=method == "median3")
                pacer.begin()
                if cached is None:
                    features = model.forward_features(torch.from_numpy(batch).to(device))
                    feature_parts.append(features.cpu().numpy())
                else:
                    features = torch.from_numpy(cached[0][start:start + len(rows)]).to(device)
                logits = model.classifier(model.feature_bn(features))
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("Non-finite held-out logits")
                margins.append((logits[:, 1] - logits[:, 0]).cpu().numpy())
                probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
                pacer.end()
                rms.append(raw_rms(raw) if cached is None else cached[1][start:start + len(rows)])
            if cached is None and cache is not None:
                cache[cache_key] = (np.concatenate(feature_parts), np.concatenate(rms))
            filename = f"predictions/{method}_step{endpoint:04d}/{condition_name(family, level)}.npz"
            path = out / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            arrays = {"margin": np.concatenate(margins), "probability": np.concatenate(probabilities),
                      "rms_score": np.concatenate(rms)}
            if any(len(value) != len(indices) or not np.isfinite(value).all() for value in arrays.values()):
                raise ValueError("Invalid prediction vector length or non-finite prediction")
            if np.any(arrays["probability"] < 0) or np.any(arrays["probability"] > 1):
                raise ValueError("Probability outside [0,1]")
            np.savez_compressed(path, row_indices=indices, **arrays)
            files.append(filename)
    return files


def checkpoint_roundtrip(model, path, batch, model_type, torch, device):
    payload = {"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    restored = new_binary_model(model_type, torch).to(device)
    restored.load_state_dict(torch.load(path, map_location=device, weights_only=True)["model_state"], strict=True)
    restored.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(batch).to(device)
        a, b = model(tensor), restored(tensor)
    if not torch.equal(a, b):
        raise ValueError("Saved checkpoint does not reproduce predictions exactly")
    return {"sha256": sha(path), "prediction_reload_exact": True}


def train_step(student, teacher, optimizer, batch, damaged, affine_reference, torch, device, cfg, clean_features=None):
    clean = None if clean_features is not None else torch.from_numpy(batch).to(device)
    corrupted = torch.from_numpy(damaged).to(device)
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        features_clean = (teacher.forward_features(clean) if clean_features is None
                          else torch.from_numpy(clean_features).to(device))
        features_damage = student.forward_features(corrupted)
        target = torch.softmax(teacher.classifier(teacher.feature_bn(features_clean)), dim=1)
    damage_logits = student.classifier(student.feature_bn(features_damage))
    clean_logits = student.classifier(student.feature_bn(features_clean))
    loss_damage = torch.nn.functional.kl_div(torch.log_softmax(damage_logits, dim=1), target, reduction="batchmean")
    loss_clean = torch.nn.functional.kl_div(torch.log_softmax(clean_logits, dim=1), target, reduction="batchmean")
    parameters = [p for p in student.parameters() if p.requires_grad]
    penalty = torch.cat([(p - reference).reshape(-1) for p, reference in zip(parameters, affine_reference)]).square().mean()
    loss = loss_damage + cfg.get("anchor_weight", 1.0) * loss_clean + cfg.get("affine_weight", 1e-3) * penalty
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite label-free adaptation loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    optimizer.step()
    return {"total": float(loss.detach()), "corruption_kl": float(loss_damage.detach()),
            "clean_anchor_kl": float(loss_clean.detach()), "affine_penalty": float(penalty.detach())}


def run(root, unit_id, mode):
    root = Path(root).resolve()
    guard = verify_guard()  # Verify before importing CUDA runtime or producing results.
    frozen_sha, bindings = verify_manifest(root)
    cfg = read_json(relative(root, "configs/protocol.json"))
    units = keyed_records(read_json(relative(root, "configs/units.json")), "units", "unit_id")
    folds = keyed_records(read_json(relative(root, "configs/folds.json")), "folds", "fold_id")
    if unit_id not in units:
        if mode != "smoke":
            raise ValueError("Unknown formal unit")
        unit = next(iter(units.values()))
    else:
        unit = units[unit_id]
    fold_id, source, replicate = str(unit["fold_id"]), unit["source"], int(unit["replicate"])
    if fold_id not in folds or source not in SOURCES or replicate not in (0, 1, 2):
        raise ValueError("Unit references invalid fold/source/replicate")
    if int(cfg.get("steps", 1000)) != 1000 or list(cfg.get("endpoints", [100, 300, 1000])) != [100, 300, 1000]:
        raise ValueError("Frozen training endpoints changed")
    for key, expected in (("anchor_weight", 1.0), ("affine_weight", 1e-3), ("lr", 1e-3), ("train_batch", 32)):
        if cfg.get(key, expected) != expected:
            raise ValueError("Frozen update setting changed: " + key)
    if int(cfg.get("threads", 2)) > 2 or float(cfg.get("gpu_memory_gib", 1)) > 1:
        raise ValueError("Requested resources exceed worker caps")
    import torch
    torch.set_num_threads(2)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(min(1.0, 1024**3 / torch.cuda.get_device_properties(device).total_memory), device)
    seed = int(unit["target_seed"])
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    x = np.load(relative(root, "inputs/acceleration.npy"), mmap_mode="r", allow_pickle=False)
    if x.dtype != np.float32 or x.ndim != 3 or x.shape[1:] != (3, 256) or len(x) != 40920:
        raise ValueError("Expected normalized float32 acceleration (N,3,256)")
    metadata, scales = load_metadata(relative(root, "inputs/metadata.csv"), len(x))
    train_name, test_name = f"inputs/folds/{fold_id}/train_indices.npy", f"inputs/folds/{fold_id}/test_indices.npy"
    checkpoint_name = f"inputs/checkpoints/{source}_seed{replicate}.pt"
    if any(name not in bindings for name in (train_name, test_name, checkpoint_name)):
        raise ValueError("Unit input not pinned in frozen manifest")
    train, test = (np.load(relative(root, name), allow_pickle=False) for name in (train_name, test_name))
    split = verify_indices(train, test, len(x), metadata)
    model_type = load_model_type(root)
    teacher, checkpoint = load_source(root, source, replicate, model_type, torch)
    teacher = teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    teacher_hash = state_hash(teacher.state_dict())
    out = root / ("results" if mode == "formal" else "smoke") / unit_id
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("Existing unit artifacts must be retained; use a distinct attempt directory")
    out.mkdir(parents=True, exist_ok=True)
    pacer = Pacer(root, device, torch, cfg.get("duty_cycle", 0.2))
    started = time.time()
    atomic_json(out / "RUNNING.json", {"unit_id": unit_id, "mode": mode, "frozen_sha256": frozen_sha,
                                      "status": "RUNNING", "started_unix": started, "guard": guard})
    files, checkpoint_checks, loss_rows, evaluation_cache = [], {}, [], {}
    eval_batch = min(128, int(cfg.get("eval_batch", 128)))
    realization_seed = int(cfg.get("evaluation_seed_base", 74103))
    if mode == "smoke":
        import io
        import unittest
        import test_device_reference
        test_stream = io.StringIO()
        suite = unittest.defaultTestLoader.loadTestsFromModule(test_device_reference)
        tested = unittest.TextTestRunner(stream=test_stream, verbosity=2).run(suite)
        test_receipt = {"tests_run": tested.testsRun, "failures": len(tested.failures),
                        "errors": len(tested.errors), "skipped": len(tested.skipped),
                        "log": test_stream.getvalue(), "guarded": True}
        atomic_json(out / "synthetic_tests.json", test_receipt)
        files.append("synthetic_tests.json")
        if not tested.wasSuccessful() or tested.skipped or tested.testsRun != 10:
            raise ValueError("All ten synthetic and model tests must run and pass in the guarded runtime")
        source_checks = []
        for candidate in SOURCES:
            for rep in range(3):
                name = f"inputs/checkpoints/{candidate}_seed{rep}.pt"
                if name not in bindings:
                    raise ValueError("Smoke requires all twelve pinned source checkpoints")
                _, check = load_source(root, candidate, rep, model_type, torch)
                source_checks.append(check)
        test = test[:min(9, len(test))]
        train = train[:min(32, len(train))]
        for family, level in CONDITIONS:
            a, _ = make_inputs(x, scales, test, family, level, realization_seed)
            b = np.concatenate([make_inputs(x, scales, rows, family, level, realization_seed)[0] for rows in np.array_split(test, 3)])
            if not np.array_equal(a, b):
                raise ValueError("Corruption depends on evaluation batch partition")
        atomic_json(out / "smoke_source_checks.json", source_checks)
        files.append("smoke_source_checks.json")
        batch, _ = make_inputs(x, scales, test)
        pacer.begin()
        with torch.no_grad():
            tensor = torch.from_numpy(batch).to(device)
            full = teacher(tensor)
            features = teacher.forward_features(tensor).cpu().numpy()
            cached_logits = teacher.classifier(teacher.feature_bn(torch.from_numpy(features).to(device)))
            if not torch.equal(full, cached_logits):
                raise ValueError("Frozen feature cache changes forward predictions")
        pacer.end()
    if mode == "formal":
        for method in ("frozen", "median3"):
            files += evaluate(teacher, x, scales, test, out, method, 0, eval_batch, realization_seed, device, torch, pacer, cache=evaluation_cache)
    bn_model, moments, training_features = pooled_bn(teacher, x, scales, train, eval_batch, device, torch, pacer)
    atomic_json(out / "bn_population_moments.json", moments)
    files.append("bn_population_moments.json")
    files += evaluate(bn_model, x, scales, test, out, "bn_moments", 0, eval_batch, realization_seed, device, torch, pacer,
                      conditions=None if mode == "formal" else [("clean", 0)], cache=evaluation_cache)
    batch, _ = make_inputs(x, scales, test[:min(8, len(test))])
    bn_checkpoint_name = "checkpoints/bn_moments.pt"
    pacer.begin()
    checkpoint_checks["bn_moments"] = checkpoint_roundtrip(bn_model, out / bn_checkpoint_name, batch, model_type, torch, device)
    pacer.end()
    files.append(bn_checkpoint_name)
    del bn_model
    student = copy.deepcopy(teacher).eval()
    parameters = configure_affine(student)
    references = [p.detach().clone() for p in parameters]
    frozen_names = ("feature_bn.weight", "feature_bn.bias")
    frozen_before = state_hash(student.state_dict(), frozen_names)
    optimizer = torch.optim.Adam(parameters, lr=1e-3)
    # This generator is source-independent: matched fold/rep gets identical sampled rows and damage.
    index_rng = np.random.default_rng(np.random.SeedSequence([seed, 7102]))
    train_rng = np.random.default_rng(np.random.SeedSequence([seed, 88119]))
    training_draw_hash = hashlib.sha256()
    train_steps = 1000 if mode == "formal" else 2
    for step in range(1, train_steps + 1):
        rows = index_rng.choice(train, size=32, replace=len(train) < 32)
        family = TRAIN_FAMILIES[int(train_rng.integers(len(TRAIN_FAMILIES)))]
        damage_seed = int(train_rng.integers(0, 2**31))
        training_draw_hash.update(np.asarray(rows, dtype="<i8").tobytes())
        training_draw_hash.update(f"{family}:{damage_seed};".encode("ascii"))
        clean, _ = make_inputs(x, scales, rows)
        damaged, _ = make_inputs(x, scales, rows, family, 1, damage_seed)
        pacer.begin()
        losses = train_step(student, teacher, optimizer, clean, damaged, references, torch, device, cfg,
                            clean_features=training_features[rows])
        pacer.end()
        loss_rows.append({"step": step, "family": family, **losses})
        if step in (100, 300, 1000) or mode == "smoke" and step == train_steps:
            if state_hash(student.state_dict(), frozen_names) != frozen_before:
                raise ValueError("Frozen encoder/classifier/BN running statistics changed")
            batch, _ = make_inputs(x, scales, test[:min(8, len(test))])
            name = f"checkpoints/adapted_step{step:04d}.pt"
            pacer.begin()
            checkpoint_checks[str(step)] = checkpoint_roundtrip(student, out / name, batch, model_type, torch, device)
            pacer.end()
            files.append(name)
            files += evaluate(student, x, scales, test, out, "adapted", step, eval_batch, realization_seed, device, torch, pacer,
                              conditions=None if mode == "formal" else [("clean", 0), ("gaussian", 1)], cache=evaluation_cache)
            atomic_json(out / "PROGRESS.json", {"status": "RUNNING", "unit_id": unit_id, "step": step,
                                               "frozen_sha256": frozen_sha, "last_losses": losses})
    if state_hash(teacher.state_dict()) != teacher_hash or sha(relative(root, checkpoint["path"])) != checkpoint["sha256"]:
        raise ValueError("Frozen source teacher changed")
    affine_changed = any(not torch.equal(p, r) for p, r in zip(parameters, references))
    if state_hash(student.state_dict(), frozen_names) != frozen_before:
        raise ValueError("Frozen parameters changed at completion")
    atomic_json(out / "losses.json", loss_rows)
    files.append("losses.json")
    receipt = {"status": "COMPLETE", "unit_id": unit_id, "mode": mode, "frozen_sha256": frozen_sha,
               "unit": unit, "split": split, "source_checkpoint": checkpoint, "guard": guard,
               "device": torch.cuda.get_device_name(device), "torch_version": torch.__version__,
               "numpy_version": np.__version__, "python_version": sys.version,
               "trainable_parameter_count": 16, "trainable_names": list(frozen_names),
               "teacher_state_sha256": teacher_hash, "frozen_state_before_sha256": frozen_before,
               "frozen_state_after_sha256": state_hash(student.state_dict(), frozen_names),
               "frozen_state_unchanged": True, "checkpoint_reload_checks": checkpoint_checks,
               "affine_changed": affine_changed,
               "affine_delta_l2": float(torch.cat([(p.detach() - r).reshape(-1) for p, r in zip(parameters, references)]).norm()),
               "target_labels_read": False, "truth_metrics_computed": False,
               "training_corruptions": list(TRAIN_FAMILIES), "training_severity": "low_only",
               "rotation_seen_during_training": False, "evaluation_seed": realization_seed,
               "frozen_feature_cache": True,
               "matched_training_draws_sha256": training_draw_hash.hexdigest(),
               "prediction_condition_count": 13,
               "formal_prediction_artifact_count": sum(name.endswith(".npz") for name in files),
               "steps": train_steps, "elapsed_seconds": time.time() - started,
               "compute_seconds": pacer.compute_seconds, "pacing_sleep_seconds": pacer.sleep_seconds,
               "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
               "input_sha256": bindings, "output_sha256": {name: sha(out / name) for name in files}}
    atomic_json(out / "COMPLETE.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--mode", choices=("formal", "smoke"), required=True)
    args = parser.parse_args()
    if Path(args.unit).name != args.unit or args.unit in (".", ".."):
        raise ValueError("Unsafe unit name")
    try:
        receipt = run(args.root, args.unit, args.mode)
        print(json.dumps({key: receipt[key] for key in ("status", "unit_id", "mode", "steps", "elapsed_seconds")}), flush=True)
    except Exception as error:
        # Retain an ordinary failure receipt; never erase failed attempts.
        out = args.root / ("results" if args.mode == "formal" else "smoke") / args.unit
        if out.is_dir() and not (out / "ABORTED.json").exists() and not (out / "COMPLETE.json").exists():
            atomic_json(out / "ABORTED.json", {"status": "ABORTED", "unit_id": args.unit,
                                              "error_type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
