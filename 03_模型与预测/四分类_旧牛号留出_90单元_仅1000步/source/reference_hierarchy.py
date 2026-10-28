"""Exploratory hierarchical four-state collar-to-tail transfer.

States are STANDING, LYING, WALKING and FEEDING.  A shared constrained CCNN
encoder feeds three conditional heads: posture (UPRIGHT/LYING), locomotion
(NON_WALKING/WALKING), and upright-nonwalking state (STANDING/FEEDING).
Public collars initialise only the encoder and locomotion head.  Posture and
feeding heads are always target-domain initialisations.  The script is isolated
from production models and writes only a new run directory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import collar_transfer_binary_206_20260822 as binary
import source_selective_transfer_20260823 as selective
from DATASET_X.metrics import selection_score


pilot = binary.pilot
STATE_NAMES = ("standing", "lying", "walking", "feeding")
POSTURE_NAMES = ("upright", "lying")
LOCOMOTION_NAMES = ("non_walking", "walking")
FEEDING_HEAD_NAMES = ("standing", "feeding")
STATE_BY_CODE = {"STANDING": 0, "LYING": 1, "WALKING": 2, "FEEDING": 3}
METHODS = ("scratch", "tail_ssl", "jbb", "act", "ufro", "multi")
SOURCE_METHODS = ("jbb", "act", "ufro", "multi")
ENCODER_PREFIXES = selective.ENCODER_PREFIXES

DEFAULT_TARGET_ROOT = Path(
    r"PATH_X"
)
DEFAULT_PRETRAIN_ROOT = (
    DEFAULT_TARGET_ROOT / "runs" / "source_selective_20260823_full_v1"
)
DEFAULT_OUTPUT = REPO_ROOT / "runs" / "hierarchical_four_state_transfer_20260823"


@dataclass
class Targets:
    state: np.ndarray
    posture: np.ndarray
    posture_mask: np.ndarray
    walking: np.ndarray
    feeding: np.ndarray
    feeding_mask: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-root", type=Path, default=DEFAULT_TARGET_ROOT)
    parser.add_argument("--pretrain-root", type=Path, default=DEFAULT_PRETRAIN_ROOT)
    parser.add_argument("--public-root", type=Path, default=selective.DEFAULT_PUBLIC_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--label-fractions", default="0.10,0.25,0.50,1.00")
    parser.add_argument("--test-cows", default="COWID_02,COWID_03")
    parser.add_argument("--validation-cow", default="COWID_11")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--min-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--posture-loss-weight", type=float, default=1.0)
    parser.add_argument("--walking-loss-weight", type=float, default=1.0)
    parser.add_argument("--feeding-loss-weight", type=float, default=1.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dry-run", action="store_true", help="audit inputs only")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="one seed, budget and cow; two epochs; reuses frozen pretraining",
    )
    return parser.parse_args()


def strings(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def ints(value: str) -> list[int]:
    return [int(value) for value in strings(value)]


def floats(value: str) -> list[float]:
    return [float(value) for value in strings(value)]


def build_targets(original_code: np.ndarray) -> Targets:
    code = np.asarray(original_code).astype(str)
    unknown = sorted(set(np.unique(code).tolist()) - set(STATE_BY_CODE))
    if unknown:
        raise ValueError(f"unknown target codes: {unknown}")
    state = np.asarray([STATE_BY_CODE[value] for value in code], dtype=np.int64)
    posture = (code == "LYING").astype(np.int64)
    posture_mask = code != "WALKING"
    walking = (code == "WALKING").astype(np.int64)
    feeding_mask = np.isin(code, ["STANDING", "FEEDING"])
    feeding = (code == "FEEDING").astype(np.int64)
    return Targets(state, posture, posture_mask, walking, feeding, feeding_mask)


class HierarchicalDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, targets: Targets, indices: np.ndarray) -> None:
        self.X = torch.from_numpy(X[indices])
        self.state = torch.from_numpy(targets.state[indices])
        self.posture = torch.from_numpy(targets.posture[indices])
        self.posture_mask = torch.from_numpy(targets.posture_mask[indices])
        self.walking = torch.from_numpy(targets.walking[indices])
        self.feeding = torch.from_numpy(targets.feeding[indices])
        self.feeding_mask = torch.from_numpy(targets.feeding_mask[indices])

    def __len__(self) -> int:
        return len(self.state)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return (
            self.X[index],
            self.state[index],
            self.posture[index],
            self.posture_mask[index],
            self.walking[index],
            self.feeding[index],
            self.feeding_mask[index],
        )


def make_loader(
    X: np.ndarray,
    targets: Targets,
    original_code: np.ndarray,
    indices: np.ndarray,
    *,
    batch_size: int,
    seed: int,
    training: bool,
) -> Any:
    dataset = HierarchicalDataset(X, targets, indices)
    generator = torch.Generator().manual_seed(seed)
    if not training:
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    codes = np.asarray(original_code)[indices]
    counts = {code: int(np.sum(codes == code)) for code in STATE_BY_CODE}
    if any(value <= 0 for value in counts.values()):
        raise ValueError(f"training subset lacks a four-state code: {counts}")
    weights = torch.tensor([1.0 / counts[str(code)] for code in codes], dtype=torch.double)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights, num_samples=len(indices), replacement=True, generator=generator
    )
    drop_last = len(indices) > batch_size and len(indices) % batch_size == 1
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, drop_last=drop_last
    )


def mlp_head(in_features: int, hidden: int = 6) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_features, hidden), nn.ReLU(), nn.Linear(hidden, 2))


class HierarchicalCCNN(nn.Module):
    def __init__(self, BaseModel: Any) -> None:
        super().__init__()
        base = BaseModel(in_channels=3, n_classes=3)
        self.ccnn_linear = base.ccnn_linear
        self.ccnn_nonlinear = base.ccnn_nonlinear
        self.nonlinear_convs = base.nonlinear_convs
        self.n_features = int(base.n_features)
        self.posture_bn = nn.BatchNorm1d(self.n_features)
        self.walking_bn = nn.BatchNorm1d(self.n_features)
        self.feeding_bn = nn.BatchNorm1d(self.n_features)
        self.posture_head = mlp_head(self.n_features)
        self.walking_head = mlp_head(self.n_features)
        self.feeding_head = mlp_head(self.n_features)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        linear_sequence = self.ccnn_linear(x)
        middle = (linear_sequence.shape[-1] - 1) // 2
        linear = linear_sequence.sort(dim=-1).values[..., middle]
        nonlinear = self.ccnn_nonlinear(x)
        for conv in self.nonlinear_convs:
            nonlinear = torch.tanh(conv(nonlinear))
        nonlinear = nonlinear.abs().mean(dim=-1)
        return torch.cat((linear, nonlinear), dim=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.forward_features(x)
        return {
            "posture_logits": self.posture_head(self.posture_bn(features)),
            "walking_logits": self.walking_head(self.walking_bn(features)),
            "feeding_logits": self.feeding_head(self.feeding_bn(features)),
        }

    @staticmethod
    def compose_probabilities(outputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        posture = torch.softmax(outputs["posture_logits"], dim=1)
        walking = torch.softmax(outputs["walking_logits"], dim=1)
        feeding = torch.softmax(outputs["feeding_logits"], dim=1)
        non_walking = walking[:, 0]
        upright_given_non_walking = posture[:, 0]
        gate = non_walking * upright_given_non_walking
        state = torch.stack(
            (
                gate * feeding[:, 0],
                non_walking * posture[:, 1],
                walking[:, 1],
                gate * feeding[:, 1],
            ),
            dim=1,
        )
        return {"posture": posture, "walking": walking, "feeding": feeding, "state": state}

    @torch.no_grad()
    def project_constraints(self) -> None:
        self.ccnn_linear.project()
        self.ccnn_nonlinear.project()


def load_safe_state(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if set(payload) != {"model_state", "metadata_json"}:
        raise ValueError(f"unsafe checkpoint contract: {path}")
    metadata = json.loads(payload["metadata_json"])
    return payload["model_state"], metadata


def initialise_model(
    BaseModel: Any,
    method: str,
    *,
    pretrain_root: Path,
    seed: int,
    fold: int,
    test_cow: str,
) -> tuple[HierarchicalCCNN, dict[str, Any]]:
    model = HierarchicalCCNN(BaseModel)
    audit: dict[str, Any] = {"method": method, "loaded_keys": [], "reset_heads": []}
    if method == "scratch":
        audit["reset_heads"] = ["posture", "walking", "feeding"]
        return model, audit
    if method == "tail_ssl":
        path = pretrain_root / f"seed{seed}" / "tail_ssl" / f"fold{fold}_{test_cow}" / "checkpoint.pt"
        state, metadata = load_safe_state(path)
        if int(metadata.get("seed", -1)) != int(seed):
            raise ValueError(f"tail SSL seed metadata mismatch: {path}")
        if int(metadata.get("fold", -1)) != int(fold):
            raise ValueError(f"tail SSL fold metadata mismatch: {path}")
        if str(metadata.get("test_cow", "")) != str(test_cow):
            raise ValueError(f"tail SSL test-cow metadata mismatch: {path}")
        expected_encoder = {
            name for name in model.state_dict() if name.startswith(ENCODER_PREFIXES)
        }
        if set(state) != expected_encoder:
            raise ValueError(
                "tail SSL encoder key contract mismatch: "
                f"missing={sorted(expected_encoder - set(state))}, "
                f"extra={sorted(set(state) - expected_encoder)}"
            )
        for name, value in state.items():
            model.state_dict()[name].copy_(value)
        audit.update(
            checkpoint=str(path),
            checkpoint_sha256=pilot.sha256_file(path),
            checkpoint_metadata=metadata,
            loaded_keys=sorted(state),
            reset_heads=["posture", "walking", "feeding"],
        )
        return model, audit
    if method not in SOURCE_METHODS:
        raise ValueError(f"unknown method: {method}")
    path = pretrain_root / f"seed{seed}" / "pretrain" / method / "checkpoint.pt"
    state, metadata = load_safe_state(path)
    if str(metadata.get("source", "")) != method:
        raise ValueError(f"public checkpoint source metadata mismatch: {path}")
    if int(metadata.get("seed", -1)) != int(seed):
        raise ValueError(f"public checkpoint seed metadata mismatch: {path}")
    source_model = binary.make_binary_model(BaseModel, torch)
    expected_source = set(source_model.state_dict())
    if set(state) != expected_source:
        raise ValueError(
            "public source state key contract mismatch: "
            f"missing={sorted(expected_source - set(state))}, "
            f"extra={sorted(set(state) - expected_source)}"
        )
    loaded: list[str] = []
    for name, value in state.items():
        target_name: str | None = None
        if name.startswith(ENCODER_PREFIXES):
            target_name = name
        elif name.startswith("feature_bn."):
            target_name = "walking_bn." + name.removeprefix("feature_bn.")
        elif name.startswith("classifier."):
            target_name = "walking_head." + name.removeprefix("classifier.")
        if target_name is not None:
            target = model.state_dict()[target_name]
            if tuple(target.shape) != tuple(value.shape):
                raise ValueError(f"source tensor shape mismatch: {name} -> {target_name}")
            target.copy_(value)
            loaded.append(target_name)
    if len(loaded) != len(state):
        raise ValueError("not every public source tensor was consumed")
    if not any(name.startswith("walking_head.") for name in loaded):
        raise ValueError(f"source walking head was not loaded: {path}")
    audit.update(
        checkpoint=str(path),
        checkpoint_sha256=pilot.sha256_file(path),
        checkpoint_metadata=metadata,
        loaded_keys=sorted(loaded),
        reset_heads=["posture", "feeding"],
    )
    return model, audit


def masked_losses(
    outputs: Mapping[str, torch.Tensor],
    posture: torch.Tensor,
    posture_mask: torch.Tensor,
    walking: torch.Tensor,
    feeding: torch.Tensor,
    feeding_mask: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    if bool(posture_mask.any()):
        posture_loss = torch.nn.functional.cross_entropy(
            outputs["posture_logits"][posture_mask], posture[posture_mask]
        )
    else:
        posture_loss = outputs["posture_logits"].sum() * 0.0
    walking_loss = torch.nn.functional.cross_entropy(outputs["walking_logits"], walking)
    if bool(feeding_mask.any()):
        feeding_loss = torch.nn.functional.cross_entropy(
            outputs["feeding_logits"][feeding_mask], feeding[feeding_mask]
        )
    else:
        # A class-balanced sampler makes this rare, but a stochastic mini-batch
        # may contain only LYING/WALKING.  Preserve the masked-loss contract
        # without manufacturing a target or interrupting the epoch.
        feeding_loss = outputs["feeding_logits"].sum() * 0.0
    total = (
        args.posture_loss_weight * posture_loss
        + args.walking_loss_weight * walking_loss
        + args.feeding_loss_weight * feeding_loss
    )
    return {"total": total, "posture": posture_loss, "walking": walking_loss, "feeding": feeding_loss}


@torch.no_grad()
def predict(model: HierarchicalCCNN, loader: Any, device: torch.device) -> dict[str, np.ndarray]:
    collected: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "state_target",
            "posture_target",
            "posture_mask",
            "walking_target",
            "feeding_target",
            "feeding_mask",
            "posture_probability",
            "walking_probability",
            "feeding_probability",
            "state_probability",
        )
    }
    model.eval()
    for inputs, state, posture, posture_mask, walking, feeding, feeding_mask in loader:
        outputs = model(inputs.to(device))
        probability = model.compose_probabilities(outputs)
        collected["state_target"].append(state.numpy())
        collected["posture_target"].append(posture.numpy())
        collected["posture_mask"].append(posture_mask.numpy())
        collected["walking_target"].append(walking.numpy())
        collected["feeding_target"].append(feeding.numpy())
        collected["feeding_mask"].append(feeding_mask.numpy())
        for name in ("posture", "walking", "feeding", "state"):
            collected[f"{name}_probability"].append(probability[name].cpu().numpy())
    result = {name: np.concatenate(values) for name, values in collected.items()}
    state_probability = result["state_probability"]
    if not np.isfinite(state_probability).all():
        raise ValueError("non-finite state probabilities")
    if not np.allclose(state_probability.sum(axis=1), 1.0, atol=1e-6, rtol=0.0):
        raise ValueError("four-state probabilities do not sum to one")
    return result


def metrics(prediction: Mapping[str, np.ndarray], original_code: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_recall_fscore_support,
    )

    state_true = prediction["state_target"]
    state_probability = prediction["state_probability"]
    state_pred = state_probability.argmax(axis=1)
    labels = list(range(4))
    precision, recall, f1, support = precision_recall_fscore_support(
        state_true, state_pred, labels=labels, zero_division=0
    )
    posture_mask = prediction["posture_mask"].astype(bool)
    posture_true = prediction["posture_target"][posture_mask]
    posture_pred = prediction["posture_probability"][posture_mask].argmax(axis=1)
    posture_macro_f1 = float(
        f1_score(posture_true, posture_pred, labels=[0, 1], average="macro", zero_division=0)
    )
    posture_lying_support = int(np.sum(posture_true == 1))
    posture_upright_support = int(np.sum(posture_true == 0))
    posture_selection_eligible = (
        posture_lying_support >= 10 and posture_upright_support >= 10
    )
    posture_selection_reason = (
        "eligible_at_least_10_lying_and_10_upright_nonwalking"
        if posture_selection_eligible
        else (
            "excluded_from_selection_insufficient_validation_support:"
            f"lying={posture_lying_support},upright_nonwalking={posture_upright_support}"
        )
    )
    walking_true = prediction["walking_target"]
    walking_ap = float(
        average_precision_score(walking_true, prediction["walking_probability"][:, 1])
    )
    feeding_mask = prediction["feeding_mask"].astype(bool)
    feeding_true = prediction["feeding_target"][feeding_mask]
    feeding_positive = int(np.sum(feeding_true == 1))
    feeding_negative = int(np.sum(feeding_true == 0))
    feeding_ap = (
        float(
            average_precision_score(
                feeding_true, prediction["feeding_probability"][feeding_mask, 1]
            )
        )
        if feeding_positive > 0 and feeding_negative > 0
        else None
    )
    feeding_selection_eligible = feeding_positive >= 10 and feeding_negative >= 10
    feeding_selection_reason = (
        "eligible_at_least_10_feeding_and_10_standing"
        if feeding_selection_eligible
        else (
            "excluded_from_selection_insufficient_validation_support:"
            f"feeding={feeding_positive},standing={feeding_negative}"
        )
    )
    selected = selection_score(
        posture_macro_f1 if posture_selection_eligible else None,
        walking_ap,
        {"FEEDING": feeding_ap} if feeding_selection_eligible else {},
    )
    code = np.asarray(original_code).astype(str)
    pure_standing = code == "STANDING"
    true_feeding = code == "FEEDING"
    feeding_destinations = {
        STATE_NAMES[index]: int(np.sum(state_pred[true_feeding] == index)) for index in labels
    }
    def subset_rate(mask: np.ndarray, predicted_class: int) -> float | None:
        return float(np.mean(state_pred[mask] == predicted_class)) if np.any(mask) else None

    return {
        "n": int(len(state_true)),
        "mcc": float(matthews_corrcoef(state_true, state_pred)),
        "macro_f1": float(f1_score(state_true, state_pred, labels=labels, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(state_true, state_pred)),
        "per_state": {
            STATE_NAMES[index]: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in labels
        },
        "confusion_matrix_order": list(STATE_NAMES),
        "confusion_matrix": confusion_matrix(state_true, state_pred, labels=labels).tolist(),
        "posture_macro_f1": posture_macro_f1,
        "posture_selection_lying_support": posture_lying_support,
        "posture_selection_upright_nonwalking_support": posture_upright_support,
        "posture_selection_eligible": posture_selection_eligible,
        "posture_selection_reason": posture_selection_reason,
        "walking_average_precision": walking_ap,
        "feeding_average_precision": feeding_ap,
        "feeding_selection_positive_support": feeding_positive,
        "feeding_selection_negative_support": feeding_negative,
        "feeding_selection_eligible": feeding_selection_eligible,
        "feeding_selection_reason": feeding_selection_reason,
        "selection": selected,
        "pure_standing_recall": subset_rate(pure_standing, 0),
        "feeding_recall": subset_rate(true_feeding, 3),
        "feeding_misclassification_destinations": feeding_destinations,
        "feeding_to_standing_rate": subset_rate(true_feeding, 0),
        "probability_row_sum_max_abs_error": float(
            np.max(np.abs(state_probability.sum(axis=1) - 1.0))
        ),
    }


def finetune(
    model: HierarchicalCCNN,
    train_loader: Any,
    validation_loader: Any,
    validation_codes: np.ndarray,
    *,
    args: argparse.Namespace,
    epochs: int,
    min_epochs: int,
    patience: int,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    model.to(device)
    best_state = copy.deepcopy(model.state_dict())
    best_score = -math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    if min_epochs < 1 or min_epochs > epochs:
        raise ValueError("need 1 <= min_epochs <= epochs")
    for epoch in range(1, epochs + 1):
        model.train()
        totals = {"total": 0.0, "posture": 0.0, "walking": 0.0, "feeding": 0.0}
        seen = 0
        for inputs, _, posture, posture_mask, walking, feeding, feeding_mask in train_loader:
            inputs = inputs.to(device)
            posture = posture.to(device)
            posture_mask = posture_mask.to(device).bool()
            walking = walking.to(device)
            feeding = feeding.to(device)
            feeding_mask = feeding_mask.to(device).bool()
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            losses = masked_losses(
                outputs,
                posture,
                posture_mask,
                walking,
                feeding,
                feeding_mask,
                args,
            )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.project_constraints()
            count = len(inputs)
            for name in totals:
                totals[name] += float(losses[name].detach().cpu()) * count
            seen += count
        val_prediction = predict(model, validation_loader, device)
        val_metrics = metrics(val_prediction, validation_codes)
        score = val_metrics["selection"]["selection_score"]
        if score is None or not np.isfinite(float(score)):
            raise ValueError("validation selection score unavailable")
        history.append(
            {
                "epoch": epoch,
                **{f"train_{name}_loss": value / max(seen, 1) for name, value in totals.items()},
                "validation_selection_score": float(score),
                "validation_posture_macro_f1": val_metrics["posture_macro_f1"],
                "validation_posture_selection_eligible": val_metrics[
                    "posture_selection_eligible"
                ],
                "validation_posture_selection_reason": val_metrics[
                    "posture_selection_reason"
                ],
                "validation_walking_average_precision": val_metrics["walking_average_precision"],
                "validation_feeding_average_precision": val_metrics["feeding_average_precision"],
                "validation_feeding_selection_eligible": val_metrics[
                    "feeding_selection_eligible"
                ],
                "validation_feeding_selection_reason": val_metrics[
                    "feeding_selection_reason"
                ],
                "checkpoint_eligible": epoch >= min_epochs,
            }
        )
        if epoch < min_epochs:
            continue
        if float(score) > best_score + 1e-12:
            best_score = float(score)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return {
        "best_epoch": best_epoch,
        "best_validation_selection_score": best_score,
        "selection_objective": (
            "DATASET_X.metrics.selection_score(posture_macro_f1, walking_average_precision, "
            "posture only with >=10 LYING and >=10 UPRIGHT_NONWALKING; feeding AP only "
            "with >=10 FEEDING and >=10 STANDING; unsupported components are None/{})"
        ),
        "minimum_checkpoint_epoch": min_epochs,
    }, history


def prediction_frame(
    inventory: pd.DataFrame,
    indices: np.ndarray,
    prediction: Mapping[str, np.ndarray],
    *,
    method: str,
    seed: int,
    fold: int,
    fraction: float,
) -> pd.DataFrame:
    frame = inventory.iloc[indices].reset_index(drop=True).copy()
    frame["state_true"] = prediction["state_target"]
    frame["state_pred"] = prediction["state_probability"].argmax(axis=1)
    frame["state_true_name"] = [STATE_NAMES[int(value)] for value in frame["state_true"]]
    frame["state_pred_name"] = [STATE_NAMES[int(value)] for value in frame["state_pred"]]
    for index, name in enumerate(POSTURE_NAMES):
        frame[f"p_posture_{name}_given_non_walking"] = prediction[
            "posture_probability"
        ][:, index]
    for index, name in enumerate(LOCOMOTION_NAMES):
        frame[f"p_locomotion_{name}"] = prediction["walking_probability"][:, index]
    for index, name in enumerate(FEEDING_HEAD_NAMES):
        frame[f"p_feeding_head_{name}"] = prediction["feeding_probability"][:, index]
    for index, name in enumerate(STATE_NAMES):
        frame[f"p_state_{name}"] = prediction["state_probability"][:, index]
    frame["method"] = method
    frame["seed"] = seed
    frame["fold"] = fold
    frame["label_fraction"] = fraction
    return frame


def four_counts(codes: np.ndarray) -> dict[str, int]:
    values = np.asarray(codes).astype(str)
    return {name: int(np.sum(values == name)) for name in STATE_BY_CODE}


def build_plans(
    tail: Any,
    *,
    seeds: list[int],
    fractions: list[float],
    test_cows: list[str],
    validation_cow: str,
    out: Path,
) -> tuple[dict[tuple[int, float, int], dict[str, Any]], dict[str, Any]]:
    plans: dict[tuple[int, float, int], dict[str, Any]] = {}
    checks = []
    for seed in seeds:
        for fold, test_cow in enumerate(test_cows, start=1):
            previous: set[str] | None = None
            for fraction in sorted(fractions):
                plan = selective.split_plan(
                    tail,
                    seed=seed,
                    fraction=fraction,
                    test_cow=test_cow,
                    validation_cow=validation_cow,
                )
                train_ids = set(tail.ids[plan["train_indices"]].tolist())
                if previous is not None and not previous.issubset(train_ids):
                    raise ValueError("non-nested cow×original_code budgets")
                previous = train_ids
                manifest = dict(plan["manifest"])
                manifest.update(
                    fold=fold,
                    train_four_state_counts=four_counts(tail.original_code[plan["train_indices"]]),
                    validation_four_state_counts=four_counts(
                        tail.original_code[plan["validation_indices"]]
                    ),
                    test_four_state_counts=four_counts(tail.original_code[plan["test_indices"]]),
                )
                for role in ("train", "validation", "test"):
                    counts = manifest[f"{role}_four_state_counts"]
                    if min(counts.values()) <= 0:
                        raise ValueError(f"{role} split lacks a four-state class: {counts}")
                budget = f"p{int(round(fraction * 100)):03d}"
                run_root = out / f"seed{seed}" / budget / f"fold{fold}_{test_cow}"
                pilot.write_json(run_root / "split.json", manifest)
                plan["manifest"] = manifest
                plans[(seed, fraction, fold)] = plan
            checks.append({"seed": seed, "fold": fold, "status": "PASS"})
    return plans, {"status": "PASS", "checks": checks}


def main() -> int:
    args = parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.out}")
    for root in (args.target_root, args.pretrain_root, args.public_root):
        if not root.exists():
            raise FileNotFoundError(root)
    seeds = ints(args.seeds)
    fractions = sorted(set(floats(args.label_fractions)))
    test_cows = strings(args.test_cows)
    methods = strings(args.methods)
    if set(methods) - set(METHODS):
        raise ValueError(f"unknown methods: {sorted(set(methods) - set(METHODS))}")
    if "scratch" not in methods:
        raise ValueError("scratch is required")
    if args.validation_cow in test_cows:
        raise ValueError("validation cow cannot be a test cow")
    if args.smoke:
        seeds, fractions, test_cows = seeds[:1], fractions[:1], test_cows[:1]
        epochs, min_epochs, patience = 2, 1, 2
    else:
        epochs, min_epochs, patience = args.epochs, args.min_epochs, args.patience

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    selective.set_seed(seeds[0], torch)
    args.out.mkdir(parents=True, exist_ok=False)
    manifest_path = args.target_root / "DATASET_MANIFEST.json"
    verification_path = args.target_root / "VERIFICATION.json"
    dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    verification_checks = {
        "verified_true": verification.get("verified") is True,
        "DATASET_X": verification.get("DATASET_X", {}).get("status") == "PASS",
        "problems_empty": verification.get("DATASET_X", {}).get("problems") == [],
    }
    if not all(verification_checks.values()):
        raise ValueError(f"DATASET_X verification contract failed: {verification_checks}")
    annotations_relative = Path(str(dataset_manifest["included_annotations_file"]))
    annotations_path = (
        annotations_relative
        if annotations_relative.is_absolute()
        else args.target_root / annotations_relative
    )
    actual_annotations_hash = pilot.sha256_file(annotations_path)
    expected_annotations_hash = str(dataset_manifest["included_annotations_sha256"])
    annotation_rows = len(pd.read_csv(annotations_path, usecols=["event_id"]))
    sessions_path = (
        args.target_root / "datasets" / "DATASET_X" / "supervised_cache" / "sessions.csv"
    )
    samples_path = (
        args.target_root / "datasets" / "DATASET_X" / "supervised_cache" / "samples.csv"
    )
    included_sessions_actual = int(
        pd.read_csv(sessions_path, usecols=["status"])["status"].eq("included").sum()
    )
    supervised_points_actual = len(pd.read_csv(samples_path, usecols=["sample_id"]))
    manifest_checks = {
        "included_annotations_sha256": actual_annotations_hash == expected_annotations_hash,
        "included_mother_annotations": annotation_rows
        == int(dataset_manifest["included_mother_annotations"]),
        "included_sessions": included_sessions_actual == int(dataset_manifest["included_sessions"]),
        "supervised_points": supervised_points_actual == int(dataset_manifest["supervised_points"]),
        "verification_supervised_points": supervised_points_actual
        == int(verification["supervised_points"]),
    }
    if not all(manifest_checks.values()):
        raise ValueError(f"DATASET_MANIFEST contract failed: {manifest_checks}")
    tail, inventory, exclusions, tail_report = binary.build_tail_bundle(args.target_root)
    usable_input_checks = {
        "tail_included_annotation_rows": tail_report["included_annotation_rows"]
        == annotation_rows,
        "usable_windows_match_inventory": tail_report["usable_windows"] == len(inventory),
        "usable_plus_excluded_equals_body_annotations": (
            tail_report["usable_windows"] + tail_report["excluded_windows"]
            == tail_report["body_state_annotation_rows"]
        ),
    }
    if not all(usable_input_checks.values()):
        raise ValueError(f"usable target input counts failed: {usable_input_checks}")
    targets = build_targets(tail.original_code)
    inventory.to_csv(args.out / "tail_window_inventory.csv", index=False, encoding="utf-8-sig")
    exclusions.to_csv(args.out / "tail_window_exclusions.csv", index=False, encoding="utf-8-sig")
    plans, nested_audit = build_plans(
        tail,
        seeds=seeds,
        fractions=fractions,
        test_cows=test_cows,
        validation_cow=args.validation_cow,
        out=args.out,
    )

    checkpoint_audit = []
    for seed in seeds:
        for method in methods:
            if method in SOURCE_METHODS:
                path = args.pretrain_root / f"seed{seed}" / "pretrain" / method / "checkpoint.pt"
                _, metadata = load_safe_state(path)
                if str(metadata.get("source", "")) != method or int(
                    metadata.get("seed", -1)
                ) != seed:
                    raise ValueError(f"public checkpoint metadata mismatch: {path}")
                checkpoint_audit.append(
                    {"seed": seed, "method": method, "path": str(path), "sha256": pilot.sha256_file(path), "metadata": metadata}
                )
        if "tail_ssl" in methods:
            for fold, test_cow in enumerate(test_cows, start=1):
                path = args.pretrain_root / f"seed{seed}" / "tail_ssl" / f"fold{fold}_{test_cow}" / "checkpoint.pt"
                _, metadata = load_safe_state(path)
                expected_train_cows = plans[(seed, fractions[0], fold)]["manifest"][
                    "train_cows"
                ]
                expected_metadata = {
                    "seed": seed,
                    "fold": fold,
                    "test_cow": test_cow,
                    "validation_cow": args.validation_cow,
                    "training_cows": sorted(expected_train_cows),
                }
                actual_metadata = {
                    "seed": int(metadata.get("seed", -1)),
                    "fold": int(metadata.get("fold", -1)),
                    "test_cow": str(metadata.get("test_cow", "")),
                    "validation_cow": str(metadata.get("validation_cow", "")),
                    "training_cows": sorted(str(value) for value in metadata.get("training_cows", [])),
                }
                if actual_metadata != expected_metadata:
                    raise ValueError(
                        f"tail SSL split metadata mismatch: expected={expected_metadata}, "
                        f"actual={actual_metadata}"
                    )
                checkpoint_audit.append(
                    {"seed": seed, "method": "tail_ssl", "fold": fold, "path": str(path), "sha256": pilot.sha256_file(path), "metadata": metadata}
                )

    # Shape-contract audit: exercise every requested initialisation during the
    # dry run without entering a training loop.
    BaseModel = pilot.load_model_class(args.public_root)
    initialisation_checks = []
    for seed in seeds:
        for fold, test_cow in enumerate(test_cows, start=1):
            for method in methods:
                model, audit = initialise_model(
                    BaseModel,
                    method,
                    pretrain_root=args.pretrain_root,
                    seed=seed,
                    fold=fold,
                    test_cow=test_cow,
                )
                model.eval()
                with torch.no_grad():
                    probe = model.compose_probabilities(model(torch.zeros(2, 3, 256)))
                composition_error = float(
                    torch.max(torch.abs(probe["state"].sum(dim=1) - 1.0))
                )
                if not math.isfinite(composition_error) or composition_error > 1e-6:
                    raise ValueError(
                        f"four-state probability composition failed: {composition_error}"
                    )
                initialisation_checks.append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "method": method,
                        "parameters": int(sum(value.numel() for value in model.parameters())),
                        "loaded_keys": len(audit["loaded_keys"]),
                        "probability_sum_max_abs_error": composition_error,
                        "status": "PASS",
                    }
                )

    protocol = {
        "status": "exploratory_hierarchical_four_state",
        "created_at": pd.Timestamp.now().isoformat(),
        "states": list(STATE_NAMES),
        "mapping": STATE_BY_CODE,
        "heads": {
            "locomotion": "unconditional NON_WALKING vs WALKING; public-transfer head",
            "posture": (
                "UPRIGHT=(STANDING+FEEDING) vs LYING conditional on original "
                "NON_WALKING; WALKING rows are masked"
            ),
            "conditional_feeding": "STANDING vs FEEDING; loss masked to those original codes",
        },
        "probability_composition": {
            "walking": "p_walking",
            "lying": "p_nonwalking*p_lying_given_nonwalking",
            "feeding": "p_nonwalking*p_upright_given_nonwalking*p_feeding",
            "standing": "p_nonwalking*p_upright_given_nonwalking*p_nonfeeding",
        },
        "source_transfer": "public encoder + walking BN/head only; posture and feeding heads target-initialised",
        "methods": methods,
        "seeds": seeds,
        "label_fractions": fractions,
        "test_cows": test_cows,
        "validation_cow": args.validation_cow,
        "selection_objective": (
            "DATASET_X.metrics.selection_score(posture_macro_f1, walking_average_precision, "
            "supported_components_only); current validation support leaves walking AP only"
        ),
        "posture_selection_gate": {
            "minimum_lying_windows": 10,
            "minimum_upright_nonwalking_windows": 10,
            "insufficient_support_action": "report posture metrics but pass None to selection_score",
        },
        "feeding_selection_gate": {
            "minimum_feeding_positive_windows": 10,
            "minimum_standing_negative_windows": 10,
            "insufficient_support_action": "report feeding metrics but pass empty event dict to selection_score",
        },
        "budget_protocol": "nested by training cow and original four-class code",
        "paired_sampling_protocol": (
            "within each seed/fold/budget every method rebuilds its training loader with "
            "the same fixed sampler seed, so methods receive an identical sampled index sequence"
        ),
        "hyperparameters": {
            "epochs": epochs,
            "min_epochs": min_epochs,
            "patience": patience,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "posture_loss_weight": args.posture_loss_weight,
            "walking_loss_weight": args.walking_loss_weight,
            "feeding_loss_weight": args.feeding_loss_weight,
            "checkpoint_candidates_begin_at_epoch": min_epochs,
        },
        "tail": tail_report,
        "target_verification": {
            "verification_checks": verification_checks,
            "manifest_checks": manifest_checks,
            "usable_input_checks": usable_input_checks,
            "included_annotations_path": str(annotations_path),
            "included_annotations_sha256": actual_annotations_hash,
            "included_annotation_rows": annotation_rows,
            "included_sessions": included_sessions_actual,
            "supervised_points": supervised_points_actual,
        },
        "checkpoint_inputs": checkpoint_audit,
        "hashes": {
            "dataset_manifest": pilot.sha256_file(manifest_path),
            "verification": pilot.sha256_file(verification_path),
            "script": pilot.sha256_file(Path(__file__)),
            "binary_helper": pilot.sha256_file(Path(binary.__file__)),
            "source_selective_helper": pilot.sha256_file(Path(selective.__file__)),
        },
        "environment": pilot.environment_report(torch, device),
        "limits": [
            "two development test cows only",
            "validation feeding and lying support is two windows each",
            "annotation fractions are not video minutes",
            "no final four-state generalisation claim",
        ],
    }
    pilot.write_json(args.out / "protocol.json", protocol)
    leakage_free = True
    feeding_retained = int(np.sum(tail.original_code == "FEEDING")) > 0
    for plan in plans.values():
        train_ids = set(tail.ids[plan["train_indices"]].tolist())
        val_ids = set(tail.ids[plan["validation_indices"]].tolist())
        test_ids = set(tail.ids[plan["test_indices"]].tolist())
        leakage_free &= not bool(train_ids & val_ids or train_ids & test_ids or val_ids & test_ids)
        feeding_retained &= plan["manifest"]["train_four_state_counts"]["FEEDING"] > 0
        feeding_retained &= plan["manifest"]["validation_four_state_counts"]["FEEDING"] > 0
        feeding_retained &= plan["manifest"]["test_four_state_counts"]["FEEDING"] > 0
    gap_safe = bool(
        (inventory["window_stop_exclusive"] - inventory["window_start_index"])
        .eq(binary.WINDOW_FRAMES)
        .all()
        and inventory["labeled_run_frames"].ge(binary.WINDOW_FRAMES).all()
    )
    nested_ok = nested_audit.get("status") == "PASS" and all(
        item.get("status") == "PASS" for item in nested_audit.get("checks", [])
    )
    checkpoint_ok = len(checkpoint_audit) == len(seeds) * (
        len([method for method in methods if method in SOURCE_METHODS])
        + (len(test_cows) if "tail_ssl" in methods else 0)
    )
    initialisation_ok = all(item["status"] == "PASS" for item in initialisation_checks)
    composition_ok = all(
        item["probability_sum_max_abs_error"] <= 1e-6 for item in initialisation_checks
    )
    validation_only_selection = "test" not in protocol["selection_objective"].lower()
    probe_plan = plans[(seeds[0], fractions[0], 1)]
    probe_sampler_seed = selective.stable_seed(
        "hierarchical_four_state_paired_sampler", seeds[0], 1,
        f"p{int(round(fractions[0] * 100)):03d}",
    )
    probe_batch_size = min(args.batch_size, len(probe_plan["train_indices"]))
    probe_loader_a = make_loader(
        tail.X,
        targets,
        tail.original_code,
        probe_plan["train_indices"],
        batch_size=probe_batch_size,
        seed=probe_sampler_seed,
        training=True,
    )
    probe_loader_b = make_loader(
        tail.X,
        targets,
        tail.original_code,
        probe_plan["train_indices"],
        batch_size=probe_batch_size,
        seed=probe_sampler_seed,
        training=True,
    )
    sampler_indices_a = list(iter(probe_loader_a.sampler))
    sampler_indices_b = list(iter(probe_loader_b.sampler))
    paired_sampler_indices_equal = sampler_indices_a == sampler_indices_b

    # Use a fresh same-seed loader so the actual batch is independent of the
    # sampler comparison above but follows the identical paired rule.
    probe_loader = make_loader(
        tail.X,
        targets,
        tail.original_code,
        probe_plan["train_indices"],
        batch_size=probe_batch_size,
        seed=probe_sampler_seed,
        training=True,
    )
    probe_model = HierarchicalCCNN(BaseModel)
    probe_model.train()
    probe_batch = next(iter(probe_loader))
    (
        probe_inputs,
        _,
        probe_posture,
        probe_posture_mask,
        probe_walking,
        probe_feeding,
        probe_feeding_mask,
    ) = probe_batch
    probe_outputs = probe_model(probe_inputs)
    probe_losses = masked_losses(
        probe_outputs,
        probe_posture,
        probe_posture_mask.bool(),
        probe_walking,
        probe_feeding,
        probe_feeding_mask.bool(),
        args,
    )
    probe_losses["total"].backward()
    finite_loss_and_gradients = bool(
        torch.isfinite(probe_losses["total"])
        and all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in probe_model.parameters()
        )
    )
    all_three_supervision_paths = bool(
        probe_posture_mask.any()
        and len(probe_walking) > 0
        and probe_feeding_mask.any()
        and any(
            name.startswith("posture_head.") and parameter.grad is not None
            for name, parameter in probe_model.named_parameters()
        )
        and any(
            name.startswith("walking_head.") and parameter.grad is not None
            for name, parameter in probe_model.named_parameters()
        )
        and any(
            name.startswith("feeding_head.") and parameter.grad is not None
            for name, parameter in probe_model.named_parameters()
        )
    )

    validation_loader_probe = make_loader(
        tail.X,
        targets,
        tail.original_code,
        probe_plan["validation_indices"],
        batch_size=args.batch_size,
        seed=probe_sampler_seed,
        training=False,
    )
    validation_prediction_probe = predict(probe_model, validation_loader_probe, torch.device("cpu"))
    validation_metrics_probe = metrics(
        validation_prediction_probe,
        tail.original_code[probe_plan["validation_indices"]],
    )
    selection_components = validation_metrics_probe["selection"]["components"]
    support_gate_behavior = bool(
        not validation_metrics_probe["posture_selection_eligible"]
        and not validation_metrics_probe["feeding_selection_eligible"]
        and selection_components["posture_macro_f1"] is None
        and selection_components["event_mean_average_precision"] is None
        and validation_metrics_probe["selection"]["events_used"] == []
        and math.isclose(
            float(validation_metrics_probe["selection"]["selection_score"]),
            float(validation_metrics_probe["walking_average_precision"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )
    dry_execution_details = {
        "probe_seed": seeds[0],
        "probe_fold": 1,
        "probe_label_fraction": fractions[0],
        "paired_sampler_seed": probe_sampler_seed,
        "sampler_draws": len(sampler_indices_a),
        "paired_sampler_indices_equal": paired_sampler_indices_equal,
        "posture_masked_rows_in_batch": int(probe_posture_mask.sum()),
        "feeding_masked_rows_in_batch": int(probe_feeding_mask.sum()),
        "finite_total_loss": float(probe_losses["total"].detach()),
        "finite_loss_and_gradients": finite_loss_and_gradients,
        "all_three_supervision_paths": all_three_supervision_paths,
        "validation_posture_selection_reason": validation_metrics_probe[
            "posture_selection_reason"
        ],
        "validation_feeding_selection_reason": validation_metrics_probe[
            "feeding_selection_reason"
        ],
        "validation_selection_score": validation_metrics_probe["selection"][
            "selection_score"
        ],
        "validation_walking_average_precision": validation_metrics_probe[
            "walking_average_precision"
        ],
        "support_gate_behavior": support_gate_behavior,
    }
    check_values = {
        "verification_json_contract": all(verification_checks.values()),
        "dataset_manifest_contract": all(manifest_checks.values()),
        "usable_input_count_contract": all(usable_input_checks.values()),
        "cow_and_sample_leakage_free": leakage_free,
        "nested_budgets": nested_ok,
        "feeding_retained_in_every_split": feeding_retained,
        "gap_safe_windows": gap_safe,
        "validation_only_selection": validation_only_selection,
        "safe_checkpoint_metadata_and_count": checkpoint_ok,
        "initialisation_shapes": initialisation_ok,
        "probability_composition": composition_ok,
        "paired_sampler_indices_identical": paired_sampler_indices_equal,
        "real_batch_finite_backward": finite_loss_and_gradients,
        "real_batch_all_three_supervision_paths": all_three_supervision_paths,
        "real_validation_support_gate_behavior": support_gate_behavior,
    }
    dry_audit = {
        "status": "PASS" if all(check_values.values()) else "FAIL",
        "checks": check_values,
        "nested_budget_details": nested_audit,
        "safe_checkpoint_loads": len(checkpoint_audit),
        "initialisation_shape_checks": initialisation_checks,
        "dry_execution_details": dry_execution_details,
        "paired_sampler_rule": protocol["paired_sampling_protocol"],
        "planned_runs": len(seeds) * len(fractions) * len(test_cows) * len(methods),
    }
    pilot.write_json(args.out / "artifact_audit.json", dry_audit)
    if dry_audit["status"] != "PASS":
        raise ValueError(f"dry-run rejection checks failed: {check_values}")
    if args.dry_run:
        print(json.dumps({"protocol": protocol, "audit": dry_audit}, ensure_ascii=False, indent=2, default=float))
        return 0

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for fraction in fractions:
            budget = f"p{int(round(fraction * 100)):03d}"
            for fold, test_cow in enumerate(test_cows, start=1):
                plan = plans[(seed, fraction, fold)]
                validation_loader = make_loader(
                    tail.X, targets, tail.original_code, plan["validation_indices"],
                    batch_size=args.batch_size, seed=seed, training=False,
                )
                test_loader = make_loader(
                    tail.X, targets, tail.original_code, plan["test_indices"],
                    batch_size=args.batch_size, seed=seed, training=False,
                )
                for method in methods:
                    paired_sampler_seed = selective.stable_seed(
                        "hierarchical_four_state_paired_sampler", seed, fold, budget
                    )
                    train_loader = make_loader(
                        tail.X,
                        targets,
                        tail.original_code,
                        plan["train_indices"],
                        batch_size=min(args.batch_size, len(plan["train_indices"])),
                        seed=paired_sampler_seed,
                        training=True,
                    )
                    selective.set_seed(seed + fold * 100, torch)
                    model, init_audit = initialise_model(
                        BaseModel, method, pretrain_root=args.pretrain_root,
                        seed=seed, fold=fold, test_cow=test_cow,
                    )
                    started = time.perf_counter()
                    best, history = finetune(
                        model, train_loader, validation_loader,
                        tail.original_code[plan["validation_indices"]],
                        args=args, epochs=epochs, min_epochs=min_epochs,
                        patience=patience, device=device,
                    )
                    elapsed = time.perf_counter() - started
                    val_pred = predict(model, validation_loader, device)
                    test_pred = predict(model, test_loader, device)
                    val_metrics = metrics(val_pred, tail.original_code[plan["validation_indices"]])
                    test_metrics = metrics(test_pred, tail.original_code[plan["test_indices"]])
                    run_dir = args.out / f"seed{seed}" / budget / f"fold{fold}_{test_cow}" / method
                    run_dir.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
                    prediction_frame(
                        inventory, plan["validation_indices"], val_pred,
                        method=method, seed=seed, fold=fold, fraction=fraction,
                    ).to_csv(run_dir / "validation_predictions.csv", index=False)
                    prediction_frame(
                        inventory, plan["test_indices"], test_pred,
                        method=method, seed=seed, fold=fold, fraction=fraction,
                    ).to_csv(run_dir / "test_predictions.csv", index=False)
                    report = {
                        "method": method, "seed": seed, "fold": fold,
                        "test_cow": test_cow, "label_fraction": fraction,
                        "elapsed_seconds": elapsed, "initialisation_audit": init_audit,
                        "paired_sampler_seed": paired_sampler_seed,
                        **best, "validation": val_metrics, "test": test_metrics,
                    }
                    checkpoint_hash = selective.save_safe_checkpoint(
                        run_dir / "best.pt", model.state_dict(),
                        {"method": method, "seed": seed, "fold": fold, "label_fraction": fraction},
                        torch=torch,
                    )
                    report["checkpoint_sha256"] = checkpoint_hash
                    pilot.write_json(run_dir / "report.json", report)
                    rows.append(
                        {
                            "seed": seed, "label_fraction": fraction, "fold": fold,
                            "test_cow": test_cow, "method": method,
                            "mcc": test_metrics["mcc"], "macro_f1": test_metrics["macro_f1"],
                            "balanced_accuracy": test_metrics["balanced_accuracy"],
                            "posture_macro_f1": test_metrics["posture_macro_f1"],
                            "walking_average_precision": test_metrics["walking_average_precision"],
                            "feeding_average_precision": test_metrics["feeding_average_precision"],
                            "pure_standing_recall": test_metrics["pure_standing_recall"],
                            "feeding_recall": test_metrics["feeding_recall"],
                            "feeding_to_standing_rate": test_metrics["feeding_to_standing_rate"],
                            "feeding_predicted_standing": test_metrics["feeding_misclassification_destinations"]["standing"],
                            "feeding_predicted_lying": test_metrics["feeding_misclassification_destinations"]["lying"],
                            "feeding_predicted_walking": test_metrics["feeding_misclassification_destinations"]["walking"],
                            "feeding_predicted_feeding": test_metrics["feeding_misclassification_destinations"]["feeding"],
                            **{f"{name}_f1": test_metrics["per_state"][name]["f1"] for name in STATE_NAMES},
                        }
                    )
    results = pd.DataFrame(rows)
    results.to_csv(args.out / "results.csv", index=False)
    metric_columns = [
        "mcc",
        "macro_f1",
        "balanced_accuracy",
        "posture_macro_f1",
        "walking_average_precision",
        "feeding_average_precision",
        "pure_standing_recall",
        "feeding_recall",
        "feeding_to_standing_rate",
        "standing_f1",
        "lying_f1",
        "walking_f1",
        "feeding_f1",
    ]
    per_cow_rows = []
    for (fraction, method, cow), group in results.groupby(
        ["label_fraction", "method", "test_cow"], sort=True
    ):
        row: dict[str, Any] = {
            "label_fraction": float(fraction),
            "method": str(method),
            "test_cow": str(cow),
            "n_seeds": int(group["seed"].nunique()),
        }
        for metric_name in metric_columns:
            row[f"{metric_name}_seed_mean"] = float(group[metric_name].mean())
            row[f"{metric_name}_seed_sd"] = (
                float(group[metric_name].std(ddof=1)) if len(group) > 1 else 0.0
            )
        per_cow_rows.append(row)
    per_cow = pd.DataFrame(per_cow_rows)
    per_cow.to_csv(args.out / "per_cow_summary.csv", index=False)
    seed_cow_equal = (
        results.groupby(["label_fraction", "method", "seed"], as_index=False)[
            metric_columns
        ]
        .mean()
    )
    seed_cow_equal["n_cows"] = len(test_cows)
    seed_cow_equal.to_csv(args.out / "per_seed_cow_equal.csv", index=False)
    curve_rows = []
    for (fraction, method), group in per_cow.groupby(
        ["label_fraction", "method"], sort=True
    ):
        row = {
            "label_fraction": float(fraction),
            "method": str(method),
            "n_cows": int(group["test_cow"].nunique()),
            "aggregation": "seed_mean_within_cow_then_equal_weight_across_cows",
        }
        for metric_name in metric_columns:
            values = group[f"{metric_name}_seed_mean"]
            row[f"{metric_name}_cow_equal_mean"] = float(values.mean())
            row[f"{metric_name}_cow_sd"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        curve_rows.append(row)
    curve = pd.DataFrame(curve_rows)
    curve.to_csv(args.out / "curve_summary.csv", index=False)
    pilot.write_json(
        args.out / "summary.json",
        {
            "status": "completed_exploratory_hierarchical_four_state",
            "runs": len(results),
            "aggregation": "seed mean within cow, then equal cow mean",
            "n_cows": len(test_cows),
        },
    )
    expected_runs = len(seeds) * len(fractions) * len(test_cows) * len(methods)
    report_paths = sorted(args.out.glob("seed*/p*/fold*/*/report.json"))
    checkpoint_paths = sorted(args.out.glob("seed*/p*/fold*/*/best.pt"))
    history_paths = sorted(args.out.glob("seed*/p*/fold*/*/history.csv"))
    prediction_paths = sorted(args.out.glob("seed*/p*/fold*/*/*_predictions.csv"))
    expected_state_keys = set(HierarchicalCCNN(BaseModel).state_dict())
    checkpoint_restore_ok = True
    for checkpoint_path in checkpoint_paths:
        state, _ = load_safe_state(checkpoint_path)
        checkpoint_restore_ok &= set(state) == expected_state_keys
    backward_history_ok = len(history_paths) == expected_runs
    for history_path in history_paths:
        history_frame = pd.read_csv(history_path)
        backward_history_ok &= bool(
            len(history_frame)
            and np.isfinite(history_frame["train_total_loss"].to_numpy()).all()
        )
    runtime_checks = {
        "expected_run_count": len(results) == expected_runs,
        "loader_and_prediction_outputs": len(prediction_paths) == 2 * expected_runs,
        "loss_backward_histories": backward_history_ok,
        "weights_only_checkpoint_restore": (
            len(checkpoint_paths) == expected_runs and checkpoint_restore_ok
        ),
        "reports_complete": len(report_paths) == expected_runs,
        "summary_contract": all(
            (args.out / name).exists()
            for name in (
                "results.csv",
                "per_cow_summary.csv",
                "per_seed_cow_equal.csv",
                "curve_summary.csv",
                "summary.json",
            )
        ),
    }
    artifact_audit = json.loads((args.out / "artifact_audit.json").read_text(encoding="utf-8"))
    artifact_audit["runtime_checks"] = runtime_checks
    artifact_audit["status"] = "PASS" if all(runtime_checks.values()) else "FAIL"
    pilot.write_json(args.out / "artifact_audit.json", artifact_audit)
    if artifact_audit["status"] != "PASS":
        raise ValueError(f"runtime artifact checks failed: {runtime_checks}")
    print(json.dumps({"status": "complete", "runs": len(results)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
