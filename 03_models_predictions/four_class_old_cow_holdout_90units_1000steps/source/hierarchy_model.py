"""Legacy-compatible 828-parameter hierarchy; no data loading or task side effects."""
from __future__ import annotations

import hashlib
from typing import Mapping

import torch
from torch import nn
import torch.nn.functional as F

STATE_NAMES = ("standing", "lying", "walking", "feeding")
HEAD_NAMES = ("posture", "walking", "feeding")
ENCODER_PREFIXES = ("ccnn_linear.", "ccnn_nonlinear.", "nonlinear_convs.")


def tensor_hash(value):
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256(f"{array.dtype}|{array.shape}|".encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def state_hash(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        digest.update(tensor_hash(state[name]).encode())
    return digest.hexdigest()


def mlp_head(in_features, hidden=6):
    return nn.Sequential(nn.Linear(in_features, hidden), nn.ReLU(), nn.Linear(hidden, 2))


class HierarchicalCCNN(nn.Module):
    def __init__(self, BaseModel):
        super().__init__()
        # The original three-class construction is intentional RNG consumption.
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

    def forward_features(self, x):
        linear_sequence = self.ccnn_linear(x)
        middle = (linear_sequence.shape[-1] - 1) // 2
        linear = linear_sequence.sort(dim=-1).values[..., middle]
        nonlinear = self.ccnn_nonlinear(x)
        for conv in self.nonlinear_convs:
            nonlinear = torch.tanh(conv(nonlinear))
        nonlinear = nonlinear.abs().mean(dim=-1)
        return torch.cat((linear, nonlinear), dim=1)

    def forward(self, x):
        features = self.forward_features(x)
        return {
            "posture_logits": self.posture_head(self.posture_bn(features)),
            "walking_logits": self.walking_head(self.walking_bn(features)),
            "feeding_logits": self.feeding_head(self.feeding_bn(features)),
        }

    @staticmethod
    def compose_probabilities(outputs: Mapping):
        posture = torch.softmax(outputs["posture_logits"], dim=1)
        walking = torch.softmax(outputs["walking_logits"], dim=1)
        feeding = torch.softmax(outputs["feeding_logits"], dim=1)
        non_walking = walking[:, 0]
        gate = non_walking * posture[:, 0]
        state = torch.stack((gate * feeding[:, 0], non_walking * posture[:, 1],
                             walking[:, 1], gate * feeding[:, 1]), dim=1)
        return {"posture": posture, "walking": walking, "feeding": feeding, "state": state}

    @torch.no_grad()
    def project_constraints(self):
        self.ccnn_linear.project()
        self.ccnn_nonlinear.project()


def masked_losses(outputs, states):
    """Three conditional CE means; never map NON_WALKING to STANDING."""
    if states.ndim != 1 or states.dtype != torch.long:
        raise ValueError("states must be a one-dimensional int64 tensor")
    if states.numel() == 0 or bool(((states < 0) | (states > 3)).any()):
        raise ValueError("training labels must be known four-state labels")
    posture_mask = states != 2
    feeding_mask = (states == 0) | (states == 3)
    posture_loss = (F.cross_entropy(outputs["posture_logits"][posture_mask],
                                   (states[posture_mask] == 1).long())
                    if bool(posture_mask.any()) else outputs["posture_logits"].sum() * 0.0)
    walking_loss = F.cross_entropy(outputs["walking_logits"], (states == 2).long())
    feeding_loss = (F.cross_entropy(outputs["feeding_logits"][feeding_mask],
                                   (states[feeding_mask] == 3).long())
                    if bool(feeding_mask.any()) else outputs["feeding_logits"].sum() * 0.0)
    return {"total": posture_loss + walking_loss + feeding_loss,
            "posture": posture_loss, "walking": walking_loss, "feeding": feeding_loss}


def initialise(BaseModel, method, target_seed, source_state=None):
    if method not in {"scratch", "act", "ufro"}:
        raise ValueError("Unknown initialization method")
    torch.manual_seed(int(target_seed))
    model = HierarchicalCCNN(BaseModel)
    fresh = {k: v.detach().clone() for k, v in model.state_dict().items()}
    selected = {k: v.detach().clone() for k, v in fresh.items()}
    copied = {}
    if method == "scratch":
        if source_state is not None:
            raise ValueError("Scratch must not consume a source checkpoint")
    else:
        if source_state is None:
            raise ValueError("Transfer initialization requires source state")
        # Exact 14-key binary state contract, derived without constructing an
        # additional random network that would perturb RNG state.
        expected = {k for k in fresh if k.startswith(ENCODER_PREFIXES)}
        expected |= {"feature_bn." + k.removeprefix("walking_bn.")
                     for k in fresh if k.startswith("walking_bn.")}
        expected |= {"classifier." + k.removeprefix("walking_head.")
                     for k in fresh if k.startswith("walking_head.")}
        if set(source_state) != expected or len(expected) != 14:
            raise ValueError("Source binary state key contract mismatch")
        for source_name, value in source_state.items():
            if source_name.startswith(ENCODER_PREFIXES):
                target_name = source_name
            elif source_name.startswith("feature_bn."):
                target_name = "walking_bn." + source_name.removeprefix("feature_bn.")
            else:
                target_name = "walking_head." + source_name.removeprefix("classifier.")
            target = selected[target_name]
            if target.shape != value.shape or target.dtype != value.dtype:
                raise ValueError(f"Source shape/dtype mismatch: {source_name}")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"Nonfinite source state: {source_name}")
            selected[target_name] = value.detach().cpu().clone()
            copied[target_name] = source_name
    model.load_state_dict(selected, strict=True)
    if sum(p.numel() for p in model.parameters()) != 828:
        raise ValueError("Hierarchy parameter contract changed")
    if not all(p.requires_grad for p in model.parameters()):
        raise ValueError("All target parameters must be trainable")
    for key in selected:
        if key.startswith(("posture_", "feeding_")) and not torch.equal(selected[key], fresh[key]):
            raise AssertionError("A target-private tensor was overwritten")
    audit = {
        "method": method, "target_seed": int(target_seed), "parameter_count": 828,
        "initial_state_sha256": state_hash(selected),
        "fresh_state_sha256": state_hash(fresh), "all_parameters_trainable": True,
        "state_order": list(STATE_NAMES), "copied_keys": sorted(copied),
        "keys": [{"key": key, "source_key": copied.get(key),
                  "action": "copied" if key in copied else "fresh",
                  "shape": list(value.shape), "dtype": str(value.dtype),
                  "fresh_sha256": tensor_hash(fresh[key]), "selected_sha256": tensor_hash(value)}
                 for key, value in selected.items()],
    }
    return model, audit
