"""Dependency-free exponential moving average for Lightning checkpoints."""
from __future__ import annotations
from contextlib import contextmanager
import torch


class ExponentialMovingAverage:
    def __init__(self, parameters, decay: float = 0.999):
        params = [p for p in parameters if p.requires_grad]
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0,1)")
        self.decay = float(decay)
        self.num_updates = 0
        self.shadow_params = [p.detach().clone() for p in params]
        self.collected_params = None

    @torch.no_grad()
    def update(self, parameters, decay: float | None = None):
        params = [p for p in parameters if p.requires_grad]
        if len(params) != len(self.shadow_params):
            raise RuntimeError("EMA parameter count changed after initialization")
        d = self.decay if decay is None else float(decay)
        one_minus = 1.0 - d
        for shadow, param in zip(self.shadow_params, params):
            shadow.sub_(one_minus * (shadow - param.detach()))
        self.num_updates += 1

    @torch.no_grad()
    def store(self, parameters):
        params = [p for p in parameters if p.requires_grad]
        self.collected_params = [p.detach().clone() for p in params]

    @torch.no_grad()
    def copy_to(self, parameters):
        params = [p for p in parameters if p.requires_grad]
        for param, shadow in zip(params, self.shadow_params):
            param.data.copy_(shadow.data)

    @torch.no_grad()
    def restore(self, parameters):
        if self.collected_params is None:
            return
        params = [p for p in parameters if p.requires_grad]
        for param, saved in zip(params, self.collected_params):
            param.data.copy_(saved.data)
        self.collected_params = None

    def state_dict(self):
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow_params": [p.detach().clone() for p in self.shadow_params],
        }

    def load_state_dict(self, state_dict):
        self.decay = float(state_dict.get("decay", self.decay))
        self.num_updates = int(state_dict.get("num_updates", 0))
        shadows = state_dict.get("shadow_params")
        if shadows is not None:
            if len(shadows) != len(self.shadow_params):
                raise RuntimeError("EMA state is incompatible with this model")
            for dst, src in zip(self.shadow_params, shadows):
                dst.data.copy_(src.to(device=dst.device, dtype=dst.dtype))

    def to(self, *args, **kwargs):
        self.shadow_params = [p.to(*args, **kwargs) for p in self.shadow_params]
        if self.collected_params is not None:
            self.collected_params = [p.to(*args, **kwargs) for p in self.collected_params]
        return self

    @contextmanager
    def average_parameters(self, parameters):
        self.store(parameters)
        self.copy_to(parameters)
        try:
            yield
        finally:
            self.restore(parameters)
