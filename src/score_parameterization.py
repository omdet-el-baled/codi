"""Score-output parameterizations for the diffusion model.

The neural network can predict either:

* ``physical``: the physical score s_theta directly.
* ``whitened``: u_theta = S_t^H s_theta.

The whitened target used by DSM is simply -z, which remains O(1) across time.
Reverse samplers always consume the physical score; conversion is analytical.
"""
from __future__ import annotations

from enum import Enum
import torch


class ScoreParameterization(str, Enum):
    PHYSICAL = "physical"
    WHITENED = "whitened"

    @classmethod
    def parse(cls, value: str) -> "ScoreParameterization":
        try:
            return cls(str(value).lower())
        except ValueError as exc:
            allowed = ", ".join(v.value for v in cls)
            raise ValueError(
                f"Unknown score parameterization {value!r}. Allowed: {allowed}."
            ) from exc


def network_output_to_physical(
    output: torch.Tensor,
    *,
    sde,
    t: torch.Tensor,
    parameterization: ScoreParameterization,
    std_modes: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert network output to the physical score used by reverse dynamics."""
    if parameterization is ScoreParameterization.PHYSICAL:
        return output
    if parameterization is ScoreParameterization.WHITENED:
        return sde.solve_SH(output, t, std_modes=std_modes)
    raise AssertionError(parameterization)


def network_output_to_whitened(
    output: torch.Tensor,
    *,
    sde,
    t: torch.Tensor,
    parameterization: ScoreParameterization,
    std_modes: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert network output to S_t^H s_theta for the DSM loss."""
    if parameterization is ScoreParameterization.WHITENED:
        return output
    if parameterization is ScoreParameterization.PHYSICAL:
        return sde.apply_SH(output, t, std_modes=std_modes)
    raise AssertionError(parameterization)
