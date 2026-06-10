"""Global numerical configuration for the Adam simulators.

Keeps the rest of the codebase device-agnostic: the same functions run on CPU
or GPU unchanged. Call :func:`enable_x64` *before* importing jax-heavy modules
if you want float64 (recommended for the ODE/SDE integrators).
"""
from __future__ import annotations

import jax


def enable_x64(flag: bool = True) -> None:
    """Toggle double precision globally. Must run before arrays are created."""
    jax.config.update("jax_enable_x64", flag)


def devices() -> list:
    """Return the JAX devices currently visible (CPU or GPU)."""
    return jax.devices()


def default_backend() -> str:
    """Name of the active backend, e.g. ``'cpu'`` or ``'gpu'``."""
    return jax.default_backend()
