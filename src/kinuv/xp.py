"""Array-module dispatch. Do not import jax at package import."""

from __future__ import annotations

import numpy as np


def is_jax(x) -> bool:
    return type(x).__module__.startswith("jax")


def numpy_or_jax(*xs):
    """``jax.numpy`` if any argument is a JAX array, else ``numpy``."""
    for x in xs:
        if is_jax(x):
            import jax.numpy as jnp

            return jnp
    return np
