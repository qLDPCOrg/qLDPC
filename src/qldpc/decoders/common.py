# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for decoders."""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt


def _get_external_caller_stacklevel() -> int:
    """Find the stacklevel of the first caller outside qLDPC implementation modules.

    Passing this stacklevel to warnings.warn attributes a warning to the user code that called into
    qLDPC, however deeply nested the call that emits the warning.  Test modules are co-located with
    the modules that they test (as qldpc.<...>_test), so they are treated as external callers.
    """
    stacklevel = 1
    frame = inspect.currentframe()
    if frame is None:  # pragma: no cover
        return 2
    frame = frame.f_back
    while frame is not None:
        module = str(frame.f_globals.get("__name__", ""))
        if not module.startswith("qldpc.") or module.endswith("_test"):
            break
        stacklevel += 1
        frame = frame.f_back
    return stacklevel


def _get_deprecated_alias(module_name: str, name: str, aliases: Mapping[str, type]) -> Any:
    """Retrieve the replacement for a deprecated name, and warn that the name is deprecated.

    This function backs module-level __getattr__ functions (PEP 562).  A deprecated name thereby
    refers to the same object as its replacement, which preserves isinstance checks and subclassing,
    while still warning whenever the deprecated name is accessed.
    """
    if name not in aliases:
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
    replacement = aliases[name]
    warnings.warn(
        f"{name} is deprecated; use {replacement.__name__} instead",
        DeprecationWarning,
        stacklevel=_get_external_caller_stacklevel(),
    )
    return replacement


def with_erasure_bits(
    errors: npt.NDArray[np.int_], erased: npt.NDArray[np.bool_] | bool
) -> npt.NDArray[np.int_]:
    """Append an erasure bit to each inferred error, in the dtype of that error.

    A decoder that signals erasure reports it in the last entry of every error it infers, set to 1
    for a syndrome that the error does not reproduce.  Accepts one error with one flag, or a batch
    of errors with one flag per error.
    """
    flags = np.asarray(erased, dtype=errors.dtype).reshape(errors.shape[:-1] + (1,))
    return np.hstack([errors, flags])
