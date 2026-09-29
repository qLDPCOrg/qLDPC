# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for decoders."""

from __future__ import annotations

import inspect

import numpy as np
import numpy.typing as npt


def _get_external_caller_stacklevel() -> int:
    """Find the first caller outside qLDPC implementation modules."""
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
