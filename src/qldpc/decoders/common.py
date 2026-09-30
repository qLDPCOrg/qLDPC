# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for decoders."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


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
