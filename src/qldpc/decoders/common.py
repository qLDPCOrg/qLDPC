"""Shared helpers for decoders.

Copyright 2025 The qLDPC Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

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
