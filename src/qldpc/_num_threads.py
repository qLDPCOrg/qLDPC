# SPDX-License-Identifier: Apache-2.0

"""The number of threads of qLDPC's searches, which the environment variable QLDPC_NUM_THREADS sets.

qLDPC reads ``QLDPC_NUM_THREADS`` when it is imported, before any of its modules imports numba.
If the variable is set, it must be a positive integer, and searches whose tuning leaves
``num_threads`` at ``None`` use that many threads (see :mod:`qldpc.codes.distance._tuning`).

Numba never reads ``QLDPC_NUM_THREADS``; it reads its own thread count, which its parallel code,
such as some of galois's, uses, from ``NUMBA_NUM_THREADS`` when numba is imported.  So the variable
sets numba's count too only if qLDPC is imported before numba, and so before galois:

- If numba has not been imported, qLDPC sets the environment variable ``NUMBA_NUM_THREADS`` to the
  same value.  If ``NUMBA_NUM_THREADS`` held another number, qLDPC warns that
  ``QLDPC_NUM_THREADS`` replaces it.
- If numba has already been imported with another count, qLDPC warns that numba's parallel code
  keeps that count.  To give it the same count, import qLDPC before numba and galois, or also set
  ``NUMBA_NUM_THREADS`` to the same value before Python starts.  ``numba.set_num_threads`` would
  change the count only for the calling thread, so qLDPC does not call it.

If the variable is not set, qLDPC changes nothing, and searches use numba's thread count,
``numba.config.NUMBA_NUM_THREADS``.
"""

from __future__ import annotations

import os
import sys
import warnings
from collections.abc import Mapping, MutableMapping
from typing import Any

# The environment variable that sets the number of threads of qLDPC's searches.
NUM_THREADS_VARIABLE = "QLDPC_NUM_THREADS"

# The environment variable from which numba reads its thread count when it is imported.
_NUMBA_VARIABLE = "NUMBA_NUM_THREADS"


def read_num_threads(environ: MutableMapping[str, str], modules: Mapping[str, Any]) -> int | None:
    """Read ``QLDPC_NUM_THREADS`` from an environment, and give numba the same count if it can.

    Args:
        environ: The environment, such as ``os.environ``, whose ``NUMBA_NUM_THREADS`` this sets.
        modules: The imported modules, such as ``sys.modules``, which say whether numba is imported.

    Returns:
        The number of threads, or None if the variable is not set.

    Raises:
        ValueError: If the variable is set to anything but a positive integer.
    """
    value = environ.get(NUM_THREADS_VARIABLE)
    if value is None:
        return None
    if not (value.isascii() and value.isdigit() and int(value) > 0):
        raise ValueError(f"{NUM_THREADS_VARIABLE} must be a positive integer, not {value!r}")
    num_threads = int(value)

    # Give numba the same count, or warn that it keeps the count that it has.
    numba = modules.get("numba")
    if numba is None:
        numba_value = environ.get(_NUMBA_VARIABLE)
        if numba_value is not None and _as_count(numba_value) != num_threads:
            warnings.warn(
                f"{NUM_THREADS_VARIABLE}={num_threads} replaces {_NUMBA_VARIABLE}={numba_value}",
                RuntimeWarning,
                stacklevel=2,
            )
        environ[_NUMBA_VARIABLE] = str(num_threads)
    elif numba.config.NUMBA_NUM_THREADS != num_threads:
        warnings.warn(
            f"numba was imported before qLDPC, with {numba.config.NUMBA_NUM_THREADS} threads, which "
            f"its parallel code, such as that of galois, keeps.  qLDPC's searches use "
            f"{NUM_THREADS_VARIABLE}={num_threads} threads.  For numba to use them too, import "
            f"qLDPC before numba and galois, or also set {_NUMBA_VARIABLE}={num_threads} before "
            "Python starts.",
            RuntimeWarning,
            stacklevel=2,
        )
    return num_threads


def _as_count(value: str) -> int | None:
    """A thread count as an integer, or None if the value is not a nonnegative integer."""
    value = value.strip()
    return int(value) if value.isascii() and value.isdigit() else None


#: The number of threads of qLDPC's searches that ``QLDPC_NUM_THREADS`` set when qLDPC was
#: imported, or None if it was not set.
NUM_THREADS = read_num_threads(os.environ, sys.modules)
