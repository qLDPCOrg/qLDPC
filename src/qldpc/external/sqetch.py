# SPDX-License-Identifier: Apache-2.0

"""Optional integration with the upstream ``sqetch`` distance estimator.

The upstream package is a GPU-only randomized information-set decoder for binary CSS codes.  It
estimates one Pauli sector at a time using stabilizers and logical operators of the opposite Pauli
type.  For subsystem codes, omitting opposite-type gauge generators makes the estimator search for
dressed rather than bare logical operators.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Any

import galois
import numpy as np
import numpy.typing as npt

from qldpc.objects import PAULIS_XZ, Pauli, PauliXZ, PauliXZLike

if TYPE_CHECKING:
    from qldpc.codes import CSSCode

# PyPI rejects direct-URL dependencies, so sqetch cannot be a qldpc extra.
_INSTALL_COMMAND = (
    "pip install 'sqetch[gpu] @ git+https://github.com/a7b/yarn.git"
    "@e9ce9d0fcecc973988558bfee27fab6d6b8d7f97#subdirectory=sqetch'"
)


def is_installed() -> bool:
    """Return whether the optional upstream package is discoverable."""
    return importlib.util.find_spec("sqetch") is not None


def _get_sqetch() -> Any:
    """Import the optional upstream dependency or raise an actionable error."""
    try:
        import sqetch
    except ModuleNotFoundError as error:
        if error.name != "sqetch":
            raise
        raise ModuleNotFoundError(
            "The sqetch distance backend requires the optional 'sqetch' package. "
            f"Install it with `{_INSTALL_COMMAND}`."
        ) from error
    return sqetch


def _get_binary_matrices(
    code: CSSCode, pauli: PauliXZ
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Build binary matrices for dressed-distance estimation in one Pauli sector."""
    if code.field is not galois.GF2:
        raise ValueError("The sqetch distance backend only supports CSS codes over GF(2).")
    if pauli not in PAULIS_XZ:
        raise ValueError(f"sqetch distance bounds require Pauli.X or Pauli.Z, got {pauli!r}.")

    check_pauli = pauli.swap_xz()
    if code.is_subsystem_code:
        # Dressed logicals need only commute with opposite-type stabilizers.  Adding opposite-type
        # gauge generators would restrict the search to bare logicals.
        check_matrix = code.get_stabilizer_ops(check_pauli, canonicalized=True)
    else:
        check_matrix = code.get_matrix(check_pauli)
    logical_matrix = code.get_logical_ops(check_pauli)
    return (
        np.asarray(check_matrix, dtype=np.uint8),
        np.asarray(logical_matrix, dtype=np.uint8),
    )


def get_distance_bound(
    code: CSSCode,
    num_trials: int = 1,
    pauli: PauliXZLike = Pauli.Z,
    *,
    cutoff: int | None = None,
    d_target: int | None = None,
    k_sub: int = 64,
    batch_size: int = 50_000,
    seed: int | None = None,
    device: int = 0,
) -> int:
    """Estimate one CSS distance sector with the optional upstream ``sqetch`` backend.

    ``sqetch`` samples candidate logical operators and reports the lowest weight it observes,
    producing a randomized upper bound.  Its ``d_target`` option stops after finding a candidate
    strictly below the target, while qLDPC's ``cutoff`` convention includes equality.  Therefore,
    qLDPC translates ``cutoff`` to ``d_target = cutoff + 1`` unless the caller provides
    ``d_target``.

    Args:
        code: Binary CSS code whose distance is being estimated.
        num_trials: Number of randomized trials.
        pauli: Pauli sector to estimate (Pauli.X or Pauli.Z, or equivalently a case-insensitive
            "X" or "Z" string).  Opposite-Pauli stabilizers and logical operators are passed to
            ``sqetch``; subsystem codes use dressed-distance semantics.
        cutoff: Stop once a bound is at most this value.
        d_target: Backend-specific strict early-stop target.  This takes precedence over ``cutoff``.
        k_sub: Dimension of the per-trial null-space sketch.
        batch_size: Number of GPU trials per kernel launch.
        seed: Optional random seed.
        device: CUDA device ordinal.

    Raises:
        ModuleNotFoundError: If the optional dependency is not installed.
        RuntimeError: If no nontrivial logical operator is found.
        ValueError: If the code or backend arguments are unsupported.

    Returns:
        An observed nontrivial logical weight, which is an upper bound on the requested distance.
    """
    if num_trials <= 0:
        raise ValueError(f"num_trials must be positive, got {num_trials}.")
    if cutoff is not None and cutoff < 0:
        raise ValueError(f"cutoff must be nonnegative, got {cutoff}.")
    if k_sub <= 0:
        raise ValueError(f"k_sub must be positive, got {k_sub}.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if device < 0:
        raise ValueError(f"device must be nonnegative, got {device}.")
    if d_target is None and cutoff is not None:
        d_target = cutoff + 1

    h_check, logical = _get_binary_matrices(code, Pauli.coerce_xz(pauli))
    sqetch = _get_sqetch()
    try:
        result = sqetch.estimate_distance(
            h_check,
            logical,
            num_trials=num_trials,
            d_target=d_target,
            k_sub=k_sub,
            batch_size=batch_size,
            seed=seed,
            device=device,
        )
    except ModuleNotFoundError as error:
        if error.name != "torch":
            raise
        raise RuntimeError(
            "The sqetch backend requires PyTorch with CUDA extension support. "
            f"Install it with `{_INSTALL_COMMAND}`."
        ) from error
    if result.best_weight is None:
        raise RuntimeError(
            "sqetch found no nontrivial logical representative in "
            f"{result.trials_run} trials; increase num_trials or adjust k_sub."
        )
    return int(result.best_weight)
