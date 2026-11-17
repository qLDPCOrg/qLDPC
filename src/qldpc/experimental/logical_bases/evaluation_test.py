# SPDX-License-Identifier: Apache-2.0

"""Unit tests for checking bases of logical operators and measuring their weights and overlaps."""

import numpy as np
import pytest

from qldpc import codes
from qldpc.objects import Pauli, PauliXZ

from .evaluation import LogicalOpsMetrics, evaluate_logical_ops


def test_metrics() -> None:
    """Weights, loads, and overlaps are counted from the supports of the operators."""
    metrics = LogicalOpsMetrics.from_ops([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]])
    np.testing.assert_array_equal(metrics.weights, [2, 2, 1])
    np.testing.assert_array_equal(metrics.loads, [1, 2, 1, 1])
    np.testing.assert_array_equal(metrics.overlaps, [[2, 1, 0], [1, 2, 0], [0, 0, 1]])
    assert (metrics.max_weight, metrics.total_weight, metrics.max_load) == (2, 5, 2)
    assert metrics.num_overlapping_pairs == 1

    metrics = LogicalOpsMetrics.from_ops(np.zeros((0, 3), dtype=int))
    assert (metrics.max_weight, metrics.total_weight, metrics.max_load) == (0, 0, 0)


def test_evaluate_code_basis(stabilizer_code: codes.CSSCode) -> None:
    """By default, the code's own basis is evaluated, as a pair or one Pauli type at a time."""
    metrics_x, metrics_z = evaluate_logical_ops(stabilizer_code)
    paulis_and_metrics: list[tuple[PauliXZ, LogicalOpsMetrics]] = [
        (Pauli.X, metrics_x),
        (Pauli.Z, metrics_z),
    ]
    for pauli, metrics in paulis_and_metrics:
        ops = stabilizer_code.get_logical_ops(pauli).view(np.ndarray)
        np.testing.assert_array_equal(metrics.overlaps, ops.astype(int) @ ops.T.astype(int))
        single = evaluate_logical_ops(stabilizer_code, pauli=pauli)
        np.testing.assert_array_equal(single.overlaps, metrics.overlaps)


def test_evaluate_invalid_bases(stabilizer_code: codes.CSSCode) -> None:
    """Matrices that are not bases of logical operators, or not paired, are rejected."""
    k, n = stabilizer_code.dimension, len(stabilizer_code)
    ops = stabilizer_code.get_logical_ops().view(np.ndarray).copy()

    with pytest.raises(ValueError, match="shape"):
        evaluate_logical_ops(stabilizer_code, ops[:, :n])
    with pytest.raises(ValueError, match="binary"):
        evaluate_logical_ops(stabilizer_code, 2 * ops)
    mixed = ops.copy()
    mixed[0, n:] = ops[k, n:]
    with pytest.raises(ValueError, match="first k logical operators"):
        evaluate_logical_ops(stabilizer_code, mixed)

    not_logical = ops[:k, :n].copy()
    not_logical[0, np.flatnonzero(stabilizer_code.matrix_z[0])[0]] ^= 1
    with pytest.raises(ValueError, match="not logical operators"):
        evaluate_logical_ops(stabilizer_code, not_logical, pauli=Pauli.X)

    dependent = ops[:k, :n].copy()
    dependent[1] = dependent[0]
    with pytest.raises(ValueError, match="independent"):
        evaluate_logical_ops(stabilizer_code, dependent, pauli=Pauli.X)

    # Each sector is still a basis, but X-type operator 0 now anticommutes with Z-type operator 1.
    unpaired = ops.copy()
    unpaired[0] ^= unpaired[1]
    evaluate_logical_ops(stabilizer_code, unpaired[:k, :n], pauli=Pauli.X)
    with pytest.raises(ValueError, match="anticommute"):
        evaluate_logical_ops(stabilizer_code, unpaired)


def test_evaluate_dressed_ops(subsystem_code: codes.CSSCode) -> None:
    """Dressed operators are paired by their logical classes, not by physical commutation."""
    n = len(subsystem_code)
    ops = subsystem_code.get_logical_ops().view(np.ndarray).copy()
    gauge_x = subsystem_code.get_gauge_ops(Pauli.X).view(np.ndarray)[0]
    gauge_z = subsystem_code.get_gauge_ops(Pauli.Z).view(np.ndarray)[0]
    assert gauge_x @ gauge_z % 2 == 1
    ops[0, :n] ^= gauge_x
    ops[subsystem_code.dimension, n:] ^= gauge_z

    # The dressed operators 0 now commute physically, although their classes anticommute.
    assert ops[0, :n] @ ops[subsystem_code.dimension, n:] % 2 == 0
    evaluate_logical_ops(subsystem_code, ops)
    with pytest.raises(ValueError, match="not logical operators"):
        evaluate_logical_ops(subsystem_code, ops, logicals="bare")
