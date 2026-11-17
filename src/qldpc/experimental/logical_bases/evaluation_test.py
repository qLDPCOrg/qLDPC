# SPDX-License-Identifier: Apache-2.0

"""Unit tests for checking bases of logical operators and measuring their weights and overlaps."""

import numpy as np
import pytest
import scipy.sparse

from qldpc import codes
from qldpc.objects import Pauli, PauliXZ

from .evaluation import LogicalOpsMetrics, evaluate_logical_ops


def test_metrics() -> None:
    """Weights, loads, and overlaps are counted from the supports of the operators."""
    metrics = LogicalOpsMetrics._from_valid_ops(
        np.array([[1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 0, 1]])
    )
    np.testing.assert_array_equal(metrics.weights, [2, 3, 1])
    np.testing.assert_array_equal(metrics.loads, [2, 2, 1, 1])
    np.testing.assert_array_equal(metrics.overlaps, [[2, 2, 0], [2, 3, 0], [0, 0, 1]])
    assert (metrics.max_weight, metrics.total_weight, metrics.max_load) == (3, 6, 2)
    assert metrics.num_overlapping_pairs == 1

    metrics = LogicalOpsMetrics._from_valid_ops(np.zeros((0, 3), dtype=np.uint8))
    assert (metrics.max_weight, metrics.total_weight, metrics.max_load) == (0, 0, 0)


def test_evaluate_code_basis(stabilizer_code: codes.CSSCode) -> None:
    """By default, the code's own basis is evaluated, complete or one Pauli type at a time."""
    metrics = evaluate_logical_ops(stabilizer_code)
    # the two Pauli types differ, so that swapping x and z would fail
    assert not np.array_equal(metrics.x.overlaps, metrics.z.overlaps)
    paulis_and_metrics: list[tuple[PauliXZ, LogicalOpsMetrics]] = [
        (Pauli.X, metrics.x),
        (Pauli.Z, metrics.z),
    ]
    for pauli, metrics_of_type in paulis_and_metrics:
        ops = stabilizer_code.get_logical_ops(pauli).view(np.ndarray)
        np.testing.assert_array_equal(metrics_of_type.overlaps, ops.astype(int) @ ops.T.astype(int))
        single = LogicalOpsMetrics.from_ops(stabilizer_code, pauli)
        np.testing.assert_array_equal(single.overlaps, metrics_of_type.overlaps)

    # sparse input gives the same metrics
    sparse_ops = scipy.sparse.csr_array(stabilizer_code.get_logical_ops().view(np.ndarray))
    np.testing.assert_array_equal(
        evaluate_logical_ops(stabilizer_code, sparse_ops).x.overlaps, metrics.x.overlaps
    )

    # a code without logical qubits has an empty basis
    code = codes.CSSCode([[1, 1]], [[1, 1]])
    metrics = evaluate_logical_ops(code)
    assert metrics.x.weights.shape == metrics.z.weights.shape == (0,)


def test_evaluate_invalid_bases(stabilizer_code: codes.CSSCode) -> None:
    """Matrices that are not bases of logical operators, or not paired, are rejected.

    So is an invalid Pauli type.
    """
    k, n = stabilizer_code.dimension, len(stabilizer_code)
    ops = stabilizer_code.get_logical_ops().view(np.ndarray).copy()

    with pytest.raises(ValueError, match="shape"):
        evaluate_logical_ops(stabilizer_code, ops[:, :n])

    # without a logical qubit, the other checks would accept a row too many
    with pytest.raises(ValueError):
        evaluate_logical_ops(codes.CSSCode([[1, 1]], [[1, 1]]), np.zeros((1, 4), dtype=int))
    # a repeated row keeps the classes independent, so only the shape check rejects it
    with pytest.raises(ValueError, match="shape"):
        LogicalOpsMetrics.from_ops(stabilizer_code, Pauli.X, np.vstack([ops[:k, :n], ops[:1, :n]]))
    with pytest.raises(ValueError, match="binary"):
        evaluate_logical_ops(stabilizer_code, 2 * ops)
    with pytest.raises(ValueError):
        LogicalOpsMetrics.from_ops(stabilizer_code, Pauli.Y)  # type: ignore[arg-type]
    for row, other_half in [(0, slice(n, None)), (k, slice(None, n))]:
        mixed = ops.copy()
        mixed[row, other_half] = 1
        with pytest.raises(ValueError, match="first k logical operators"):
            evaluate_logical_ops(stabilizer_code, mixed)

    # flip a bit of one Z-type operator on the support of an X-type stabilizer
    not_logical = ops.copy()
    not_logical[k, n + np.flatnonzero(stabilizer_code.matrix_x[0])[0]] ^= 1
    with pytest.raises(ValueError, match="Z-type operators are not logical"):
        evaluate_logical_ops(stabilizer_code, not_logical)

    # the rows of a basis of one type stay independent, but two of them now have the same class
    dependent = ops[:k, :n].copy()
    dependent[1] = dependent[0] ^ stabilizer_code.matrix_x.view(np.ndarray)[0]
    with pytest.raises(ValueError, match="independent"):
        LogicalOpsMetrics.from_ops(stabilizer_code, Pauli.X, dependent)

    # each half is still a basis, but X-type operator 0 now anticommutes with Z-type operator 1
    unpaired = ops.copy()
    unpaired[0] ^= unpaired[1]
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

    # X-type operator 0 and Z-type operator 0 now commute physically, but their classes anticommute
    assert ops[0, :n] @ ops[subsystem_code.dimension, n:] % 2 == 0
    evaluate_logical_ops(subsystem_code, ops)

    # each half of the dressed operators is rejected when bare operators are required
    original = subsystem_code.get_logical_ops().view(np.ndarray)
    for row, name in [(0, "X"), (subsystem_code.dimension, "Z")]:
        one_half = original.copy()
        one_half[row] = ops[row]
        with pytest.raises(ValueError, match=f"{name}-type operators are not logical"):
            evaluate_logical_ops(subsystem_code, one_half, logicals="bare")

    # the X-type half of these dressed operators is likewise accepted as a basis of one type, and
    # rejected when bare operators are required
    ops_x = ops[: subsystem_code.dimension, :n]
    LogicalOpsMetrics.from_ops(subsystem_code, Pauli.X, ops_x)
    with pytest.raises(ValueError, match="X-type operators are not logical"):
        LogicalOpsMetrics.from_ops(subsystem_code, Pauli.X, ops_x, logicals="bare")
