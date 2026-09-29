# SPDX-License-Identifier: Apache-2.0

"""Tests for the optional sqetch distance backend."""

from __future__ import annotations

import builtins
import itertools
import sys
import types
import unittest.mock
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
import pytest

from qldpc import codes, external
from qldpc.objects import Pauli, PauliXZ, PauliXZLike


def test_is_installed() -> None:
    """Report whether the optional upstream package is discoverable."""
    with unittest.mock.patch("importlib.util.find_spec", return_value=unittest.mock.Mock()):
        assert external.sqetch.is_installed()
    with unittest.mock.patch("importlib.util.find_spec", return_value=None):
        assert not external.sqetch.is_installed()


def _row_span(rows: npt.NDArray[np.uint8]) -> set[tuple[int, ...]]:
    """Enumerate the binary span of a small matrix for an independent test oracle."""
    rows = np.asarray(rows, dtype=np.uint8)
    return {
        tuple(np.asarray(coefficients, dtype=np.uint8) @ rows % 2)
        for coefficients in itertools.product((0, 1), repeat=len(rows))
    }


def _dressed_distance_oracle(code: codes.CSSCode, pauli: Literal[Pauli.X, Pauli.Z]) -> int:
    """Find the minimum-weight target-type operator modulo target-type gauges."""
    check_pauli = cast(Literal[Pauli.X, Pauli.Z], pauli.swap_xz())
    check = np.asarray(code.get_stabilizer_ops(check_pauli), dtype=np.uint8)
    quotient = np.vstack(
        [
            np.asarray(code.get_stabilizer_ops(pauli), dtype=np.uint8),
            np.asarray(code.get_gauge_ops(pauli), dtype=np.uint8),
        ]
    )
    target_span = _row_span(quotient)
    distances = [
        sum(vector)
        for vector in itertools.product((0, 1), repeat=len(code))
        if not np.any(check @ np.asarray(vector, dtype=np.uint8) % 2) and vector not in target_span
    ]
    return min(distances)


def test_binary_matrix_conversion() -> None:
    """Convert both CSS distance sectors to the upstream matrix convention."""
    code = codes.SteaneCode()

    h_check, logical = external.sqetch._get_binary_matrices(code, Pauli.Z)
    assert h_check.dtype == np.uint8
    assert logical.dtype == np.uint8
    assert np.array_equal(h_check, code.matrix_x)
    assert np.array_equal(logical, code.get_logical_ops(Pauli.X))

    h_check, logical = external.sqetch._get_binary_matrices(code, Pauli.X)
    assert np.array_equal(h_check, code.matrix_z)
    assert np.array_equal(logical, code.get_logical_ops(Pauli.Z))


def test_subsystem_matrix_conversion() -> None:
    """Use only stabilizer checks so sqetch searches dressed subsystem logicals."""
    code = codes.BaconShorCode(3)
    h_check, logical = external.sqetch._get_binary_matrices(code, Pauli.Z)
    expected = code.get_stabilizer_ops(Pauli.X, canonicalized=True)
    assert np.array_equal(h_check, expected)
    assert np.array_equal(logical, code.get_logical_ops(Pauli.X))


def test_subsystem_dressed_distance_regression() -> None:
    """Match an independent dressed-distance oracle on a nontrivial subsystem code."""
    code = codes.CSSCode([[1, 1, 0], [1, 0, 1]], [[1, 0, 0]])
    h_check, logical = external.sqetch._get_binary_matrices(code, Pauli.Z)
    candidate = (0, 1, 1)

    assert _dressed_distance_oracle(code, Pauli.Z) == 2
    assert np.array_equal(h_check, [[0, 1, 1]])
    assert np.array_equal(logical, [[0, 0, 1]])
    assert not np.any(h_check @ np.asarray(candidate, dtype=np.uint8) % 2)
    target_span = _row_span(
        np.vstack([code.get_stabilizer_ops(Pauli.Z), code.get_gauge_ops(Pauli.Z)])
    )
    assert candidate not in target_span


def test_matrix_conversion_rejects_unsupported_inputs() -> None:
    """Reject nonbinary and non-CSS distance sectors before importing the optional package."""
    with pytest.raises(ValueError, match="only supports CSS codes over GF\\(2\\)"):
        external.sqetch._get_binary_matrices(codes.BaconShorCode(3, field=3), Pauli.Z)

    with pytest.raises(ValueError, match=r"require Pauli\.X or Pauli\.Z"):
        external.sqetch._get_binary_matrices(codes.SteaneCode(), cast(PauliXZ, Pauli.Y))


def test_get_distance_bound_forwards_options(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forward qLDPC's bound options and translate inclusive cutoff semantics."""
    result = types.SimpleNamespace(best_weight=3, trials_run=17)
    estimate = unittest.mock.Mock(return_value=result)
    monkeypatch.setitem(sys.modules, "sqetch", types.SimpleNamespace(estimate_distance=estimate))

    code = codes.SteaneCode()
    assert (
        external.sqetch.get_distance_bound(
            code,
            num_trials=17,
            pauli=Pauli.Z,
            cutoff=3,
            k_sub=5,
            batch_size=11,
            seed=7,
            device=2,
        )
        == 3
    )

    h_check, logical = estimate.call_args.args
    assert h_check.dtype == np.uint8
    assert logical.dtype == np.uint8
    assert estimate.call_args.kwargs == {
        "num_trials": 17,
        "d_target": 4,
        "k_sub": 5,
        "batch_size": 11,
        "seed": 7,
        "device": 2,
    }


def test_get_distance_bound_validates_and_reports_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate trial arguments and provide a useful missing-dependency error."""
    monkeypatch.setitem(sys.modules, "sqetch", None)
    code = codes.SteaneCode()
    with pytest.raises(ModuleNotFoundError, match="optional 'sqetch' package"):
        external.sqetch.get_distance_bound(code)

    with pytest.raises(ValueError, match="num_trials must be positive"):
        external.sqetch.get_distance_bound(code, num_trials=0)
    with pytest.raises(ValueError, match="cutoff must be nonnegative"):
        external.sqetch.get_distance_bound(code, cutoff=-1)
    with pytest.raises(ValueError, match="k_sub must be positive"):
        external.sqetch.get_distance_bound(code, k_sub=0)
    with pytest.raises(ValueError, match="batch_size must be positive"):
        external.sqetch.get_distance_bound(code, batch_size=0)
    with pytest.raises(ValueError, match="device must be nonnegative"):
        external.sqetch.get_distance_bound(code, device=-1)


def test_get_distance_bound_accepts_string_pauli(monkeypatch: pytest.MonkeyPatch) -> None:
    """Accept a case-insensitive "X" or "Z" string in place of Pauli.X or Pauli.Z."""
    result = types.SimpleNamespace(best_weight=3, trials_run=1)
    estimate = unittest.mock.Mock(return_value=result)
    monkeypatch.setitem(sys.modules, "sqetch", types.SimpleNamespace(estimate_distance=estimate))

    code = codes.SurfaceCode(3, rotated=False)
    cases: list[tuple[PauliXZ, PauliXZLike]] = [(Pauli.X, "x"), (Pauli.Z, "Z")]
    for pauli, string in cases:
        external.sqetch.get_distance_bound(code, pauli=pauli)
        expected_args = estimate.call_args.args
        external.sqetch.get_distance_bound(code, pauli=string)
        for actual, expected in zip(estimate.call_args.args, expected_args, strict=True):
            assert np.array_equal(actual, expected)


def test_get_distance_bound_requires_observed_logical(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report a failed sqetch search instead of returning a false numeric bound."""
    result = types.SimpleNamespace(best_weight=None, trials_run=4)
    monkeypatch.setitem(
        sys.modules,
        "sqetch",
        types.SimpleNamespace(estimate_distance=unittest.mock.Mock(return_value=result)),
    )
    with pytest.raises(RuntimeError, match="found no nontrivial"):
        external.sqetch.get_distance_bound(codes.SteaneCode())


def test_get_distance_bound_reports_missing_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Translate the upstream missing-PyTorch error into an actionable qLDPC error."""

    def estimate_distance(*args: object, **kwargs: object) -> None:
        """Raise the error emitted when sqetch's optional GPU dependency is absent."""
        raise ModuleNotFoundError("No module named 'torch'", name="torch")

    monkeypatch.setitem(
        sys.modules,
        "sqetch",
        types.SimpleNamespace(estimate_distance=estimate_distance),
    )
    with pytest.raises(RuntimeError, match="PyTorch with CUDA"):
        external.sqetch.get_distance_bound(codes.SteaneCode())


def test_get_distance_bound_preserves_unrelated_import_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not hide import failures unrelated to the optional PyTorch dependency."""

    def estimate_distance(*args: object, **kwargs: object) -> None:
        """Raise an unrelated optional-dependency error."""
        raise ModuleNotFoundError("No module named 'other'", name="other")

    monkeypatch.setitem(
        sys.modules,
        "sqetch",
        types.SimpleNamespace(estimate_distance=estimate_distance),
    )
    with pytest.raises(ModuleNotFoundError, match="other"):
        external.sqetch.get_distance_bound(codes.SteaneCode())


def test_get_sqetch_preserves_import_errors_for_other_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not relabel an import failure raised while loading another dependency."""
    original_import = builtins.__import__

    def import_sqetch(
        name: str,
        globals_: dict[str, object] | None = None,
        locals_: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> types.ModuleType:
        if name == "sqetch":
            raise ModuleNotFoundError("No module named 'other'", name="other")
        return original_import(name, globals_, locals_, fromlist, level)

    assert import_sqetch("types") is types
    monkeypatch.setattr(builtins, "__import__", import_sqetch)
    with pytest.raises(ModuleNotFoundError, match="other"):
        external.sqetch._get_sqetch()
