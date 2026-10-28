# SPDX-License-Identifier: Apache-2.0

"""Tests for pymatching decoder builders and observable decoding."""

from __future__ import annotations

import builtins
import concurrent.futures
import pickle
import sys
import threading
import types
import unittest.mock
from collections.abc import Callable
from typing import Any, Literal, cast

import galois
import numpy as np
import pymatching as pymatching_package
import pytest
import stim

from qldpc import decoders
from qldpc.conftest import run_python
from qldpc.decoders.external import pymatching
from qldpc.decoders.external.pymatching import (
    _get_decoder_mwpm,
    _get_observable_decoder_mwpm,
)


def test_mwpm_error_builders() -> None:
    """MWPM error builders decode matrices and detector error models."""
    matrix = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int32)
    dem = decoders.DetectorErrorModelArrays.from_arrays(
        matrix, None, np.array([0.1, 0.2, 0.3])
    ).to_dem()
    error = np.array([1, 0, 0])
    syndrome = matrix @ error % 2

    for pcm_or_dem in [matrix, dem]:
        decoded = np.asarray(_get_decoder_mwpm(pcm_or_dem).decode(syndrome), dtype=int)
        assert decoded.shape == error.shape
        assert np.array_equal(matrix @ decoded % 2, syndrome)

    assert _get_decoder_mwpm(matrix).decode(syndrome).shape == (3,)
    with pytest.raises(ValueError, match="cannot infer errors"):
        _get_decoder_mwpm(matrix, enable_correlations=True)
    # the erasure-bit guard of every builder accepts add_erasure_bit, and rejects a True value
    erasure_checked_builder = cast(Callable[..., object], _get_decoder_mwpm)
    with pytest.raises(ValueError, match="The MWPM decoder cannot signal erasure"):
        erasure_checked_builder(matrix, add_erasure_bit=True)
    assert erasure_checked_builder(matrix, add_erasure_bit=False)


def test_mwpm_observable_builders() -> None:
    """Ordinary and correlated matching predict DEM observable flips."""
    dem = stim.DetectorErrorModel("error(0.1) D0 L0\nerror(0.2) D1 L1")
    syndromes = np.array([[1, 0], [0, 1]], dtype=int)
    observable_decoder = _get_observable_decoder_mwpm(dem)
    assert np.array_equal(observable_decoder.decode_observables(syndromes[0]), syndromes[0])
    assert isinstance(observable_decoder, decoders.BatchDetailedObservableDecoder)
    detailed = observable_decoder.decode_observables_detailed_batch(syndromes)
    assert [result.observable_flips.tolist() for result in detailed] == [[1, 0], [0, 1]]
    weights = [result.diagnostics["pymatching.objective_weight"] for result in detailed]
    assert weights == pytest.approx(np.log([9, 4]))
    single = observable_decoder.decode_observables_detailed(syndromes[1])
    assert single.diagnostics["pymatching.objective_weight"] == pytest.approx(np.log(4))
    assert observable_decoder.decode_observables_detailed_batch(syndromes[:0]) == ()
    batch_decoder = cast(decoders.BatchObservableDecoder, observable_decoder)
    assert np.array_equal(batch_decoder.decode_observables_batch(syndromes), syndromes)

    correlated_dem = stim.DetectorErrorModel("""
        error(0.02) D0 D1 ^ D2 D3
        error(0.3) D2 L0
        error(0.3) D3
    """)
    correlated_decoder = _get_observable_decoder_mwpm(correlated_dem, enable_correlations=True)
    assert np.array_equal(correlated_decoder.decode_observables(np.ones(4, dtype=int)), [0])
    assert isinstance(correlated_decoder, decoders.DetailedObservableDecoder)
    assert not correlated_decoder.decode_observables_detailed(np.ones(4, dtype=int)).diagnostics


def test_matching_builder_validation() -> None:
    """MWPM validates weights, decomposition, and non-graphlike errors."""
    dem = stim.DetectorErrorModel("error(0.1) D0 ^ D1 L0\nerror(0.2) D0")
    with pytest.raises(ValueError, match="Cannot set error weights"):
        _get_decoder_mwpm(dem, weights=[1.0, 1.0])
    with pytest.raises(ValueError, match="Cannot set error probabilities"):
        decoders.mwpm(error_probabilities=[0.1, 0.2]).build_observable_decoder(dem)

    decomposed = _get_decoder_mwpm(dem, decompose_errors=True)
    assert getattr(decomposed, "_infers_decomposed_errors", False)

    graphlike_dem = stim.DetectorErrorModel("error(0.1) D0 L0\nerror(0.1) D0 D1")
    decoder = _get_decoder_mwpm(graphlike_dem, decompose_errors=True)
    assert not getattr(decoder, "_infers_decomposed_errors", False)

    matrix = galois.GF(2)([[1, 1], [1, 0], [1, 0]])
    syndrome = np.array([1, 0, 0], dtype=int)
    with pytest.raises(ValueError, match="column 0 of the parity check matrix addresses 3"):
        _get_decoder_mwpm(matrix).decode(syndrome)
    decoder = _get_decoder_mwpm(matrix, ignore_non_graphlike_errors=True)
    assert np.array_equal(decoder.decode(syndrome), [0, 1])

    assert not decoders.mwpm().options["enable_correlations"]
    equivalent_default = cast(Literal["smallest-weight"], b"smallest-weight".decode())
    assert decoders.mwpm(
        enable_correlations=True, merge_strategy=equivalent_default
    ).predicts_observables_natively
    with pytest.raises(ValueError, match=r"decompose_errors=True.*enable_correlations=True"):
        decoders.mwpm(enable_correlations=True, decompose_errors=True)
    faults_matrix = {"faults_matrix": np.eye(2, dtype=int)}
    with pytest.raises(ValueError, match=r"faults_matrix is reserved.*build_observable_decoder"):
        decoders.mwpm(backend_options=faults_matrix)
    with pytest.raises(ValueError, match="faults_matrix is reserved"):
        _get_decoder_mwpm(np.eye(2, dtype=int), backend_options=faults_matrix)
    with pytest.raises(ValueError, match="not supported with enable_correlations=True"):
        decoders.mwpm(enable_correlations=True, backend_options={"backend_extension": 12})
    with pytest.raises(TypeError, match="unexpected keyword argument 'merge_stratgy'"):
        decoders.mwpm(merge_stratgy="disallow")  # type: ignore[call-arg]


def test_matching_backend_options() -> None:
    """Deferred and immediate builders forward backend_options in both modes."""
    matrix = np.eye(2, dtype=int)
    dem = stim.DetectorErrorModel("error(0.1) D0 L0")
    backend_options = {"backend_extension": 12}
    calls: list[dict[str, object]] = []

    def load_from_check_matrix(
        self: object, check_matrix: object = None, *, backend_extension: int = 0, **kwargs: object
    ) -> None:
        """Accept an option that a future PyMatching release might add."""
        calls.append({"backend_extension": backend_extension, **kwargs})

    with unittest.mock.patch.object(
        pymatching_package.Matching, "load_from_check_matrix", load_from_check_matrix
    ):
        _get_decoder_mwpm(matrix, backend_options=backend_options)
        decoders.mwpm(backend_options=backend_options).build(matrix)
        observable = decoders.mwpm(backend_options=backend_options).build_observable_decoder(dem)
    assert isinstance(observable, pymatching.MatchingObservableDecoder)
    assert [call["backend_extension"] for call in calls] == [12, 12, 12]
    assert calls[-1]["faults_matrix"] is not None


def test_matching_backend_option_validation() -> None:
    """PyMatching ignores unknown options, so names absent from its signature are rejected."""
    with pytest.raises(ValueError, match=r"Unsupported MWPM backend option\(s\) \['typo'\]"):
        decoders.mwpm(backend_options={"typo": 1}).build(np.eye(2, dtype=int))

    class FutureMatching:
        def load_from_check_matrix(
            self, check_matrix: object = None, *, new_option: int = 0, **kwargs: object
        ) -> None:
            """Accept an option that a future PyMatching release might add."""

    future_pymatching = types.ModuleType("pymatching")
    vars(future_pymatching)["Matching"] = FutureMatching
    pymatching._validate_backend_options(future_pymatching, {"new_option": 1})
    with pytest.raises(ValueError, match="kwargs"):
        pymatching._validate_backend_options(future_pymatching, {"kwargs": 1})


def test_matching_protocol_adapter() -> None:
    """The error-mode matching is a PyMatching decoder with qLDPC batch methods."""
    assert pymatching.__getattr__("Matching") is pymatching.Matching
    with pytest.raises(AttributeError, match="has no attribute"):
        pymatching.__getattr__("NotAMatchingDecoder")

    matrix = np.array([[1, 1, 0], [0, 1, 1]], dtype=int)
    syndromes = np.array([[1, 0], [1, 1]], dtype=int)
    decoder = _get_decoder_mwpm(matrix)
    assert isinstance(decoder, pymatching.Matching)
    assert isinstance(decoder, pymatching_package.Matching)
    assert isinstance(decoder, decoders.BatchErrorDecoder)
    assert np.array_equal(decoder.decode_errors_batch(syndromes), decoder.decode_batch(syndromes))
    assert isinstance(decoder, decoders.BatchDetailedErrorDecoder)
    detailed = decoder.decode_errors_detailed_batch(syndromes)
    assert np.array_equal([result.error for result in detailed], decoder.decode_batch(syndromes))
    assert [result.diagnostics["pymatching.objective_weight"] for result in detailed] == [1.0, 1.0]
    single = decoder.decode_errors_detailed(syndromes[0])
    assert np.array_equal(single.error, decoder.decode(syndromes[0]))
    assert decoder.decode_errors_detailed_batch(syndromes[:0]) == ()
    assert pickle.loads(pickle.dumps(pymatching.Matching)) is pymatching.Matching  # noqa: S301


def test_pymatching_import_is_lazy() -> None:
    """Importing the integration does not import pymatching until a matching is requested."""
    code = """
import sys
import types
import qldpc.decoders.external.pymatching
assert "pymatching" not in sys.modules
"""
    run_python(code)


def test_pymatching_protocol_adapter_is_created_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Concurrent first uses share one adapter class, as pickling requires."""
    # create the original classes, which monkeypatch restores after the test
    pymatching._get_matching_type()
    num_threads = 8

    def get_adapter_type(barrier: threading.Barrier) -> type[Any]:
        barrier.wait()
        return pymatching._get_matching_type()

    switch_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)  # switch threads often, so that an unguarded race is likely
    try:
        for _ in range(5):
            monkeypatch.setattr(pymatching, "_MATCHING_TYPE", None)
            monkeypatch.delattr(pymatching, "Matching")
            barriers = [threading.Barrier(num_threads)] * num_threads
            with concurrent.futures.ThreadPoolExecutor(num_threads) as executor:
                adapter_types = set(executor.map(get_adapter_type, barriers))
            assert adapter_types == {pymatching.Matching}
    finally:
        sys.setswitchinterval(switch_interval)


def test_pymatching_missing_dependency_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing required dependency identifies a broken qLDPC installation."""
    monkeypatch.setitem(sys.modules, "pymatching", None)
    with pytest.raises(ModuleNotFoundError, match="required qLDPC dependency"):
        pymatching._get_observable_decoder_mwpm(
            stim.DetectorErrorModel("error(0.1) D0 L0"),
            enable_correlations=True,
        )


def test_pymatching_transitive_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """An import failure inside PyMatching is preserved instead of mislabeled as a missing extra."""

    def import_with_missing_dependency(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        del globals, locals, fromlist, level
        assert name == "pymatching"
        raise ModuleNotFoundError("No module named 'backend_dependency'", name="backend_dependency")

    monkeypatch.delitem(sys.modules, "pymatching")
    monkeypatch.setattr(builtins, "__import__", import_with_missing_dependency)
    with pytest.raises(ModuleNotFoundError, match="backend_dependency"):
        pymatching._get_pymatching()
