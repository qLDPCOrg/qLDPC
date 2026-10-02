# SPDX-License-Identifier: Apache-2.0

"""Tests for the optional Frontier observable decoder."""

from __future__ import annotations

import dataclasses
import itertools
import math
import subprocess
import sys
import types
from typing import Any

import numpy as np
import pytest
import stim

from qldpc import codes, decoders
from qldpc.decoders.external import frontier


@dataclasses.dataclass(frozen=True)
class _Column:
    family: str
    index: int
    label: str
    instruction_offset: int
    prior_probs: tuple[float, float]
    detector_response_masks: tuple[int, int]
    logical_response_masks: tuple[int, int]
    detector_support_mask: int
    original_column_index: int = -1


@dataclasses.dataclass(frozen=True)
class _Model:
    columns: tuple[_Column, ...]
    layout: tuple[_Column, ...]
    num_detectors: int
    num_observables: int = 1
    backward_columns: tuple[_Column, ...] | None = None
    backward_layout: tuple[_Column, ...] | None = None


@dataclasses.dataclass(frozen=True)
class _Result:
    status: str
    logical_hat: int | None


_Call = tuple[str, _Model, list[int], dict[str, object]]


def _decode_exactly(model: _Model, syndrome: np.ndarray, **options: object) -> _Result:
    """Return the most likely observable flips, as unpruned Frontier would."""
    target = sum(int(bit) << index for index, bit in enumerate(syndrome))
    class_probs: dict[int, float] = {}
    for outcomes in itertools.product((0, 1), repeat=len(model.columns)):
        detectors = logicals = 0
        prob = 1.0
        for column, outcome in zip(model.columns, outcomes):
            detectors ^= column.detector_response_masks[outcome]
            logicals ^= column.logical_response_masks[outcome]
            prob *= column.prior_probs[outcome]
        if detectors == target:
            class_probs[logicals] = class_probs.get(logicals, 0) + prob
    if not class_probs:
        return _Result("no_path", None)
    return _Result("ok", max(class_probs, key=lambda logicals: class_probs[logicals]))


def _optimize_column_order(
    columns: list[_Column], *, num_detectors: int
) -> tuple[list[_Column], tuple[int, ...]]:
    """Reorder columns, reindexing them as Frontier does."""
    ordering = tuple(reversed(range(len(columns))))
    return [
        dataclasses.replace(columns[old], index=new) for new, old in enumerate(ordering)
    ], ordering


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[_Call]:
    """Install a substitute for Frontier that decodes exactly, and record its decoding calls."""
    calls: list[_Call] = []

    def get_decode_func(name: str) -> Any:
        def decode(model: _Model, syndrome: np.ndarray, **options: object) -> _Result:
            calls.append((name, model, syndrome.tolist(), options))
            return _decode_exactly(model, syndrome)

        return decode

    progressive = types.ModuleType("frontier.progressive")
    progressive.__dict__.update(
        ProgressiveColumn=_Column,
        build_frontier_layout=lambda columns, *, num_detectors: tuple(columns),
        optimize_column_order=_optimize_column_order,
    )
    package = types.ModuleType("frontier")
    package.__dict__.update(
        FrontierModel=_Model,
        decode_frontier=get_decode_func("forward"),
        decode_frontier_committee=get_decode_func("committee"),
        progressive=progressive,
    )
    monkeypatch.setitem(sys.modules, "frontier", package)
    monkeypatch.setitem(sys.modules, "frontier.progressive", progressive)
    return calls


def test_frontier_import_is_lazy() -> None:
    """Importing the integration and configuring a decoder does not import Frontier."""
    code = """
import sys
from qldpc import decoders
decoders.frontier(committee=True)
assert "frontier" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_frontier_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid settings are rejected before Frontier is imported."""
    monkeypatch.setitem(sys.modules, "frontier", None)
    dem = stim.DetectorErrorModel("error(0.1) D0 L0")
    for options, message in [
        ({"K": 0}, "K must be positive"),
        ({"Delta": -1}, "Delta must be non-negative"),
        ({"Delta": math.nan}, "Delta must be non-negative"),
        ({"score_alpha": math.inf}, "score_alpha must be finite"),
        ({"metric_mode": "maxlog"}, "metric_mode must be one of"),
        ({"int_metric_scale": 0}, "int_metric_scale must be positive"),
        ({"column_order": "random"}, "column_order must be one of"),
    ]:
        with pytest.raises(ValueError, match=message):
            decoders.get_observable_decoder_frontier(dem, **options)  # type: ignore[arg-type]


def test_frontier_decoding(calls: list[_Call]) -> None:
    """Frontier receives every error mechanism of a model, and its predictions are converted."""
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 L0
        error(0.2) D0 D1
        error(0.3) D1 L1
        detector D2
    """)
    decoder = decoders.get_observable_decoder_frontier(
        dem, K=11, Delta=4.5, column_order="time_order", add_erasure_bit=True
    )
    assert isinstance(decoder, decoders.FrontierObservableDecoder)
    assert decoder.has_erasure_bit
    assert [
        (column.prior_probs, column.detector_response_masks, column.logical_response_masks)
        for column in decoder.model.columns
    ] == [
        ((0.9, 0.1), (0, 0b001), (0, 0b01)),
        ((0.8, 0.2), (0, 0b011), (0, 0b00)),
        ((0.7, 0.3), (0, 0b010), (0, 0b10)),
    ]

    # the last entry of a prediction is the erasure flag, which Frontier signals with "no_path"
    syndromes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1]])
    expected = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]]
    assert [decoder.decode_observables(syndrome).tolist() for syndrome in syndromes] == expected
    assert calls[0][0] == "forward"
    assert calls[0][3] == {
        "K": 11,
        "Delta": 4.5,
        "score_alpha": 0.8,
        "metric_mode": "logsumexp_float",
        "int_metric_scale": 1024,
    }

    with pytest.raises(ValueError, match="Expected a syndrome of shape"):
        decoder.decode_observables(np.zeros(2, dtype=int))


def test_frontier_scan_orders(calls: list[_Call]) -> None:
    """Frontier reorders columns on request, and the committee's reverse scan is precomputed."""
    dem = stim.DetectorErrorModel("error(0.1) D0\nerror(0.2) D0 D1\nerror(0.3) D1")
    labels = ["error_0", "error_1", "error_2"]

    model = decoders.get_observable_decoder_frontier(dem, column_order="time_order").model
    assert [column.label for column in model.columns] == labels
    assert model.backward_columns is None

    # with the substitute for Frontier, deadline reordering reverses each scan
    decoder = decoders.get_observable_decoder_frontier(dem, committee=True)
    assert [column.label for column in decoder.model.columns] == labels[::-1]
    assert [column.index for column in decoder.model.backward_columns] == [0, 1, 2]
    assert [column.label for column in decoder.model.backward_columns] == labels[::-1]
    assert decoder.model.backward_layout == decoder.model.backward_columns
    assert decoder.decode_observables(np.array([1, 0])).tolist() == []
    assert calls[-1][0] == "committee"


def test_frontier_degenerate_models(calls: list[_Call]) -> None:
    """Models without detectors or error mechanisms are decoded with a nonempty Frontier model."""
    decoder = decoders.get_observable_decoder_frontier(stim.DetectorErrorModel("error(0.9) L0"))
    assert decoder.decode_observables(np.zeros(0, dtype=int)).tolist() == [1]
    assert calls[-1][1].num_detectors == 1
    assert calls[-1][2] == [0]

    decoder = decoders.get_observable_decoder_frontier(
        stim.DetectorErrorModel(""), add_erasure_bit=True
    )
    assert decoder.model.columns == ()
    assert decoder.decode_observables(np.zeros(0, dtype=int)).tolist() == [0]


def test_frontier_with_generic_decoding_apis(calls: list[_Call]) -> None:
    """Frontier settings are accepted wherever an observable decoder can be compiled."""
    dem = stim.DetectorErrorModel("error(0.1) D0 L0\nerror(0.2) D0 D1\nerror(0.3) D1 L1")
    settings = decoders.frontier(add_erasure_bit=True)
    syndrome = np.array([1, 0])

    assert decoders.decode_observables(dem, syndrome, decoder=settings).tolist() == [1, 0, 0]

    sinter_decoder = decoders.SinterDecoder(decoder=settings).compile_decoder_for_dem(dem)
    assert sinter_decoder.num_erasure_bits == 1
    shots = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    assert sinter_decoder.decode_shots(shots).tolist() == [[1, 0, 0], [0, 1, 0]]
    assert sinter_decoder.decode_shots(np.zeros((0, 2), dtype=np.uint8)).shape == (0, 3)

    code = codes.RepetitionCode(3)
    num_calls = len(calls)
    logical_error_rate, _ = code.get_logical_error_rate_func(
        num_samples=10, max_error_rate=0.1, decoder=settings
    )(0.1)
    assert 0 <= logical_error_rate <= 1
    assert len(calls) > num_calls


def test_frontier_unexpected_status() -> None:
    """Malformed Frontier results are reported instead of being decoded."""
    model = _Model(columns=(), layout=(), num_detectors=1, num_observables=1)
    decoder = frontier.FrontierObservableDecoder(
        model, lambda *args, **kwargs: _Result("timeout", None), num_detectors=1, decode_options={}
    )
    with pytest.raises(RuntimeError, match="unexpected status: 'timeout'"):
        decoder.decode_observables(np.zeros(1, dtype=int))

    decoder.decode_func = lambda *args, **kwargs: _Result("ok", None)
    with pytest.raises(ValueError, match="status='ok' without logical_hat"):
        decoder.decode_observables(np.zeros(1, dtype=int))


def test_frontier_missing_installation(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing Frontier package raises an error that shows how to install it."""
    dem = stim.DetectorErrorModel("error(0.1) D0 L0")
    monkeypatch.setitem(sys.modules, "frontier", None)
    with pytest.raises(
        ModuleNotFoundError, match=r"Install it with `pip install 'frontier @ git\+"
    ):
        decoders.get_observable_decoder_frontier(dem)

    # an import error within an installed Frontier package is not mistaken for a missing package
    monkeypatch.setitem(sys.modules, "frontier", types.ModuleType("frontier"))
    monkeypatch.setitem(sys.modules, "frontier.progressive", None)
    with pytest.raises(ModuleNotFoundError) as error:
        decoders.get_observable_decoder_frontier(dem)
    assert "Install it with" not in str(error.value)
