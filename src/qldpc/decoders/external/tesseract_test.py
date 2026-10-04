# SPDX-License-Identifier: Apache-2.0

"""Tests for the optional Tesseract decoder adapter."""

from __future__ import annotations

import builtins
import importlib
import importlib.util
import inspect
import subprocess
import sys
import types
from typing import Any, cast

import galois
import numpy as np
import numpy.typing as npt
import pytest
import scipy.sparse
import stim

from qldpc import codes, decoders
from qldpc.codes import code_capacity
from qldpc.decoders.construction.resolution import _get_error_decoder, _get_observable_decoder
from qldpc.decoders.external.tesseract import _get_decoder_tesseract


class _FakeTesseractConfig:
    """API-faithful stand-in for upstream TesseractConfig."""

    dem: stim.DetectorErrorModel
    merge_errors: bool
    pqlimit: int

    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)

    def compile_decoder(self) -> _FakeTesseractDecoder:
        return _FakeTesseractDecoder(self)


class _FakeTesseractDecoder:
    """Small exact decoder with Tesseract's Python method names and result state."""

    def __init__(self, config: _FakeTesseractConfig) -> None:
        self.config = config
        self.dem_arrays = decoders.DetectorErrorModelArrays(config.dem, simplify=False)
        self.lookup = decoders.custom.LookupDecoder(config.dem, max_weight=config.dem.num_errors)
        self.low_confidence_flag = False
        self.predicted_errors_buffer: list[int] = []

    def decode_to_errors(self, syndrome: npt.NDArray[np.bool_]) -> list[int]:
        syndrome = np.asarray(syndrome, dtype=np.uint8)
        self.predicted_errors_buffer = []
        self.low_confidence_flag = False
        if not np.any(syndrome):
            return self.predicted_errors_buffer
        if self.config.pqlimit == 0:
            self.low_confidence_flag = True
            return self.predicted_errors_buffer

        error = np.asarray(self.lookup.decode(syndrome.astype(int)), dtype=np.uint8)
        self.predicted_errors_buffer = np.flatnonzero(error).tolist()
        reproduced = np.asarray(self.dem_arrays.detector_flip_matrix @ error).ravel() % 2
        self.low_confidence_flag = not np.array_equal(reproduced, syndrome)
        return self.predicted_errors_buffer

    def decode(self, syndrome: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
        indices = self.decode_to_errors(syndrome)
        error = np.zeros(self.dem_arrays.num_errors, dtype=np.uint8)
        np.bitwise_xor.at(error, indices, 1)
        flips = np.asarray(self.dem_arrays.observable_flip_matrix @ error).ravel() % 2
        return flips.astype(bool)

    def decode_batch(self, syndromes: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
        return np.asarray([self.decode(syndrome) for syndrome in syndromes], dtype=bool)


@pytest.fixture
def fake_tesseract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a controlled stand-in for the optional upstream package."""
    monkeypatch.setitem(
        sys.modules,
        "tesseract_decoder",
        types.SimpleNamespace(
            tesseract=types.SimpleNamespace(TesseractConfig=_FakeTesseractConfig),
            utils=types.SimpleNamespace(
                DetectorOrderMethod=types.SimpleNamespace(
                    BFS="BFS",
                    Coordinate="Coordinate",
                    Index="Index",
                )
            ),
        ),
    )


def test_tesseract_matrix_error_decoding(fake_tesseract: None) -> None:
    """Dense, sparse, and field matrices decode to errors in column order."""
    matrix = np.array([[1, 1, 0], [0, 1, 1]], dtype=int)
    syndromes = np.array([[1, 0], [0, 1]], dtype=int)
    expected = np.array([[1, 0, 0], [0, 0, 1]], dtype=np.uint8)

    matrix_inputs = [
        matrix,
        scipy.sparse.csc_matrix(matrix),
        scipy.sparse.dok_matrix(matrix),
        scipy.sparse.lil_matrix(matrix),
        galois.GF2(matrix),
    ]
    for matrix_input in matrix_inputs:
        decoder = _get_decoder_tesseract(
            matrix_input,
            error_channel=np.array([0.1, 0.2, 0.3]),
        )
        assert np.array_equal(decoder.decode_errors(syndromes[0]), expected[0])

    assert isinstance(decoder, decoders.ErrorDecoder)
    assert not decoders.supports_batch_decoding(decoder)
    assert isinstance(decoder, decoders.BatchObservableDecoder)
    assert np.array_equal(decoder.decode(syndromes[0]), expected[0])
    assert np.array_equal(decoders.batch_decode_errors(decoder, syndromes), expected)
    assert decoders.batch_decode_errors(decoder, syndromes[:0]).shape == (0, 3)
    assert np.array_equal(decoder.decoder.dem_arrays.error_probs, [0.1, 0.2, 0.3])

    with pytest.warns(DeprecationWarning, match="error_rate=0.25.*error_channel=0.25"):
        decoder = _get_decoder_tesseract(matrix, error_rate=0.25)
    assert np.array_equal(decoder.decoder.dem_arrays.error_probs, [0.25, 0.25, 0.25])
    decoder = _get_decoder_tesseract(matrix, error_channel=0.4)
    assert np.array_equal(decoder.decoder.dem_arrays.error_probs, [0.4, 0.4, 0.4])

    # Merging equal columns maps their aggregate probability to one representative original index,
    # which can be a less likely physical correction. qLDPC preserves column identity by default.
    decoder = _get_decoder_tesseract(
        np.array([[1, 1]], dtype=int),
        error_channel=[0.1, 0.4],
    )
    assert not decoder.config.merge_errors
    assert np.array_equal(decoder.decode_errors(np.array([1], dtype=int)), [0, 1])
    assert _get_decoder_tesseract(matrix, merge_errors=True).config.merge_errors


def test_tesseract_dem_error_and_observable_decoding(
    fake_tesseract: None,
) -> None:
    """A DEM supports original-index errors and native observable predictions."""
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 L0
        error(0) D1 L1
        error(0.2) D1 L1
    """)
    decoder = _get_decoder_tesseract(dem)
    syndromes = np.array([[1, 0], [0, 1]], dtype=int)

    # mechanisms of a DEM are merged by default, since merged mechanisms are interchangeable
    assert decoder.config.merge_errors
    assert not _get_decoder_tesseract(dem, merge_errors=False).config.merge_errors
    assert decoder.num_errors == 3
    assert np.array_equal(decoder.decode_errors(syndromes[1]), [0, 0, 1])
    assert np.array_equal(decoder.decode_observables(syndromes[0]), [1, 0])
    assert np.array_equal(decoder.decode_observables(syndromes[1]), [0, 1])
    assert np.array_equal(decoder.decode_observables_batch(syndromes), syndromes)
    assert decoder.decode_observables_batch(syndromes[:0]).shape == (0, 2)


def test_tesseract_erasure_bits(fake_tesseract: None) -> None:
    """Tesseract low-confidence results become optional qLDPC erasure flags."""
    dem = stim.DetectorErrorModel("error(0.1) D0 L0")
    decoder = _get_decoder_tesseract(dem, pqlimit=0, add_erasure_bit=True)
    syndromes = np.array([[1], [0]], dtype=int)

    assert np.array_equal(decoder.decode_errors(syndromes[0]), [0, 1])
    assert np.array_equal(decoder.decode_errors(syndromes[1]), [0, 0])
    assert np.array_equal(decoders.batch_decode_errors(decoder, syndromes), [[0, 1], [0, 0]])
    assert np.array_equal(decoder.decode_observables(syndromes[0]), [0, 1])
    assert np.array_equal(decoder.decode_observables_batch(syndromes), [[0, 1], [0, 0]])
    assert decoders.batch_decode_errors(decoder, syndromes[:0]).shape == (0, 2)
    assert decoder.decode_observables_batch(syndromes[:0]).shape == (0, 2)

    # detailed results flag low-confidence erasure whether or not erasure bits are requested
    for add_erasure_bit in [False, True]:
        decoder = _get_decoder_tesseract(dem, pqlimit=0, add_erasure_bit=add_erasure_bit)
        detailed_error = decoder.decode_errors_detailed(syndromes[0])
        detailed_observables = decoder.decode_observables_detailed(syndromes[0])
        assert np.array_equal(detailed_error.error, decoder.decode_errors(syndromes[0]))
        assert np.array_equal(
            detailed_observables.observable_flips, decoder.decode_observables(syndromes[0])
        )
        assert detailed_error.erasure and detailed_observables.erasure


def test_tesseract_options_and_validation(fake_tesseract: None) -> None:
    """Construction forwards typed options and rejects unsupported inputs."""
    assert (
        list(inspect.signature(decoders.tesseract).parameters)
        == (list(inspect.signature(decoders.external.TesseractDecoder).parameters)[1:])
    )

    options: dict[str, Any] = {
        "det_beam": 8,
        "beam_climbing": True,
        "no_revisit_dets": False,
        "verbose": True,
        "merge_errors": False,
        "pqlimit": 123,
        "det_orders": [[1, 0]],
        "det_penalty": 0.5,
        "create_visualization": True,
        "sparsify_errors": True,
        "sparsify_base_degree": 2,
        "sparsify_max_degree": 4,
        "sparsify_reactivate_limit": 7,
    }
    config = _get_decoder_tesseract(np.eye(2, dtype=int), **options).config
    assert {name: getattr(config, name) for name in options} == options

    decoder = _get_decoder_tesseract(
        np.eye(2, dtype=int),
        num_det_orders=3,
        det_order_method="bfs",
        seed=11,
    )
    assert decoder.config.num_det_orders == 3
    assert decoder.config.det_order_method == "BFS"
    assert decoder.config.seed == 11

    with pytest.raises(ValueError, match="Unknown Tesseract det_order_method"):
        _get_decoder_tesseract(np.eye(1, dtype=int), det_order_method=cast(Any, "random"))
    with pytest.raises(ValueError, match="only supports binary"):
        _get_decoder_tesseract(galois.GF(3)(np.eye(2, dtype=int)))
    with pytest.raises(ValueError, match="only 0 and 1"):
        _get_decoder_tesseract(np.array([[0, 2]], dtype=int))
    with pytest.raises(ValueError, match="binary integers"):
        _get_decoder_tesseract(np.array([[0.0, 1.0]]))
    with pytest.raises(ValueError, match="two-dimensional"):
        _get_decoder_tesseract(np.array([0, 1], dtype=int))
    with pytest.raises(ValueError, match="error probabilities of shape"):
        _get_decoder_tesseract(np.eye(2, dtype=int), error_channel=[0.1])
    with (
        pytest.warns(DeprecationWarning, match="error_rate=nan.*error_channel=nan"),
        pytest.raises(ValueError, match="finite and between 0 and 1"),
    ):
        _get_decoder_tesseract(np.eye(1, dtype=int), error_rate=np.nan)
    with pytest.raises(ValueError, match=r"supplies its own.*error_channel=\[0.2\] cannot"):
        _get_decoder_tesseract(
            stim.DetectorErrorModel("error(0.1) D0"),
            error_channel=[0.2],
        )
    with pytest.raises(ValueError, match=r"supplies its own.*error_rate=0.2 cannot"):
        _get_decoder_tesseract(
            stim.DetectorErrorModel("error(0.1) D0"),
            error_rate=0.2,
        )
    with pytest.warns(DeprecationWarning, match="error_rate=0.2.*error_channel=0.2"):
        decoder = _get_decoder_tesseract(np.eye(1, dtype=int), error_rate=0.2)
    assert np.array_equal(decoder.decoder.dem_arrays.error_probs, [0.2])

    decoder = _get_decoder_tesseract(np.eye(2, dtype=int))
    with pytest.raises(ValueError, match=r"shape \(2,\)"):
        decoder.decode_errors(np.array([1], dtype=int))
    with pytest.raises(ValueError, match=r"shape \(num_shots, 2\)"):
        decoder.decode_observables_batch(np.array([1, 0], dtype=int))


def test_tesseract_rejects_invalid_backend_error_index(
    fake_tesseract: None,
) -> None:
    """An invalid upstream result cannot silently corrupt a dense inferred error."""
    decoder = _get_decoder_tesseract(np.eye(1, dtype=int))
    decoder.decoder.decode_to_errors = lambda syndrome: [1]
    with pytest.raises(ValueError, match="outside the provided"):
        decoder.decode_errors(np.array([1], dtype=int))


def test_tesseract_preset_merge_defaults(fake_tesseract: None) -> None:
    """Preset helpers validate names, probabilities, and input-dependent merge defaults."""
    spec = decoders.tesseract_preset()
    assert not spec.build(np.eye(2, dtype=int)).config.merge_errors
    assert spec.build(stim.DetectorErrorModel("error(0.1) D0")).config.merge_errors

    with pytest.warns(DeprecationWarning, match="error_rate=0.2.*error_channel=0.2") as warnings:
        deprecated_spec = decoders.tesseract_preset(error_rate=0.2)
    assert warnings[0].filename == __file__
    assert deprecated_spec.options["error_channel"] == 0.2
    with pytest.raises(ValueError, match="cannot both be specified"):
        decoders.tesseract_preset(error_channel=0.1, error_rate=0.2)
    with pytest.raises(ValueError, match="Unknown Tesseract preset"):
        decoders.tesseract_preset(cast(Any, "medium-beam"))
    with pytest.raises(ValueError, match="Unknown Tesseract sparsify preset"):
        decoders.tesseract_preset(sparsify=cast(Any, "generic"))


def test_tesseract_specs_sinter_and_code_capacity(
    fake_tesseract: None,
) -> None:
    """Decoder specifications predict observables natively in all workflows."""
    dem = stim.DetectorErrorModel("error(0.1) D0 L0")
    spec = decoders.tesseract(det_beam=7)
    assert spec.predicts_observables_natively
    assert isinstance(_get_error_decoder(dem, decoder=spec), decoders.external.TesseractDecoder)
    observable_decoder = _get_observable_decoder(dem, decoder=spec)
    assert isinstance(observable_decoder, decoders.external.TesseractDecoder)
    assert np.array_equal(observable_decoder.decode_observables(np.array([1])), [1])

    compiled = decoders.SinterDecoder(decoder=spec).compile_decoder_for_dem(dem)
    assert isinstance(compiled.observable_decoder, decoders.external.TesseractDecoder)
    assert np.array_equal(compiled.decode_shots(np.array([[1]], dtype=np.uint8)), [[1]])

    code = codes.RepetitionCode(3)
    observable_matrix = code.field([[1, 0, 0]])
    native_capacity_decoder = code_capacity.get_code_capacity_decoder(
        code.matrix, observable_matrix, spec
    )
    assert isinstance(native_capacity_decoder.decoder, decoders.external.TesseractDecoder)
    capacity_decoder = code_capacity.get_code_capacity_decoder(
        code.matrix,
        observable_matrix,
        decoders.SinterDecoder(decoder=decoders.tesseract()),
    )
    for bit in range(3):
        error = code.field.Zeros(3)
        error[bit] = 1
        assert native_capacity_decoder.get_failure_and_erasure(error) == (False, False)
        assert capacity_decoder.get_failure_and_erasure(error) == (False, False)


_REAL_PACKAGE_CHECKS = """
import numpy as np
import stim

from qldpc import decoders
from qldpc.decoders.construction.resolution import _get_error_decoder, _get_observable_decoder
from qldpc.decoders.external.tesseract import _get_decoder_tesseract

# A matrix keeps the most likely of two identical columns.
decoder = _get_decoder_tesseract(np.array([[1, 1]]), error_channel=[0.1, 0.4])
assert np.array_equal(decoder.decode_errors(np.array([1])), [0, 1])

# A DEM merges equivalent mechanisms, so their combined probability selects the logical class.
dem = stim.DetectorErrorModel("error(0.3) D0 L0\\nerror(0.3) D0 L0\\nerror(0.4) D0")
decoder = _get_decoder_tesseract(dem)
syndromes = np.array([[1], [0]])
assert np.array_equal(decoder.decode_observables_batch(syndromes), [[1], [0]])
assert np.array_equal(decoders.batch_decode_errors(decoder, syndromes), [[1, 0, 0], [0, 0, 0]])

# A syndrome that no error explains is flagged, with generated detector orders.
dem = stim.DetectorErrorModel("detector D0\\ndetector D1\\nerror(0.1) D0 L0")
decoder = _get_decoder_tesseract(
    dem, add_erasure_bit=True, num_det_orders=2, det_order_method="bfs", seed=3
)
assert np.array_equal(decoder.decode_observables(np.array([0, 1])), [0, 1])
assert np.array_equal(decoder.decode_errors(np.array([1, 0])), [1, 0])

# Inferred circuit-level errors reproduce their syndromes, and Sinter compiles native decoding.
circuit = stim.Circuit.generated(
    "repetition_code:memory", distance=3, rounds=3, after_clifford_depolarization=0.02
)
dem = circuit.detector_error_model()
shots = circuit.compile_detector_sampler(seed=0).sample(50).astype(np.uint8)
errors = decoders.batch_decode_errors(_get_decoder_tesseract(dem), shots)
matrix = decoders.DetectorErrorModelArrays(dem, simplify=False).detector_flip_matrix
assert np.array_equal(matrix @ errors.T % 2, shots.T)
compiled = decoders.SinterDecoder(decoder=decoders.tesseract()).compile_decoder_for_dem(dem)
assert compiled.decode_shots(shots).shape == (50, dem.num_observables)
"""


def test_real_tesseract_package() -> None:
    """Exercise the published Tesseract package whenever it is installed.

    The checks run in a subprocess, which runs an empty script if the package is absent, so the
    statement coverage of this module does not depend on the optional installation.
    """
    installed = importlib.util.find_spec("tesseract_decoder") is not None
    script = _REAL_PACKAGE_CHECKS if installed else ""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_tesseract_import_is_lazy() -> None:
    """Importing qLDPC's integration does not import the optional backend."""
    code = """
import sys
import qldpc.decoders.external.tesseract
assert "tesseract_decoder" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_tesseract_missing_dependency_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing backend produces an actionable error without hiding nested failures."""
    monkeypatch.setitem(sys.modules, "tesseract_decoder", None)
    with pytest.raises(ModuleNotFoundError, match=r"qldpc\[tesseract\]") as exc_info:
        _get_decoder_tesseract(np.eye(1, dtype=int))
    message = str(exc_info.value)
    assert "CPython 3.12-3.14 on macOS arm64 and Linux x86-64" in message
    assert "github.com/quantumlib/tesseract-decoder#installation" in message

    monkeypatch.delitem(sys.modules, "tesseract_decoder")
    original_import = builtins.__import__

    def import_tesseract(
        name: str,
        globals_: dict[str, object] | None = None,
        locals_: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> types.ModuleType:
        if name == "tesseract_decoder":
            raise ModuleNotFoundError(
                "No module named 'nested_dependency'", name="nested_dependency"
            )
        return original_import(name, globals_, locals_, fromlist, level)

    assert import_tesseract("types") is types
    monkeypatch.setattr(builtins, "__import__", import_tesseract)
    with pytest.raises(ModuleNotFoundError, match="nested_dependency"):
        importlib.import_module("qldpc.decoders.external.tesseract")._get_tesseract()
