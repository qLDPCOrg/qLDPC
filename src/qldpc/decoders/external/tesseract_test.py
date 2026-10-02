# SPDX-License-Identifier: Apache-2.0

"""Tests for the optional Tesseract decoder adapter."""

from __future__ import annotations

import builtins
import importlib.util
import inspect
import itertools
import math
import pathlib
import subprocess
import sys
import tomllib
import types
from collections.abc import Iterator
from typing import Any, cast

import galois
import numpy as np
import numpy.typing as npt
import packaging.requirements
import pytest
import scipy.sparse
import stim

from qldpc import codes, decoders
from qldpc.codes import code_capacity
from qldpc.decoders.external import tesseract


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

        matrix = self.dem_arrays.detector_flip_matrix
        probabilities = self.dem_arrays.error_probs
        best_cost = math.inf
        for value in range(1 << self.dem_arrays.num_errors):
            error = np.array(
                [(value >> index) & 1 for index in range(self.dem_arrays.num_errors)],
                dtype=np.uint8,
            )
            if not np.array_equal(np.asarray(matrix @ error).ravel() % 2, syndrome):
                continue
            cost = 0.0
            for probability in probabilities[error.astype(bool)]:
                if probability == 0:
                    cost = math.inf
                    break
                cost += math.log((1 - probability) / probability)
            if cost < best_cost:
                best_cost = cost
                self.predicted_errors_buffer = np.flatnonzero(error).tolist()

        self.low_confidence_flag = best_cost == math.inf
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
def fake_tesseract(monkeypatch: pytest.MonkeyPatch) -> Iterator[types.SimpleNamespace]:
    """Install a controlled stand-in for the optional upstream package."""
    backend = types.SimpleNamespace(
        tesseract=types.SimpleNamespace(TesseractConfig=_FakeTesseractConfig),
        utils=types.SimpleNamespace(
            DetectorOrderMethod=types.SimpleNamespace(
                BFS="BFS",
                Coordinate="Coordinate",
                Index="Index",
            )
        ),
    )
    monkeypatch.setitem(sys.modules, "tesseract_decoder", backend)
    yield backend


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
    with pytest.raises(ModuleNotFoundError, match=r"qldpc\[tesseract\]"):
        decoders.get_decoder_tesseract(np.eye(1, dtype=int))

    # the error explains whether the extra can install the package on this platform
    message = tesseract._get_missing_tesseract_message("CPython", (3, 13), "linux", "x86_64")
    assert "Install it with `pip install 'qldpc[tesseract]'`" in message
    assert "glibc" in message
    message = tesseract._get_missing_tesseract_message("CPython", (3, 11), "win32", "AMD64")
    assert "cannot install it for CPython 3.11 on win32 AMD64" in message
    assert "github.com/quantumlib/tesseract-decoder" in message

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
        tesseract._get_tesseract()


def test_tesseract_extra_covers_supported_platforms() -> None:
    """The tesseract extra installs the package everywhere that it publishes wheels.

    The extra may also attempt an installation that fails elsewhere, but must not skip a supported
    platform, and must skip Python versions and platforms that qLDPC's development installs use.
    """
    with open(pathlib.Path(__file__).parents[4] / "pyproject.toml", "rb") as file:
        requirements = tomllib.load(file)["project"]["optional-dependencies"]["tesseract"]
    (requirement,) = (packaging.requirements.Requirement(line) for line in requirements)
    marker = requirement.marker
    assert requirement.name == "tesseract-decoder" and marker is not None

    def is_installed(implementation: str, version: str, system: str, machine: str) -> bool:
        return marker.evaluate(
            {
                "platform_python_implementation": implementation,
                "python_version": version,
                "python_full_version": f"{version}.0",
                "sys_platform": system,
                "platform_machine": machine,
            }
        )

    for implementation, minor, system, machine in itertools.product(
        ["CPython", "PyPy"],
        range(11, 16),
        ["darwin", "linux", "win32"],
        ["arm64", "aarch64", "x86_64", "AMD64"],
    ):
        if tesseract._is_supported_platform(implementation, (3, minor), system, machine):
            assert is_installed(implementation, f"3.{minor}", system, machine), (system, machine)

    assert not is_installed("CPython", "3.11", "linux", "x86_64")
    assert not is_installed("CPython", "3.14", "win32", "AMD64")
    assert not is_installed("CPython", "3.14", "linux", "aarch64")


def test_tesseract_matrix_error_decoding(fake_tesseract: types.SimpleNamespace) -> None:
    """Dense, sparse, and field matrices decode to errors in column order."""
    del fake_tesseract
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
        decoder = decoders.get_decoder_tesseract(
            matrix_input,
            error_channel=np.array([0.1, 0.2, 0.3]),
        )
        assert isinstance(decoder, decoders.BatchErrorDecoder)
        assert isinstance(decoder, decoders.BatchObservableDecoder)
        assert np.array_equal(decoder.decode_errors(syndromes[0]), expected[0])
        assert np.array_equal(decoder.decode(syndromes[0]), expected[0])
        assert np.array_equal(decoder.decode_errors_batch(syndromes), expected)
        assert np.array_equal(decoder.decode_batch(syndromes), expected)
        assert decoder.decode_errors_batch(syndromes[:0]).shape == (0, 3)
        assert np.array_equal(decoder.decoder.dem_arrays.error_probs, [0.1, 0.2, 0.3])

    decoder = decoders.get_decoder_tesseract(matrix, error_rate=0.25)
    assert np.array_equal(decoder.decoder.dem_arrays.error_probs, [0.25, 0.25, 0.25])

    # Merging equal columns maps their aggregate probability to one representative original index,
    # which can be a less likely physical correction. qLDPC preserves column identity by default.
    decoder = decoders.get_decoder_tesseract(
        np.array([[1, 1]], dtype=int),
        error_channel=[0.1, 0.4],
    )
    assert not decoder.config.merge_errors
    assert np.array_equal(decoder.decode_errors(np.array([1], dtype=int)), [0, 1])
    assert decoders.get_decoder_tesseract(matrix, merge_errors=True).config.merge_errors


def test_tesseract_dem_error_and_observable_decoding(
    fake_tesseract: types.SimpleNamespace,
) -> None:
    """A DEM supports original-index errors and native observable predictions."""
    del fake_tesseract
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 L0
        error(0) D1 L1
        error(0.2) D1 L1
    """)
    decoder = decoders.get_decoder_tesseract(dem)
    syndromes = np.array([[1, 0], [0, 1]], dtype=int)

    # mechanisms of a DEM are merged by default, since merged mechanisms are interchangeable
    assert decoder.config.merge_errors
    assert not decoders.get_decoder_tesseract(dem, merge_errors=False).config.merge_errors
    assert decoder.num_errors == 3
    assert np.array_equal(decoder.decode_errors(syndromes[1]), [0, 0, 1])
    assert np.array_equal(decoder.decode_observables(syndromes[0]), [1, 0])
    assert np.array_equal(decoder.decode_observables(syndromes[1]), [0, 1])
    assert np.array_equal(decoder.decode_observables_batch(syndromes), syndromes)
    assert decoder.decode_observables_batch(syndromes[:0]).shape == (0, 2)


def test_tesseract_erasure_bits(fake_tesseract: types.SimpleNamespace) -> None:
    """Tesseract low-confidence results become optional qLDPC erasure flags."""
    del fake_tesseract
    dem = stim.DetectorErrorModel("error(0.1) D0 L0")
    decoder = decoders.get_decoder_tesseract(dem, pqlimit=0, add_erasure_bit=True)
    syndromes = np.array([[1], [0]], dtype=int)

    assert np.array_equal(decoder.decode_errors(syndromes[0]), [0, 1])
    assert np.array_equal(decoder.decode_errors(syndromes[1]), [0, 0])
    assert np.array_equal(decoder.decode_errors_batch(syndromes), [[0, 1], [0, 0]])
    assert np.array_equal(decoder.decode_observables(syndromes[0]), [0, 1])
    assert np.array_equal(decoder.decode_observables_batch(syndromes), [[0, 1], [0, 0]])
    assert decoder.decode_errors_batch(syndromes[:0]).shape == (0, 2)
    assert decoder.decode_observables_batch(syndromes[:0]).shape == (0, 2)


def test_tesseract_options_and_validation(fake_tesseract: types.SimpleNamespace) -> None:
    """Construction forwards typed options and rejects unsupported inputs."""
    del fake_tesseract
    assert inspect.signature(decoders.get_decoder_tesseract).parameters == (
        inspect.signature(decoders.TesseractDecoder).parameters
    )

    decoder = decoders.get_decoder_tesseract(
        np.eye(2, dtype=int),
        det_beam=8,
        beam_climbing=True,
        no_revisit_dets=False,
        verbose=True,
        merge_errors=False,
        pqlimit=123,
        det_orders=[[1, 0]],
        det_penalty=0.5,
        create_visualization=True,
        sparsify_errors=True,
        sparsify_base_degree=2,
        sparsify_max_degree=4,
        sparsify_reactivate_limit=7,
    )
    config = decoder.config
    assert config.det_beam == 8
    assert config.beam_climbing
    assert not config.no_revisit_dets
    assert config.verbose
    assert not config.merge_errors
    assert config.pqlimit == 123
    assert config.det_orders == [[1, 0]]
    assert config.det_penalty == 0.5
    assert config.create_visualization
    assert config.sparsify_errors
    assert config.sparsify_base_degree == 2
    assert config.sparsify_max_degree == 4
    assert config.sparsify_reactivate_limit == 7

    decoder = decoders.get_decoder_tesseract(
        np.eye(2, dtype=int),
        num_det_orders=3,
        det_order_method="bfs",
        seed=11,
    )
    assert decoder.config.num_det_orders == 3
    assert decoder.config.det_order_method == "BFS"
    assert decoder.config.seed == 11

    with pytest.raises(ValueError, match="Unknown Tesseract det_order_method"):
        decoders.get_decoder_tesseract(np.eye(1, dtype=int), det_order_method=cast(Any, "random"))
    with pytest.raises(ValueError, match="only supports binary"):
        decoders.get_decoder_tesseract(galois.GF(3)(np.eye(2, dtype=int)))
    with pytest.raises(ValueError, match="only 0 and 1"):
        decoders.get_decoder_tesseract(np.array([[0, 2]], dtype=int))
    with pytest.raises(ValueError, match="binary integers"):
        decoders.get_decoder_tesseract(np.array([[0.0, 1.0]]))
    with pytest.raises(ValueError, match="two-dimensional"):
        decoders.get_decoder_tesseract(np.array([0, 1], dtype=int))
    with pytest.raises(ValueError, match="error probabilities of shape"):
        decoders.get_decoder_tesseract(np.eye(2, dtype=int), error_channel=[0.1])
    with pytest.raises(ValueError, match="finite and between 0 and 1"):
        decoders.get_decoder_tesseract(np.eye(1, dtype=int), error_rate=np.nan)
    with pytest.raises(ValueError, match="Cannot specify an error_channel"):
        decoders.get_decoder_tesseract(
            stim.DetectorErrorModel("error(0.1) D0"),
            error_channel=[0.2],
        )

    decoder = decoders.get_decoder_tesseract(np.eye(2, dtype=int))
    with pytest.raises(ValueError, match=r"shape \(2,\)"):
        decoder.decode_errors(np.array([1], dtype=int))
    with pytest.raises(ValueError, match=r"shape \(num_shots, 2\)"):
        decoder.decode_errors_batch(np.array([1, 0], dtype=int))


def test_tesseract_rejects_invalid_backend_error_index(
    fake_tesseract: types.SimpleNamespace,
) -> None:
    """An invalid upstream result cannot silently corrupt a dense inferred error."""
    del fake_tesseract
    decoder = decoders.get_decoder_tesseract(np.eye(1, dtype=int))
    decoder.decoder.decode_to_errors = lambda syndrome: [1]
    with pytest.raises(ValueError, match="outside the provided"):
        decoder.decode_errors(np.array([1], dtype=int))


def test_tesseract_specs_sinter_and_code_capacity(
    fake_tesseract: types.SimpleNamespace,
) -> None:
    """Typed settings use native observables through resolution, Sinter, and code capacity."""
    del fake_tesseract
    dem = stim.DetectorErrorModel("error(0.1) D0 L0")
    spec = decoders.tesseract(det_beam=7)
    assert spec.predicts_observables_natively
    assert isinstance(decoders.get_error_decoder(dem, decoder=spec), decoders.TesseractDecoder)
    observable_decoder = decoders.get_observable_decoder(dem, decoder=spec)
    assert isinstance(observable_decoder, decoders.TesseractDecoder)
    assert np.array_equal(observable_decoder.decode_observables(np.array([1])), [1])

    compiled = decoders.SinterDecoder(decoder=spec).compile_decoder_for_dem(dem)
    assert isinstance(compiled.observable_decoder, decoders.TesseractDecoder)
    assert np.array_equal(compiled.decode_shots(np.array([[1]], dtype=np.uint8)), [[1]])

    code = codes.RepetitionCode(3)
    observable_matrix = code.field([[1, 0, 0]])
    capacity_decoder = code_capacity.get_code_capacity_decoder(
        code.matrix,
        observable_matrix,
        decoders.SinterDecoder(decoder=decoders.tesseract()),
    )
    for bit in range(3):
        error = code.field.Zeros(3)
        error[bit] = 1
        assert capacity_decoder.get_failure_and_erasure(error) == (False, False)


_REAL_PACKAGE_CHECKS = """
import numpy as np
import stim

from qldpc import decoders

# A matrix keeps the most likely of two identical columns.
decoder = decoders.get_decoder_tesseract(np.array([[1, 1]]), error_channel=[0.1, 0.4])
assert np.array_equal(decoder.decode_errors(np.array([1])), [0, 1])

# A DEM merges equivalent mechanisms, so their combined probability selects the logical class.
dem = stim.DetectorErrorModel("error(0.3) D0 L0\\nerror(0.3) D0 L0\\nerror(0.4) D0")
decoder = decoders.get_decoder_tesseract(dem)
syndromes = np.array([[1], [0]])
assert np.array_equal(decoder.decode_observables_batch(syndromes), [[1], [0]])
assert np.array_equal(decoder.decode_errors_batch(syndromes), [[1, 0, 0], [0, 0, 0]])

# A syndrome that no error explains is flagged, with generated detector orders.
dem = stim.DetectorErrorModel("detector D0\\ndetector D1\\nerror(0.1) D0 L0")
decoder = decoders.get_decoder_tesseract(
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
errors = decoders.get_decoder_tesseract(dem).decode_errors_batch(shots)
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
