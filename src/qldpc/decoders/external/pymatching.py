# SPDX-License-Identifier: Apache-2.0

"""Settings and observable adapter for the pymatching package.

qLDPC imports this integration module while initializing its public decoder API.  Importing
PyMatching and ldpc eagerly adds roughly 0.18 seconds (about 25 percent) to ``import qldpc`` in
fresh-process development benchmarks.  The error-decoder subclass is therefore created on first use
in the private lazy-backend section at the bottom of this module.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np
import numpy.typing as npt
import scipy.sparse
import stim

from qldpc.math import IntegerArray

from ..common import _erasure_bit_support
from ..construction.specs import _is_default_value, decoder_spec
from ..dems import DetectorErrorModelArrays
from ..protocols import BatchErrorDecoder, ObservableDecoder

_PcmOrDem: TypeAlias = IntegerArray | stim.DetectorErrorModel
_FAULTS_MATRIX_MESSAGE = (
    "MWPM faults_matrix is reserved for observable decoding: use"
    " decoders.mwpm(...).build_observable_decoder(dem), which derives it from the observables of"
    " the detector error model"
)

if TYPE_CHECKING:
    import pymatching

    class Matching(pymatching.Matching, BatchErrorDecoder): ...


# Public decoder and settings


class MatchingObservableDecoder(ObservableDecoder):
    """Observable decoder backed by a Matching that predicts observable flips."""

    def __init__(self, matching: Any, *, enable_correlations: bool = False) -> None:
        self.matching = matching
        self.enable_correlations = enable_correlations

    def decode_observables(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode one syndrome to predicted observable flips."""
        return np.asarray(
            self.matching.decode(syndrome, enable_correlations=self.enable_correlations),
            dtype=np.uint8,
        )

    def decode_observables_batch(self, syndromes: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode a batch of syndromes to predicted observable flips."""
        return np.asarray(
            self.matching.decode_batch(syndromes, enable_correlations=self.enable_correlations),
            dtype=np.uint8,
        )


@_erasure_bit_support("MWPM", supported=False)
def _get_decoder_mwpm(
    pcm_or_dem: _PcmOrDem,
    *,
    enable_correlations: bool = False,
    decompose_errors: bool = False,
    ignore_non_graphlike_errors: bool = False,
    weights: float | npt.NDArray[np.floating] | Sequence[float] | None = None,
    error_probabilities: float | npt.NDArray[np.floating] | Sequence[float] | None = None,
    repetitions: int | None = None,
    timelike_weights: float | npt.NDArray[np.floating] | Sequence[float] | None = None,
    measurement_error_probabilities: float
    | npt.NDArray[np.floating]
    | Sequence[float]
    | None = None,
    merge_strategy: Literal[
        "disallow", "independent", "smallest-weight", "keep-original", "replace"
    ] = "smallest-weight",
    use_virtual_boundary_node: bool = False,
    backend_options: Mapping[str, object] | None = None,
) -> BatchErrorDecoder:
    """Build a minimum-weight perfect matching (MWPM) decoder.

    Args:
        pcm_or_dem: A parity-check matrix or detector error model (DEM) to decode.
        enable_correlations: Whether to use PyMatching's correlated-matching mode.  Correlated
            matching predicts observable flips, so it requires ``build_observable_decoder`` and
            rejects every other non-default option.
        decompose_errors: Whether to apply decompositions suggested by a detector error model.
        ignore_non_graphlike_errors: Whether to ignore errors that trigger more than two detectors
            after any requested decomposition.
        weights: Weight of each error mechanism for a matrix input.
        error_probabilities: Error probabilities for a matrix input.
        repetitions: Number of repeated matching rounds.
        timelike_weights: Weight assigned to timelike edges.
        measurement_error_probabilities: Measurement-error probabilities for repeated rounds.
        merge_strategy: Strategy used when merging duplicate matching edges.
        use_virtual_boundary_node: Whether to use a virtual boundary node.
        backend_options: Additional options for ``pymatching.Matching.load_from_check_matrix``
            that are not listed above.  PyMatching ignores names that it does not recognize, so
            names absent from its signature are rejected when the decoder is built.
            ``faults_matrix`` is reserved for observable decoding and cannot be specified here.

    Returns:
        A ``pymatching.Matching`` subclass that is also a
        :class:`~qldpc.decoders.protocols.BatchErrorDecoder`, which maps a syndrome to an inferred
        physical error.

    This decoder cannot signal erasure, so ``add_erasure_bit=True`` is rejected.  If
    ``decompose_errors=True`` splits a DEM error mechanism, an inferred error addresses the
    resulting components rather than the original error mechanisms, so it cannot be converted back
    into observable flips of that DEM.

    See the `PyMatching documentation <https://pymatching.readthedocs.io/>`_ and
    `arXiv:2105.13082 <https://arxiv.org/abs/2105.13082>`_.
    """
    if enable_correlations:
        raise ValueError(
            "Correlated matching (enable_correlations=True) cannot infer errors; it can only"
            " predict observable flips, as with decoders.mwpm(...).build_observable_decoder(dem)"
        )
    return _build_matching(
        pcm_or_dem,
        decompose_errors=decompose_errors,
        ignore_non_graphlike_errors=ignore_non_graphlike_errors,
        predict_observables=False,
        weights=weights,
        error_probabilities=error_probabilities,
        repetitions=repetitions,
        timelike_weights=timelike_weights,
        measurement_error_probabilities=measurement_error_probabilities,
        merge_strategy=merge_strategy,
        use_virtual_boundary_node=use_virtual_boundary_node,
        backend_options=backend_options,
    )


def _get_observable_decoder_mwpm(
    dem: stim.DetectorErrorModel,
    *,
    decompose_errors: bool = False,
    ignore_non_graphlike_errors: bool = False,
    enable_correlations: bool = False,
    backend_options: Mapping[str, object] | None = None,
    **decoder_args: object,
) -> MatchingObservableDecoder:
    """Build an MWPM decoder that predicts DEM observable flips natively.

    In correlated-matching mode, PyMatching constructs the matching directly from the DEM, so the
    other matching-construction arguments are unused.
    """
    if enable_correlations:
        pymatching = _get_pymatching()
        matching = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
        return MatchingObservableDecoder(matching, enable_correlations=True)
    return MatchingObservableDecoder(
        _build_matching(
            dem,
            decompose_errors=decompose_errors,
            ignore_non_graphlike_errors=ignore_non_graphlike_errors,
            predict_observables=True,
            backend_options=backend_options,
            **decoder_args,
        )
    )


def _validate_mwpm_options(
    options: dict[str, object], _explicitly_provided: frozenset[str]
) -> dict[str, object]:
    """Reject options that would change error decoding or break correlated matching."""
    backend_options = options["backend_options"]
    if isinstance(backend_options, Mapping) and "faults_matrix" in backend_options:
        raise ValueError(_FAULTS_MATRIX_MESSAGE)
    if not options["enable_correlations"]:
        return options
    parameters = inspect.signature(_get_decoder_mwpm).parameters
    for name, value in options.items():
        if name != "enable_correlations" and not _is_default_value(value, parameters[name].default):
            raise ValueError(
                f"The MWPM option {name}={value!r} is not supported with enable_correlations=True"
            )
    return options


_MWPM_SETTINGS_RETURNS = (
    "Decoder settings.  Their ``build(pcm_or_dem)`` method takes a parity-check matrix or "
    "detector error model (DEM) and returns a ``pymatching.Matching`` subclass that is also a "
    ":class:`~qldpc.decoders.protocols.BatchErrorDecoder`, which maps a syndrome to an inferred"
    " physical error.  Their ``build_observable_decoder(dem)`` method returns a "
    ":class:`~qldpc.decoders.external.pymatching.MatchingObservableDecoder`, which predicts the"
    " observable flips of a DEM natively: DEM probabilities provide matching weights, and DEM "
    "observable targets provide the faults matrix."
)

mwpm = decoder_spec(
    "mwpm",
    _get_decoder_mwpm,
    _get_observable_decoder_mwpm,
    option_transform=_validate_mwpm_options,
    returns=_MWPM_SETTINGS_RETURNS,
)


# Private matching helpers


def _build_matching(
    pcm_or_dem: _PcmOrDem,
    *,
    decompose_errors: bool,
    ignore_non_graphlike_errors: bool,
    predict_observables: bool,
    backend_options: Mapping[str, object] | None = None,
    **decoder_args: object,
) -> Any:
    """Build a Matching that predicts errors or, from a DEM, observable flips."""
    pymatching = _get_pymatching()
    backend_options = dict(backend_options or {})
    _validate_backend_options(pymatching, backend_options)
    decoder_args |= backend_options
    infers_decomposed_errors = False
    if isinstance(pcm_or_dem, stim.DetectorErrorModel):
        dem_arrays = DetectorErrorModelArrays(pcm_or_dem, decompose_errors=decompose_errors)
        pcm = dem_arrays.detector_flip_matrix
        if decoder_args.get("weights") is not None:
            raise ValueError("Cannot set error weights when initializing a MWPM decoder from a DEM")
        decoder_args["weights"] = np.log((1 - dem_arrays.error_probs) / dem_arrays.error_probs)
        if predict_observables:
            decoder_args["faults_matrix"] = dem_arrays.observable_flip_matrix
        elif decompose_errors:
            infers_decomposed_errors = _splits_errors(pcm_or_dem, dem_arrays)
    else:
        pcm = pcm_or_dem

    detectors_per_error = np.asarray((pcm != 0).sum(axis=0)).ravel()
    error_is_not_graphlike = detectors_per_error > 2
    if ignore_non_graphlike_errors:
        if np.any(error_is_not_graphlike):
            mask = np.ones(pcm.shape[1])
            mask[error_is_not_graphlike] = 0
            pcm = pcm @ scipy.sparse.diags(mask)
    elif np.any(error_is_not_graphlike):
        column = int(np.argmax(error_is_not_graphlike))
        raise ValueError(
            "The provided parity check matrix or detector error model contains a non-graphlike"
            f" error: column {column} of the parity check matrix addresses"
            f" {detectors_per_error[column]} detectors, which may occur (for example) due to the"
            " presence of a Pauli-Y error that flips both X and Z detectors.  Try decomposing"
            " non-graphlike errors by passing 'decompose_errors=True' to the decoder, which splits"
            " errors along the decompositions that the detector error model suggests; stim provides"
            " those suggestions for a circuit via"
            " circuit.detector_error_model(decompose_errors=True).  If that does not work either,"
            " you can try 'ignore_non_graphlike_errors=True'"
        )

    matching = pymatching.Matching() if predict_observables else _get_matching_type()()
    matching.load_from_check_matrix(pcm, **decoder_args)
    if infers_decomposed_errors:
        matching._infers_decomposed_errors = True
    return matching


def _validate_backend_options(pymatching: Any, backend_options: Mapping[str, object]) -> None:
    """Reject backend options that load_from_check_matrix would silently ignore."""
    if "faults_matrix" in backend_options:
        raise ValueError(_FAULTS_MATRIX_MESSAGE)
    parameters = inspect.signature(pymatching.Matching.load_from_check_matrix).parameters
    supported = {
        name
        for name, parameter in parameters.items()
        if name not in ("self", "check_matrix")
        and parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }
    if unsupported := sorted(set(backend_options) - supported):
        raise ValueError(
            f"Unsupported MWPM backend option(s) {unsupported}: they are not parameters of"
            " pymatching.Matching.load_from_check_matrix, which would silently ignore them"
        )


def _splits_errors(
    dem: stim.DetectorErrorModel, decomposed_arrays: DetectorErrorModelArrays
) -> bool:
    """Whether decomposing a detector error model splits any error mechanism."""
    merged_arrays = DetectorErrorModelArrays(dem)
    return (
        merged_arrays.num_errors != decomposed_arrays.num_errors
        or (merged_arrays.detector_flip_matrix != decomposed_arrays.detector_flip_matrix).nnz > 0
        or (merged_arrays.observable_flip_matrix != decomposed_arrays.observable_flip_matrix).nnz
        > 0
    )


# Lazy backend class
#
# Matching is an error decoder only when it has no faults matrix.  Create that protocol-compatible
# subclass on first use instead of importing PyMatching during every qldpc import.

_MATCHING_TYPE: type[Any] | None = None


def _get_pymatching() -> Any:
    """Import PyMatching or report a broken qLDPC installation."""
    try:
        import pymatching
    except ModuleNotFoundError as error:
        if error.name != "pymatching":
            raise
        raise ModuleNotFoundError(
            "PyMatching is a required qLDPC dependency but is not installed. Reinstall qLDPC"
        ) from None
    return pymatching


def _get_matching_type() -> type[Any]:
    """Return the protocol-compatible PyMatching subclass, creating it on first use."""
    global _MATCHING_TYPE
    if _MATCHING_TYPE is None:
        pymatching_module = _get_pymatching()
        matching_type = type(
            "Matching",
            (pymatching_module.Matching, BatchErrorDecoder),
            {
                "__module__": __name__,
                "__doc__": "A pymatching.Matching that is also a BatchErrorDecoder.",
            },
        )
        matching_type.__qualname__ = matching_type.__name__
        Matching = matching_type
        globals()["Matching"] = Matching
        _MATCHING_TYPE = Matching
    return _MATCHING_TYPE


def __getattr__(name: str) -> Any:
    """Load the protocol-compatible PyMatching subclass only when requested."""
    if name != "Matching":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return _get_matching_type()
