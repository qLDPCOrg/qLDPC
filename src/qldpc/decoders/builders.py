# SPDX-License-Identifier: Apache-2.0

"""Immediate construction of decoders from matrices and detector error models."""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import Any, ParamSpec, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt
import scipy.sparse
import stim

from qldpc._util import format_docstring
from qldpc.math import IntegerArray

from .common import PLACEHOLDER_ERROR_RATE
from .custom import GUFDecoder, ILPDecoder, RelayBPDecoder
from .dems import DetectorErrorModelArrays
from .lookup import LookupDecoder, ObservableLookupDecoder
from .protocols import BatchErrorDecoder, ErrorDecoder, ObservableDecoder

__all__ = [
    "MatchingObservableDecoder",
    "get_decoder_bf",
    "get_decoder_bp_lsd",
    "get_decoder_bp_osd",
    "get_decoder_guf",
    "get_decoder_ilp",
    "get_decoder_lookup",
    "get_decoder_mwpm",
    "get_decoder_rbp",
    "get_error_decoder_mwpm",
    "get_min_sum_bp_decoder",
    "get_observable_decoder_lookup",
    "get_observable_decoder_mwpm",
    "get_relay_bp_decoder",
]

PcmOrDem: TypeAlias = IntegerArray | stim.DetectorErrorModel

_Parameters = ParamSpec("_Parameters")
_Decoder = TypeVar("_Decoder", bound=ErrorDecoder)


def _erasure_bit_support(
    display_name: str,
    *,
    supported: bool,
) -> Callable[[Callable[_Parameters, _Decoder]], Callable[_Parameters, _Decoder]]:
    """Declare and enforce whether a decoder builder supports an erasure bit."""

    def decorator(
        decoder_builder: Callable[_Parameters, _Decoder],
    ) -> Callable[_Parameters, _Decoder]:
        message = (
            f"The {display_name} decoder cannot signal erasure, so it does not accept the"
            " add_erasure_bit argument"
        )

        @functools.wraps(decoder_builder)
        def checked_builder(*args: _Parameters.args, **kwargs: _Parameters.kwargs) -> _Decoder:
            add_erasure_bit = bool(kwargs.get("add_erasure_bit"))
            if not supported:
                if add_erasure_bit:
                    raise ValueError(message)
                kwargs.pop("add_erasure_bit", None)

            decoder = decoder_builder(*args, **kwargs)
            if add_erasure_bit and not getattr(decoder, "has_erasure_bit", False):
                raise ValueError(message)
            return decoder

        return checked_builder

    return decorator


@_erasure_bit_support("BP_OSD", supported=False)
@format_docstring(PLACEHOLDER_ERROR_RATE=PLACEHOLDER_ERROR_RATE)
def get_decoder_bp_osd(
    pcm_or_dem: PcmOrDem,
    *,
    error_rate: float = PLACEHOLDER_ERROR_RATE,
    error_channel: npt.NDArray[np.floating] | Sequence[float] | None = None,
    **decoder_args: object,
) -> ErrorDecoder:
    """Build a belief-propagation with ordered-statistics (BP+OSD) decoder.

    Args:
        pcm_or_dem: A parity-check matrix or detector error model (DEM) to decode.
        error_rate: The i.i.d. probability of each error in a matrix.  Ignored for a DEM.
            Default: {PLACEHOLDER_ERROR_RATE}.
        error_channel: The probability of each error mechanism.  For a matrix, defaults to
            ``[error_rate] * num_errors``.  For a DEM, defaults to its error probabilities.  An
            explicit value overrides either default.
        **decoder_args: Additional keyword arguments passed to ``ldpc.BpOsdDecoder``.

    Returns:
        An ``ldpc.BpOsdDecoder`` subclass that is also an
        :class:`~qldpc.decoders.protocols.ErrorDecoder`.

    This decoder cannot signal erasure, so ``add_erasure_bit=True`` is rejected.

    See ``help(ldpc.BpOsdDecoder)``, the
    `ldpc decoder documentation <https://software.roffe.eu/ldpc/quantum_decoder.html>`_, and
    `arXiv:2005.07016 <https://arxiv.org/abs/2005.07016>`_.
    """
    from . import adapters

    pcm, error_channel = _to_ldpc_inputs(pcm_or_dem, error_rate, error_channel)
    return adapters.BpOsdDecoder(pcm, error_channel=error_channel, **decoder_args)


@_erasure_bit_support("BP_LSD", supported=False)
@format_docstring(PLACEHOLDER_ERROR_RATE=PLACEHOLDER_ERROR_RATE)
def get_decoder_bp_lsd(
    pcm_or_dem: PcmOrDem,
    *,
    error_rate: float = PLACEHOLDER_ERROR_RATE,
    error_channel: npt.NDArray[np.floating] | Sequence[float] | None = None,
    **decoder_args: object,
) -> ErrorDecoder:
    """Build a belief-propagation with localized-statistics (BP+LSD) decoder.

    Args:
        pcm_or_dem: A parity-check matrix or detector error model (DEM) to decode.
        error_rate: The i.i.d. probability of each error in a matrix.  Ignored for a DEM.
            Default: {PLACEHOLDER_ERROR_RATE}.
        error_channel: The probability of each error mechanism.  For a matrix, defaults to
            ``[error_rate] * num_errors``.  For a DEM, defaults to its error probabilities.  An
            explicit value overrides either default.
        **decoder_args: Additional keyword arguments passed to
            ``ldpc.bplsd_decoder.BpLsdDecoder``.

    Returns:
        An ``ldpc.bplsd_decoder.BpLsdDecoder`` subclass that is also an
        :class:`~qldpc.decoders.protocols.ErrorDecoder`.

    This decoder cannot signal erasure, so ``add_erasure_bit=True`` is rejected.

    See ``help(ldpc.bplsd_decoder.BpLsdDecoder)``, the
    `ldpc decoder documentation <https://software.roffe.eu/ldpc/quantum_decoder.html>`_, and
    `arXiv:2406.18655 <https://arxiv.org/abs/2406.18655>`_.
    """
    from . import adapters

    pcm, error_channel = _to_ldpc_inputs(pcm_or_dem, error_rate, error_channel)
    return adapters.BpLsdDecoder(pcm, error_channel=error_channel, **decoder_args)


@_erasure_bit_support("BF", supported=False)
@format_docstring(PLACEHOLDER_ERROR_RATE=PLACEHOLDER_ERROR_RATE)
def get_decoder_bf(
    pcm_or_dem: PcmOrDem,
    *,
    error_rate: float = PLACEHOLDER_ERROR_RATE,
    error_channel: npt.NDArray[np.floating] | Sequence[float] | None = None,
    **decoder_args: object,
) -> ErrorDecoder:
    """Build a belief-find (BF) decoder.

    Args:
        pcm_or_dem: A parity-check matrix or detector error model (DEM) to decode.
        error_rate: The i.i.d. probability of each error in a matrix.  Ignored for a DEM.
            Default: {PLACEHOLDER_ERROR_RATE}.
        error_channel: The probability of each error mechanism.  For a matrix, defaults to
            ``[error_rate] * num_errors``.  For a DEM, defaults to its error probabilities.  An
            explicit value overrides either default.
        **decoder_args: Additional keyword arguments passed to ``ldpc.BeliefFindDecoder``.

    Returns:
        An ``ldpc.BeliefFindDecoder`` subclass that is also an
        :class:`~qldpc.decoders.protocols.ErrorDecoder`.

    This decoder cannot signal erasure, so ``add_erasure_bit=True`` is rejected.

    See ``help(ldpc.BeliefFindDecoder)``, the
    `ldpc decoder documentation <https://software.roffe.eu/ldpc/quantum_decoder.html>`_,
    `arXiv:1709.06218 <https://arxiv.org/abs/1709.06218>`_,
    `arXiv:2103.08049 <https://arxiv.org/abs/2103.08049>`_, and
    `arXiv:2209.01180 <https://arxiv.org/abs/2209.01180>`_.
    """
    from . import adapters

    pcm, error_channel = _to_ldpc_inputs(pcm_or_dem, error_rate, error_channel)
    return adapters.BeliefFindDecoder(pcm, error_channel=error_channel, **decoder_args)


def _to_ldpc_inputs(
    pcm_or_dem: PcmOrDem,
    error_rate: float,
    error_channel: npt.NDArray[np.floating] | Sequence[float] | None,
) -> tuple[IntegerArray, list[float]]:
    """Convert backend input to the matrix and probabilities expected by ldpc."""
    if isinstance(pcm_or_dem, stim.DetectorErrorModel):
        dem_arrays = DetectorErrorModelArrays(pcm_or_dem)
        pcm = dem_arrays.detector_flip_matrix
        error_channel = dem_arrays.error_probs if error_channel is None else error_channel
    else:
        pcm = pcm_or_dem
        error_channel = [error_rate] * pcm.shape[1] if error_channel is None else error_channel
    if pcm.dtype.kind in "biu":
        pcm = pcm.astype(np.uint8, copy=False)
    return pcm, list(error_channel)


@_erasure_bit_support("MWPM", supported=False)
def get_decoder_mwpm(
    pcm_or_dem: PcmOrDem,
    *,
    decompose_errors: bool = False,
    ignore_non_graphlike_errors: bool = False,
    **decoder_args: object,
) -> BatchErrorDecoder:
    """Build a minimum-weight perfect matching (MWPM) error decoder.

    Args:
        pcm_or_dem: A parity-check matrix or detector error model (DEM) to decode.
        decompose_errors: Whether to apply decompositions suggested by a DEM.
        ignore_non_graphlike_errors: Whether to ignore errors that trigger more than two detectors
            after any requested decomposition.
        **decoder_args: Additional keyword arguments passed to
            ``pymatching.Matching.load_from_check_matrix``.

    Returns:
        A ``pymatching.Matching`` subclass that is also a
        :class:`~qldpc.decoders.protocols.BatchErrorDecoder`.

    This decoder cannot signal erasure, so ``add_erasure_bit=True`` is rejected.  It always maps a
    syndrome to an inferred physical error, even when built from a DEM.  To predict the observable
    flips of a DEM natively, use :func:`get_observable_decoder_mwpm`, normally through
    :func:`qldpc.decoders.retrieval.get_observable_decoder` with ``decoder=decoders.mwpm(...)``.

    If ``decompose_errors=True`` splits a DEM error mechanism, the inferred vector addresses the
    resulting components rather than the original error mechanisms.  It therefore cannot be
    converted back into observable flips of that DEM.

    See the `PyMatching documentation <https://pymatching.readthedocs.io/>`_ and
    `arXiv:2105.13082 <https://arxiv.org/abs/2105.13082>`_.
    """
    return _build_matching(
        pcm_or_dem,
        decompose_errors=decompose_errors,
        ignore_non_graphlike_errors=ignore_non_graphlike_errors,
        predict_observables=False,
        **decoder_args,
    )


def _build_matching(
    pcm_or_dem: PcmOrDem,
    *,
    decompose_errors: bool,
    ignore_non_graphlike_errors: bool,
    predict_observables: bool,
    **decoder_args: object,
) -> Any:
    """Build a Matching that predicts errors or, from a DEM, observable flips."""
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

    import pymatching

    from . import adapters

    matching = pymatching.Matching() if predict_observables else adapters.Matching()
    matching.load_from_check_matrix(pcm, **decoder_args)
    if infers_decomposed_errors:
        matching._infers_decomposed_errors = True
    return matching


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


def get_error_decoder_mwpm(
    pcm_or_dem: PcmOrDem,
    *,
    enable_correlations: bool = False,
    **decoder_args: Any,
) -> BatchErrorDecoder:
    """Build the error-decoding mode used by an MWPM :class:`DecoderSpec`.

    Args:
        pcm_or_dem: A parity-check matrix or detector error model to decode.
        enable_correlations: Whether to enable correlated matching.  Correlated matching predicts
            observable flips and therefore is not available in this error-decoding mode.
        **decoder_args: Arguments forwarded to :func:`get_decoder_mwpm`.

    Returns:
        An MWPM decoder that infers physical errors.
    """
    if enable_correlations:
        raise ValueError(
            "Correlated matching (enable_correlations=True) cannot infer errors; it can only"
            " predict observable flips, as with decoders.get_observable_decoder"
        )
    return get_decoder_mwpm(pcm_or_dem, **decoder_args)


def get_observable_decoder_mwpm(
    dem: stim.DetectorErrorModel,
    *,
    decompose_errors: bool = False,
    ignore_non_graphlike_errors: bool = False,
    enable_correlations: bool = False,
    **decoder_args: object,
) -> MatchingObservableDecoder:
    """Build an MWPM decoder that predicts DEM observable flips natively.

    Args:
        dem: The detector error model to decode.
        decompose_errors: Whether to apply decompositions suggested by the DEM.
        ignore_non_graphlike_errors: Whether to ignore errors that trigger more than two detectors
            after any requested decomposition.
        enable_correlations: Whether to use PyMatching's correlated-matching mode.
        **decoder_args: Additional matching-construction arguments.  PyMatching constructs
            correlated matchings directly from ``dem``; in that mode these arguments are unused.

    Returns:
        An observable decoder backed by ``pymatching.Matching``.

    Unlike :func:`get_decoder_mwpm`, this builder returns observable flips rather than inferred
    physical errors.  DEM probabilities provide matching weights, and DEM observable targets provide
    the faults matrix.
    """
    if enable_correlations:
        import pymatching

        matching = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
        return MatchingObservableDecoder(matching, enable_correlations=True)
    return MatchingObservableDecoder(
        _build_matching(
            dem,
            decompose_errors=decompose_errors,
            ignore_non_graphlike_errors=ignore_non_graphlike_errors,
            predict_observables=True,
            **decoder_args,
        )
    )


@_erasure_bit_support("RBP", supported=True)
def get_decoder_rbp(
    pcm_or_dem: PcmOrDem,
    error_priors: npt.NDArray[np.floating] | Sequence[float] | None = None,
    **decoder_args: object,
) -> RelayBPDecoder:
    """Build a Relay-BP decoder.

    Args:
        pcm_or_dem: A parity-check matrix or detector error model (DEM) to decode.
        error_priors: Prior probabilities for each error.  A DEM supplies these by default.
        **decoder_args: Arguments passed to
            :class:`~qldpc.decoders.custom.relay_bp.RelayBPDecoder`, including the backend class
            ``name``, observable matrix, and ``add_erasure_bit``.

    Returns:
        A :class:`~qldpc.decoders.custom.relay_bp.RelayBPDecoder`, which can infer errors and, when
        observable metadata is available, predict observable flips.

    With ``add_erasure_bit=True``, the decoder appends a flag set when the inferred error does not
    reproduce the syndrome.

    See the `relay-bp package documentation <https://pypi.org/project/relay-bp>`_ and
    `arXiv:2506.01779 <https://arxiv.org/abs/2506.01779>`_.
    """
    return RelayBPDecoder(pcm_or_dem, error_priors, **decoder_args)  # type: ignore[arg-type]


def _get_relay_decoder(
    pcm_or_dem: PcmOrDem,
    *,
    decoder_class_prefix: str,
    precision: str,
    **decoder_args: Any,
) -> RelayBPDecoder:
    """Build a RelayBPDecoder from a class-name prefix and precision."""
    return get_decoder_rbp(pcm_or_dem, name=f"{decoder_class_prefix}{precision}", **decoder_args)


def get_relay_bp_decoder(pcm_or_dem: PcmOrDem, **decoder_args: Any) -> RelayBPDecoder:
    """Build the ``RelayDecoder`` backend selected by a ``relay_bp`` :class:`DecoderSpec`.

    The specification supplies a ``precision`` suffix and forwards all other options to
    :func:`get_decoder_rbp`.  This public builder exists so deferred specifications have a stable,
    pickleable construction path.
    """
    return _get_relay_decoder(pcm_or_dem, decoder_class_prefix="RelayDecoder", **decoder_args)


def get_min_sum_bp_decoder(pcm_or_dem: PcmOrDem, **decoder_args: Any) -> RelayBPDecoder:
    """Build the ``MinSumBPDecoder`` backend selected by a ``min_sum_bp`` :class:`DecoderSpec`.

    The specification supplies a ``precision`` suffix and forwards all other options to
    :func:`get_decoder_rbp`.  This public builder exists so deferred specifications have a stable,
    pickleable construction path.
    """
    return _get_relay_decoder(pcm_or_dem, decoder_class_prefix="MinSumBPDecoder", **decoder_args)


@_erasure_bit_support("lookup", supported=True)
def get_decoder_lookup(pcm_or_dem: PcmOrDem, **decoder_args: object) -> LookupDecoder:
    """Build a lookup table that maps syndromes to inferred errors.

    Args:
        pcm_or_dem: A parity-check matrix or detector error model (DEM) to decode.  A DEM supplies
            default error probabilities and observable metadata.
        **decoder_args: Arguments passed to :class:`~qldpc.decoders.lookup.LookupDecoder`, including
            the required ``max_weight`` and optional erasure, confidence, and symplectic settings.

    Returns:
        A :class:`~qldpc.decoders.lookup.LookupDecoder`.

    ``add_erasure_bit=True`` appends a flag for syndromes absent from the table.  A positive
    ``confidence_ratio`` also enables the flag and erases ambiguous syndromes.  This error builder
    returns a representative physical error; use :func:`get_observable_decoder_lookup` to return
    observable flips directly.
    """
    return LookupDecoder(pcm_or_dem, **decoder_args)  # type: ignore[arg-type]


def get_observable_decoder_lookup(
    dem: stim.DetectorErrorModel, **decoder_args: object
) -> ObservableDecoder:
    """Build a lookup table that maps DEM syndromes directly to observable flips.

    Args:
        dem: The detector error model whose detectors and observables define the table.
        **decoder_args: Arguments passed to
            :class:`~qldpc.decoders.lookup.ObservableLookupDecoder`, including ``max_weight`` and
            optional erasure, confidence, and post-selection settings.

    Returns:
        An :class:`~qldpc.decoders.lookup.ObservableLookupDecoder`.
    """
    return ObservableLookupDecoder(dem, **decoder_args)  # type: ignore[call-overload]


@_erasure_bit_support("ILP", supported=True)
def get_decoder_ilp(
    pcm_or_dem: PcmOrDem,
    *,
    add_erasure_bit: bool = False,
    **decoder_args: object,
) -> ILPDecoder:
    """Build an integer-linear-program (ILP) decoder.

    Args:
        pcm_or_dem: A parity-check matrix or detector error model to decode.  A DEM is converted to
            its dense detector-flip matrix.
        add_erasure_bit: Whether to append a flag when the solver cannot produce an error that
            reproduces the syndrome.
        **decoder_args: Arguments passed to ``cvxpy.Problem.solve`` by
            :class:`~qldpc.decoders.custom.ilp.ILPDecoder`.

    Returns:
        An :class:`~qldpc.decoders.custom.ilp.ILPDecoder`.

    ILP decoding supports prime fields.  Without an erasure bit, an unexplained syndrome is rejected
    rather than returned as an ordinary inferred error.
    """
    return ILPDecoder(_to_pcm(pcm_or_dem), add_erasure_bit=add_erasure_bit, **decoder_args)


@_erasure_bit_support("GUF", supported=True)
def get_decoder_guf(pcm_or_dem: PcmOrDem, **decoder_args: object) -> GUFDecoder:
    """Build a generalized union-find (GUF) decoder.

    Args:
        pcm_or_dem: A parity-check matrix or detector error model to decode.  A DEM is converted to
            its dense detector-flip matrix.
        **decoder_args: Arguments passed to
            :class:`~qldpc.decoders.custom.guf.GUFDecoder`, including ``max_weight``,
            ``symplectic``, and ``add_erasure_bit``.

    Returns:
        A :class:`~qldpc.decoders.custom.guf.GUFDecoder`.

    With ``add_erasure_bit=True``, the decoder appends a flag when its search is exhausted without
    finding an error that reproduces the syndrome.  Supplying ``max_weight`` can make the search
    exponential.  See `arXiv:2103.08049 <https://arxiv.org/abs/2103.08049>`_.
    """
    return GUFDecoder(_to_pcm(pcm_or_dem), **decoder_args)  # type: ignore[arg-type]


def _to_pcm(pcm_or_dem: PcmOrDem) -> IntegerArray:
    """Return a parity-check matrix, densifying a detector error model."""
    if isinstance(pcm_or_dem, stim.DetectorErrorModel):
        return DetectorErrorModelArrays(pcm_or_dem).detector_flip_matrix.toarray()
    return pcm_or_dem
