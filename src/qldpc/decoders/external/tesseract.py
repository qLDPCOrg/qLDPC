# SPDX-License-Identifier: Apache-2.0

"""Adapter and builder for the optional Tesseract decoder."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, cast

import galois
import numpy as np
import numpy.typing as npt
import scipy.sparse
import stim

from qldpc._util import format_docstring
from qldpc.math import IntegerArray

from ..common import (
    PLACEHOLDER_ERROR_RATE,
    _deprecate_error_rate_option,
    _erasure_bit_support,
    _get_matrix_error_channel,
    _reject_dem_error_probabilities,
    with_erasure_bits,
)
from ..construction.specs import DecoderSpec, decoder_spec
from ..dems import DetectorErrorModelArrays
from ..protocols import BatchObservableDecoder, ErrorDecoder

TesseractDetectorOrderMethod = Literal["bfs", "index", "coordinate"]

_DETECTOR_ORDER_NAMES: dict[TesseractDetectorOrderMethod, str] = {
    "bfs": "BFS",
    "coordinate": "Coordinate",
    "index": "Index",
}


class TesseractDecoder(ErrorDecoder, BatchObservableDecoder):
    """Wrapper for the Tesseract search-based decoder.

    Requires the optional ``tesseract-decoder`` package, which can be installed with
    ``pip install 'qldpc[tesseract]'``.

    Tesseract natively decodes a binary Stim detector error model.  A binary parity-check matrix is
    converted to a detector error model with one error mechanism per matrix column.  The
    ``decode_errors`` method returns an inferred error in matrix-column or flattened DEM-error
    order.  The ``decode_observables`` and ``decode_observables_batch`` methods return Tesseract's
    native observable predictions.

    If initialized with ``add_erasure_bit=True``, the decoder appends a bit to every inferred error
    and observable prediction.  The bit is set when Tesseract reports low confidence because its
    search did not converge within the configured beam or priority-queue limits.

    Tesseract can merge error mechanisms with identical detector and observable flips, combining
    their probabilities and reporting the first merged mechanism.  By default, qLDPC merges for a
    detector error model, where merged mechanisms are interchangeable and combining probabilities
    identifies the most likely logical class.  A parity-check matrix has no observables, so merging
    could report a column other than the most likely one, and it is disabled by default.

    See the `Tesseract documentation
    <https://github.com/quantumlib/tesseract-decoder#python-interface>`_ and
    `arXiv:2503.10988 <https://arxiv.org/abs/2503.10988>`_.
    """

    @format_docstring(PLACEHOLDER_ERROR_RATE=PLACEHOLDER_ERROR_RATE)
    def __init__(
        self,
        pcm_or_dem: IntegerArray | stim.DetectorErrorModel,
        *,
        error_channel: float | npt.NDArray[np.floating] | Sequence[float] | None = None,
        add_erasure_bit: bool = False,
        det_beam: int = 5,
        beam_climbing: bool = False,
        no_revisit_dets: bool = True,
        verbose: bool = False,
        merge_errors: bool | None = None,
        pqlimit: int = 200_000,
        det_orders: Sequence[Sequence[int]] | None = None,
        det_penalty: float = 0.0,
        create_visualization: bool = False,
        sparsify_errors: bool = False,
        sparsify_base_degree: int = -1,
        sparsify_max_degree: int = -1,
        sparsify_reactivate_limit: int = -1,
        num_det_orders: int | None = None,
        det_order_method: TesseractDetectorOrderMethod | None = None,
        seed: int | None = None,
        error_rate: float | None = None,
    ) -> None:
        """Initialize a Tesseract decoder.

        Args:
            pcm_or_dem: A binary parity-check matrix or detector error model to decode.
            error_channel: One probability for every matrix-column error, or one probability per
                column.  Defaults to {PLACEHOLDER_ERROR_RATE}.  A DEM supplies its own
                probabilities, so neither probability argument can be specified with one.
            add_erasure_bit: Whether to append Tesseract's low-confidence flag to each result.
            det_beam: Beam-search cutoff.
            beam_climbing: Whether to retry with increasing beam sizes.
            no_revisit_dets: Whether to avoid revisiting equal residual detector sets.
            verbose: Whether Tesseract prints decoding diagnostics.
            merge_errors: Whether to merge error mechanisms with identical detector and observable
                flips, or None to merge only when decoding a detector error model.
            pqlimit: Maximum number of nodes pushed into the priority queue.
            det_orders: Explicit detector traversal permutations, or None to generate them.
            det_penalty: Additional cost for each residual detection event.
            create_visualization: Whether to retain visualization data.
            sparsify_errors: Whether to activate selected high-degree errors per shot.
            sparsify_base_degree: Maximum degree of errors that always remain active.
            sparsify_max_degree: Maximum degree of optional errors, or -1 for no maximum.
            sparsify_reactivate_limit: Maximum optional errors reactivated per shot, or -1 for
                Tesseract's heuristic.
            num_det_orders: Number of generated detector orders, or None for Tesseract's default.
            det_order_method: Generated detector-order method, or None for Tesseract's default.
            seed: Seed for generated detector orders, or None for Tesseract's default.
            error_rate: Deprecated i.i.d. matrix error probability. Use ``error_channel`` instead.
        """
        backend = _get_tesseract()
        is_dem = isinstance(pcm_or_dem, stim.DetectorErrorModel)
        dem, num_errors = _get_dem_and_num_errors(pcm_or_dem, error_channel, error_rate)
        backend_order_method = (
            None
            if det_order_method is None
            else getattr(
                backend.utils.DetectorOrderMethod,
                _get_detector_order_name(det_order_method),
            )
        )
        literal_orders = None if det_orders is None else [list(order) for order in det_orders]
        self.config = backend.tesseract.TesseractConfig(
            dem=dem,
            det_beam=det_beam,
            beam_climbing=beam_climbing,
            no_revisit_dets=no_revisit_dets,
            verbose=verbose,
            merge_errors=is_dem if merge_errors is None else merge_errors,
            pqlimit=pqlimit,
            det_orders=literal_orders,
            det_penalty=det_penalty,
            create_visualization=create_visualization,
            sparsify_errors=sparsify_errors,
            sparsify_base_degree=sparsify_base_degree,
            sparsify_max_degree=sparsify_max_degree,
            sparsify_reactivate_limit=sparsify_reactivate_limit,
            num_det_orders=num_det_orders,
            det_order_method=backend_order_method,
            seed=seed,
        )
        self.decoder = self.config.compile_decoder()
        self.has_erasure_bit = add_erasure_bit
        self.num_errors = num_errors
        self.num_detectors = dem.num_detectors
        self.num_observables = dem.num_observables

    def decode_errors(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode one syndrome to an inferred error in matrix-column or DEM-error order."""
        predicted_indices = np.asarray(
            self.decoder.decode_to_errors(self._validate_syndrome(syndrome)), dtype=int
        )
        if np.any((predicted_indices < 0) | (predicted_indices >= self.num_errors)):
            raise ValueError(
                "Tesseract returned an error index outside the provided matrix or detector error"
                " model"
            )
        error = np.zeros(self.num_errors, dtype=int)
        np.bitwise_xor.at(error, predicted_indices, 1)
        if self.has_erasure_bit:
            return with_erasure_bits(error, bool(self.decoder.low_confidence_flag))
        return error

    decode = decode_errors

    def decode_observables(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode one syndrome to Tesseract's native observable-flip prediction."""
        prediction = np.asarray(self.decoder.decode(self._validate_syndrome(syndrome)), dtype=int)
        if self.has_erasure_bit:
            return with_erasure_bits(prediction, bool(self.decoder.low_confidence_flag))
        return prediction

    def decode_observables_batch(self, syndromes: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode a batch of syndromes to native observable-flip predictions."""
        validated_syndromes = self._validate_syndromes(syndromes)
        output_size = self.num_observables + int(self.has_erasure_bit)
        if len(validated_syndromes) == 0:
            return np.empty((0, output_size), dtype=int)
        if self.has_erasure_bit:
            return np.asarray(
                [self.decode_observables(syndrome) for syndrome in validated_syndromes], dtype=int
            )
        return np.asarray(self.decoder.decode_batch(validated_syndromes), dtype=int)

    def _validate_syndrome(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.bool_]:
        """Convert one correctly shaped syndrome to the dtype required by Tesseract."""
        validated_syndrome = np.asarray(syndrome, dtype=bool)
        expected_shape = (self.num_detectors,)
        if validated_syndrome.shape != expected_shape:
            raise ValueError(
                "A Tesseract syndrome must have shape"
                f" {expected_shape}, but got {validated_syndrome.shape}"
            )
        return validated_syndrome

    def _validate_syndromes(self, syndromes: npt.NDArray[np.int_]) -> npt.NDArray[np.bool_]:
        """Convert correctly shaped batched syndromes to the dtype required by Tesseract."""
        validated_syndromes = np.asarray(syndromes, dtype=bool)
        if validated_syndromes.ndim != 2 or validated_syndromes.shape[1] != self.num_detectors:
            expected_shape = f"(num_shots, {self.num_detectors})"
            raise ValueError(
                "Tesseract syndromes must have shape"
                f" {expected_shape}, but got {validated_syndromes.shape}"
            )
        return validated_syndromes


@_erasure_bit_support("Tesseract", supported=True)
def _get_decoder_tesseract(
    pcm_or_dem: IntegerArray | stim.DetectorErrorModel, **decoder_args: Any
) -> TesseractDecoder:
    """Build a Tesseract search-based decoder.

    The options are those of :class:`~qldpc.decoders.external.tesseract.TesseractDecoder`.

    Returns:
        A :class:`~qldpc.decoders.external.tesseract.TesseractDecoder`, which infers errors and
        predicts observable flips natively.

    Tesseract requires the optional ``tesseract-decoder`` package, which can be installed with
    ``pip install 'qldpc[tesseract]'``.
    """
    return TesseractDecoder(pcm_or_dem, **decoder_args)


_TESSERACT_SETTINGS_RETURNS = (
    "Decoder settings.  Their ``build(pcm_or_dem)`` and ``build_observable_decoder(dem)`` "
    "methods take a binary parity-check matrix or detector error model and return a "
    ":class:`~qldpc.decoders.external.tesseract.TesseractDecoder`, which infers errors and "
    "predicts observable flips natively."
)

tesseract = decoder_spec(
    "tesseract",
    _get_decoder_tesseract,
    _get_decoder_tesseract,
    signature_source=TesseractDecoder,
    option_transform=_deprecate_error_rate_option,
    returns=_TESSERACT_SETTINGS_RETURNS,
)


def tesseract_preset(
    preset: Literal["long-beam", "short-beam"] = "long-beam",
    *,
    sparsify: Literal["surface-code-like", "color-code-like"] | None = None,
    error_channel: float | npt.NDArray[np.floating] | Sequence[float] | None = None,
    add_erasure_bit: bool = False,
    error_rate: float | None = None,
) -> DecoderSpec[TesseractDecoder]:
    """Configure one of Tesseract's named Sinter presets.

    Args:
        preset: ``"long-beam"`` or ``"short-beam"``, which set the beam-search cutoff, the
            priority-queue limit, and the number of generated detector orders.
        sparsify: None, or ``"surface-code-like"`` or ``"color-code-like"`` to sparsify errors with
            the base degree of the corresponding Tesseract preset.
        error_channel: One probability for every matrix-column error, or one probability per
            column.  A detector error model supplies its own probabilities.
        add_erasure_bit: Whether to append Tesseract's low-confidence flag to each result.
        error_rate: Deprecated i.i.d. matrix error probability.  Use ``error_channel`` instead.

    Returns:
        Settings for :func:`decoders.tesseract <qldpc.decoders.tesseract>` that reproduce the
        preset.
    """
    beam_presets = {
        "long-beam": (20, 1_000_000, 21),
        "short-beam": (15, 200_000, 16),
    }
    sparsify_base_degrees = {
        None: -1,
        "surface-code-like": 2,
        "color-code-like": 3,
    }
    try:
        det_beam, pqlimit, num_det_orders = beam_presets[preset]
    except KeyError:
        raise ValueError(
            f"Unknown Tesseract preset {preset!r}; expected 'long-beam' or 'short-beam'"
        ) from None
    try:
        sparsify_base_degree = sparsify_base_degrees[sparsify]
    except KeyError:
        raise ValueError(
            f"Unknown Tesseract sparsify preset {sparsify!r}; expected None,"
            " 'surface-code-like', or 'color-code-like'"
        ) from None

    explicitly_provided: set[str] = set()
    if error_channel is not None:
        explicitly_provided.add("error_channel")
    if error_rate is not None:
        explicitly_provided.add("error_rate")
    probability_options = _deprecate_error_rate_option(
        {"error_channel": error_channel, "error_rate": error_rate},
        frozenset(explicitly_provided),
    )
    return tesseract(
        error_channel=cast(
            float | npt.NDArray[np.floating] | Sequence[float] | None,
            probability_options["error_channel"],
        ),
        add_erasure_bit=add_erasure_bit,
        det_beam=det_beam,
        beam_climbing=True,
        pqlimit=pqlimit,
        sparsify_errors=sparsify is not None,
        sparsify_base_degree=sparsify_base_degree,
        num_det_orders=num_det_orders,
        det_order_method="index",
        seed=2_384_753,
    )


def _get_tesseract() -> Any:
    """Import the optional Tesseract package or raise an actionable error."""
    try:
        import tesseract_decoder
    except ModuleNotFoundError as error:
        if error.name != "tesseract_decoder":
            raise
        raise ModuleNotFoundError(
            "The Tesseract decoder requires the optional 'tesseract-decoder' package.  Install it"
            " with `pip install 'qldpc[tesseract]'`.  The pinned release publishes wheels for"
            " CPython 3.12-3.14 on macOS arm64 and Linux x86-64; for other environments, see"
            " https://github.com/quantumlib/tesseract-decoder#installation"
        ) from error
    return tesseract_decoder


def _get_dem_and_num_errors(
    pcm_or_dem: IntegerArray | stim.DetectorErrorModel,
    error_channel: float | npt.NDArray[np.floating] | Sequence[float] | None,
    error_rate: float | None,
) -> tuple[stim.DetectorErrorModel, int]:
    """Convert an input to Tesseract's DEM while preserving its error indexing."""
    if isinstance(pcm_or_dem, stim.DetectorErrorModel):
        _reject_dem_error_probabilities(error_channel, error_rate)
        return pcm_or_dem, pcm_or_dem.num_errors

    _validate_binary_matrix(pcm_or_dem)
    probabilities = cast(
        npt.NDArray[np.floating],
        _get_matrix_error_channel(pcm_or_dem, error_channel, error_rate),
    )
    dem_arrays = DetectorErrorModelArrays.from_arrays(pcm_or_dem, None, probabilities)
    return dem_arrays.to_dem(), dem_arrays.num_errors


def _validate_binary_matrix(matrix: IntegerArray) -> None:
    """Reject matrix inputs that Stim cannot represent as a binary DEM."""
    if len(matrix.shape) != 2:
        raise ValueError(
            f"A Tesseract parity-check matrix must be two-dimensional, got {matrix.shape}"
        )
    if isinstance(matrix, galois.FieldArray) and type(matrix).order != 2:
        raise ValueError("The Tesseract decoder only supports binary parity-check matrices")
    values = (
        matrix.tocoo().data
        if isinstance(matrix, scipy.sparse.sparray | scipy.sparse.spmatrix)
        else matrix
    )
    array = np.asarray(values)
    if not (np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_)):
        raise ValueError("A Tesseract parity-check matrix must contain binary integers")
    if np.any((array != 0) & (array != 1)):
        raise ValueError("A Tesseract parity-check matrix must contain only 0 and 1")


def _get_detector_order_name(method: TesseractDetectorOrderMethod) -> str:
    """Translate qLDPC's typed detector-order name to Tesseract's enum attribute."""
    if method not in _DETECTOR_ORDER_NAMES:
        options = ", ".join(repr(option) for option in _DETECTOR_ORDER_NAMES)
        raise ValueError(
            f"Unknown Tesseract det_order_method {method!r}; expected one of {options}"
        )
    return _DETECTOR_ORDER_NAMES[method]
