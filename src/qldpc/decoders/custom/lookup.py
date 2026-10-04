# SPDX-License-Identifier: Apache-2.0

"""Lookup-table decoder classes."""

from __future__ import annotations

import collections
import warnings
from collections.abc import Callable, Collection, Iterator, Sequence
from typing import TypeAlias, cast, overload

import galois
import numpy as np
import numpy.typing as npt
import scipy.sparse
import stim

from qldpc import math
from qldpc._util import get_external_caller_stacklevel
from qldpc.math import IntegerArray

from ..common import _erasure_bit_support, with_erasure_bits
from ..construction.specs import decoder_spec
from ..dems import DetectorErrorModelArrays
from ..protocols import ErrorDecoder, ObservableDecoder
from ._lookup_enumeration import _iter_errors_and_syndromes

_LOOKUP_CHUNK_SIZE = 4096


_ErrorVector: TypeAlias = npt.NDArray[np.int_] | Sequence[int]
_ErrorLogProbability: TypeAlias = Callable[[_ErrorVector], float]
_ErrorChannel: TypeAlias = npt.NDArray[np.floating] | Sequence[float] | _ErrorLogProbability | None


# Lookup decoders


class _LookupDecoderBase:
    """Shared implementation of lookup-table decoders.

    See help(LookupDecoder) for details.
    """

    _outputs_observables = False

    def __init__(
        self,
        pcm_or_dem: IntegerArray | stim.DetectorErrorModel,
        max_weight: int,
        *,
        error_channel: (
            npt.NDArray[np.floating]
            | Sequence[float]
            | Callable[[npt.NDArray[np.int_] | Sequence[int]], float]
            | None
        ) = None,
        observable_flip_matrix: IntegerArray | None = None,
        post_select: Collection[int] = (),
        add_erasure_bit: bool | None = None,  # falsy by default
        confidence_ratio: float | None = None,
        probability_cutoff: float = 0,
        symplectic: bool = False,
        penalty_func: Callable[[npt.NDArray[np.int_] | Sequence[int]], float] | None = None,
    ) -> None:
        error_channel = _prepare_error_channel(error_channel, penalty_func)
        if confidence_ratio is not None and not confidence_ratio >= 0:  # also rejects NaN
            raise ValueError("A LookupDecoder confidence_ratio must be a non-negative number")
        if confidence_ratio:  # a positive confidence_ratio signals erasure via the erasure bit
            if add_erasure_bit is False:
                raise ValueError(
                    "A positive confidence_ratio signals erasure with the erasure bit, so it cannot"
                    " be combined with add_erasure_bit=False"
                )
            add_erasure_bit = True  # auto-enable the erasure bit used to signal erasure
        add_erasure_bit = bool(add_erasure_bit)  # a default of None is treated as False

        (
            pcm,
            observable_flip_matrix,
            error_log_probability,
            syndrome_mask,
            default_correction,
            independent_error_channel,
        ) = _organize_lookup_table_initialization_data(
            pcm_or_dem,
            error_channel,
            observable_flip_matrix,
            self._outputs_observables,
            post_select,
            add_erasure_bit,
            probability_cutoff,
            symplectic=symplectic,
        )
        if observable_flip_matrix is not None and error_log_probability is None:
            raise ValueError(
                "Grouping errors by observable flip with a LookupDecoder requires providing a"
                " stim.DetectorErrorModel or error_channel"
            )
        if confidence_ratio and observable_flip_matrix is None:
            raise ValueError(
                "Using a positive confidence_ratio with a LookupDecoder requires grouping errors by"
                " observable flip, which requires a stim.DetectorErrorModel with observables or an"
                " observable_flip_matrix"
            )

        self._save_interface_metadata(pcm, observable_flip_matrix, self._outputs_observables)
        self.syndrome_mask = syndrome_mask
        self.has_erasure_bit = add_erasure_bit
        self.default_correction = default_correction

        # start working on the decoding map from syndrome -> error, stored with packed keys/values
        num_syndrome_bits = pcm.shape[0] if syndrome_mask is None else int(syndrome_mask.sum())
        self._syndrome_packer = _FieldVectorPacker(self.field.order, num_syndrome_bits, pcm.dtype)
        self._prediction_packer = _FieldVectorPacker(
            self.field.order, len(default_correction), default_correction.dtype
        )
        self._packed_syndrome_to_prediction: dict[bytes, bytes] = {}

        # enumerate errors and their syndromes, and use them to build the decoding map
        errors_and_syndromes = _iter_errors_and_syndromes(
            pcm,
            max_weight,
            syndrome_mask,
            symplectic,
            error_channel=independent_error_channel,
            probability_cutoff=probability_cutoff,
        )
        if observable_flip_matrix is None:
            self._build_syndrome_map_from_errors(errors_and_syndromes, error_log_probability)
        else:
            assert error_log_probability is not None
            self._build_syndrome_map_from_observable_flips(
                errors_and_syndromes,
                error_log_probability,
                pcm,
                observable_flip_matrix,
                symplectic,
                confidence_ratio,
            )

    def __len__(self) -> int:
        """The number of entries in this lookup table."""
        return len(self._packed_syndrome_to_prediction)

    def _build_syndrome_map_from_observable_flips(
        self,
        errors_and_syndromes: Iterator[tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]],
        error_log_probability: _ErrorLogProbability,
        pcm: IntegerArray,
        observable_flip_matrix: IntegerArray,
        symplectic: bool,
        confidence_ratio: float | None,
    ) -> None:
        """Populate the lookup table, mapping each syndrome to its most likely observable flip.

        Builds a lookup table that maps each syndrome to an error that induces the most likely
        observable flips (or to representative errors with those flips).
        """

        get_observable_flip = _build_observable_flip_func(pcm, observable_flip_matrix, symplectic)

        # For each "key" = (syndrome, observable_flip) combination, identify:
        # 1. The net log-probability of each key.
        # 2. The most likely error for each key.
        # 3. The log-probability of the most likely error for each key.
        # Probabilities are accumulated in log-space (via logaddexp) to avoid the underflow that
        # would otherwise arise from summing the tiny probabilities of individual errors.
        num_observables = observable_flip_matrix.shape[0]
        observable_flip_packer = _FieldVectorPacker(self.field.order, num_observables, pcm.dtype)
        error_packer = _FieldVectorPacker(self.field.order, pcm.shape[1], pcm.dtype)
        net_log_probs: dict[bytes, dict[bytes, float]] = collections.defaultdict(dict)
        most_likely_errors: dict[tuple[bytes, bytes], bytes] = {}
        most_likely_error_log_probs: dict[tuple[bytes, bytes], float] = {}
        for error, syndrome_array in errors_and_syndromes:
            syndrome = self._syndrome_packer.pack(syndrome_array)
            obs_flip = observable_flip_packer.pack(get_observable_flip(error))
            log_prob = error_log_probability(error)
            net_log_probs[syndrome][obs_flip] = float(
                np.logaddexp(net_log_probs[syndrome].get(obs_flip, -np.inf), log_prob)
            )
            if self._outputs_observables:
                continue  # representative errors are only needed to decode to errors
            # Record the first error for each key (so it always has a representative, even when all
            # of its errors have zero probability), then keep the most likely one thereafter.  A tie
            # in probability resolves toward the lighter error, since enumeration runs from heavy to
            # light and so reaches the lightest error of a tie last.
            key = (syndrome, obs_flip)
            if key not in most_likely_errors or log_prob >= most_likely_error_log_probs[key]:
                most_likely_error_log_probs[key] = log_prob
                most_likely_errors[key] = error_packer.pack(error)

        # Identify the most likely observable_flip for each syndrome, and map the syndrome to the
        # most likely error with that (syndrome, observable_flip) combination.  If a
        # confidence_ratio is set, we instead omit any syndrome whose most likely observable flip is
        # not confidence_ratio times as likely as the rest, so that it decodes to erasure (via
        # default_correction) just like a syndrome that was never enumerated.
        log_confidence_ratio = float(np.log(confidence_ratio)) if confidence_ratio else -np.inf
        for syndrome, obs_flip_to_log_prob in net_log_probs.items():
            most_likely_obs_flip = max(obs_flip_to_log_prob, key=obs_flip_to_log_prob.__getitem__)
            if confidence_ratio:
                log_prob_top = obs_flip_to_log_prob[most_likely_obs_flip]
                other_log_probs = [
                    log_prob
                    for obs_flip, log_prob in obs_flip_to_log_prob.items()
                    if obs_flip != most_likely_obs_flip
                ]
                log_prob_rest = (
                    float(np.logaddexp.reduce(other_log_probs)) if other_log_probs else -np.inf
                )
                # confident iff prob_top >= confidence_ratio * prob_rest (compared in log-space),
                # which ignores competing flips of zero probability at every confidence_ratio
                if log_prob_top < log_confidence_ratio + log_prob_rest:
                    continue  # omit the ambiguous syndrome, leaving it to decode as erasure
            if self._outputs_observables:
                prediction = observable_flip_packer.unpack(most_likely_obs_flip)
            else:
                prediction = error_packer.unpack(most_likely_errors[syndrome, most_likely_obs_flip])
            self._store_prediction(syndrome, prediction)

    def _build_syndrome_map_from_errors(
        self,
        errors_and_syndromes: Iterator[tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]],
        error_log_probability: _ErrorLogProbability | None,
    ) -> None:
        """Populate the lookup table, mapping each syndrome to its likeliest error.

        Errors are enumerated in decreasing weight, so ties in log probability (or, with no error
        channel, all errors) resolve in favor of the lowest-weight error for each syndrome.
        """
        best_log_probabilities: dict[bytes, float] = {}
        for error, syndrome in errors_and_syndromes:
            key = self._syndrome_packer.pack(syndrome)
            if error_log_probability is None:
                self._store_prediction(key, error)
            elif (log_probability := error_log_probability(error)) >= best_log_probabilities.get(
                key, -np.inf
            ):
                best_log_probabilities[key] = log_probability
                self._store_prediction(key, error)

    def _save_interface_metadata(
        self,
        pcm: IntegerArray,
        observable_flip_matrix: IntegerArray | None,
        output_observables: bool,
    ) -> None:
        """Record dimensions and the field so prebuilt-decoder compatibility can be checked."""
        self.num_detectors = pcm.shape[0]
        self.num_observables = (
            observable_flip_matrix.shape[0]
            if output_observables and observable_flip_matrix is not None
            else 0
        )
        self.field = type(pcm) if isinstance(pcm, galois.FieldArray) else galois.GF2

    def _store_prediction(self, packed_syndrome: bytes, prediction: npt.NDArray[np.int_]) -> None:
        """Record a prediction for a packed syndrome, appending an erasure bit if needed."""
        prediction = self._maybe_add_erasure_bit(prediction)
        self._packed_syndrome_to_prediction[packed_syndrome] = self._prediction_packer.pack(
            prediction
        )

    def _maybe_add_erasure_bit(self, error: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Append a trivial (zero) erasure bit to an error if this decoder tracks erasure bits."""
        if not self.has_erasure_bit:
            return error
        return with_erasure_bits(error, False)

    def _decode(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Look up the configured error or observable-flip prediction."""
        retained_syndrome = self._remove_post_selected_bits(syndrome)
        if retained_syndrome is None:
            return self.default_correction.copy()
        packed = self._packed_syndrome_to_prediction.get(
            self._syndrome_packer.pack(retained_syndrome)
        )
        if packed is None:
            return self.default_correction.copy()
        return self._prediction_packer.unpack(packed)

    def _decode_batch(self, syndromes: npt.NDArray[np.integer]) -> npt.NDArray[np.int_]:
        """Look up a batch of syndromes and return unpacked predictions."""
        packed_syndromes, passes_post_selection = self._pack_retained_syndromes(syndromes)
        return self._prediction_packer.unpack_rows(
            self._lookup_packed_predictions(packed_syndromes, passes_post_selection)
        )

    def _lookup_packed_predictions(
        self,
        packed_syndromes: npt.NDArray[np.uint8],
        passes_post_selection: npt.NDArray[np.bool_] | None = None,
    ) -> npt.NDArray[np.uint8]:
        """Map packed syndrome rows directly to packed prediction rows.

        Rows that fail post-selection (False in the passes_post_selection mask) get the default
        prediction.
        """
        packed_syndromes = np.ascontiguousarray(packed_syndromes)
        num_predictions = len(packed_syndromes)
        num_prediction_bytes = self._prediction_packer.num_packed_bytes
        packed_predictions = np.empty((num_predictions, num_prediction_bytes), dtype=np.uint8)
        packed_default = self._prediction_packer.pack(self.default_correction)
        lookup_prediction = self._packed_syndrome_to_prediction.get
        packed_syndrome_dtype = np.dtype((np.void, self._syndrome_packer.num_packed_bytes))
        # Bound temporary Python byte keys while retaining the throughput of bulk conversion.
        for start in range(0, num_predictions, _LOOKUP_CHUNK_SIZE):
            stop = min(start + _LOOKUP_CHUNK_SIZE, num_predictions)
            if self._syndrome_packer.num_packed_bytes:
                # A void view preserves trailing zero bytes; tolist() constructs the keys in C.
                packed_syndromes_chunk = cast(
                    list[bytes],
                    packed_syndromes[start:stop].view(packed_syndrome_dtype).reshape(-1).tolist(),
                )
            else:
                packed_syndromes_chunk = [b""] * (stop - start)
            packed_predictions[start:stop] = np.frombuffer(
                b"".join(
                    [
                        lookup_prediction(packed_syndrome, packed_default)
                        for packed_syndrome in packed_syndromes_chunk
                    ]
                ),
                dtype=np.uint8,
            ).reshape(stop - start, num_prediction_bytes)
        if passes_post_selection is not None:
            packed_predictions[~passes_post_selection] = np.frombuffer(
                packed_default, dtype=np.uint8
            )
        return packed_predictions

    def _pack_retained_syndromes(
        self, syndromes: npt.NDArray[np.integer]
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.bool_] | None]:
        """Pack the retained bits of syndrome rows.

        Also returns a mask of the rows that are trivial on every post-selected bit, or None if this
        decoder does not post-select.  The other rows are absent from the lookup table.
        """
        syndromes = syndromes.view(np.ndarray)
        if syndromes.ndim != 2 or syndromes.shape[1] != self.num_detectors:
            raise ValueError(
                f"Expected syndromes of shape (num_syndromes, {self.num_detectors}),"
                f" but got {syndromes.shape}"
            )
        if self.syndrome_mask is None:
            return self._syndrome_packer.pack_rows(syndromes), None
        passes_post_selection = ~np.any(syndromes[:, ~self.syndrome_mask], axis=1)
        return self._syndrome_packer.pack_rows(
            syndromes[:, self.syndrome_mask]
        ), passes_post_selection

    def _remove_post_selected_bits(
        self, syndrome: npt.NDArray[np.int_]
    ) -> npt.NDArray[np.int_] | None:
        """Drop post-selected bits from a syndrome, or return None if any of them is nonzero."""
        syndrome = syndrome.view(np.ndarray)
        if self.syndrome_mask is None:
            return syndrome
        retained_syndrome = syndrome[self.syndrome_mask]
        if np.count_nonzero(retained_syndrome) != np.count_nonzero(syndrome):
            return None
        return retained_syndrome

    def _get_syndrome_key(self, syndrome: npt.NDArray[np.int_]) -> tuple[int, ...] | None:
        """Return the retained syndrome key, or None when a post-selected bit is nontrivial."""
        retained_syndrome = self._remove_post_selected_bits(syndrome)
        return None if retained_syndrome is None else tuple(retained_syndrome.tolist())

    def _stack_predictions(self, predictions: list[npt.NDArray[np.int_]]) -> npt.NDArray[np.int_]:
        """Stack predictions, one per row, into a 2D array, even if there are no predictions."""
        return np.array(predictions, dtype=self.default_correction.dtype).reshape(
            len(predictions), len(self.default_correction)
        )


class LookupDecoder(_LookupDecoderBase, ErrorDecoder):
    """Decoder based on a lookup table that maps syndromes to errors.

    Accepts a parity check matrix (PCM) or detector error model (DEM) for ``pcm_or_dem``.  If
    provided a DEM, this decoder extracts a PCM, ``error_channel``, and ``observable_flip_matrix``
    from the DEM, which are used as described below.

    In addition to a PCM, this decoder needs to be initialized with some choice of ``max_weight``.
    The decoder enumerates all errors with ``weight <= max_weight`` in order of decreasing weight.
    For each ``error``, the decoder computes the corresponding ``syndrome``, and nominally adds an
    ``syndrome -> error`` entry to the lookup table, overriding any past entry for ``syndrome``.

    The optional ``error_channel`` specifies the physical error distribution in one of two forms.
    An array-like channel contains the independent probability that each primitive error mechanism
    (one column of the PCM) is nonzero.  Over a field of order ``q``, a mechanism with probability
    ``p`` is zero with probability ``1 - p`` and takes each of its ``q - 1`` nonzero values with
    probability ``p / (q - 1)``.  Thus, entries may have different probabilities even though the
    channel factorizes over primitive mechanisms.  With ``symplectic=True``, the X and Z components
    are separate primitive mechanisms and are independent under an array-like channel.

    A callable ``error_channel`` describes a general, potentially correlated distribution.  It is
    called with each complete field-valued error vector considered by the decoder and must return
    the normalized natural log probability ``log(P(error))``.  It may return ``-np.inf`` for an
    impossible error; finite values must be non-positive.  The decoder rejects NaN, positive values,
    and positive infinity, but cannot verify that the full distribution sums to one.  A callable can
    distinguish nonzero field values, represent an arbitrary local Pauli channel, or correlate error
    mechanisms in ways that an array-like channel cannot.

    If an error distribution is supplied, a candidate ``syndrome -> new_error`` entry encountered
    during enumeration overrides a past entry only when the new error is at least as probable.
    Errors are enumerated in decreasing weight, so an equal probability resolves in favor of the
    lighter error.  With no ``error_channel``, every enumerated error is treated as equally likely.

    If provided an ``observable_flip_matrix`` (shape ``num_observables × num_primitive_errors``),
    this decoder maps each syndrome to an error that induces the most likely observable flip for
    that syndrome, which may be different from the single most likely error.  Concretely: errors
    consistent with a given syndrome are grouped by their observable flip value; the total
    probability of each group is the sum of the probabilities of its member errors, restricted to
    the errors of ``weight <= max_weight`` that this decoder enumerates.  This decoder then assigns
    each ``syndrome`` the highest-probability individual ``error`` from the group with the highest
    total probability.

    If provided a ``post_select`` collection of syndrome-bit (i.e., detector) indices, this decoder
    post-selects on those bits being trivial: when constructing the lookup table, it ignores
    syndromes that are nonzero on the post-selected bits, and it drops those bits from the syndrome
    keys in the lookup table.  For consistency with the post-selection options in sinter, syndromes
    passed to ``.decode`` should still contain all syndrome bits.  A syndrome that is
    nonzero on a post-selected bit is one that the lookup table was never given, so it decodes
    identically to a syndrome that was never enumerated.

    If initialized with ``add_erasure_bit=True``, this decoder appends a bit to all decoded errors.
    If asked to decode a syndrome that was not observed when constructing the lookup table, the
    erasure bit is set to 1.  The erasure bit is set to 0 otherwise.

    If initialized with a positive ``confidence_ratio`` (which requires observables, from a
    detector error model or an ``observable_flip_matrix``), this decoder only assigns a syndrome its
    most likely observable flip if that flip is at least ``confidence_ratio`` times as likely as all
    other flips combined.  Otherwise, the syndrome decodes to erasure, so a positive
    ``confidence_ratio`` sets ``add_erasure_bit=True``.  With ``confidence_ratio=np.inf``, every
    syndrome with a competing flip that can occur decodes to erasure.

    A positive ``probability_cutoff`` omits every error whose probability is below the cutoff,
    which allows ignoring very-low-probability events.  When combined with ``confidence_ratio``,
    confidence is computed from the retained errors.  A positive cutoff is not supported with a
    callable ``error_channel``.

    If initialized with ``symplectic=True``, this decoder treats the provided parity check matrix as
    that of a ``QuditCode``, with the first and last half of the columns denoting, respectively, the
    ``[X|Z]`` support of a stabilizer.  Decoded errors are likewise vectors that indicate
    ``[X|Z]`` support.
    """

    def __init__(
        self,
        pcm_or_dem: IntegerArray | stim.DetectorErrorModel,
        max_weight: int,
        *,
        error_channel: (
            npt.NDArray[np.floating]
            | Sequence[float]
            | Callable[[npt.NDArray[np.int_] | Sequence[int]], float]
            | None
        ) = None,
        observable_flip_matrix: IntegerArray | None = None,
        predict_observable_flips: bool = False,
        post_select: Collection[int] = (),
        add_erasure_bit: bool | None = None,
        confidence_ratio: float | None = None,
        probability_cutoff: float = 0,
        symplectic: bool = False,
        penalty_func: Callable[[npt.NDArray[np.int_] | Sequence[int]], float] | None = None,
    ) -> None:
        """Initialize an error lookup table."""
        advice = (
            "use decoders.lookup(max_weight=...).build_observable_decoder(dem) instead"
            if isinstance(pcm_or_dem, stim.DetectorErrorModel)
            else (
                "use ObservableLookupDecoder(matrix, max_weight=..., observable_flip_matrix=...,"
                " error_channel=...).decode_observables(syndrome) instead"
            )
        )
        _warn_deprecated_observable_prediction(predict_observable_flips, advice)
        self._outputs_observables = predict_observable_flips
        self.predict_observable_flips = predict_observable_flips
        self.decode_returns_observables = predict_observable_flips
        super().__init__(
            pcm_or_dem,
            max_weight,
            error_channel=error_channel,
            observable_flip_matrix=observable_flip_matrix,
            post_select=post_select,
            add_erasure_bit=add_erasure_bit,
            confidence_ratio=confidence_ratio,
            probability_cutoff=probability_cutoff,
            symplectic=symplectic,
            penalty_func=penalty_func,
        )

    def decode_errors(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode an error syndrome and return an inferred error."""
        return self._decode(syndrome)

    def decode_errors_batch(self, syndromes: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode a batch of syndromes, one per row, and return inferred errors, one per row.

        The batch must have shape ``(num_syndromes, num_detectors)``.  A syndrome that is missing
        from the lookup table, or that is nonzero on a post-selected bit, is decoded to the default
        correction, as in ``.decode``.
        """
        return self._decode_batch(syndromes)

    decode_batch = decode_errors_batch


class ObservableLookupDecoder(_LookupDecoderBase):
    """Decoder based on a lookup table that maps syndromes directly to observable flips.

    An ObservableLookupDecoder builds the same lookup table as a LookupDecoder, and accepts the same
    options (see help(LookupDecoder)).  However, rather than mapping each syndrome to a
    representative error, an ObservableLookupDecoder maps each syndrome to its most likely
    observable flip, a vector of length ``num_observables`` over the field of the parity check
    matrix.  Decode with the ``.decode_observables`` method.

    ``decoders.lookup(...).build_observable_decoder(dem)`` builds this decoder from a detector error
    model.  A matrix-based instance additionally requires an ``observable_flip_matrix`` whose rows
    specify which errors flip which observables, and an ``error_channel`` for grouping errors.
    If initialized with ``add_erasure_bit=True``, this decoder appends an erasure bit to each
    predicted observable flip.
    """

    _outputs_observables = True

    @overload
    def __init__(
        self,
        pcm_or_dem: stim.DetectorErrorModel,
        max_weight: int,
        *,
        post_select: Collection[int] = (),
        add_erasure_bit: bool | None = None,
        confidence_ratio: float | None = None,
        probability_cutoff: float = 0,
        symplectic: bool = False,
    ) -> None: ...

    @overload
    def __init__(
        self,
        pcm_or_dem: IntegerArray,
        max_weight: int,
        *,
        observable_flip_matrix: IntegerArray,
        error_channel: (
            npt.NDArray[np.floating]
            | Sequence[float]
            | Callable[[npt.NDArray[np.int_] | Sequence[int]], float]
            | None
        ) = None,
        post_select: Collection[int] = (),
        add_erasure_bit: bool | None = None,
        confidence_ratio: float | None = None,
        probability_cutoff: float = 0,
        symplectic: bool = False,
        penalty_func: Callable[[npt.NDArray[np.int_] | Sequence[int]], float] | None = None,
    ) -> None: ...

    def __init__(
        self,
        pcm_or_dem: IntegerArray | stim.DetectorErrorModel,
        max_weight: int,
        *,
        observable_flip_matrix: IntegerArray | None = None,
        error_channel: (
            npt.NDArray[np.floating]
            | Sequence[float]
            | Callable[[npt.NDArray[np.int_] | Sequence[int]], float]
            | None
        ) = None,
        post_select: Collection[int] = (),
        add_erasure_bit: bool | None = None,
        confidence_ratio: float | None = None,
        probability_cutoff: float = 0,
        symplectic: bool = False,
        penalty_func: Callable[[npt.NDArray[np.int_] | Sequence[int]], float] | None = None,
    ) -> None:
        super().__init__(
            pcm_or_dem,
            max_weight,
            error_channel=error_channel,
            observable_flip_matrix=observable_flip_matrix,
            post_select=post_select,
            add_erasure_bit=add_erasure_bit,
            confidence_ratio=confidence_ratio,
            probability_cutoff=probability_cutoff,
            symplectic=symplectic,
            penalty_func=penalty_func,
        )

    def decode_observables(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode a syndrome and return predicted observable flips."""
        return self._decode(syndrome)

    def decode_observables_batch(self, syndromes: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode a batch of syndromes, one per row, and return observable flips, one per row.

        The batch must have shape ``(num_syndromes, num_detectors)``.  A syndrome that is missing
        from the lookup table, or that is nonzero on a post-selected bit, is decoded to the default
        prediction, as in ``.decode_observables``.
        """
        return self._decode_batch(syndromes)

    def decode_shots_bit_packed(
        self, bit_packed_detection_event_data: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.uint8]:
        """Predict little-endian bit-packed flips from bit-packed binary detection events.

        Observable flips occupy ``ceil(num_observables / 8)`` bytes per row.  When configured, an
        erasure is signalled in one additional whole byte, as required by Sinter.
        """
        if self.field.order != 2:
            raise ValueError("Bit-packed lookup decoding is only available over GF(2)")
        packed_shots = np.ascontiguousarray(bit_packed_detection_event_data, dtype=np.uint8)
        num_detector_bytes = -(-self.num_detectors // 8)
        if packed_shots.ndim != 2 or packed_shots.shape[1] != num_detector_bytes:
            raise ValueError(
                f"Expected bit-packed syndromes of shape (num_syndromes, {num_detector_bytes}),"
                f" but got {packed_shots.shape}"
            )
        if self.syndrome_mask is None:
            packed_predictions = self._lookup_packed_predictions(packed_shots)
        else:
            shots = np.unpackbits(packed_shots, count=self.num_detectors, bitorder="little", axis=1)
            packed_predictions = self._lookup_packed_predictions(
                *self._pack_retained_syndromes(shots)
            )
        if not self.has_erasure_bit:
            return packed_predictions

        erasure_byte, erasure_bit = divmod(self.num_observables, 8)
        erased = (packed_predictions[:, erasure_byte] >> erasure_bit) & 1
        # clear the erasure bit from the observable flips
        packed_predictions[:, erasure_byte] &= (1 << erasure_bit) - 1
        num_observable_bytes = -(-self.num_observables // 8)
        return np.column_stack([packed_predictions[:, :num_observable_bytes], erased])


# Weighted lookup decoders


class _WeightedLookupDecoderBase(_LookupDecoderBase):
    """Shared implementation of weighted lookup-table decoders.

    See help(WeightedLookupDecoder) for details.
    """

    def __init__(
        self,
        pcm_or_dem: IntegerArray | stim.DetectorErrorModel,
        max_weight: int,
        *,
        observable_flip_matrix: IntegerArray | None = None,
        post_select: Collection[int] = (),
        add_erasure_bit: bool = False,
        symplectic: bool = False,
    ) -> None:
        pcm, observable_flip_matrix, _, syndrome_mask, default_correction, _ = (
            _organize_lookup_table_initialization_data(
                pcm_or_dem,
                None,
                observable_flip_matrix,
                self._outputs_observables,
                post_select,
                add_erasure_bit,
                symplectic=symplectic,
            )
        )

        self._save_interface_metadata(pcm, observable_flip_matrix, self._outputs_observables)
        self.syndrome_mask = syndrome_mask
        self.has_erasure_bit = add_erasure_bit
        self.default_correction = default_correction
        self.symplectic = symplectic

        # Record all errors consistent with each syndrome, together with the output to return if
        # that error is selected: an observable-flip prediction if requested (else the error
        # itself), with a trivial erasure bit appended.  The output is precomputed here so that
        # decode() only has to select the minimum-penalty candidate error.
        self.syndrome_to_candidates: dict[
            tuple[int, ...], list[tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]]
        ] = collections.defaultdict(list)
        get_observable_flip = None
        if self._outputs_observables:
            assert observable_flip_matrix is not None  # primarily for type-checking reasons
            get_observable_flip = _build_observable_flip_func(
                pcm, observable_flip_matrix, symplectic
            )
        for error, syndrome in _iter_errors_and_syndromes(
            pcm, max_weight, syndrome_mask, symplectic
        ):
            output = (
                error
                if get_observable_flip is None
                else get_observable_flip(error).astype(pcm.dtype)
            )
            self.syndrome_to_candidates[tuple(syndrome.tolist())].append(
                (error, self._maybe_add_erasure_bit(output))
            )

    def __len__(self) -> int:
        """The number of entries in this lookup table."""
        return len(self.syndrome_to_candidates)

    def _decode_weighted(
        self,
        syndrome: npt.NDArray[np.int_],
        penalty_func: Callable[[npt.NDArray[np.int_]], float] | None = lambda vec: int(
            np.count_nonzero(vec)
        ),
    ) -> npt.NDArray[np.int_]:
        """Look up the minimum-penalty configured prediction."""
        key = self._get_syndrome_key(syndrome)
        if key is None or key not in self.syndrome_to_candidates:
            return self.default_correction.copy()

        candidates = self.syndrome_to_candidates[key]
        if penalty_func is None:
            output = candidates[-1][1]
        else:
            # an equal penalty resolves in favor of the lighter candidate error
            output = min(
                candidates,
                key=lambda candidate: (
                    penalty_func(candidate[0]),
                    _error_weight(candidate[0], self.symplectic),
                ),
            )[1]
        return output.copy()


class WeightedLookupDecoder(_WeightedLookupDecoderBase, LookupDecoder):
    """Lookup-table decoder that maps syndromes to errors, with a penalty function chosen later.

    A WeightedLookupDecoder is a LookupDecoder that, when initialized, records *all* errors of
    ``weight <= max_weight`` that are consistent with each syndrome.  The WeightedLookupDecoder then
    minimizes a penalty function that is provided to the ``.decode`` method.  A
    WeightedLookupDecoder can thereby be initialized once, and subsequently asked to decode with
    different penalty functions.  The default penalty function is the Hamming weight of an error.

    The ``pcm_or_dem``, ``max_weight``, ``observable_flip_matrix``, ``post_select``,
    ``add_erasure_bit``, and ``symplectic`` options behave as they do for a LookupDecoder; see
    help(LookupDecoder).
    """

    def __init__(
        self,
        pcm_or_dem: IntegerArray | stim.DetectorErrorModel,
        max_weight: int,
        *,
        observable_flip_matrix: IntegerArray | None = None,
        predict_observable_flips: bool = False,
        post_select: Collection[int] = (),
        add_erasure_bit: bool = False,
        symplectic: bool = False,
    ) -> None:
        _warn_deprecated_observable_prediction(
            predict_observable_flips,
            "use WeightedObservableLookupDecoder(...).decode_observables(syndrome, penalty_func)"
            " instead",
        )
        self._outputs_observables = predict_observable_flips
        self.predict_observable_flips = predict_observable_flips
        self.decode_returns_observables = predict_observable_flips
        super().__init__(
            pcm_or_dem,
            max_weight,
            observable_flip_matrix=observable_flip_matrix,
            post_select=post_select,
            add_erasure_bit=add_erasure_bit,
            symplectic=symplectic,
        )

    def decode_errors(
        self,
        syndrome: npt.NDArray[np.int_],
        penalty_func: Callable[[npt.NDArray[np.int_]], float] | None = lambda vec: int(
            np.count_nonzero(vec)
        ),
    ) -> npt.NDArray[np.int_]:
        """Decode an error syndrome and return an inferred error."""
        return self._decode_weighted(syndrome, penalty_func)

    def decode_errors_batch(
        self,
        syndromes: npt.NDArray[np.int_],
        penalty_func: Callable[[npt.NDArray[np.int_]], float] | None = lambda vec: int(
            np.count_nonzero(vec)
        ),
    ) -> npt.NDArray[np.int_]:
        """Decode a batch of syndromes, one per row, and return inferred errors."""
        return self._stack_predictions(
            [self._decode_weighted(syndrome, penalty_func) for syndrome in syndromes]
        )

    decode_batch = decode_errors_batch

    def decode(
        self,
        syndrome: npt.NDArray[np.int_],
        penalty_func: Callable[[npt.NDArray[np.int_]], float] | None = lambda vec: int(
            np.count_nonzero(vec)
        ),
    ) -> npt.NDArray[np.int_]:
        """Decode an error syndrome and return an inferred error (alias for decode_errors)."""
        return self.decode_errors(syndrome, penalty_func)


class WeightedObservableLookupDecoder(_WeightedLookupDecoderBase):
    """Weighted lookup-table decoder that maps syndromes to observable flips.

    A WeightedObservableLookupDecoder records the same candidate errors as a WeightedLookupDecoder
    (see help(WeightedLookupDecoder)).  Its ``.decode_observables`` method selects the candidate
    error that minimizes a provided penalty function, and returns the observable flip that this
    error induces.  A WeightedObservableLookupDecoder is built from a detector error model, or from
    a parity check matrix together with an ``observable_flip_matrix``.
    """

    _outputs_observables = True

    @overload
    def __init__(
        self,
        pcm_or_dem: stim.DetectorErrorModel,
        max_weight: int,
        *,
        post_select: Collection[int] = (),
        add_erasure_bit: bool = False,
        symplectic: bool = False,
    ) -> None: ...

    @overload
    def __init__(
        self,
        pcm_or_dem: IntegerArray,
        max_weight: int,
        *,
        observable_flip_matrix: IntegerArray,
        post_select: Collection[int] = (),
        add_erasure_bit: bool = False,
        symplectic: bool = False,
    ) -> None: ...

    def __init__(
        self,
        pcm_or_dem: IntegerArray | stim.DetectorErrorModel,
        max_weight: int,
        *,
        observable_flip_matrix: IntegerArray | None = None,
        post_select: Collection[int] = (),
        add_erasure_bit: bool = False,
        symplectic: bool = False,
    ) -> None:
        super().__init__(
            pcm_or_dem,
            max_weight,
            observable_flip_matrix=observable_flip_matrix,
            post_select=post_select,
            add_erasure_bit=add_erasure_bit,
            symplectic=symplectic,
        )

    def decode_observables(
        self,
        syndrome: npt.NDArray[np.int_],
        penalty_func: Callable[[npt.NDArray[np.int_]], float] | None = lambda vec: int(
            np.count_nonzero(vec)
        ),
    ) -> npt.NDArray[np.int_]:
        """Decode a syndrome and return predicted observable flips."""
        return self._decode_weighted(syndrome, penalty_func)

    def decode_observables_batch(
        self,
        syndromes: npt.NDArray[np.int_],
        penalty_func: Callable[[npt.NDArray[np.int_]], float] | None = lambda vec: int(
            np.count_nonzero(vec)
        ),
    ) -> npt.NDArray[np.int_]:
        """Decode a batch of syndromes, one per row, and return predicted observable flips."""
        return self._stack_predictions(
            [self._decode_weighted(syndrome, penalty_func) for syndrome in syndromes]
        )


# Decoder specifications


@_erasure_bit_support("lookup", supported=True)
def _get_decoder_lookup(
    pcm_or_dem: IntegerArray | stim.DetectorErrorModel, **decoder_args: object
) -> LookupDecoder:
    """Build a lookup-table decoder.

    The options, including the required ``max_weight``, an independent or callable correlated
    ``error_channel``, and optional erasure, confidence, probability-cutoff, post-selection, and
    symplectic options, are those of :class:`~qldpc.decoders.custom.lookup.LookupDecoder`.

    Returns:
        A :class:`~qldpc.decoders.custom.lookup.LookupDecoder` that maps syndromes to representative
        errors.

    ``add_erasure_bit=True`` appends a flag for syndromes absent from the table.  A positive
    ``confidence_ratio`` also enables the flag and erases ambiguous syndromes.
    """
    return LookupDecoder(pcm_or_dem, **decoder_args)  # type: ignore[arg-type]


def _get_observable_decoder_lookup(
    dem: stim.DetectorErrorModel, **decoder_args: object
) -> ObservableDecoder:
    """Build a lookup table that maps DEM syndromes directly to observable flips."""
    return ObservableLookupDecoder(dem, **decoder_args)  # type: ignore[call-overload]


_LOOKUP_SPEC_DOC = """Configure a bounded-weight lookup-table decoder.

Args:
    max_weight: Maximum error weight to enumerate; required to bound construction.
    error_channel: Independent error probabilities for each matrix column, or a callable
        returning the normalized log probability of a complete error.  A DEM supplies its
        own probabilities and does not accept this option.
    observable_flip_matrix: Observable values of matrix-column errors.  A DEM supplies its
        own observables.  An error decoder with observables selects an error from the most
        likely observable class, which need not contain the most likely individual error.
    post_select: Syndrome indices that must be zero; other syndromes decode as absent.
    add_erasure_bit: Append a flag for absent syndromes, or None to enable it automatically
        when ``confidence_ratio`` is positive.
    confidence_ratio: Minimum likelihood ratio of the best observable class to all others;
        an ambiguous syndrome is erased.  Requires observable metadata.
    probability_cutoff: Ignore errors below this probability; not supported with a callable
        ``error_channel``.
    symplectic: Treat ``[X|Z]`` columns as two error components per qudit.

Returns:
    A decoder specification.  ``build(pcm_or_dem)`` returns a
    :class:`~qldpc.decoders.custom.lookup.LookupDecoder` that infers an error.
    ``build_observable_decoder(dem)`` returns an
    :class:`~qldpc.decoders.custom.lookup.ObservableLookupDecoder` that predicts observable
    flips natively for a binary DEM.

See :class:`~qldpc.decoders.custom.lookup.LookupDecoder` for error-channel and field
conventions, and :class:`~qldpc.decoders.custom.lookup.ObservableLookupDecoder` for the
observable output contract.
"""


lookup = decoder_spec(
    "lookup",
    _get_decoder_lookup,
    _get_observable_decoder_lookup,
    signature_source=LookupDecoder,
    exclude=frozenset({"predict_observable_flips"}),
    doc=_LOOKUP_SPEC_DOC,
)


# Lookup-table construction


def _organize_lookup_table_initialization_data(
    pcm_or_dem: IntegerArray | stim.DetectorErrorModel,
    error_channel: _ErrorChannel,
    observable_flip_matrix: IntegerArray | None,
    output_observables: bool,
    post_select: Collection[int],
    add_erasure_bit: bool,
    probability_cutoff: float = 0,
    *,
    symplectic: bool = False,
) -> tuple[
    IntegerArray,
    IntegerArray | None,
    _ErrorLogProbability | None,
    npt.NDArray[np.bool_] | None,
    npt.NDArray[np.int_],
    npt.NDArray[np.floating] | None,
]:
    """Organize and validate the inputs to a LookupDecoder."""
    if not np.isfinite(probability_cutoff) or not 0 <= probability_cutoff <= 1:
        raise ValueError(
            "A LookupDecoder probability_cutoff must be a finite probability between 0 and 1,"
            " inclusive"
        )
    if isinstance(pcm_or_dem, stim.DetectorErrorModel):
        if symplectic:
            raise ValueError(
                "A LookupDecoder cannot be symplectic when built from a stim.DetectorErrorModel,"
                " whose error mechanisms are not [X|Z] qudit errors"
            )
        if error_channel is not None or observable_flip_matrix is not None:
            raise ValueError(
                "Cannot specify an error_channel or observable_flip_matrix when providing a"
                " stim.DetectorErrorModel to a LookupDecoder"
            )
        dem_arrays = DetectorErrorModelArrays(pcm_or_dem, simplify=False)
        pcm = dem_arrays.detector_flip_matrix
        error_channel = dem_arrays.error_probs
        # errors are grouped by observable flip if there are observables, or if predicting
        # observable flips, which are trivial (empty) if there are no observables
        if dem_arrays.num_observables > 0 or output_observables:
            observable_flip_matrix = dem_arrays.observable_flip_matrix
    else:
        pcm = pcm_or_dem

    independent_error_channel: npt.NDArray[np.floating] | None = None
    error_log_probability: _ErrorLogProbability | None = None
    if callable(error_channel):
        error_log_probability = error_channel
    elif error_channel is not None:
        independent_error_channel = np.asarray(error_channel, dtype=float)
        expected_shape = (pcm.shape[1],)
        if independent_error_channel.shape != expected_shape:
            raise ValueError(
                f"A LookupDecoder error_channel must have shape {expected_shape}, but got"
                f" {independent_error_channel.shape}"
            )
        if not np.all((0 <= independent_error_channel) & (independent_error_channel <= 1)):
            raise ValueError(
                "An array-like LookupDecoder error_channel must contain finite probabilities"
                " between 0 and 1, inclusive"
            )
        error_log_probability = _build_error_log_probability(
            independent_error_channel,
            (type(pcm) if isinstance(pcm, galois.FieldArray) else galois.GF2).order,
        )
    if probability_cutoff and independent_error_channel is None:
        raise ValueError(
            "A positive LookupDecoder probability_cutoff requires a stim.DetectorErrorModel or"
            " an array-like independent error_channel"
        )

    # build the mask of syndrome bits to keep (None if not post-selecting)
    syndrome_mask: npt.NDArray[np.bool_] | None = None
    if len(post_select):
        syndrome_mask = np.ones(pcm.shape[0], dtype=bool)
        syndrome_mask[list(post_select)] = False

    # build the default output returned for syndromes absent from the lookup table
    if output_observables:
        if observable_flip_matrix is None:
            raise ValueError(
                "A lookup decoder that predicts observable flips requires an"
                " observable_flip_matrix when it is built from a parity check matrix"
            )
        output_length = observable_flip_matrix.shape[0]
    else:
        output_length = pcm.shape[1]
    default_correction = np.zeros(output_length, dtype=pcm.dtype)
    if add_erasure_bit:
        default_correction = np.hstack([default_correction, np.ones(1, dtype=pcm.dtype)])

    return (
        pcm,
        observable_flip_matrix,
        error_log_probability,
        syndrome_mask,
        default_correction,
        independent_error_channel,
    )


def _prepare_error_channel(
    error_channel: _ErrorChannel,
    penalty_func: _ErrorLogProbability | None,
) -> _ErrorChannel:
    """Validate a callable error channel, or replace a deprecated penalty function with one."""
    if penalty_func is not None:
        if error_channel is not None:
            raise ValueError(
                "Cannot specify both an error_channel and a penalty_func in a LookupDecoder"
            )
        warnings.warn(
            "LookupDecoder penalty_func is deprecated; pass a callable error_channel that returns"
            " the full error log probability instead",
            DeprecationWarning,
            stacklevel=get_external_caller_stacklevel(),
        )
        # legacy penalties are not validated as normalized log probabilities
        return lambda error: -float(penalty_func(error))

    if not callable(error_channel):
        return error_channel
    get_log_probability = error_channel

    def error_log_probability(error: _ErrorVector) -> float:
        """Return the log probability of an error, rejecting invalid values."""
        log_probability = float(get_log_probability(error))
        if not log_probability <= 0:  # also rejects NaN
            raise ValueError(
                "A callable LookupDecoder error_channel must return a log probability that is"
                " non-positive or -inf"
            )
        return log_probability

    return error_log_probability


def _build_observable_flip_func(
    pcm: IntegerArray,
    observable_flip_matrix: IntegerArray,
    symplectic: bool,
) -> Callable[[npt.NDArray[np.int_]], npt.NDArray[np.int_]]:
    """Build the map that takes an error to the observable flips that it induces.

    The observable flip matrix is interpreted over the same field as the parity check matrix,
    which is GF(2) unless the parity check matrix is a galois.FieldArray.  Plain integer entries
    are reduced modulo the order of a prime field; for an extension field, they must already be
    valid integer representations of field elements.  A galois.FieldArray over a different field
    is rejected: the errors that get enumerated take their values from the parity check matrix's
    field, so an observable over any other field cannot say what they flip.

    With ``symplectic=True``, an error assigns both an X and a Z component to each qudit, and
    the flip that it induces in an observable is their symplectic product,
    ``observable @ symplectic_conjugate(error)``.  That product is obtained by multiplying the
    error by -symplectic_conjugate(observable_flip_matrix), in the same way that
    _iter_errors_and_syndromes obtains a syndrome from a parity check matrix.
    """
    field = type(pcm) if isinstance(pcm, galois.FieldArray) else galois.GF2
    if isinstance(observable_flip_matrix, galois.FieldArray) and (
        type(observable_flip_matrix) is not field
    ):
        raise ValueError(
            f"An observable flip matrix over {type(observable_flip_matrix).name} cannot be"
            f" paired with a parity check matrix over {field.name}"
        )

    if not symplectic and field.is_prime_field:
        # A prime field is the integers modulo its order, so the product can be taken over the
        # integers, which is faster than field arithmetic and keeps a sparse matrix sparse.
        #
        # The cast widens a narrow dtype, whose own wrap-around does not commute with reducing
        # modulo an odd order.
        integer_matrix = (
            observable_flip_matrix.view(np.ndarray)
            if isinstance(observable_flip_matrix, galois.FieldArray)
            else observable_flip_matrix
        ).astype(int)
        order = field.order

        def get_integer_flip(error: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            """Map an error to the observable flips that it induces."""
            return np.asarray(integer_matrix @ error) % order

        return get_integer_flip

    dense_matrix = (
        observable_flip_matrix.todense()
        if isinstance(observable_flip_matrix, scipy.sparse.spmatrix | scipy.sparse.sparray)
        else observable_flip_matrix
    )
    matrix_values = np.asarray(dense_matrix, dtype=int)
    if field.is_prime_field:
        matrix_values = matrix_values % field.order
    matrix = field(matrix_values)
    if symplectic:
        matrix = -math.symplectic_conjugate(matrix)

    def get_field_flip(error: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Map an error to the observable flips that it induces."""
        return (matrix @ error.view(field)).view(np.ndarray)

    return get_field_flip


def _build_error_log_probability(
    error_channel: npt.NDArray[np.floating],
    field_order: int = 2,
) -> _ErrorLogProbability:
    """Construct a full-error log probability from independent mechanism probabilities."""
    with np.errstate(divide="ignore"):  # a probability of 0 or 1 yields a -inf log, which is ok
        log_probs = np.log(error_channel / (field_order - 1))
        log_non_probs = np.log(1 - error_channel)

    def error_log_probability(error: _ErrorVector) -> float:
        """Return the full independent-channel log probability of an error."""
        events = np.asarray(error).astype(bool)
        return float(np.sum(log_probs[events]) + np.sum(log_non_probs[~events]))

    return error_log_probability


# Supporting objects


class _FieldVectorPacker:
    """Pack fixed-length vectors over a finite field into compact, hashable bytes.

    Over GF(2), vectors are bit-packed with np.packbits.  Over any other field, each entry is stored
    as the smallest unsigned integer that holds the integer representation of a field element that
    galois uses, which is lossless for both prime and extension fields.  Fields with more than 2**64
    elements are out of scope, since _iter_errors_and_syndromes builds a tuple of every element.
    """

    def __init__(self, field_order: int, vector_length: int, unpacked_dtype: npt.DTypeLike) -> None:
        self.field_order = field_order
        self.vector_length = vector_length
        self.unpacked_dtype = np.dtype(unpacked_dtype)
        self.element_dtype = np.dtype(np.min_scalar_type(field_order - 1)).newbyteorder("<")
        self.num_packed_bytes = (
            -(-vector_length // 8)
            if field_order == 2
            else vector_length * self.element_dtype.itemsize
        )

    def pack(self, vector: npt.NDArray[np.integer]) -> bytes:
        """Pack a vector of field elements."""
        if self.field_order == 2:
            return np.packbits(vector, bitorder="little").tobytes()
        return np.asarray(vector, dtype=self.element_dtype).tobytes()

    def unpack(self, packed: bytes) -> npt.NDArray[np.int_]:
        """Unpack a vector into a new array."""
        if self.field_order == 2:
            buffer = np.frombuffer(packed, dtype=np.uint8)
            bits = np.unpackbits(buffer, count=self.vector_length, bitorder="little")
            return bits.astype(self.unpacked_dtype)
        return np.frombuffer(packed, dtype=self.element_dtype).astype(self.unpacked_dtype)

    def pack_rows(self, vectors: npt.NDArray[np.integer]) -> npt.NDArray[np.uint8]:
        """Pack a two-dimensional array of vectors, one vector per row."""
        if self.field_order == 2:
            return np.packbits(vectors, bitorder="little", axis=1)
        elements = np.ascontiguousarray(vectors, dtype=self.element_dtype)
        return elements.view(np.uint8).reshape(len(vectors), self.num_packed_bytes)

    def unpack_rows(self, packed: npt.NDArray[np.uint8]) -> npt.NDArray[np.int_]:
        """Unpack fixed-width byte rows into vectors, one vector per row."""
        if self.field_order == 2:
            bits = np.unpackbits(packed, count=self.vector_length, bitorder="little", axis=1)
            return bits.astype(self.unpacked_dtype)
        elements = np.ascontiguousarray(packed).view(self.element_dtype)
        return elements.reshape(len(packed), self.vector_length).astype(self.unpacked_dtype)


def _error_weight(error: npt.NDArray[np.int_], symplectic: bool) -> int:
    """The weight of an error: the number of qudits, or of bits, that it addresses nontrivially.

    This is the weight that ``_iter_errors_and_syndromes`` enumerates by, so ranking errors by it
    keeps the lighter of two that a penalty scores equally.  A symplectic error assigns both an X
    and a Z component to each qudit, and a qudit carrying both counts once, so the number of
    nonzero entries would count it twice.
    """
    if symplectic:
        return int(math.symplectic_weight(np.asarray(error)))
    return int(np.count_nonzero(error))


# Deprecated compatibility helpers


def _warn_deprecated_observable_prediction(enabled: bool, advice: str) -> None:
    """Warn about the legacy mode in which an error decoder predicts observable flips."""
    if enabled:
        warnings.warn(
            f"predict_observable_flips=True is deprecated; {advice}",
            DeprecationWarning,
            stacklevel=get_external_caller_stacklevel(),
        )
