# SPDX-License-Identifier: Apache-2.0

"""Lookup-table decoder classes."""

from __future__ import annotations

import collections
import itertools
import warnings
from collections.abc import Callable, Collection, Iterator, MutableMapping, Sequence
from typing import cast, overload

import galois
import numpy as np
import numpy.typing as npt
import scipy.sparse
import stim

from qldpc import math
from qldpc._util import get_external_caller_stacklevel
from qldpc.math import IntegerArray

from ..common import _erasure_bit_support, with_erasure_bits
from ..dems import DetectorErrorModelArrays
from ..protocols import ErrorDecoder, ObservableDecoder

_LOOKUP_BATCH_SIZE = 4096


class _LookupDecoderBase:
    """Shared implementation of lookup-table decoders.

    See help(LookupDecoder) for details.
    """

    def __init__(
        self,
        pcm_or_dem: IntegerArray | stim.DetectorErrorModel,
        max_weight: int,
        *,
        error_channel: npt.NDArray[np.floating] | Sequence[float] | None = None,
        penalty_func: Callable[[npt.NDArray[np.int_] | Sequence[int]], float] | None = None,
        observable_flip_matrix: IntegerArray | None = None,
        predict_observable_flips: bool = False,
        post_select: Collection[int] = (),
        add_erasure_bit: bool | None = None,  # falsy by default
        confidence_ratio: float | None = None,
        symplectic: bool = False,
    ) -> None:
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

        pcm, observable_flip_matrix, penalty_func, syndrome_mask, default_correction = (
            self._organize_lookup_table_initialization_data(
                pcm_or_dem,
                error_channel,
                penalty_func,
                observable_flip_matrix,
                predict_observable_flips,
                post_select,
                add_erasure_bit,
            )
        )
        if observable_flip_matrix is not None and penalty_func is None:
            raise ValueError(
                "Grouping errors by observable flip with a LookupDecoder requires providing a"
                " stim.DetectorErrorModel, error_channel, or penalty_func"
            )
        if confidence_ratio and observable_flip_matrix is None:
            raise ValueError(
                "Using a positive confidence_ratio with a LookupDecoder requires grouping errors by"
                " observable flip, which requires a stim.DetectorErrorModel with observables or an"
                " observable_flip_matrix"
            )

        # save attributes; decode_returns_observables declares whether a decode method returns
        # observable flips, which it does with the deprecated predict_observable_flips=True
        self.predict_observable_flips = predict_observable_flips
        self.decode_returns_observables = predict_observable_flips
        self._save_interface_metadata(pcm, observable_flip_matrix, predict_observable_flips)
        self.syndrome_mask = syndrome_mask
        self.has_erasure_bit = add_erasure_bit
        self.default_correction = default_correction

        # start working on the decoding map from syndrome -> error, stored with packed keys/values
        num_syndrome_bits = pcm.shape[0] if syndrome_mask is None else int(syndrome_mask.sum())
        self._syndrome_codec = _VectorCodec(self.field.order, num_syndrome_bits, pcm.dtype)
        self._output_codec = _VectorCodec(
            self.field.order, len(default_correction), default_correction.dtype
        )
        self._syndrome_to_error: dict[bytes, bytes] = {}
        self._syndrome_to_error_view = _PackedLookupTable(self)

        if observable_flip_matrix is None:
            self._build_syndrome_map_from_errors(
                pcm, max_weight, penalty_func, syndrome_mask, symplectic
            )
        else:
            assert penalty_func is not None  # primarily for type-checking reasons
            self._build_syndrome_map_from_observable_flips(
                pcm,
                max_weight,
                penalty_func,
                observable_flip_matrix,
                predict_observable_flips,
                syndrome_mask,
                confidence_ratio,
                symplectic,
            )

    def _build_syndrome_map_from_errors(
        self,
        pcm: IntegerArray,
        max_weight: int,
        penalty_func: Callable[[npt.NDArray[np.int_] | Sequence[int]], float] | None,
        syndrome_mask: npt.NDArray[np.bool_] | None,
        symplectic: bool,
    ) -> None:
        """Populate syndrome_to_error, mapping each syndrome to its likeliest error.

        Errors are enumerated in decreasing weight, so ties in penalty (or, with no penalty
        function, all errors) resolve in favor of the lowest-weight error for each syndrome.
        """
        error_penalty: dict[bytes, float] = {}
        for error, syndrome in _LookupDecoderBase._iter_error_and_syndrome_arrays(
            pcm, max_weight, syndrome_mask, symplectic
        ):
            key = self._syndrome_codec.pack_trusted(syndrome)
            if penalty_func is None:
                self._store(key, error)
            elif (error_weight := penalty_func(error)) <= error_penalty.get(key, np.inf):
                error_penalty[key] = error_weight
                self._store(key, error)

    def _store(self, key: bytes, prediction: npt.NDArray[np.int_]) -> None:
        """Record a packed table entry, appending an erasure bit if needed."""
        prediction = self._maybe_add_erasure_bit(prediction)
        self._syndrome_to_error[key] = self._output_codec.pack_trusted(prediction)

    @property
    def syndrome_to_error(
        self,
    ) -> MutableMapping[tuple[int, ...], npt.NDArray[np.int_]]:
        """A mutable mapping from syndromes to the predictions of this lookup table.

        Keys are tuples of syndrome entries (excluding post-selected bits), and values are
        predictions (errors or observable flips, with an erasure bit if configured).  The table is
        stored in a compact packed format, so this view unpacks entries on read: a value retrieved
        from this mapping is a fresh copy, and editing it in place does not affect decoding.  To
        change an entry, assign the edited array back to the mapping.  Assigning and deleting
        entries do affect decoding.
        """
        return self._syndrome_to_error_view

    def _save_interface_metadata(
        self,
        pcm: IntegerArray,
        observable_flip_matrix: IntegerArray | None,
        predict_observable_flips: bool,
    ) -> None:
        """Record dimensions and the field so prebuilt-decoder compatibility can be checked."""
        self.num_detectors = pcm.shape[0]
        self.num_observables = (
            observable_flip_matrix.shape[0]
            if predict_observable_flips and observable_flip_matrix is not None
            else 0
        )
        self.field = type(pcm) if isinstance(pcm, galois.FieldArray) else galois.GF2

    def _build_syndrome_map_from_observable_flips(
        self,
        pcm: IntegerArray,
        max_weight: int,
        penalty_func: Callable[[npt.NDArray[np.int_] | Sequence[int]], float],
        observable_flip_matrix: IntegerArray,
        predict_observable_flips: bool,
        syndrome_mask: npt.NDArray[np.bool_] | None,
        confidence_ratio: float | None,
        symplectic: bool,
    ) -> None:
        """Populate syndrome_to_error, mapping each syndrome to its most likely observable flip.

        Builds a lookup table that maps each syndrome to an error that induces the most likely
        observable flips (or, if predict_observable_flips, to the observable flips themselves).
        """

        get_observable_flip = _LookupDecoderBase._build_observable_flip_func(
            pcm, observable_flip_matrix, symplectic
        )

        # For each "key" = (syndrome, observable_flip) combination, identify:
        # 1. The net log-probability of each key.
        # 2. The most likely error for each key.
        # 3. The log-probability of the most likely error for each key.
        # Probabilities are accumulated in log-space (via logaddexp) to avoid the underflow that
        # would otherwise arise from summing the tiny probabilities of individual errors.
        num_observables = observable_flip_matrix.shape[0]
        flip_codec = _VectorCodec(self.field.order, num_observables, pcm.dtype)
        error_codec = _VectorCodec(self.field.order, pcm.shape[1], pcm.dtype)
        net_log_probs: dict[bytes, dict[bytes, float]] = collections.defaultdict(dict)
        most_likely_errors: dict[tuple[bytes, bytes], bytes] = {}
        most_likely_error_log_probs: dict[tuple[bytes, bytes], float] = {}
        for error, syndrome_array in _LookupDecoderBase._iter_error_and_syndrome_arrays(
            pcm, max_weight, syndrome_mask, symplectic
        ):
            syndrome = self._syndrome_codec.pack_trusted(syndrome_array)
            obs_flip = flip_codec.pack_trusted(get_observable_flip(error))
            log_prob = -penalty_func(error)
            net_log_probs[syndrome][obs_flip] = float(
                np.logaddexp(net_log_probs[syndrome].get(obs_flip, -np.inf), log_prob)
            )
            # Record the first error for each key (so it always has a representative, even when all
            # of its errors have zero probability), then keep the most likely one thereafter.  A tie
            # in probability resolves toward the lighter error, since enumeration runs from heavy to
            # light and so reaches the lightest error of a tie last.
            key = (syndrome, obs_flip)
            if key not in most_likely_errors or log_prob >= most_likely_error_log_probs[key]:
                most_likely_error_log_probs[key] = log_prob
                most_likely_errors[key] = error_codec.pack_trusted(error)

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
            if predict_observable_flips:
                prediction = flip_codec.unpack(most_likely_obs_flip)
            else:
                prediction = error_codec.unpack(most_likely_errors[syndrome, most_likely_obs_flip])
            self._store(syndrome, prediction)

    @staticmethod
    def _organize_lookup_table_initialization_data(
        pcm_or_dem: IntegerArray | stim.DetectorErrorModel,
        error_channel: npt.NDArray[np.floating] | Sequence[float] | None,
        penalty_func: Callable[[npt.NDArray[np.int_] | Sequence[int]], float] | None,
        observable_flip_matrix: IntegerArray | None,
        predict_observable_flips: bool,
        post_select: Collection[int],
        add_erasure_bit: bool,
    ) -> tuple[
        IntegerArray,
        IntegerArray | None,
        Callable[[npt.NDArray[np.int_] | Sequence[int]], float] | None,
        npt.NDArray[np.bool_] | None,
        npt.NDArray[np.int_],
    ]:
        """Organize and validate the inputs to a LookupDecoder."""
        if isinstance(pcm_or_dem, stim.DetectorErrorModel):
            if (
                error_channel is not None
                or penalty_func is not None
                or observable_flip_matrix is not None
            ):
                raise ValueError(
                    "Cannot specify an error_channel, penalty_func, or observable_flip_matrix when"
                    " providing a stim.DetectorErrorModel to a LookupDecoder"
                )
            dem_arrays = DetectorErrorModelArrays(pcm_or_dem, simplify=False)
            pcm = dem_arrays.detector_flip_matrix
            error_channel = dem_arrays.error_probs
            # errors are grouped by observable flip if there are observables, or if predicting
            # observable flips, which are trivial (empty) if there are no observables
            if dem_arrays.num_observables > 0 or predict_observable_flips:
                observable_flip_matrix = dem_arrays.observable_flip_matrix
        else:
            pcm = pcm_or_dem
            if error_channel is not None and penalty_func is not None:
                raise ValueError(
                    "Cannot specify both an error_channel and a penalty_func in a LookupDecoder"
                )

        # validate the explicit error channel before building its penalty function
        if error_channel is not None:
            error_channel = np.asarray(error_channel, dtype=float)
            expected_shape = (pcm.shape[1],)
            if error_channel.shape != expected_shape:
                raise ValueError(
                    f"A LookupDecoder error_channel must have shape {expected_shape}, but got"
                    f" {error_channel.shape}"
                )
            if not np.all((0 <= error_channel) & (error_channel <= 1)):
                raise ValueError(
                    "A LookupDecoder error_channel must contain finite probabilities between 0 and"
                    " 1, inclusive"
                )

        # if an explicit penalty_func was not provided, build one from the error channel
        penalty_func = penalty_func or (
            _LookupDecoderBase._build_penalty_func(error_channel)
            if error_channel is not None
            else None
        )

        # build the mask of syndrome bits to keep (None if not post-selecting)
        syndrome_mask: npt.NDArray[np.bool_] | None = None
        if len(post_select):
            syndrome_mask = np.ones(pcm.shape[0], dtype=bool)
            syndrome_mask[list(post_select)] = False

        # build the default output returned for syndromes absent from the lookup table
        if predict_observable_flips:
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

        return pcm, observable_flip_matrix, penalty_func, syndrome_mask, default_correction

    @staticmethod
    def _build_penalty_func(
        error_channel: npt.NDArray[np.floating] | Sequence[float],
    ) -> Callable[[npt.NDArray[np.int_] | Sequence[int]], float]:
        """Construct a penalty function from independent probabilities of individual errors."""
        error_channel = np.asarray(error_channel)
        with np.errstate(divide="ignore"):  # a probability of 0 or 1 yields a -inf log, which is ok
            log_probs = np.log(error_channel)
            log_non_probs = np.log(1 - error_channel)

        def penalty_func(error: npt.NDArray[np.int_] | Sequence[int]) -> float:
            """Penalize unlikely combinations of errors."""
            events = np.asarray(error).astype(bool)
            log_probability_of_error = np.sum(log_probs[events]) + np.sum(log_non_probs[~events])
            return -float(log_probability_of_error)

        return penalty_func

    @staticmethod
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

    @staticmethod
    def _iter_errors_and_syndromes(
        matrix: IntegerArray,
        max_weight: int,
        syndrome_mask: npt.NDArray[np.bool_] | None,
        symplectic: bool,
    ) -> Iterator[tuple[npt.NDArray[np.int_], tuple[int, ...]]]:
        """Iterate over all errors that this decoder considers, and their syndromes as tuples.

        See _iter_error_and_syndrome_arrays, which yields syndromes as arrays.
        """
        for error, syndrome in _LookupDecoderBase._iter_error_and_syndrome_arrays(
            matrix, max_weight, syndrome_mask, symplectic
        ):
            yield error, tuple(syndrome.tolist())

    @staticmethod
    def _iter_error_and_syndrome_arrays(
        matrix: IntegerArray,
        max_weight: int,
        syndrome_mask: npt.NDArray[np.bool_] | None,
        symplectic: bool,
    ) -> Iterator[tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]]:
        """Iterate over all errors that this decoder considers, and their associated syndromes.

        Errors are sorted in decreasing weight (number of bits or qudits addressed nontrivially).

        The syndrome_mask is a boolean mask of syndrome bits to retain, or None to keep all bits.
        When post-selecting (keep is not None), errors whose syndrome is nontrivial on any dropped
        bit are skipped, and dropped bits are omitted from the yielded syndrome.
        """
        from qldpc import codes

        dtype = matrix.dtype
        # rewrite the checks so multiplying by an error produces its syndrome
        code = codes.ClassicalCode(matrix) if not symplectic else codes.QuditCode(matrix)
        matrix = code.matrix if not symplectic else -math.symplectic_conjugate(code.matrix)
        syndrome_bits_to_drop = (
            None if syndrome_mask is None else ~syndrome_mask
        )  # post-selected bits, required to be trivial

        # identify the set of local errors that can occur
        repeat = 2 if symplectic else 1
        error_ops = tuple(itertools.product(range(code.field.order), repeat=repeat))[1:]

        block_length = matrix.shape[1] // repeat
        for weight in range(max_weight, -1, -1):
            for error_sites in itertools.combinations(range(block_length), weight):
                error_site_indices = list(error_sites)
                for local_errors in itertools.product(error_ops, repeat=weight):
                    error = code.field.Zeros((repeat, block_length))
                    error[:, error_site_indices] = np.asarray(local_errors, dtype=dtype).T
                    error = error.ravel()
                    syndrome = matrix @ error
                    if syndrome_mask is not None:
                        if np.any(syndrome[syndrome_bits_to_drop]):
                            continue
                        syndrome = syndrome[syndrome_mask]
                    yield error.view(np.ndarray).astype(dtype), syndrome.view(np.ndarray)

    def _maybe_add_erasure_bit(self, error: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Append a trivial (zero) erasure bit to an error if this decoder tracks erasure bits."""
        if not self.has_erasure_bit:
            return error
        return with_erasure_bits(error, False)

    def _get_packed_syndrome_key(self, syndrome: npt.NDArray[np.int_]) -> bytes | None:
        """Return the packed retained syndrome, or None if it cannot be in the lookup table."""
        syndrome = np.asarray(syndrome).view(np.ndarray)
        if self.syndrome_mask is not None:
            if syndrome.shape != self.syndrome_mask.shape:
                return None
            retained_syndrome = syndrome[self.syndrome_mask]
            if np.count_nonzero(retained_syndrome) != np.count_nonzero(syndrome):
                return None
            syndrome = retained_syndrome
        return self._syndrome_codec.pack(syndrome)

    def _get_syndrome_key(self, syndrome: npt.NDArray[np.int_]) -> tuple[int, ...] | None:
        """Return the retained syndrome key, or None when a post-selected bit is nontrivial."""
        syndrome = syndrome.view(np.ndarray)
        if self.syndrome_mask is not None:
            retained_syndrome = syndrome[self.syndrome_mask]
            if np.count_nonzero(retained_syndrome) != np.count_nonzero(syndrome):
                return None
            syndrome = retained_syndrome
        return tuple(syndrome.tolist())

    def __len__(self) -> int:
        """The number of entries in this lookup table."""
        return len(self._syndrome_to_error)

    def _decode(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Look up the configured error or observable-flip prediction."""
        key = self._get_packed_syndrome_key(syndrome)
        packed = None if key is None else self._syndrome_to_error.get(key)
        if packed is None:
            return self.default_correction.copy()
        return self._output_codec.unpack(packed)

    def _pack_syndrome_rows(
        self, syndromes: npt.ArrayLike
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.bool_]]:
        """Pack full syndrome rows and identify rows eligible for table lookup."""
        syndromes = np.asarray(syndromes).view(np.ndarray)
        if syndromes.ndim != 2 or syndromes.shape[1] != self.num_detectors:
            raise ValueError(
                f"Expected syndromes of shape (num_syndromes, {self.num_detectors}),"
                f" but got {syndromes.shape}"
            )

        valid = np.ones(len(syndromes), dtype=bool)
        if self.syndrome_mask is not None:
            valid &= ~np.any(syndromes[:, ~self.syndrome_mask] != 0, axis=1)
            syndromes = syndromes[:, self.syndrome_mask]
        valid &= self._syndrome_codec.valid_rows(syndromes)

        safe_syndromes = np.zeros(
            (len(syndromes), self._syndrome_codec.length), dtype=self._syndrome_codec.dtype
        )
        safe_syndromes[valid] = syndromes[valid]
        return self._syndrome_codec.pack_rows_trusted(safe_syndromes), valid

    def _lookup_packed_rows(
        self,
        packed_syndromes: npt.NDArray[np.uint8],
        valid: npt.NDArray[np.bool_] | None = None,
    ) -> npt.NDArray[np.uint8]:
        """Map packed syndrome rows directly to packed prediction rows."""
        packed_syndromes = np.ascontiguousarray(packed_syndromes, dtype=np.uint8)
        if (
            packed_syndromes.ndim != 2
            or packed_syndromes.shape[1] != self._syndrome_codec.num_bytes
        ):
            raise ValueError(
                f"Expected packed syndromes of shape"
                f" (num_syndromes, {self._syndrome_codec.num_bytes}),"
                f" but got {packed_syndromes.shape}"
            )
        if valid is not None and valid.shape != (len(packed_syndromes),):
            raise ValueError(
                f"Expected a validity mask of shape {(len(packed_syndromes),)},"
                f" but got {valid.shape}"
            )

        num_predictions = len(packed_syndromes)
        packed_predictions = np.empty(
            (num_predictions, self._output_codec.num_bytes), dtype=np.uint8
        )
        if self._output_codec.num_bytes == 0:
            return packed_predictions

        default = self._output_codec.pack_trusted(self.default_correction)
        key_dtype = np.dtype((np.void, self._syndrome_codec.num_bytes))
        # Bound temporary Python byte keys while retaining the throughput of bulk conversion.
        for start in range(0, num_predictions, _LOOKUP_BATCH_SIZE):
            stop = min(start + _LOOKUP_BATCH_SIZE, num_predictions)
            if self._syndrome_codec.num_bytes:
                # A void view preserves trailing zero bytes; tolist() constructs the keys in C.
                keys = cast(
                    list[bytes],
                    packed_syndromes[start:stop].view(key_dtype).reshape(-1).tolist(),
                )
            else:
                keys = [b""] * (stop - start)
            if valid is None or np.all(valid[start:stop]):
                values = [self._syndrome_to_error.get(key, default) for key in keys]
            else:
                values = [
                    self._syndrome_to_error.get(key, default) if is_valid else default
                    for key, is_valid in zip(keys, valid[start:stop], strict=True)
                ]
            packed_predictions[start:stop] = np.frombuffer(
                b"".join(values), dtype=np.uint8
            ).reshape(stop - start, self._output_codec.num_bytes)
        return packed_predictions

    def _decode_batch(self, syndromes: npt.ArrayLike) -> npt.NDArray[np.int_]:
        """Look up a batch of syndromes and return unpacked predictions."""
        packed_syndromes, valid = self._pack_syndrome_rows(syndromes)
        return self._output_codec.unpack_rows(self._lookup_packed_rows(packed_syndromes, valid))

    def _decode_binary_packed_batch(
        self, bit_packed_syndromes: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.uint8]:
        """Look up binary syndromes that are already packed in little-endian bit order."""
        if self.field.order != 2:
            raise ValueError("Bit-packed lookup decoding is only available over GF(2)")
        bit_packed_syndromes = np.ascontiguousarray(bit_packed_syndromes, dtype=np.uint8)
        num_syndrome_bytes = -(-self.num_detectors // 8)
        if bit_packed_syndromes.ndim != 2 or bit_packed_syndromes.shape[1] != num_syndrome_bytes:
            raise ValueError(
                f"Expected bit-packed syndromes of shape"
                f" (num_syndromes, {num_syndrome_bytes}),"
                f" but got {bit_packed_syndromes.shape}"
            )

        final_byte_bits = self.num_detectors % 8
        if final_byte_bits and len(bit_packed_syndromes):
            final_byte_mask = np.uint8((1 << final_byte_bits) - 1)
            if np.any(bit_packed_syndromes[:, -1] & ~final_byte_mask):
                bit_packed_syndromes = bit_packed_syndromes.copy()
                bit_packed_syndromes[:, -1] &= final_byte_mask

        if self.syndrome_mask is None:
            return self._lookup_packed_rows(bit_packed_syndromes)
        syndromes = np.unpackbits(
            bit_packed_syndromes,
            count=self.num_detectors,
            bitorder="little",
            axis=1,
        )
        packed_syndromes, valid = self._pack_syndrome_rows(syndromes)
        return self._lookup_packed_rows(packed_syndromes, valid)

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

    If provided an ``error_channel`` of independent probabilities for each "primitive" error
    mechanism (which associated with one column of a PCM, or one entry in a DEM), this method
    constructs a penalty function, ``penalty_func``, that penalizes unlikely errors.  In this case,
    a candidate ``syndrome -> new_error`` entry encountered during enumeration will only override a
    past entry in the lookup table if ``penalty_func(new_error) <= penalty_func(old_error)``.
    Errors are enumerated in decreasing weight, so an equal penalty resolves in favor of the
    lighter error.
    Alternatively, this decoder supports the use of a user-provided ``penalty_func``, which must map
    an error (represented as a binary vector of length ``num_primitive_error_mechanisms``) to a real
    number (i.e., a penalty).

    If provided an ``observable_flip_matrix`` (shape ``num_observables × num_primitive_errors``),
    this decoder maps each syndrome to an error that induces the most likely observable flip for
    that syndrome, which may be different from the single most likely error.  Concretely: errors
    consistent with a given syndrome are grouped by their observable flip value; the total
    probability of each group is the sum of the probabilities of its member errors, restricted to
    the errors of ``weight <= max_weight`` that this decoder enumerates.  This decoder then assigns
    each ``syndrome`` the highest-probability individual ``error`` from the group with the highest
    total probability.

    The deprecated ``predict_observable_flips=True`` option makes ``.decode`` return the most likely
    observable flip for each syndrome, rather than a representative ``error``.  Use an
    ObservableLookupDecoder instead, whose ``.decode_observables`` method returns observable flips.

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

    If initialized with a positive ``confidence_ratio`` (which then requires an
    ``observable_flip_matrix``), this decoder handles ambiguous syndromes -- those consistent with
    more than one observable flip -- by declining to guess unless one flip is clearly dominant.
    Letting ``prob_top`` and ``prob_rest`` be the net probabilities of the most likely observable
    flip and of all other flips combined (summed over the enumerated ``weight <= max_weight``
    errors, grouped as above), the decoder assigns the most likely flip iff it is at least
    ``confidence_ratio`` times as likely as the rest, i.e. ``prob_top >= confidence_ratio *
    prob_rest``.  Otherwise, the syndrome is omitted from the lookup table, so that it decodes to
    erasure, identically to a syndrome that was never enumerated.  A positive ``confidence_ratio``
    therefore auto-enables the erasure bit, setting ``add_erasure_bit=True``.  At the extreme,
    ``confidence_ratio=np.inf`` keeps only syndromes whose competing flips have zero net
    probability, erasing every syndrome with a competing flip that can actually occur.

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
        error_channel: npt.NDArray[np.floating] | Sequence[float] | None = None,
        penalty_func: Callable[[npt.NDArray[np.int_] | Sequence[int]], float] | None = None,
        observable_flip_matrix: IntegerArray | None = None,
        predict_observable_flips: bool = False,
        post_select: Collection[int] = (),
        add_erasure_bit: bool | None = None,
        confidence_ratio: float | None = None,
        symplectic: bool = False,
    ) -> None:
        """Initialize an error lookup table.

        predict_observable_flips is deprecated; use ObservableLookupDecoder for observable output.
        """
        _warn_deprecated_observable_prediction(predict_observable_flips, "ObservableLookupDecoder")
        super().__init__(
            pcm_or_dem,
            max_weight,
            error_channel=error_channel,
            penalty_func=penalty_func,
            observable_flip_matrix=observable_flip_matrix,
            predict_observable_flips=predict_observable_flips,
            post_select=post_select,
            add_erasure_bit=add_erasure_bit,
            confidence_ratio=confidence_ratio,
            symplectic=symplectic,
        )

    def decode_errors(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode an error syndrome and return an inferred error."""
        return self._decode(syndrome)

    def decode_errors_batch(self, syndromes: npt.ArrayLike) -> npt.NDArray[np.int_]:
        """Decode a 2-D batch of syndromes, one per row, and return inferred errors.

        A malformed or absent syndrome receives the same default correction as scalar decoding.
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

    An ObservableLookupDecoder is built from a detector error model, or from a parity check matrix
    together with an ``observable_flip_matrix`` whose rows specify which errors flip which
    observables.  If initialized with ``add_erasure_bit=True``, this decoder appends an erasure bit
    to each predicted observable flip.
    """

    @overload
    def __init__(
        self,
        pcm_or_dem: stim.DetectorErrorModel,
        max_weight: int,
        *,
        post_select: Collection[int] = (),
        add_erasure_bit: bool | None = None,
        confidence_ratio: float | None = None,
        symplectic: bool = False,
    ) -> None: ...

    @overload
    def __init__(
        self,
        pcm_or_dem: IntegerArray,
        max_weight: int,
        *,
        observable_flip_matrix: IntegerArray,
        error_channel: npt.NDArray[np.floating] | Sequence[float] | None = None,
        penalty_func: Callable[[npt.NDArray[np.int_] | Sequence[int]], float] | None = None,
        post_select: Collection[int] = (),
        add_erasure_bit: bool | None = None,
        confidence_ratio: float | None = None,
        symplectic: bool = False,
    ) -> None: ...

    def __init__(
        self,
        pcm_or_dem: IntegerArray | stim.DetectorErrorModel,
        max_weight: int,
        *,
        observable_flip_matrix: IntegerArray | None = None,
        error_channel: npt.NDArray[np.floating] | Sequence[float] | None = None,
        penalty_func: Callable[[npt.NDArray[np.int_] | Sequence[int]], float] | None = None,
        post_select: Collection[int] = (),
        add_erasure_bit: bool | None = None,
        confidence_ratio: float | None = None,
        symplectic: bool = False,
    ) -> None:
        super().__init__(
            pcm_or_dem,
            max_weight,
            error_channel=error_channel,
            penalty_func=penalty_func,
            observable_flip_matrix=observable_flip_matrix,
            predict_observable_flips=True,
            post_select=post_select,
            add_erasure_bit=add_erasure_bit,
            confidence_ratio=confidence_ratio,
            symplectic=symplectic,
        )

    def decode_observables(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode a syndrome and return predicted observable flips."""
        return self._decode(syndrome)

    def decode_observables_batch(self, syndromes: npt.ArrayLike) -> npt.NDArray[np.int_]:
        """Decode a 2-D batch of syndromes, one per row, and return observable flips.

        A malformed or absent syndrome receives the same default prediction as scalar decoding.
        """
        return self._decode_batch(syndromes)

    def decode_shots(self, detection_event_data: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Predict observable flips from a 2-D array of unpacked binary detection events.

        Each output row contains one bit per observable and, when configured, one final erasure bit.
        """
        if self.field.order != 2:
            raise ValueError("ObservableLookupDecoder.decode_shots is only available over GF(2)")
        return np.asarray(self._decode_batch(detection_event_data), dtype=np.uint8)

    def decode_shots_bit_packed(
        self, bit_packed_detection_event_data: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.uint8]:
        """Predict little-endian bit-packed flips from bit-packed binary detection events.

        Observable flips occupy ``ceil(num_observables / 8)`` bytes per row.  When configured, an
        erasure is signalled in one additional whole byte, as required by Sinter.
        """
        packed_predictions = self._decode_binary_packed_batch(bit_packed_detection_event_data)
        num_observable_bytes = -(-self.num_observables // 8)
        if not self.has_erasure_bit:
            return packed_predictions[:, :num_observable_bytes]

        sinter_predictions = np.empty(
            (len(packed_predictions), num_observable_bytes + 1), dtype=np.uint8
        )
        if num_observable_bytes:
            sinter_predictions[:, :num_observable_bytes] = packed_predictions[
                :, :num_observable_bytes
            ]
        erasure_byte, erasure_bit = divmod(self.num_observables, 8)
        sinter_predictions[:, -1] = (packed_predictions[:, erasure_byte] >> erasure_bit) & 1
        if erasure_bit:
            observable_mask = np.uint8((1 << erasure_bit) - 1)
            sinter_predictions[:, num_observable_bytes - 1] &= observable_mask
        return sinter_predictions


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
        predict_observable_flips: bool = False,
        post_select: Collection[int] = (),
        add_erasure_bit: bool = False,
        symplectic: bool = False,
    ) -> None:
        pcm, observable_flip_matrix, _, syndrome_mask, default_correction = (
            self._organize_lookup_table_initialization_data(
                pcm_or_dem,
                None,
                None,
                observable_flip_matrix,
                predict_observable_flips,
                post_select,
                add_erasure_bit,
            )
        )

        # save attributes; decode_returns_observables declares whether a decode method returns
        # observable flips, which it does with the deprecated predict_observable_flips=True
        self.predict_observable_flips = predict_observable_flips
        self.decode_returns_observables = predict_observable_flips
        self._save_interface_metadata(pcm, observable_flip_matrix, predict_observable_flips)
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
        if predict_observable_flips:
            assert observable_flip_matrix is not None  # primarily for type-checking reasons
            get_observable_flip = _LookupDecoderBase._build_observable_flip_func(
                pcm, observable_flip_matrix, symplectic
            )
        for error, syndrome in _LookupDecoderBase._iter_errors_and_syndromes(
            pcm, max_weight, syndrome_mask, symplectic
        ):
            output = (
                error
                if get_observable_flip is None
                else get_observable_flip(error).astype(pcm.dtype)
            )
            self.syndrome_to_candidates[syndrome].append(
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
    help(LookupDecoder).  The deprecated ``predict_observable_flips=True`` option makes ``.decode``
    return observable flips; use a WeightedObservableLookupDecoder instead.
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
            predict_observable_flips, "WeightedObservableLookupDecoder"
        )
        super().__init__(
            pcm_or_dem,
            max_weight,
            observable_flip_matrix=observable_flip_matrix,
            predict_observable_flips=predict_observable_flips,
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
        syndromes: npt.ArrayLike,
        penalty_func: Callable[[npt.NDArray[np.int_]], float] | None = lambda vec: int(
            np.count_nonzero(vec)
        ),
    ) -> npt.NDArray[np.int_]:
        """Decode a batch of syndromes with one penalty function and return inferred errors."""
        return self._stack_predictions(
            [self._decode_weighted(syndrome, penalty_func) for syndrome in np.asarray(syndromes)]
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
            predict_observable_flips=True,
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


@_erasure_bit_support("lookup", supported=True)
def get_decoder_lookup(
    pcm_or_dem: IntegerArray | stim.DetectorErrorModel, **decoder_args: object
) -> LookupDecoder:
    """Build a lookup table that maps syndromes to inferred errors.

    Args:
        pcm_or_dem: A parity-check matrix or detector error model (DEM) to decode.  A DEM supplies
            default error probabilities and observable metadata.
        **decoder_args: Arguments passed to :class:`LookupDecoder`, including the required
            ``max_weight`` and optional erasure, confidence, and symplectic settings.

    Returns:
        A :class:`LookupDecoder`.

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
        **decoder_args: Arguments passed to :class:`ObservableLookupDecoder`, including
            ``max_weight`` and optional erasure, confidence, and post-selection settings.

    Returns:
        An :class:`ObservableLookupDecoder`.
    """
    return ObservableLookupDecoder(dem, **decoder_args)  # type: ignore[call-overload]


# Private helpers


class _VectorCodec:
    """Pack fixed-length vectors over a finite field into compact, hashable bytes.

    Over GF(2), vectors are bit-packed with np.packbits.  Over any other field, each entry is stored
    as a fixed-width unsigned integer: the integer representation of a field element that galois
    uses, which is lossless for both prime and extension fields.
    """

    def __init__(self, order: int, length: int, dtype: npt.DTypeLike) -> None:
        self.order = order
        self.length = length
        self.dtype = np.dtype(dtype)
        self.width = max(1, ((order - 1).bit_length() + 7) // 8)
        self.storage_dtype = (
            np.dtype(f"<u{self.width}") if self.width in (1, 2, 4, 8) else None
        )  # None for fields too large for a native integer type
        self.num_bytes = -(-length // 8) if order == 2 else length * self.width

    def is_valid(self, vector: npt.NDArray[np.int_]) -> bool:
        """Is this a vector of the right length whose entries all represent field elements?"""
        if vector.shape != (self.length,):
            return False
        if vector.dtype.kind not in "biuO" and not np.all(np.mod(vector, 1) == 0):
            return False  # a non-integer entry is not a field element
        return self.length == 0 or bool(0 <= vector.min() and vector.max() < self.order)

    def pack(self, vector: npt.NDArray[np.int_]) -> bytes | None:
        """Pack a vector, or return None if it is not a valid vector over the field."""
        vector = np.asarray(vector).view(np.ndarray)
        if not self.is_valid(vector):
            return None
        return self.pack_trusted(
            vector if vector.dtype.kind in "biu" else np.array(vector.tolist(), dtype=object)
        )

    def pack_trusted(self, vector: npt.NDArray[np.int_]) -> bytes:
        """Pack a vector that is known to be valid."""
        if self.order == 2:
            return np.packbits(
                vector.astype(np.uint8) if vector.dtype == object else vector,
                bitorder="little",
            ).tobytes()
        if self.storage_dtype is not None:
            return np.asarray(vector).astype(self.storage_dtype).tobytes()
        return b"".join(int(value).to_bytes(self.width, "little") for value in vector)

    def unpack(self, packed: bytes) -> npt.NDArray[np.int_]:
        """Unpack a vector into a new array."""
        if self.order == 2:
            bits = np.unpackbits(
                np.frombuffer(packed, dtype=np.uint8), count=self.length, bitorder="little"
            )
            return bits.astype(self.dtype)
        if self.storage_dtype is not None:
            return np.frombuffer(packed, dtype=self.storage_dtype).astype(self.dtype)
        return np.array(
            [
                int.from_bytes(packed[index : index + self.width], "little")
                for index in range(0, len(packed), self.width)
            ],
            dtype=self.dtype,
        ).reshape(self.length)

    def valid_rows(self, vectors: npt.ArrayLike) -> npt.NDArray[np.bool_]:
        """Identify rows whose entries are valid integer representations of field elements."""
        vectors = np.asarray(vectors)
        if vectors.ndim != 2 or vectors.shape[1] != self.length:
            raise ValueError(
                f"Expected vectors of shape (num_vectors, {self.length}), but got {vectors.shape}"
            )
        if self.length == 0:
            return np.ones(len(vectors), dtype=bool)
        if vectors.dtype.kind in "biu":
            return np.all((0 <= vectors) & (vectors < self.order), axis=1)
        if vectors.dtype.kind == "f":
            integral = np.isfinite(vectors) & (vectors == np.floor(vectors))
            return np.all(integral & (0 <= vectors) & (vectors < self.order), axis=1)
        return np.array([self.is_valid(vector) for vector in vectors], dtype=bool)

    def pack_rows_trusted(self, vectors: npt.ArrayLike) -> npt.NDArray[np.uint8]:
        """Pack a two-dimensional array of valid vectors, one vector per row."""
        vectors = np.asarray(vectors)
        num_vectors = len(vectors)
        if self.num_bytes == 0:
            return np.empty((num_vectors, 0), dtype=np.uint8)
        if self.order == 2:
            return np.packbits(vectors, bitorder="little", axis=1)
        if self.storage_dtype is not None:
            stored = vectors.astype(self.storage_dtype, copy=False)
            return np.ascontiguousarray(stored).view(np.uint8).reshape(num_vectors, self.num_bytes)
        packed = b"".join(
            int(value).to_bytes(self.width, "little") for vector in vectors for value in vector
        )
        return np.frombuffer(packed, dtype=np.uint8).reshape(num_vectors, self.num_bytes)

    def unpack_rows(self, packed: npt.NDArray[np.uint8]) -> npt.NDArray[np.int_]:
        """Unpack fixed-width byte rows into vectors, one vector per row."""
        packed = np.asarray(packed, dtype=np.uint8)
        if packed.ndim != 2 or packed.shape[1] != self.num_bytes:
            raise ValueError(
                f"Expected packed vectors of shape (num_vectors, {self.num_bytes}),"
                f" but got {packed.shape}"
            )
        num_vectors = len(packed)
        if self.length == 0:
            return np.empty((num_vectors, 0), dtype=self.dtype)
        if self.order == 2:
            bits = np.unpackbits(packed, count=self.length, bitorder="little", axis=1)
            return bits.astype(self.dtype)
        if self.storage_dtype is not None:
            stored = np.ascontiguousarray(packed).view(self.storage_dtype)
            return stored.reshape(num_vectors, self.length).astype(self.dtype)
        values = [
            int.from_bytes(row[index : index + self.width], "little")
            for row in packed
            for index in range(0, self.num_bytes, self.width)
        ]
        return np.asarray(values, dtype=self.dtype).reshape(num_vectors, self.length)


class _PackedLookupTable(MutableMapping[tuple[int, ...], npt.NDArray[np.int_]]):
    """A mutable view of a lookup table that stores packed syndromes and predictions.

    See help(LookupDecoder.syndrome_to_error) for details.
    """

    def __init__(self, decoder: _LookupDecoderBase) -> None:
        self._decoder = decoder

    def _pack_key(self, syndrome: object) -> bytes | None:
        try:
            return self._decoder._syndrome_codec.pack(np.asarray(syndrome))
        except (TypeError, ValueError):
            return None

    def __getitem__(self, syndrome: tuple[int, ...]) -> npt.NDArray[np.int_]:
        key = self._pack_key(syndrome)
        if key is None or key not in self._decoder._syndrome_to_error:
            raise KeyError(syndrome)
        return self._decoder._output_codec.unpack(self._decoder._syndrome_to_error[key])

    def __setitem__(self, syndrome: tuple[int, ...], prediction: npt.NDArray[np.int_]) -> None:
        key = self._pack_key(syndrome)
        if key is None:
            raise ValueError(
                f"A lookup table syndrome must be a vector of {self._decoder._syndrome_codec.length}"
                f" elements of {self._decoder.field.name}, but got {syndrome}"
            )
        value = self._decoder._output_codec.pack(np.asarray(prediction))
        if value is None:
            raise ValueError(
                f"A lookup table prediction must be a vector of"
                f" {self._decoder._output_codec.length} elements of {self._decoder.field.name}, but"
                f" got {prediction}"
            )
        self._decoder._syndrome_to_error[key] = value

    def __delitem__(self, syndrome: tuple[int, ...]) -> None:
        key = self._pack_key(syndrome)
        if key is None or key not in self._decoder._syndrome_to_error:
            raise KeyError(syndrome)
        del self._decoder._syndrome_to_error[key]

    def __iter__(self) -> Iterator[tuple[int, ...]]:
        unpack = self._decoder._syndrome_codec.unpack
        for key in list(self._decoder._syndrome_to_error):
            yield tuple(unpack(key).tolist())

    def __len__(self) -> int:
        return len(self._decoder._syndrome_to_error)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.copy()})"

    def copy(self) -> dict[tuple[int, ...], npt.NDArray[np.int_]]:
        """Return an unpacked dictionary snapshot of this lookup table."""
        return dict(self.items())


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


def _warn_deprecated_observable_prediction(enabled: bool, replacement: str) -> None:
    """Warn about the legacy mode in which an error decoder predicts observable flips."""
    if enabled:
        warnings.warn(
            f"predict_observable_flips=True is deprecated; use {replacement} instead",
            DeprecationWarning,
            stacklevel=get_external_caller_stacklevel(),
        )
