"""Lookup-table decoder classes.

Copyright 2023 The qLDPC Authors and Infleqtion Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import collections
import itertools
from collections.abc import Callable, Collection, Iterator, Sequence

import galois
import numpy as np
import numpy.typing as npt
import stim

from qldpc import codes, math
from qldpc.math import IntegerArray

from .dems import DetectorErrorModelArrays


class LookupDecoder:
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
    past entry in the lookup table if ``penalty_func(new_error) < penalty_func(old_error)``.
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

    If initialized with ``predict_observable_flips=True``, this decoder maps each syndrome directly
    to its most likely observable flip, rather than to a representative ``error``.  In this case
    the decoded output is a binary vector of length ``num_observables``.  Predicting observable
    flips requires an ``observable_flip_matrix``.

    If provided a ``post_select`` collection of syndrome-bit (i.e., detector) indices, this decoder
    post-selects on those bits being trivial: when constructing the lookup table, it ignores
    syndromes that are nonzero on the post-selected bits. and it drops those bits from the syndrome
    keys in the lookup table.  For consistency with the post-selection options in sinter, syndromes
    passed to ``LookupDecoder.decode`` should still contain all syndrome bits.

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
    ``confidence_ratio=np.inf`` keeps only syndromes with a single consistent observable flip,
    erasing every syndrome that has any competing flip.

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

        # save attributes
        self.predict_observable_flips = predict_observable_flips
        self.syndrome_mask = syndrome_mask
        self.has_erasure_bit = add_erasure_bit
        self.default_correction = default_correction

        # start working on the decoding map from syndrome -> error
        self.syndrome_to_error: dict[tuple[int, ...], npt.NDArray[np.int_]] = {}

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
        error_penalty: dict[tuple[int, ...], float] = {}
        for error, syndrome in LookupDecoder._iter_errors_and_syndromes(
            pcm, max_weight, syndrome_mask, symplectic
        ):
            if penalty_func is None:
                self.syndrome_to_error[syndrome] = self._maybe_add_erasure_bit(error)
            elif (error_weight := penalty_func(error)) <= error_penalty.get(syndrome, np.inf):
                error_penalty[syndrome] = error_weight
                self.syndrome_to_error[syndrome] = self._maybe_add_erasure_bit(error)

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

        def _get_obs_flip(error: npt.NDArray[np.int_]) -> tuple[int, ...]:
            """Map an error to its induced observable flips."""
            if isinstance(observable_flip_matrix, galois.FieldArray):  # pragma: no cover
                error = error.view(type(observable_flip_matrix))
                obs_flip = (observable_flip_matrix @ error).view(np.ndarray)
            else:
                obs_flip = (observable_flip_matrix @ error).view(np.ndarray) % 2
            return tuple(obs_flip.tolist())

        # For each "key" = (syndrome, observable_flip) combination, identify:
        # 1. The net log-probability of each key.
        # 2. The most likely error for each key.
        # 3. The log-probability of the most likely error for each key.
        # Probabilities are accumulated in log-space (via logaddexp) to avoid the underflow that
        # would otherwise arise from summing the tiny probabilities of individual errors.
        Bitstring = tuple[int, ...]
        net_log_probs: dict[Bitstring, dict[Bitstring, float]] = collections.defaultdict(dict)
        most_likely_errors: dict[tuple[Bitstring, Bitstring], npt.NDArray[np.int_]] = {}
        most_likely_error_log_probs: dict[tuple[Bitstring, Bitstring], float] = {}
        for error, syndrome in LookupDecoder._iter_errors_and_syndromes(
            pcm, max_weight, syndrome_mask, symplectic
        ):
            obs_flip = _get_obs_flip(error)
            log_prob = -penalty_func(error)
            net_log_probs[syndrome][obs_flip] = float(
                np.logaddexp(net_log_probs[syndrome].get(obs_flip, -np.inf), log_prob)
            )
            # Record the first error for each key (so it always has a representative, even when all
            # of its errors have zero probability), then keep the most likely one thereafter.
            key = (syndrome, obs_flip)
            if key not in most_likely_errors or log_prob > most_likely_error_log_probs[key]:
                most_likely_error_log_probs[key] = log_prob
                most_likely_errors[key] = error

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
                # confident iff prob_top >= confidence_ratio * prob_rest (compared in log-space)
                if log_prob_top < log_confidence_ratio + log_prob_rest:
                    continue  # omit the ambiguous syndrome, leaving it to decode as erasure
            if predict_observable_flips:
                prediction = np.asarray(most_likely_obs_flip, dtype=pcm.dtype)
            else:
                prediction = most_likely_errors[syndrome, most_likely_obs_flip]
            self.syndrome_to_error[syndrome] = self._maybe_add_erasure_bit(prediction)

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
            ):  # pragma: no cover
                raise ValueError(
                    "Cannot specify an error_channel, penalty_func, or observable_flip_matrix when"
                    " providing a stim.DetectorErrorModel to a LookupDecoder"
                )
            dem_arrays = DetectorErrorModelArrays(pcm_or_dem, simplify=False)
            pcm = dem_arrays.detector_flip_matrix
            error_channel = dem_arrays.error_probs
            if dem_arrays.num_observables > 0:
                observable_flip_matrix = dem_arrays.observable_flip_matrix
        else:
            pcm = pcm_or_dem
            if error_channel is not None and penalty_func is not None:  # pragma: no cover
                raise ValueError(
                    "Cannot specify both an error_channel and a penalty_func in a LookupDecoder"
                )

        # if an explicit penalty_func was not provided, build one from the error channel
        penalty_func = penalty_func or (
            LookupDecoder._build_penalty_func(error_channel) if error_channel is not None else None
        )

        # build the mask of syndrome bits to keep (None if not post-selecting)
        syndrome_mask: npt.NDArray[np.bool_] | None = None
        if len(post_select):
            syndrome_mask = np.ones(pcm.shape[0], dtype=bool)
            syndrome_mask[list(post_select)] = False

        # build the default output returned for syndromes absent from the lookup table
        if predict_observable_flips:
            if observable_flip_matrix is None:  # pragma: no cover
                raise ValueError(
                    "Predicting observable flips with a LookupDecoder requires providing a"
                    " stim.DetectorErrorModel with observables or an observable_flip_matrix"
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
    def _iter_errors_and_syndromes(
        matrix: IntegerArray,
        max_weight: int,
        syndrome_mask: npt.NDArray[np.bool_] | None,
        symplectic: bool,
    ) -> Iterator[tuple[npt.NDArray[np.int_], tuple[int, ...]]]:
        """Iterate over all errors that this decoder considers, and their associated syndromes.

        Errors are sorted in decreasing weight (number of bits or qudits addressed nontrivially).

        The syndrome_mask is a boolean mask of syndrome bits to retain, or None to keep all bits.
        When post-selecting (keep is not None), errors whose syndrome is nontrivial on any dropped
        bit are skipped, and dropped bits are omitted from the yielded syndrome.
        """
        dtype = matrix.dtype
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
                    yield (
                        error.view(np.ndarray).astype(dtype),
                        tuple(syndrome.view(np.ndarray).tolist()),
                    )

    def _maybe_add_erasure_bit(self, error: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Append a trivial (zero) erasure bit to an error if this decoder tracks erasure bits."""
        if not self.has_erasure_bit:
            return error
        return np.hstack([error, np.zeros(1, dtype=error.dtype)])

    def __len__(self) -> int:
        """The number of entries in this lookup table."""
        return len(self.syndrome_to_error)

    def decode(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode an error syndrome and return an inferred error.

        If initialized with predict_observable_flips=True, return the inferred observable flip.
        """
        syndrome = syndrome.view(np.ndarray)
        if self.syndrome_mask is not None:
            syndrome = syndrome[self.syndrome_mask]
        return self.syndrome_to_error.get(tuple(syndrome.tolist()), self.default_correction).copy()


class WeightedLookupDecoder(LookupDecoder):
    """Decoder based on a lookup table that maps syndromes to errors.

    A WeightedLookupDecoder is a LookupDecoder that, when initialized, records *all* errors that are
    consistent with a given syndrome.  The WeightedLookupDecoder then minimizes a penalty function
    that is provided to the .decode method.  A WeightedLookupDecoder can thereby be initialized
    once, and subsequently asked to decode with different penalty functions.
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

        # save attributes
        self.predict_observable_flips = predict_observable_flips
        self.syndrome_mask = syndrome_mask
        self.has_erasure_bit = add_erasure_bit
        self.default_correction = default_correction

        # Record all errors consistent with each syndrome, together with the output to return if
        # that error is selected: an observable-flip prediction if requested (else the error
        # itself), with a trivial erasure bit appended.  The output is precomputed here so that
        # decode() only has to select the minimum-penalty candidate error.
        self.syndrome_to_candidates: dict[
            tuple[int, ...], list[tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]]
        ] = collections.defaultdict(list)
        for error, syndrome in LookupDecoder._iter_errors_and_syndromes(
            pcm, max_weight, syndrome_mask, symplectic
        ):
            if predict_observable_flips:
                assert observable_flip_matrix is not None  # primarily for type-checking reasons
                output = (observable_flip_matrix @ error).view(np.ndarray) % 2
            else:
                output = error
            self.syndrome_to_candidates[syndrome].append(
                (error, self._maybe_add_erasure_bit(output))
            )

    def __len__(self) -> int:
        """The number of entries in this lookup table."""
        return len(self.syndrome_to_candidates)

    def decode(
        self,
        syndrome: npt.NDArray[np.int_],
        penalty_func: Callable[[npt.NDArray[np.int_]], float] | None = lambda vec: int(
            np.count_nonzero(vec)
        ),
    ) -> npt.NDArray[np.int_]:
        """Decode an error syndrome and return an inferred error."""
        syndrome = syndrome.view(np.ndarray)
        if self.syndrome_mask is not None:
            syndrome = syndrome[self.syndrome_mask]
        key = tuple(syndrome.tolist())
        if key not in self.syndrome_to_candidates:
            return self.default_correction.copy()

        candidates = self.syndrome_to_candidates[key]
        if penalty_func is None:
            output = candidates[-1][1]
        else:
            output = min(candidates, key=lambda candidate: penalty_func(candidate[0]))[1]
        return output.copy()
