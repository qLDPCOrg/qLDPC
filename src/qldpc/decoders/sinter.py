"""Decoders for sinter to sample quantum error correction circuits.

Copyright 2025 The qLDPC Authors

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
import pathlib
import warnings
from collections.abc import Callable, Collection, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import sinter
import stim

from .dems import DetectorErrorModelArrays
from .retrieval import Decoder, get_decoder


class DecoderNotCompiledError(Exception):
    pass


class SinterDecoder(Decoder, sinter.Decoder):
    """Decoder usable by Sinter for decoding circuit errors."""

    def __init__(
        self,
        *,
        simplify: bool = True,
        decompose_errors: bool = False,
        **decoder_kwargs: object,
    ) -> None:
        """Initialize a SinterDecoder.

        A SinterDecoder is used by Sinter to decode detection events from a detector error model to
        predict observable flips.  See help(sinter.Decoder) for additional information.

        Args:
            simplify: Whether to merge equivalent errors in a DEM when compiling a decoder for
                that DEM.
            decompose_errors: Whether to decompose errors according to their suggested decomposition
                when compiling a decoder for a DEM.
            **decoder_kwargs: Arguments to pass to qldpc.decoders.get_decoder when compiling a
                custom decoder from a detector error model.
        """
        self.simplify = simplify
        self.decompose_errors = decompose_errors
        self.decoder_kwargs = decoder_kwargs
        if "priors_arg" in decoder_kwargs or "log_likelihood_priors" in decoder_kwargs:
            raise ValueError(
                "The 'priors_arg' and 'log_likelihood_priors' arguments to a SinterDecoder are"
                " DEFUNCT and should no longer be necessary.\nIf you need these arguments restored,"
                " please open an issue at https://github.com/qLDPCOrg/qLDPC/issues"
            )

    def compile_decoder_for_dem(self, dem: stim.DetectorErrorModel) -> CompiledSinterDecoder:
        """Creates a decoder preconfigured for the given detector error model.

        See help(sinter.Decoder) for additional information.
        """
        dem_arrays = DetectorErrorModelArrays(
            dem, simplify=self.simplify, decompose_errors=self.decompose_errors
        )
        decoder = get_decoder(dem_arrays.to_dem(), **self.decoder_kwargs)
        _check_decodes_errors(decoder)
        if getattr(decoder, "has_erasure_bit", False):
            dem_arrays = dem_arrays.with_erasure()
        return CompiledSinterDecoder(dem_arrays, decoder)

    def decode_via_files(
        self,
        *,
        num_shots: int,
        num_dets: int,
        num_obs: int,
        dem_path: pathlib.Path,
        dets_b8_in_path: pathlib.Path,
        obs_predictions_b8_out_path: pathlib.Path,
        tmp_dir: pathlib.Path,
    ) -> None:
        """Predict observable flips for detection events read from a file, and write them to a file.

        See help(sinter.Decoder) for additional information.

        A file of predictions holds exactly one observable flip per observable, with no room for the
        byte in which a decoder asks for a shot to be discarded, so an erased shot is reported here
        with the observable flips that its decoder predicts for it anyway.  Sample with
        sinter.collect or with qldpc.circuits.get_logical_error_and_discard_rate to have erased
        shots discarded instead of predicted.
        """
        num_detector_bytes = -(-num_dets // 8)
        num_observable_bytes = -(-num_obs // 8)
        detection_event_data = np.fromfile(
            dets_b8_in_path, dtype=np.uint8, count=num_shots * num_detector_bytes
        ).reshape(num_shots, num_detector_bytes)

        compiled_decoder = self.compile_decoder_for_dem(stim.DetectorErrorModel.from_file(dem_path))
        predicted_flips = compiled_decoder.decode_shots_bit_packed(detection_event_data)
        if predicted_flips.shape[1] not in (num_observable_bytes, num_observable_bytes + 1):
            raise ValueError(
                f"This decoder predicted {predicted_flips.shape[1]} bytes of observable flips per"
                f" shot, but {num_obs} observables take {num_observable_bytes} bytes, or"
                f" {num_observable_bytes + 1} bytes with a byte added to signal discards"
            )
        observable_flips = predicted_flips[:, :num_observable_bytes]
        observable_flips.tofile(obs_predictions_b8_out_path)

    def decode(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode an error syndrome and return an inferred error."""
        raise DecoderNotCompiledError(
            "This SinterDecoder needs to be compiled in order to decode.  Please compile with"
            " SinterDecoder.compile_decoder_for_dem"
        )


class CompiledSinterDecoder(Decoder, sinter.CompiledDecoder):
    """Decoder usable by Sinter for decoding circuit errors, compiled to a specific circuit.

    Instances of this class are meant to be constructed by a SinterDecoder, whose
    .compile_decoder_for_dem method returns a CompiledSinterDecoder.

    When the decoder being wrapped signals erasure with an erasure bit, .decode_shots appends one
    erasure bit to the observable flips of every shot, and .decode_shots_bit_packed reports those
    erasure bits in one whole byte added past the packed observable flips.  Sinter reads that added
    byte as a request to discard the shot, so an erasure becomes a discarded shot rather than a
    predicted flip of an observable that the sampled circuit does not have.
    """

    num_detectors: int
    num_observables: int
    num_erasure_bits: int = 0

    def __init__(self, dem_arrays: DetectorErrorModelArrays, decoder: Decoder) -> None:
        self.dem_arrays = dem_arrays
        self.decoder = decoder
        self.num_detectors = dem_arrays.num_detectors
        self.num_erasure_bits = int(getattr(decoder, "has_erasure_bit", False))
        self.num_observables = dem_arrays.num_observables - self.num_erasure_bits

    def decode_shots_bit_packed(
        self, bit_packed_detection_event_data: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.uint8]:
        """Predicts observable flips from the given detection events.

        This method accepts and returns bit-packed data.

        See help(sinter.CompiledDecoder) for additional information.
        """
        detection_event_data = self.unpack_detection_event_data(bit_packed_detection_event_data)
        observable_flips = self.decode_shots(detection_event_data)
        return self.pack_observable_flips(observable_flips)

    def pack_observable_flips(
        self, observable_flips: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.uint8]:
        """Bit-pack predicted observable flips, signalling erasure in one whole added byte.

        Sinter discards a shot whose bit-packed prediction is exactly one byte wider than the
        observables of the sampled circuit require, and whose extra byte is nonzero.  Erasure is
        signalled in that byte, which keeps the packed predictions aligned with the observables
        that the circuit actually reports.  A shot is erased if any erasure bit is set.

        The added byte is read by the sampler that sinter runs a decoder under, and is not part of
        the return shape that sinter documents for a compiled decoder, which is one byte per eight
        observables.  A sinter release can therefore change how the byte is read without
        contradicting its own documentation.
        """
        if not self.num_erasure_bits:
            return self.packbits(observable_flips)
        erased = np.any(observable_flips[:, self.num_observables :], axis=1)
        packed_flips = self.packbits(observable_flips[:, : self.num_observables])
        return np.hstack([packed_flips, erased.astype(np.uint8)[:, None]])

    def decode_shots(self, detection_event_data: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Predicts observable flips from the given detection events.

        This method accepts and returns boolean data.

        See help(sinter.CompiledDecoder) for additional information.
        """
        if hasattr(self.decoder, "decode_batch"):
            predicted_errors = self.decoder.decode_batch(detection_event_data)
            return predicted_errors @ self.dem_arrays.observable_flip_matrix.T % 2

        num_shots = len(detection_event_data)
        num_observables = self.dem_arrays.observable_flip_matrix.shape[0]
        observable_flips = np.zeros((num_shots, num_observables), dtype=np.uint8)
        for row, syndrome in enumerate(detection_event_data):
            predicted_errors = self.decoder.decode(syndrome)
            observable_flips[row] = self.dem_arrays.observable_flip_matrix @ predicted_errors
        return np.asarray(observable_flips, dtype=np.uint8) % 2

    def packbits(self, data: npt.NDArray[np.uint8], axis: int = -1) -> npt.NDArray[np.uint8]:
        """Bit-pack the data along an axis.

        Working with bit-packed data is more memory and compute-efficient, which is why Sinter
        generally passes around bit-packed data.
        """
        return np.packbits(np.asarray(data, dtype=np.uint8), bitorder="little", axis=axis)

    def unpack_detection_event_data(
        self, bit_packed_detection_event_data: npt.NDArray[np.uint8], axis: int = -1
    ) -> npt.NDArray[np.uint8]:
        """Unpack the bit-packed data along an axis.

        By default, bit_packed_detection_event_data is assumed to be a two-dimensional array in
        which each row contains bit-packed detection events from one sample of a detector error
        model (DEM).  In this case, the unpacked data is a boolean matrix whose entry in row ss and
        column kk specify whether detector kk was flipped in sample ss of a DEM.
        """
        return np.unpackbits(
            np.asarray(bit_packed_detection_event_data, dtype=np.uint8),
            count=self.num_detectors,
            bitorder="little",
            axis=axis,
        )

    def decode(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Alias for CompiledSinterDecoder.decode_shots.

        Predicts observable flips.
        """
        syndrome_uint8 = np.asarray(syndrome, dtype=np.uint8)
        return self.decode_shots(syndrome_uint8.reshape(1, *syndrome.shape))[0]


class TrivialDecoder(SinterDecoder):
    """A trivial decoder that unconditionally predicts null errors/logical flips."""

    def __init__(self) -> None: ...

    def compile_decoder_for_dem(self, dem: stim.DetectorErrorModel) -> CompiledSinterDecoder:
        """Creates a decoder preconfigured for the given detector error model.

        See help(sinter.Decoder) for additional information.
        """
        return CompiledTrivialDecoder(dem.num_detectors, dem.num_observables)


class CompiledTrivialDecoder(CompiledSinterDecoder):
    """A compiled trivial decoder that unconditionally predicts null errors/logical flips."""

    def __init__(self, num_detectors: int, num_observables: int) -> None:
        self.num_detectors = num_detectors  # for compatibility with the parent class
        self.num_observables = num_observables
        self.packed_observable_size = (num_observables + 7) // 8

    def decode_shots_bit_packed(
        self, bit_packed_detection_event_data: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.uint8]:
        """Unconditionally predicts no observable flips.

        This method accepts and returns bit-packed data.

        See help(sinter.CompiledDecoder) for additional information.
        """
        shape = (len(bit_packed_detection_event_data), self.packed_observable_size)
        return np.zeros(shape, dtype=np.uint8)

    def decode_shots(self, detection_event_data: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Unconditionally predicts no observable flips.

        This method accepts and returns boolean data.

        See help(sinter.CompiledDecoder) for additional information.
        """
        shape = (len(detection_event_data), self.num_observables)
        return np.zeros(shape, dtype=np.uint8)


class SubgraphDecoder(SinterDecoder):
    """Decoder usable by Sinter for decoding circuit errors.

    A SubgraphDecoder splits the Tanner graph of a detector error model into subgraphs, and decodes
    these subgraphs independently.  Each subgraph is defined by a subset of detectors, S.  When
    compiling a SubgraphDecoder for a specific detector error model D, this decoder constructs, for
    each subgraph S, a smaller detector error model ``D_S`` that restricts D to the detectors in S
    and the error mechanisms that flip the detectors in S.

    A SubgraphDecoder may optionally assign each subgraph S a set of observables, ``O_S``, in which
    case the subgraph detector error model ``D_S`` only considers (and predicts corrections for) the
    observables in ``O_S``.

    The subgraphs predict observable flips independently, and their predictions are combined by
    exclusive or.  Every observable therefore has to be assigned to the subgraphs in a way that lets
    exactly one of them predict each of its flips: if two subgraphs both witness an error mechanism
    and both own an observable that the mechanism flips, then both predict that flip and the two
    predictions cancel.  Compiling a SubgraphDecoder warns when a detector error model and a
    partition permit that, and when a detector belongs to no subgraph at all.

    As an example, a SubgraphDecoder is useful for independently decoding the X and Z sectors of a
    CSS code, where each sector owns the observables of the opposite type.
    """

    def __init__(
        self,
        subgraph_detectors: Sequence[Collection[int]],
        subgraph_observables: Sequence[Collection[int]] | None = None,
        *,
        simplify: bool = True,
        decompose_errors: bool = False,
        **decoder_kwargs: object,
    ) -> None:
        """Initialize a SinterDecoder that splits a detector error model into disjoint subgraphs.

        A SubgraphDecoder is used by Sinter to decode detection events from a detector error model
        to predict observable flips.

        See help(sinter.Decoder) for additional information.

        Args:
            subgraph_detectors: A sequence containing one set of detectors per subgraph.
            subgraph_observables: A sequence containing one set of observables per subgraph; or None
                to indicate that every subgraph should decode every observable.  Default: None.
            simplify: Whether to merge equivalent errors in a DEM when compiling a decoder for
                that DEM.
            decompose_errors: Whether to decompose errors according to their suggested decomposition
                when compiling a decoder for a DEM.
            **decoder_kwargs: Arguments to pass to qldpc.decoders.get_decoder when compiling a
                custom decoder from a detector error model.
        """
        SinterDecoder.__init__(
            self, simplify=simplify, decompose_errors=decompose_errors, **decoder_kwargs
        )

        # consistency checks
        self.num_subgraphs = len(subgraph_detectors)
        num_observable_sets = None if subgraph_observables is None else len(subgraph_observables)
        if not (num_observable_sets is None or num_observable_sets == self.num_subgraphs):
            raise ValueError(
                f"The number of detector sets ({self.num_subgraphs}) is inconsistent with the"
                f" number of observable sets ({num_observable_sets})"
            )

        self.subgraph_detectors = [sorted(dets) for dets in subgraph_detectors]
        self.subgraph_observables = (
            None if subgraph_observables is None else [sorted(obs) for obs in subgraph_observables]
        )

    def compile_decoder_for_dem(self, dem: stim.DetectorErrorModel) -> CompiledSubgraphDecoder:
        """Creates a decoder preconfigured for the given detector error model.

        See help(sinter.Decoder) for additional information.
        """
        dem_arrays = DetectorErrorModelArrays(
            dem, simplify=self.simplify, decompose_errors=self.decompose_errors
        )
        subgraph_observables = (
            [list(range(dem.num_observables)) for _ in range(self.num_subgraphs)]
            if self.subgraph_observables is None
            else [list(obs) for obs in self.subgraph_observables]
        )
        num_erasure_bits = 0

        # count, for every observable flip, the subgraphs that can predict it
        flip_observables, flip_errors = dem_arrays.observable_flip_matrix.nonzero()
        flip_predictors = np.zeros(len(flip_errors), dtype=int)
        covered_detectors = np.zeros(dem.num_detectors, dtype=bool)

        # build a decoder for each subgraph
        subgraph_decoders = []
        for ss, (detectors, observables) in enumerate(
            zip(self.subgraph_detectors, subgraph_observables)
        ):
            # identify the error mechanisms that flip these detectors
            errors = dem_arrays.detector_flip_matrix[detectors].getnnz(axis=0) != 0

            # this subgraph can predict a flip if it owns the observable and witnesses the error
            owned = np.zeros(dem.num_observables, dtype=bool)
            owned[observables] = True
            flip_predictors += owned[flip_observables] & errors[flip_errors]
            covered_detectors[detectors] = True

            # build the detector error model for this subgraph
            subgraph_dem = DetectorErrorModelArrays.from_arrays(
                dem_arrays.detector_flip_matrix[detectors][:, errors],
                dem_arrays.observable_flip_matrix[observables][:, errors],
                dem_arrays.error_probs[errors],
            ).to_detector_error_model()

            # compile the decoder for this subgraph
            subgraph_decoder = SinterDecoder.compile_decoder_for_dem(self, subgraph_dem)
            subgraph_decoders.append(subgraph_decoder)

            # collect the erasure bit of this subgraph past the observables of the whole model
            if getattr(subgraph_decoder.decoder, "has_erasure_bit", False):
                subgraph_observables[ss].append(dem.num_observables + num_erasure_bits)
                num_erasure_bits += 1

        _warn_about_subgraph_partition(
            flip_errors, flip_observables, flip_predictors, np.flatnonzero(~covered_detectors)
        )

        return CompiledSubgraphDecoder(
            self.subgraph_detectors,
            subgraph_observables,
            subgraph_decoders,
            dem.num_detectors,
            dem.num_observables,
            num_erasure_bits,
        )


class SubgraphSinterDecoder(SubgraphDecoder):
    """Deprecated alias for SubgraphDecoder."""

    def __getattribute__(self, name: str) -> Any:
        warnings.warn(
            f"{SubgraphSinterDecoder} is DEPRECATED; use {SubgraphDecoder} instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return super().__getattribute__(name)


class CompiledSubgraphDecoder(CompiledSinterDecoder):
    """Decoder usable by Sinter for decoding circuit errors, compiled to a specific circuit.

    This decoder splits a decoding problem into subgraphs that are decoded independently.

    Instances of this class are meant to be constructed by a SubgraphDecoder, whose
    .compile_decoder_for_dem method returns a CompiledSubgraphDecoder.
    See help(SubgraphDecoder).
    """

    def __init__(
        self,
        subgraph_detectors: Sequence[Sequence[int] | slice],
        subgraph_observables: Sequence[Sequence[int] | slice],
        subgraph_decoders: Sequence[CompiledSinterDecoder],
        num_detectors: int,
        num_observables: int,
        num_erasure_bits: int = 0,
    ) -> None:
        if not len(subgraph_detectors) == len(subgraph_observables) == len(subgraph_decoders):
            raise ValueError(
                "A CompiledSubgraphDecoder needs one detector set, one observable set, and one"
                f" decoder per subgraph (provided: {len(subgraph_detectors)},"
                f" {len(subgraph_observables)}, {len(subgraph_decoders)})"
            )
        self.subgraph_detectors = subgraph_detectors
        self.subgraph_observables = subgraph_observables
        self.subgraph_decoders = subgraph_decoders
        self.num_detectors = num_detectors
        self.num_observables = num_observables
        self.num_erasure_bits = num_erasure_bits

    def decode_shots(self, detection_event_data: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Predicts observable flips from the given detection events.

        This method accepts and returns boolean data.

        See help(sinter.CompiledDecoder) for additional information.
        """
        if detection_event_data.shape[1] != self.num_detectors:
            raise ValueError(
                f"Detection event data has {detection_event_data.shape[1]} detectors per shot, but"
                f" this decoder was compiled for {self.num_detectors} detectors"
            )

        # initialize predicted observable flips, followed by one erasure bit per erasing subgraph
        observable_flips = np.zeros(
            (len(detection_event_data), self.num_observables + self.num_erasure_bits),
            dtype=np.uint8,
        )

        # decode segments independently
        for detectors, observables, decoder in zip(
            self.subgraph_detectors, self.subgraph_observables, self.subgraph_decoders
        ):
            syndromes = detection_event_data[:, detectors]
            observable_flips[:, observables] ^= decoder.decode_shots(syndromes)

        return observable_flips


class SequentialWindowDecoder(SinterDecoder):
    """Decoder usable by Sinter for decoding circuit errors.

    A SequentialWindowDecoder splits a detector error model into (possibly overlapping) "windows".
    Each window is defined by two sets of detectors, which in turn define a "detection region" and
    a "commit region" for that window.  Each region consists of a (given) set of detectors and the
    (induced) set of error mechanisms that trigger those detectors.

    Windows are decoded sequentially, one by one.  To decode a window, we first decode the syndrome
    in its detection region.  We then "commit" to the decoded circuit error in the commit
    region, which entails

    (a) removing the error mechanisms in the commit region from all subsequent windows, and
    (b) emulating the active correction of committed errors by appropriately updating the syndromes
        in subsequent windows.

    The net circuit error inferred by decoding all windows is used to predict observable flips.

    A window decoder that signals erasure erases the whole shot, and all the windows share the one
    erasure bit that the compiled decoder reports.

    A SequentialWindowDecoder initialized without specifying commit regions sets the commit region
    of each window to the corresponding detection region.

    A special case of SequentialWindowDecoder is a SlidingWindowDecoder, in which case this
    decoding method is known as the "overlapping recovery method" in arXiv:quant-ph/0110143, which
    is explained more nicely in arXiv:2012.15403 and arXiv:2209.08552.
    """

    def __init__(
        self,
        detection_regions: Sequence[Collection[int]],
        commit_regions: Sequence[Collection[int]] | None = None,
        *,
        simplify: bool = True,
        decompose_errors: bool = False,
        **decoder_kwargs: object,
    ) -> None:
        """Initialize a SinterDecoder that splits a detector error model into windows.

        A SequentialWindowDecoder is used by Sinter to decode detection events from a detector error
        model to predict observable flips.

        See help(sinter.Decoder) for additional information.

        Args:
            detection_regions: A sequence containing a set of detectors for each window.
            commit_regions: A sequence containing a set of detectors for each window, or None, in
                which case the commit region of each window is equal to its detection regions.
                Default: None.  The errors triggered by a commit region must also be triggered by
                the detection region of the same window, which holds whenever the commit region is
                a subset of the detection region.
            simplify: Whether to merge equivalent errors in a DEM when compiling a decoder for
                that DEM.
            decompose_errors: Whether to decompose errors according to their suggested decomposition
                when compiling a decoder for a DEM.
            **decoder_kwargs: Arguments to pass to qldpc.decoders.get_decoder when compiling a
                custom decoder from a detector error model.
        """
        SinterDecoder.__init__(
            self, simplify=simplify, decompose_errors=decompose_errors, **decoder_kwargs
        )

        if commit_regions is not None and len(detection_regions) != len(commit_regions):
            raise ValueError(
                f"The number of detection regions ({len(detection_regions)}) is inconsistent with"
                f" the number of commit regions ({len(commit_regions)})"
            )
        self.windows = [
            (list(d_detectors), list(c_detectors))
            for d_detectors, c_detectors in zip(
                detection_regions, commit_regions or detection_regions
            )
            if d_detectors
        ]

    def compile_decoder_for_dem(
        self, dem: stim.DetectorErrorModel
    ) -> CompiledSequentialWindowDecoder:
        """Creates a decoder preconfigured for the given detector error model.

        See help(sinter.Decoder) for additional information.
        """
        dem_arrays = DetectorErrorModelArrays(
            dem, simplify=self.simplify, decompose_errors=self.decompose_errors
        )

        # identify regions and compile a decoder for each window
        window_detectors = []
        window_errors = []
        window_decoders = []
        addressed_errors = np.zeros(dem_arrays.num_errors, dtype=bool)
        for d_detectors, c_detectors in self.windows:
            # identify errors in the detection region
            d_errors = dem_arrays.detector_flip_matrix[d_detectors].getnnz(axis=0) != 0
            d_errors[addressed_errors] = False

            # compile a decoder for the detection region
            window_dem_arrays = DetectorErrorModelArrays.from_arrays(
                dem_arrays.detector_flip_matrix[d_detectors][:, d_errors],
                dem_arrays.observable_flip_matrix[:, d_errors],
                dem_arrays.error_probs[d_errors],
            )
            window_dem = window_dem_arrays.to_dem()
            window_decoder = get_decoder(window_dem, **self.decoder_kwargs)
            _check_decodes_errors(window_decoder)

            # Restricting the DEM to this window may result in several error mechanisms that are
            # equivalent, which the window_decoder will merge into one error mechanism.  In this
            # case, wrap the decoder into an _ExpandedWindowDecoder that maps decoded errors in the
            # simplified DEM to errors in the full DEM.  An erasure bit is not an error mechanism,
            # so it does not count toward the width being compared here.
            num_erasure_bits = int(getattr(window_decoder, "has_erasure_bit", False))
            test_error = window_decoder.decode(np.zeros(window_dem.num_detectors, dtype=int))
            if len(test_error) - num_erasure_bits < window_dem.num_errors:
                window_decoder = _ExpandedWindowDecoder(window_decoder, window_dem)

            # identify errors in the commit region
            c_errors = dem_arrays.detector_flip_matrix[c_detectors].getnnz(axis=0) != 0
            c_errors[addressed_errors] = False
            if (c_errors & ~d_errors).any():
                raise ValueError(
                    f"The commit region of a window (detectors {c_detectors}) is triggered by"
                    f" errors that its detection region (detectors {d_detectors}) is not, so those"
                    " errors cannot be decoded before they are committed"
                )
            c_errors_in_detection_region = np.isin(np.where(d_errors), np.where(c_errors))[0]

            # save detection region detectors, committed error data, and decoders
            window_detectors.append(d_detectors)
            window_errors.append((c_errors, c_errors_in_detection_region))
            window_decoders.append(window_decoder)

            # update the history of errors that are addressed by preceding windows
            addressed_errors |= c_errors

        return CompiledSequentialWindowDecoder(
            dem_arrays, window_detectors, window_errors, window_decoders
        )


class _ExpandedWindowDecoder(Decoder):
    """Wrapper for a decoder, to map decoded errors in a simplified DEM to errors in the full DEM.

    The SequentialWindowDecoder restricts a DEM to a "window" before passing the DEM to a decoder
    for that window.  Restricting a DEM may result in equivalent error mechanisms that end up
    getting merged, which causes the restricted + simplified DEM to have fewer errors in the window
    than the un-simplified DEM.  This wrapper expands decoded errors in the simplified DEM to
    equivalent errors in the original DEM, and passes any erasure bit through as the last entry.
    """

    def __init__(self, decoder: Decoder, window_dem: stim.DetectorErrorModel) -> None:
        self._decoder = decoder
        self.has_erasure_bit = bool(getattr(decoder, "has_erasure_bit", False))

        original_errors = DetectorErrorModelArrays.get_circuit_errors(window_dem)
        simplified_errors = DetectorErrorModelArrays.get_merged_circuit_errors(original_errors)
        self._num_original_errors = len(original_errors)

        # map each detector/observable signature to an original error index
        signature_to_original_error_index = {
            signature: original_error_index
            for original_error_index, (_, signature) in enumerate(original_errors)
        }

        # locate each simplified error among the original errors
        self._simplified_to_original_index = np.array(
            [signature_to_original_error_index[signature] for _, signature in simplified_errors],
            dtype=np.intp,
        )

    def decode(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        simplified_error = self._decoder.decode(syndrome)
        original_error = np.zeros(
            self._num_original_errors + self.has_erasure_bit, dtype=syndrome.dtype
        )
        if self.has_erasure_bit:
            original_error[-1] = simplified_error[-1]
            simplified_error = simplified_error[:-1]
        original_error[self._simplified_to_original_index] = simplified_error
        return np.asarray(original_error, dtype=syndrome.dtype)

    def decode_batch(self, syndromes: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        simplified_errors = (
            self._decoder.decode_batch(syndromes)
            if hasattr(self._decoder, "decode_batch")
            else np.array([self._decoder.decode(syndrome) for syndrome in syndromes])
        )
        original_errors = np.zeros(
            (len(syndromes), self._num_original_errors + self.has_erasure_bit),
            dtype=syndromes.dtype,
        )
        if self.has_erasure_bit:
            original_errors[:, -1] = simplified_errors[:, -1]
            simplified_errors = simplified_errors[:, :-1]
        original_errors[:, self._simplified_to_original_index] = simplified_errors
        return original_errors


class SequentialSinterDecoder(SequentialWindowDecoder):
    """Deprecated alias for SequentialWindowDecoder."""

    def __getattribute__(self, name: str) -> Any:
        warnings.warn(
            f"{SequentialSinterDecoder} is DEPRECATED; use {SequentialWindowDecoder} instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return super().__getattribute__(name)


class CompiledSequentialWindowDecoder(CompiledSinterDecoder):
    """Decoder usable by Sinter for decoding circuit errors, compiled to a specific circuit.

    This decoder splits a decoding problem into (possibly overlapping) windows that are decoded
    sequentially.

    Instances of this class are meant to be constructed by a SequentialWindowDecoder, whose
    .compile_decoder_for_dem method returns a CompiledSequentialWindowDecoder.
    See help(SequentialWindowDecoder).
    """

    def __init__(
        self,
        dem_arrays: DetectorErrorModelArrays,
        window_detectors: Sequence[Sequence[int] | slice],
        window_errors: Sequence[tuple[Sequence[int] | slice, Sequence[int] | slice]],
        window_decoders: Sequence[Decoder],
    ) -> None:
        if not len(window_detectors) == len(window_errors) == len(window_decoders):
            raise ValueError(
                "A CompiledSequentialWindowDecoder needs one detector set, one set of committed"
                f" errors, and one decoder per window (provided: {len(window_detectors)},"
                f" {len(window_errors)}, {len(window_decoders)})"
            )
        self.dem_arrays = dem_arrays
        self.window_detectors = window_detectors
        self.window_errors = window_errors
        self.window_decoders = window_decoders

        self.num_detectors = dem_arrays.num_detectors
        self.num_observables = dem_arrays.num_observables
        self.num_erasure_bits = int(
            any(getattr(decoder, "has_erasure_bit", False) for decoder in window_decoders)
        )

    def decode_shots(self, detection_event_data: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Predicts observable flips from the given detection events.

        This method accepts and returns boolean data.

        See help(sinter.CompiledDecoder) for additional information.
        """
        net_error, erased = self.decode_shots_to_error_and_erasure(detection_event_data)
        observable_flips = net_error @ self.dem_arrays.observable_flip_matrix.T % 2
        if not self.num_erasure_bits:
            return observable_flips
        return np.hstack([observable_flips, erased[:, None].astype(observable_flips.dtype)])

    def decode_shots_to_error(
        self, detection_event_data: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.uint8]:
        """Predicts a net circuit error from the given detection events.

        The error predicted for an erased shot is a guess, and this method reports no erasures, so
        use .decode_shots_to_error_and_erasure to tell the two apart.

        This method accepts and returns boolean data.
        """
        return self.decode_shots_to_error_and_erasure(detection_event_data)[0]

    def decode_shots_to_error_and_erasure(
        self, detection_event_data: npt.NDArray[np.uint8]
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.bool_]]:
        """Predicts a net circuit error, and whether any window erased, per shot.

        A shot is erased if any of its windows is, and an erased shot still commits whatever error
        its windows inferred, which no window could explain and which the windows after it are then
        decoded against.  Sinter and qldpc.circuits.get_logical_error_and_discard_rate discard an
        erased shot, so the error committed for it does not reach a rate either of them reports.

        This method accepts and returns boolean data.
        """
        num_samples, num_detectors = detection_event_data.shape
        if num_detectors != self.dem_arrays.num_detectors:
            raise ValueError(
                f"Detection event data has {num_detectors} detectors per shot, but this decoder was"
                f" compiled for {self.dem_arrays.num_detectors} detectors"
            )

        # identify the net circuit error predicted by decoding one window at a time
        net_error = np.zeros((num_samples, self.dem_arrays.num_errors), dtype=np.uint8)
        erased = np.zeros(num_samples, dtype=bool)
        detector_flip_matrix_T = self.dem_arrays.detector_flip_matrix.T
        for detectors, (errors, error_locs), decoder in zip(
            self.window_detectors, self.window_errors, self.window_decoders
        ):
            # the bare syndrome plus any corrections we have inferred so far
            syndromes = (
                detection_event_data[:, detectors]
                + net_error @ detector_flip_matrix_T[:, detectors]
            ) % 2

            # decode this syndrome and update the net error appropriately
            decoded_error = (
                decoder.decode_batch(syndromes)
                if hasattr(decoder, "decode_batch")
                else np.array([decoder.decode(syndrome) for syndrome in syndromes])
            )
            if getattr(decoder, "has_erasure_bit", False):
                erased |= decoded_error[:, -1] != 0
                decoded_error = decoded_error[:, :-1]
            net_error[:, errors] = decoded_error[:, error_locs]

        return net_error, erased


class SlidingWindowDecoder(SequentialWindowDecoder):
    """Decoder usable by Sinter for decoding circuit errors.

    A SlidingWindowDecoder is a SequentialWindowDecoder whose windows are constructed by grouping
    detectors based on a time coordinate.  The amount of overlapping rounds between adjacent windows
    is determined by the window size and stride.  For example, a window size of w and a stride of s
    indicates adjacent windows will overlap on ``w - s`` rounds.  The "commit region" for each
    window therefore corresponds to the first s rounds in the window.

    Visually::

      Time:      |------------------------------------------------------------>

      Window 1:  [ ........... Detection Region ........... ]
                 [ Commit Region ]
                     |
                     +---> 1. Decode errors in Detection Region.
                           2. Commit to errors in Commit Region.
                           3. Update syndromes in future windows based on committed errors.
                           4. Slide window forward.
                                                |
                                                v
      Window 2:                   [ ........... Detection Region ........... ]
                                  [ Commit Region ]
                                      |
                                      v
                                     ...

    If provided a sequence of subsets of detectors, construct sliding windows for each subset.  This
    functionality is used to independently decode X and Z sectors of a CSS code.

    This decoding method is known as the "overlapping recovery method" in arXiv:quant-ph/0110143,
    which is explained more nicely in arXiv:2012.15403 and arXiv:2209.08552.
    """

    def __init__(
        self,
        window_size: int,
        stride: int,
        detector_subsets: Collection[Collection[int]] | None = None,
        detector_to_time: Callable[[int], int] | None = None,
        *,
        simplify: bool = True,
        decompose_errors: bool = False,
        **decoder_kwargs: object,
    ) -> None:
        """Initialize a SinterDecoder that splits a detector error model into temporal windows.

        A SlidingWindowDecoder is used by Sinter to decode detection events from a detector error
        model to predict observable flips.

        See help(sinter.Decoder) for additional information.

        Args:
            window_size: The size of each window, measured in discrete time steps.
            stride: The number of time steps by which to slide each window forward to get the next
                window.  Equivalently, the size of each commit region.
            detector_subsets: A collection of subsets of detectors from a detector error model, or
                None.  If not None, each provided subset is decoded independently.  If None, all
                detectors are decoded together, as if the detector_subsets was a one-element list
                containing the set of all detectors.  Default: None.
            detector_to_time: A function that maps each detector to a time coordinate that is used
                to decide window boundaries, or None.  If None, the time index of each detector is
                read from its coordinates in DetectorErrorModel.get_detector_coordinates(): the
                first coordinate, unless that coordinate decreases from one detector to the next --
                which a coordinate indexing time cannot do -- in which case a later coordinate that
                varies and never decreases is read instead, provided exactly one does.  Only the
                detectors that get windowed are consulted, and one of those with no coordinates at
                all is rejected, since there is nothing to read a time index from.
                WARNING: if a detector_to_time mapping is not None, it will be assumed to be
                both valid and compatible with any detector error model that this decoder is later
                compiled to with SlidingWindowDecoder.compile_decoder_for_dem.
            simplify: Whether to merge equivalent errors in a DEM when compiling a decoder for
                that DEM.
            decompose_errors: Whether to decompose errors according to their suggested decomposition
                when compiling a decoder for a DEM.
            **decoder_kwargs: Arguments to pass to qldpc.decoders.get_decoder when compiling a
                custom decoder from a detector error model.
        """
        SinterDecoder.__init__(
            self, simplify=simplify, decompose_errors=decompose_errors, **decoder_kwargs
        )

        if not window_size >= stride > 0:
            raise ValueError(
                f"{type(self).__name__} must have window_size >= stride > 0"
                f" (provided window_size, stride: {window_size}, {stride})"
            )

        self.window_size = window_size
        self.stride = stride
        self.detector_subsets = detector_subsets
        self.detector_to_time = detector_to_time

    def compile_decoder_for_dem(
        self, dem: stim.DetectorErrorModel
    ) -> CompiledSequentialWindowDecoder:
        """Creates a decoder preconfigured for the given detector error model.

        WARNING: if this decoder was initialized with a `detector_to_time` mapping, it is assumed
        that the mapping is both valid and compatible with the detector error model provided here.

        See help(sinter.Decoder) for additional information.
        """
        # the time index mapping is specific to the given model, so keep it out of self
        detector_to_time = self.detector_to_time
        if detector_to_time is None:
            # only the detectors that get windowed need a time index, so ignore the rest
            windowed_detectors = sorted(
                {detector for detectors in self.detector_subsets for detector in detectors}
                if self.detector_subsets
                else range(dem.num_detectors)
            )
            all_coords = dem.get_detector_coordinates()
            dem_coords = {det: all_coords.get(det, []) for det in windowed_detectors}
            uncoordinated = [det for det, coords in dem_coords.items() if not coords]
            if uncoordinated:
                raise ValueError(
                    f"detector {uncoordinated[0]} has no coordinates to read a time index from."
                    "  Pass detector_to_time to assign time indices explicitly."
                )
            coordinate = _time_coordinate(dem_coords)

            def coordinate_to_time(detector: int) -> int:
                """Read a detector's time index from its coordinates in this model."""
                return int(dem_coords[detector][coordinate])

            detector_to_time = coordinate_to_time

        # construct windows defined by "detection" and "commit" regions
        self.windows = []
        for detectors in self.detector_subsets or [range(dem.num_detectors)]:
            # collect detectors according to their time index
            time_to_dets: dict[int, list[int]] = collections.defaultdict(list)
            for detector in detectors:
                time = detector_to_time(detector)
                if not isinstance(time, (int, np.integer)):
                    raise TypeError(
                        f"detector {detector} has an invalid (non-integer) time index: {time}"
                    )
                time_to_dets[int(time)].append(detector)

            # add one window at a time (except the last window)
            start_time = min(time_to_dets)
            end_time = max(time_to_dets) + 1
            max_size_of_last_window = self.window_size + self.stride - 1
            while start_time < end_time - max_size_of_last_window:
                window_time_to_dets = [
                    time_to_dets[start_time + dt] for dt in range(self.window_size)
                ]
                window = (  # defined by (detection, commit) regions
                    [det for dets in window_time_to_dets for det in dets],
                    [det for dets in window_time_to_dets[: self.stride] for det in dets],
                )
                self.windows.append(window)
                start_time += self.stride

            # add last window
            window_time_to_dets = [time_to_dets[tt] for tt in range(start_time, end_time)]
            last_dets = [det for dets in window_time_to_dets for det in dets]
            self.windows.append((last_dets, last_dets))

        # drop windows with nothing to commit, which arise from gaps between time indices
        self.windows = [(d_dets, c_dets) for d_dets, c_dets in self.windows if c_dets]

        return SequentialWindowDecoder.compile_decoder_for_dem(self, dem)


def _check_decodes_errors(decoder: Decoder) -> None:
    """Reject a decoder whose output is observable flips rather than an inferred error."""
    if getattr(decoder, "predict_observable_flips", False):
        raise ValueError(
            "A sinter decoder maps decoded circuit errors to observable flips itself, so the decoder"
            " that it wraps must predict errors rather than observable flips"
        )


def _warn_about_subgraph_partition(
    flip_errors: npt.NDArray[np.int_],
    flip_observables: npt.NDArray[np.int_],
    flip_predictors: npt.NDArray[np.int_],
    uncovered_detectors: npt.NDArray[np.int_],
) -> None:
    """Warn about a partition into subgraphs whose predictions do not add up.

    Args:
        flip_errors: The error mechanism of each observable flip in a detector error model.
        flip_observables: The observable of each of those flips.
        flip_predictors: The number of subgraphs that can predict each of those flips.
        uncovered_detectors: The detectors that belong to no subgraph.
    """
    contested = np.flatnonzero(flip_predictors > 1)
    if contested.size:
        first = contested[0]
        warnings.warn(
            f"{contested.size} observable flips of this detector error model can be predicted by"
            " more than one subgraph, and predictions are combined by exclusive or, so two"
            " subgraphs predicting the same flip cancel each other.  Assign each observable only to"
            " subgraphs whose detectors witness its flips.  For example, error mechanism"
            f" {flip_errors[first]} flips observable {flip_observables[first]}, which"
            f" {flip_predictors[first]} subgraphs can predict",
            stacklevel=3,
        )
    if uncovered_detectors.size:
        warnings.warn(
            f"{uncovered_detectors.size} detectors of this detector error model belong to no"
            " subgraph, so no decoder ever sees their detection events:"
            f" {uncovered_detectors[:10].tolist()}",
            stacklevel=3,
        )


def _time_coordinate(dem_coords: dict[int, list[float]]) -> int:
    """Which detector coordinate of a detector error model indexes time.

    Detector coordinates are assigned as a circuit is built, and a circuit runs forward, so a
    coordinate that indexes time never decreases from one detector to the next.  The first
    coordinate is used whenever it has that property.  Otherwise the first coordinate cannot be
    indexing time, and a later coordinate that varies and never decreases is used instead -- the
    circuits that stim generates place time last, for example.  If no later coordinate qualifies,
    or if several do, the first coordinate is used anyway.
    """
    detectors = sorted(dem_coords)

    def never_decreases(coordinate: int) -> bool:
        values = [dem_coords[det][coordinate] for det in detectors]
        return all(before <= after for before, after in itertools.pairwise(values))

    def varies(coordinate: int) -> bool:
        return len({dem_coords[det][coordinate] for det in detectors}) > 1

    if never_decreases(0):
        return 0
    num_coordinates = min(len(dem_coords[det]) for det in detectors)
    candidates = [
        coordinate
        for coordinate in range(1, num_coordinates)
        if varies(coordinate) and never_decreases(coordinate)
    ]
    return candidates[0] if len(candidates) == 1 else 0
