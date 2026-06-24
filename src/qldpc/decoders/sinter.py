"""Decoders for sinter to sample quantum error correction circuits

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
            simplify: Whether merge equivalent errors in a DEM when compiling a decoder for that DEM.
            decompose_errors: Whether to decompose errors according to their suggested decomposition
                when compiling a decoder for a DEM.
            **decoder_kwargs: Arguments to pass to qldpc.decoders.get_decoder when compiling a
                custom decoder from a detector error model.
        """
        self.simplify = simplify
        self.decompose_errors = decompose_errors
        self.decoder_kwargs = decoder_kwargs
        if (
            "priors_arg" in decoder_kwargs or "log_likelihood_priors" in decoder_kwargs
        ):  # pragma: no cover
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
        if getattr(decoder, "has_erasure_bit", False):
            dem_arrays = dem_arrays.with_erasure()
        return CompiledSinterDecoder(dem_arrays, decoder)

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
    """

    num_detectors: int

    def __init__(self, dem_arrays: DetectorErrorModelArrays, decoder: Decoder) -> None:
        self.dem_arrays = dem_arrays
        self.decoder = decoder
        self.num_detectors = dem_arrays.num_detectors

    def decode_shots_bit_packed(
        self, bit_packed_detection_event_data: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.uint8]:
        """Predicts observable flips from the given detection events.

        This method accepts and returns bit-packed data.

        See help(sinter.CompiledDecoder) for additional information.
        """
        detection_event_data = self.unpack_detection_event_data(bit_packed_detection_event_data)
        observable_flips = self.decode_shots(detection_event_data)
        return self.packbits(observable_flips)

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
        """Alias for CompiledSinterDecoder.decode_shots.  Predicts observable flips."""
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
    each subgraph S, a smaller detector error model D_S that restricts D to the detectors in S and
    the error mechanisms that flip the detectors in S.

    A SubgraphDecoder may optionally assign each subgraph S a set of observables, O_S, in which case
    the subgraph detector error model D_S only considers (and predicts corrections for) the
    observables in O_S.

    As an example, a SubgraphDecoder is useful for independently decoding the X and Z sectors of a
    CSS code.
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
            simplify: Whether merge equivalent errors in a DEM when compiling a decoder for that DEM.
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
        num_observables = dem.num_observables

        # build a decoder for each subgraph
        subgraph_decoders = []
        for ss, (detectors, observables) in enumerate(
            zip(self.subgraph_detectors, subgraph_observables)
        ):
            # identify the error mechanisms that flip these detectors
            errors = dem_arrays.detector_flip_matrix[detectors].getnnz(axis=0) != 0

            # build the detector error model for this subgraph
            subgraph_dem = DetectorErrorModelArrays.from_arrays(
                dem_arrays.detector_flip_matrix[detectors][:, errors],
                dem_arrays.observable_flip_matrix[observables][:, errors],
                dem_arrays.error_probs[errors],
            ).to_detector_error_model()

            # compile the decoder for this subgraph
            subgraph_decoder = SinterDecoder.compile_decoder_for_dem(self, subgraph_dem)
            subgraph_decoders.append(subgraph_decoder)

            if getattr(subgraph_decoder.decoder, "has_erasure_bit", False):
                subgraph_observables[ss].append(num_observables)
                num_observables += 1

        return CompiledSubgraphDecoder(
            self.subgraph_detectors,
            subgraph_observables,
            subgraph_decoders,
            dem.num_detectors,
            num_observables,
        )


class SubgraphSinterDecoder(SubgraphDecoder):  # pragma: no cover
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
    ) -> None:
        assert len(subgraph_detectors) == len(subgraph_observables) == len(subgraph_decoders)
        self.subgraph_detectors = subgraph_detectors
        self.subgraph_observables = subgraph_observables
        self.subgraph_decoders = subgraph_decoders
        self.num_detectors = num_detectors
        self.num_observables = num_observables

    def decode_shots(self, detection_event_data: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Predicts observable flips from the given detection events.

        This method accepts and returns boolean data.

        See help(sinter.CompiledDecoder) for additional information.
        """
        assert detection_event_data.shape[1] == self.num_detectors

        # initialize predicted observable flips
        observable_flips = np.zeros(
            (len(detection_event_data), self.num_observables), dtype=np.uint8
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
    in its its detection region.  We then "commit" to the decoded circuit error in the commit
    region, which entails
    (a) removing the error mechanisms in the commit region from all subsequent windows, and
    (b) emulating the active correction of committed errors by appropriately updating the syndromes
        in subsequent windows.
    The net circuit error inferred by decoding all windows is used to predict observable flips.

    A SequentialWindowDecoder initialized without specifying commit regions sets the commit region of
    each window to the corresponding detection region.

    A special case of SequentialWindowDecoder is a SlidingWindowDecoder, in which case this
    decoding method is known as the "overlapping recovery method" in arXiv:quant-ph/0110143, which is
    explained more nicely in arXiv:2012.15403 and arXiv:2209.08552.
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
                Default: None.
            simplify: Whether merge equivalent errors in a DEM when compiling a decoder for that DEM.
            decompose_errors: Whether to decompose errors according to their suggested decomposition
                when compiling a decoder for a DEM.
            **decoder_kwargs: Arguments to pass to qldpc.decoders.get_decoder when compiling a
                custom decoder from a detector error model.
        """
        SinterDecoder.__init__(
            self, simplify=simplify, decompose_errors=decompose_errors, **decoder_kwargs
        )

        assert commit_regions is None or len(detection_regions) == len(commit_regions)
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
            if getattr(window_decoder, "has_erasure_bit", False):
                raise NotImplementedError(
                    f"{type(self)} does not yet support decoding with erasure.\n"
                    "If you would like to see this feature, please file an issue at "
                    "https://github.com/qLDPCOrg/qLDPC/issues"
                )

            # Restricting the DEM to this window may result in several error mechanisms that are
            # equivalent, which the window_decoder will merge into one error mechanism.  In this
            # case, wrap the decoder into an _ExpandingDecoder that maps decoded errors in the
            # simflified DEM to errors in the full DEM.
            test_error = window_decoder.decode(np.zeros(window_dem.num_detectors, dtype=int))
            if len(test_error) < window_dem.num_errors:
                window_decoder = _ExpandedWindowDecoder(window_decoder, window_dem)

            # identify errors in the commit region
            c_errors = dem_arrays.detector_flip_matrix[c_detectors].getnnz(axis=0) != 0
            c_errors[addressed_errors] = False
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

    The SequentialWindowDecoder restricts a DEM to a "window" before passing the DEM to a decoder for
    that window.  Restricting a DEM may result in equivalent error mechanisms that end up getting
    merged, which causes the restricted + simplified DEM to have fewer errors in the window than the
    un-simplified DEM.  This wrapper expands decoded errors in the simplified DEM to equivalent
    errors in the original DEM.
    """

    def __init__(self, decoder: Decoder, window_dem: stim.DetectorErrorModel) -> None:
        self._decoder = decoder

        original_errors = DetectorErrorModelArrays.get_circuit_errors(window_dem)
        simplified_errors = DetectorErrorModelArrays.get_merged_circuit_errors(original_errors)
        self._num_original_errors = len(original_errors)

        # map each detector/observable signature to an original error index
        signature_to_original_error_index = {
            signature: original_error_index
            for original_error_index, (_, signature) in enumerate(original_errors)
        }

        # map each detector/observable signature to a simplified error index
        signature_to_simplified_error_index = {
            signature: simplified_error_index
            for simplified_error_index, (_, signature) in enumerate(simplified_errors)
        }

        if signature_to_simplified_error_index.keys() != signature_to_original_error_index.keys():
            raise ValueError("Incompatible error sets")  # pragma: no cover

        self._simplified_to_original_index = np.full(len(simplified_errors), -1, dtype=np.intp)
        for signature, original_error_index in signature_to_original_error_index.items():
            simplified_error_index = signature_to_simplified_error_index[signature]
            self._simplified_to_original_index[simplified_error_index] = original_error_index

    def decode(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        simplified_error = self._decoder.decode(syndrome)
        original_error = np.zeros(self._num_original_errors, dtype=syndrome.dtype)
        original_error[self._simplified_to_original_index] = simplified_error
        return np.asarray(original_error, dtype=syndrome.dtype)

    def decode_batch(self, syndromes: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        simplified_errors = (
            self._decoder.decode_batch(syndromes)
            if hasattr(self._decoder, "decode_batch")
            else np.array([self._decoder.decode(syndrome) for syndrome in syndromes])
        )
        original_errors = np.zeros(
            (len(syndromes), self._num_original_errors), dtype=syndromes.dtype
        )
        original_errors[:, self._simplified_to_original_index] = simplified_errors
        return original_errors


class SequentialSinterDecoder(SequentialWindowDecoder):  # pragma: no cover
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
        assert len(window_detectors) == len(window_errors) == len(window_decoders)
        self.dem_arrays = dem_arrays
        self.window_detectors = window_detectors
        self.window_errors = window_errors
        self.window_decoders = window_decoders

        self.num_detectors = dem_arrays.num_detectors

    def decode_shots(self, detection_event_data: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Predicts observable flips from the given detection events.

        This method accepts and returns boolean data.

        See help(sinter.CompiledDecoder) for additional information.
        """
        return (
            self.decode_shots_to_error(detection_event_data)
            @ self.dem_arrays.observable_flip_matrix.T
            % 2
        )

    def decode_shots_to_error(
        self, detection_event_data: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.uint8]:
        """Predicts a net circuit error from the given detection events.

        This method accepts and returns boolean data.
        """
        num_samples, num_detectors = detection_event_data.shape
        assert num_detectors == self.dem_arrays.num_detectors

        # identify the net circuit error predicted by decoding one window at a time
        net_error = np.zeros((num_samples, self.dem_arrays.num_errors), dtype=np.uint8)
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
            net_error[:, errors] = decoded_error[:, error_locs]

        return net_error


class SlidingWindowDecoder(SequentialWindowDecoder):
    """Decoder usable by Sinter for decoding circuit errors.

    A SlidingWindowDecoder is a SequentialWindowDecoder whose windows are constructed by grouping
    detectors based on a time coordinate.  The amount of overlapping rounds between adjacent windows
    is determined by the window size and stride.  For example, a window size of w and a stride of s
    indicates adjacent windows will overlap on w - s rounds.  The "commit region" for each window
    therefore corresponds to the first s rounds in the window.

    Visually:

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
                its first coordinate in DetectorErrorModel.get_detector_coordinates().
                WARNING: if a detector_to_time mapping is not None, it will be assumed to be
                both valid compatible with any detector error model that this decoder is later
                compiled to with SlidingWindowDecoder.compile_decoder_for_dem.
            simplify: Whether merge equivalent errors in a DEM when compiling a decoder for that DEM.
            decompose_errors: Whether to decompose errors according to their suggested decomposition
                when compiling a decoder for a DEM.
            **decoder_kwargs: Arguments to pass to qldpc.decoders.get_decoder when compiling a
                custom decoder from a detector error model.
        """
        SinterDecoder.__init__(
            self, simplify=simplify, decompose_errors=decompose_errors, **decoder_kwargs
        )

        if not window_size >= stride > 0:  # pragma: no cover
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
        if not self.detector_to_time:
            dem_coords = dem.get_detector_coordinates()
            self.detector_to_time = lambda det: int(dem_coords[det][0])

        # construct windows defined by "detection" and "commit" regions
        self.windows = []
        for detectors in self.detector_subsets or [range(dem.num_detectors)]:
            # collect detectors according to their time index
            time_to_dets: dict[int, list[int]] = collections.defaultdict(list)
            for detector in detectors:
                time = self.detector_to_time(detector)
                if not isinstance(time, int):  # pragma: no cover
                    raise ValueError(
                        f"detector {detector} has an invalid (non-integer) time index: {time}"
                    )
                time_to_dets[time].append(detector)

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

        return SequentialWindowDecoder.compile_decoder_for_dem(self, dem)
