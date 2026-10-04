# SPDX-License-Identifier: Apache-2.0

"""Sequential and sliding-window Sinter decoders."""

from __future__ import annotations

import collections
import itertools
from collections.abc import Callable, Collection, Sequence

import numpy as np
import numpy.typing as npt
import stim

from ..adapters.error_decoders import match_error_decoder_to_dem
from ..construction.resolution import _resolve_error_decoder
from ..construction.specs import DeferredErrorDecoderInput
from ..dems import DetectorErrorModelArrays
from ..protocols import ErrorDecoder, batch_decode_errors
from .core import CompiledSinterDecoder, SinterDecoder


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

    _prebuilt_decoder_rejection_reason = (
        "a SequentialWindowDecoder builds a new error decoder for each window"
    )

    # the __init__ method of a window decoder only accepts inputs for error decoders
    decoder_input: DeferredErrorDecoderInput

    def __init__(
        self,
        detection_regions: Sequence[Collection[int]],
        commit_regions: Sequence[Collection[int]] | None = None,
        *,
        simplify: bool = True,
        decompose_errors: bool = False,
        decoder: DeferredErrorDecoderInput = None,
        **decoder_kwargs: object,
    ) -> None:
        """Initialize an observable decoder that splits a detector error model into windows.

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
            decoder: A specification such as ``decoders.bp_osd(...)``, or a constructor that builds
                an error decoder from a detector error model, or None to
                select the default error decoder.  Windows commit the errors that they infer, so
                they require error decoders.  A prebuilt decoder is rejected, because an inner
                decoder is built for each window.
            **decoder_kwargs: Deprecated arguments to pass to qldpc.decoders.get_decoder.
        """
        SinterDecoder.__init__(
            self,
            simplify=simplify,
            decompose_errors=decompose_errors,
            decoder=decoder,
            **decoder_kwargs,
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
            window_decoder = _resolve_error_decoder(
                window_dem,
                self.decoder_input,
                self.decoder_kwargs.copy(),
                warn_deprecated=False,
            )
            # Windows commit inferred errors, so they need error decoders.  Restricting the DEM to
            # this window may leave equivalent error mechanisms, which the window_decoder may merge
            # into one, so align the errors that it infers with the error mechanisms of the window.
            window_decoder = match_error_decoder_to_dem(window_decoder, window_dem)

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
        window_decoders: Sequence[ErrorDecoder],
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
            decoded_error = batch_decode_errors(decoder, syndromes)
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
        decoder: DeferredErrorDecoderInput = None,
        **decoder_kwargs: object,
    ) -> None:
        """Initialize an observable decoder that splits a model into temporal windows.

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
                first coordinate unless that coordinate decreases from one detector to the next.
                Otherwise, a later coordinate that varies and never decreases is read instead,
                provided exactly one qualifies.  A constant first coordinate is retained because a
                later monotone coordinate may merely enumerate checks within one round.  Pass an
                explicit detector_to_time mapping when this fallback does not match the model's
                coordinate convention.  Only the detectors that get windowed are consulted, and one
                of those with no coordinates at all is rejected, since there is nothing to read a
                time index from.  A non-None ``detector_to_time`` mapping is assumed to be valid and
                compatible with every detector error model that this decoder is later compiled to
                with ``SlidingWindowDecoder.compile_decoder_for_dem``.
            simplify: Whether to merge equivalent errors in a DEM when compiling a decoder for
                that DEM.
            decompose_errors: Whether to decompose errors according to their suggested decomposition
                when compiling a decoder for a DEM.
            decoder: A specification such as ``decoders.bp_osd(...)``, or a constructor that builds
                an error decoder from a detector error model, or None to
                select the default error decoder.  Windows commit the errors that they infer, so
                they require error decoders.  A prebuilt decoder is rejected, because an inner
                decoder is built for each window.
            **decoder_kwargs: Deprecated arguments to pass to qldpc.decoders.get_decoder.
        """
        SinterDecoder.__init__(
            self,
            simplify=simplify,
            decompose_errors=decompose_errors,
            decoder=decoder,
            **decoder_kwargs,
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

        .. warning::
            If this decoder was initialized with a ``detector_to_time`` mapping, the mapping is
            assumed to be valid and compatible with the detector error model provided here.

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


def _time_coordinate(dem_coords: dict[int, list[float]]) -> int:
    """Which detector coordinate of a detector error model indexes time.

    Detector coordinates are assigned as a circuit is built, and a circuit runs forward, so a
    coordinate that indexes time never decreases from one detector to the next.  The first
    coordinate is used whenever it has that property, even if it is constant: a later monotone
    coordinate may enumerate checks within one round.  Otherwise a later coordinate that varies and
    never decreases is used instead -- the circuits that stim generates place time last, for
    example.  If no later coordinate qualifies, or if several do, the first coordinate is used as a
    fallback.
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
