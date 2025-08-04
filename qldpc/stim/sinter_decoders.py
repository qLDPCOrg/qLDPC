from __future__ import annotations

import collections
import dataclasses

import numpy as np
import numpy.typing as npt
import scipy
import scipy.sparse
import sinter
import stim

from qldpc import decoders


class CompiledSinterDecoder(sinter.CompiledDecoder):
    """Decoder usable by Sinter for decoding circuit errors, compiled to a specific circuit."""

    def __init__(self, dem_arrays: DemArrays, decoder: decoders.Decoder) -> None:
        self.dem_arrays = dem_arrays
        self.decoder = decoder
        self.num_detectors = dem_arrays.detector_flip_matrix.shape[0]

    def decode_shots_bit_packed(
        self, bit_packed_detection_event_data: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.uint8]:
        """Predicts observable flips from the given detection events.

        See help(sinter.CompiledDecoder) for additional information.
        """
        observable_flips = []
        for bit_packed_syndrome in bit_packed_detection_event_data:
            syndrome = np.unpackbits(
                bit_packed_syndrome, count=self.num_detectors, bitorder="little"
            )
            pred_errors = self.decoder.decode(self.dem_arrays.syndrome_map @ syndrome)
            obs_pred = (self.dem_arrays.observables_flip_matrix @ pred_errors) % 2
            observable_flips.append(np.packbits(obs_pred.astype(np.uint8), bitorder="little"))

        return np.array(observable_flips)


class SinterDecoder(sinter.Decoder):
    """Decoder usable by Sinter for decoding circuit errors."""

    def __init__(self, *, priors_arg: str | None = None, **decoder_kwargs: object) -> None:
        """Initialize a SinterDecoder.

        Args:
            prior_args: The keyword argument to which to pass priors about circuit-level error
                likelihoods when constructing a decoder with qldpc.decoders.get_decoder.
            decoder_kwargs: Arguments to pass to qldpc.decoders.get_decoder when compiling a
                custom decoder from a detector error model.

        """
        self.priors_arg = priors_arg
        self.decoder_kwargs = decoder_kwargs

    def compile_decoder_for_dem(self, dem: stim.DetectorErrorModel) -> sinter.CompiledDecoder:
        """Creates a decoder preconfigured for the given detector error model.

        See help(sinter.Decoder) for additional information.
        """
        dem_arrays = DemArrays(dem)
        priors_kwargs = {self.priors_arg: list(dem_arrays.error_probs)} if self.priors_arg else {}
        decoder = decoders.get_decoder(
            dem_arrays.detector_flip_matrix, **priors_kwargs, **self.decoder_kwargs
        )
        return CompiledSinterDecoder(dem_arrays, decoder)


@dataclasses.dataclass(frozen=True)
class CircuitLevelError:
    """A circuit-level error, identified by the detectors and observables that it flips."""

    detectors: tuple[stim.DemTargetWithCoords, ...]
    observables: tuple[stim.DemTarget, ...]


# TODO: think more about this and refactor
class DemArrays:
    """Representation of a stim.DetectorErrorModel by a collection of arrays."""

    error_probs: npt.NDArray[np.float64]  # the probability of occurrence for each error
    detector_flip_matrix: scipy.sparse.csc_matrix  # maps errors to detector flips
    observables_flip_matrix: scipy.sparse.csc_matrix  # maps errors to observable flips

    # matrix to pre-process syndromes (detection events)
    syndrome_map: scipy.sparse.csc_matrix  # TODO: explain why this is necessary

    def __init__(self, dem: stim.DetectorErrorModel) -> None:
        """Initialize the arrays of a given detector error model."""
        errors = DemArrays._collect_errors(dem)
        (
            self.error_probs,
            self.detector_flip_matrix,
            self.observables_flip_matrix,
            self.syndrome_map,
        ) = DemArrays._arrays_from_errors(errors, dem.num_detectors, dem.num_observables)

    @staticmethod
    def _collect_errors(dem: stim.DetectorErrorModel) -> dict[CircuitLevelError, float]:
        """Convert a stim.DetectorErrorModel into a dictionary mapping errors to likelihoods."""
        det_coords: dict[int, list[float]] = dem.get_detector_coordinates()
        errors: dict[CircuitLevelError, float] = collections.defaultdict(float)
        for instruction in dem.flattened():
            if instruction.type == "error":
                prob = instruction.args_copy()[0]
                targets = instruction.targets_copy()
                detectors = tuple(
                    stim.DemTargetWithCoords(dem_target=target, coords=det_coords[target.val])
                    for target in targets
                    if target.is_relative_detector_id()
                )
                if len(detectors) > 0:
                    observables = tuple(
                        target for target in targets if target.is_logical_observable_id()
                    )
                    error = CircuitLevelError(detectors, observables)
                    errors[error] = errors[error] * (1 - prob) + prob * (1 - errors[error])
        return errors

    @staticmethod
    def _arrays_from_errors(
        errors: dict[CircuitLevelError, float], num_detectors: int, num_observables: int
    ) -> tuple[
        npt.NDArray[np.float64],
        scipy.sparse.csc_matrix,
        scipy.sparse.csc_matrix,
        scipy.sparse.csc_matrix,
    ]:
        """Construct DemArrays instance data from a dictionary mapping errors to likelihoods."""
        detectors: list[stim.DemTarget] = []
        detector_index_map: dict[stim.DemTarget, int] = {}
        det_row_idx: list[int] = []
        det_col_idx: list[int] = []

        observables: list[int] = list(range(num_observables))
        obs_row_idx: list[int] = []
        obs_col_idx: list[int] = []

        error_probs: list[float] = []

        for erorr_index, (erorr_witnesses, error_prob) in enumerate(errors.items()):
            error_probs.append(error_prob)

            for det in erorr_witnesses.detectors:
                det_val = det.dem_target.val
                if det not in detectors:
                    detector_index_map[det_val] = len(detectors)
                    detectors += [det]
                det_row_idx += [detector_index_map[det_val]]
                det_col_idx += [erorr_index]

            for obs in erorr_witnesses.observables:
                obs_row_idx += [obs.val]
                obs_col_idx += [erorr_index]

        syndrome_map = scipy.sparse.csc_matrix(
            (
                np.ones(len(detector_index_map), dtype=int),
                (list(detector_index_map.values()), list(detector_index_map.keys())),
            ),
            shape=(len(detectors), num_detectors),
        )
        detector_flip_matrix = scipy.sparse.csc_matrix(
            (np.ones(len(det_row_idx), dtype=int), (det_row_idx, det_col_idx)),
            shape=(len(detectors), len(errors)),
        )
        observables_flip_matrix = scipy.sparse.csc_matrix(
            (np.ones(len(obs_row_idx), dtype=int), (obs_row_idx, obs_col_idx)),
            shape=(len(observables), len(errors)),
        )
        return np.array(error_probs), detector_flip_matrix, observables_flip_matrix, syndrome_map
