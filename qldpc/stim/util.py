from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import scipy.sparse
import stim

from qldpc.objects import Pauli, PauliXZ


@dataclasses.dataclass(frozen=True)
class CircuitLevelError:
    dets: tuple[stim.DemTargetWithCoords, ...]
    obs: tuple[stim.DemTarget, ...]
    basis: PauliXZ


@dataclasses.dataclass
class CheckMatrices:
    check_map: scipy.sparse.csc_matrix
    check_matrix: scipy.sparse.csc_matrix
    obs_matrix: scipy.sparse.csc_matrix
    priors: npt.NDArray[np.float64]


def _det_basis_coord(det: stim.DemTargetWithCoords) -> PauliXZ:
    """
    Returns the basis of the detector based on the 1st coordinate (1 == X, 2 == Z)
    """
    if det.coords[0] == 1:
        return Pauli.X
    elif det.coords[0] == 2:
        return Pauli.Z
    else:
        raise ValueError(f"Invalid basis: {det.coords[0]} (must be 1 or 2)")


def _prior_dict_to_matrices(
    prior_dict: dict[CircuitLevelError, float], num_detectors: int, num_obs: int
) -> CheckMatrices:
    det_list: list[stim.DemTarget] = []
    det_map: dict[stim.DemTarget, int] = {}
    det_row_idx: list[int] = []
    det_col_idx: list[int] = []

    obs_list: list[int] = list(range(num_obs))
    obs_row_idx: list[int] = []
    obs_col_idx: list[int] = []

    priors: list[float] = []

    for i, (c_err, prior) in enumerate(prior_dict.items()):
        priors.append(prior)

        for det in c_err.dets:
            det_val = det.dem_target.val
            if det not in det_list:
                det_map[det_val] = len(det_list)
                det_list += [det]
            det_row_idx += [det_map[det_val]]
            det_col_idx += [i]

        for obs in c_err.obs:
            obs_row_idx += [obs.val]
            obs_col_idx += [i]

    # Resulting check matrix may have fewer dets than original
    check_map = scipy.sparse.csc_matrix(
        (np.ones(len(det_map)), (list(det_map.values()), list(det_map.keys()))),
        shape=(len(det_list), num_detectors),
    )
    check_matrix = scipy.sparse.csc_matrix(
        (np.ones(len(det_row_idx)), (det_row_idx, det_col_idx)),
        shape=(len(det_list), len(prior_dict)),
    )
    obs_matrix = scipy.sparse.csc_matrix(
        (np.ones(len(obs_row_idx)), (obs_row_idx, obs_col_idx)),
        shape=(len(obs_list), len(prior_dict)),
    )

    return CheckMatrices(check_map, check_matrix, obs_matrix, np.array(priors))


def detector_error_model_to_css_checks(
    dem: stim.DetectorErrorModel,
    fn_det_basis: Callable[[stim.DemTargetWithCoords], PauliXZ] = _det_basis_coord,
) -> tuple[CheckMatrices, CheckMatrices]:
    """
    Convert a stim.DetectorErrorModel into separate Z/X check matrices

    Args:
        dem: stim.DetectorErrorModel
            The detector error model to convert
        fn_det_basis: Callable[[stim.DemTargetWithCoords], PauliXZ]
            A function that takes a detector and returns the basis of the CSS stabilizer it checks (Z/X)
            By default, the 1st coordinate of the detector is used to determine the basis (1 == Z, 2 == X)
    returns:
        tuple[CheckMatrices, CheckMatrices]
            The Z and X check matrices
    """
    det_coords: dict[int, list[float]] = dem.get_detector_coordinates()

    error_priors: dict[CircuitLevelError, float] = {}
    for instr in dem.flattened():
        if instr.type == "error":
            prior = instr.args_copy()[0]
            dets: list[stim.DemTarget] = []
            obs: list[stim.DemTarget] = []

            for targ in instr.targets_copy():
                if targ.is_relative_detector_id():
                    det = stim.DemTargetWithCoords(dem_target=targ, coords=det_coords[targ.val])
                    dets.append(det)
                elif targ.is_logical_observable_id():
                    obs.append(targ)

            if len(dets) > 0:
                error = CircuitLevelError(tuple(dets), tuple(obs), Pauli.Z)
                error_priors[error] = error_priors.setdefault(error, 0) + prior

    check_matrices = _prior_dict_to_matrices(
        error_priors, dem.num_detectors, dem.num_observables
    )

    return check_matrices
