from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.sparse import csc_matrix
from stim import DemTarget, DemTargetWithCoords, DetectorErrorModel

from qldpc.objects import Pauli, PauliXZ


@dataclass(frozen=True)
class CircuitLevelError:
    dets: tuple[DemTargetWithCoords, ...]
    obs: tuple[DemTarget, ...]
    basis: PauliXZ


@dataclass
class CheckMatrices:
    check_map: csc_matrix
    check_matrix: csc_matrix
    obs_matrix: csc_matrix
    priors: npt.NDArray[np.float64]


def _det_basis_coord(det: DemTargetWithCoords) -> PauliXZ:
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
    det_list: list[DemTarget] = []
    det_map: dict[DemTarget, int] = {}
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
    check_map = csc_matrix(
        (np.ones(len(det_map)), (list(det_map.values()), list(det_map.keys()))),
        shape=(len(det_list), num_detectors),
    )
    check_matrix = csc_matrix(
        (np.ones(len(det_row_idx)), (det_row_idx, det_col_idx)),
        shape=(len(det_list), len(prior_dict)),
    )
    obs_matrix = csc_matrix(
        (np.ones(len(obs_row_idx)), (obs_row_idx, obs_col_idx)),
        shape=(len(obs_list), len(prior_dict)),
    )

    return CheckMatrices(check_map, check_matrix, obs_matrix, np.array(priors))


def detector_error_model_to_css_checks(
    dem: DetectorErrorModel,
    fn_det_basis: Callable[[DemTargetWithCoords], PauliXZ] = _det_basis_coord,
) -> tuple[CheckMatrices, CheckMatrices]:
    """
    Convert a DetectorErrorModel into separate Z/X check matrices

    args:
        dem: DetectorErrorModel
            The detector error model to convert
        fn_det_basis: Callable[[DemTargetWithCoords], PauliXZ]
            A function that takes a detector and returns the basis of the CSS stabilizer it checks (Z/X)
            By default, the 1st coordinate of the detector is used to determine the basis (1 == Z, 2 == X)
    returns:
        tuple[CheckMatrices, CheckMatrices]
            The Z and X check matrices
    """
    det_coords: dict[int, list[float]] = dem.get_detector_coordinates()

    z_error_priors: dict[CircuitLevelError, float] = {}
    x_error_priors: dict[CircuitLevelError, float] = {}
    for instr in dem.flattened():
        if instr.type == "error":
            prior = instr.args_copy()[0]
            z_dets: list[DemTarget] = []
            x_dets: list[DemTarget] = []
            obs: list[DemTarget] = []
            for targ in instr.targets_copy():
                if targ.is_relative_detector_id():
                    det = DemTargetWithCoords(dem_target=targ, coords=det_coords[targ.val])
                    basis = fn_det_basis(det)
                    if basis is Pauli.Z:
                        z_dets.append(det)
                    elif basis is Pauli.X:
                        x_dets.append(det)
                    else:
                        raise ValueError(f"Invalid basis: {basis}")
                elif targ.is_logical_observable_id():
                    obs.append(targ)

            if len(z_dets) > 0:
                z_error = CircuitLevelError(tuple(z_dets), tuple(obs), Pauli.Z)
                z_error_priors[z_error] = z_error_priors.setdefault(z_error, 0) + prior

            if len(x_dets) > 0:
                x_error = CircuitLevelError(tuple(x_dets), tuple(obs), Pauli.X)
                x_error_priors[x_error] = x_error_priors.setdefault(x_error, 0) + prior

    z_check_matrices = _prior_dict_to_matrices(
        z_error_priors, dem.num_detectors, dem.num_observables
    )
    x_check_matrices = _prior_dict_to_matrices(
        x_error_priors, dem.num_detectors, dem.num_observables
    )

    return z_check_matrices, x_check_matrices
