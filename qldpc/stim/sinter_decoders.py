from __future__ import annotations

import pathlib
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import sinter
from stim import DemTargetWithCoords, DetectorErrorModel

from qldpc.decoders import Decoder, get_decoder
from qldpc.objects import Pauli, PauliXZ
from qldpc.stim.util import (
    CheckMatrices,
    _det_basis_coord,
    detector_error_model_to_css_checks,
)


class CompiledSinterDecoder(sinter.CompiledDecoder):
    def __init__(self, check_matrices: CheckMatrices, decoder: Decoder) -> None:
        self.check_matrices = check_matrices
        self.decoder = decoder

    def decode_shots_bit_packed(
        self, bit_packed_detection_event_data: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.uint8]:
        obs_flip_data = []
        for shot_data in bit_packed_detection_event_data:
            unpacked_data = np.unpackbits(
                shot_data,
                bitorder="little",
                count=self.check_matrices.check_map.shape[1],
            )
            pred_errors = self.decoder.decode(self.check_matrices.check_map @ unpacked_data)
            obs_pred = (self.check_matrices.obs_matrix @ pred_errors) % 2
            obs_flip_data.append(np.packbits(obs_pred.astype(np.uint8), bitorder="little"))

        return np.array(obs_flip_data)


class SinterDecoder(sinter.Decoder):
    """
    Base class for running sinter experiments using a qldpc.decoders.Decoder via qldpc.decoders.get_decoder.

    NOTE: Currently assumes a CSS code experiment in a specified basis and therefore separates decoding into the resulting X or Z sub-problem.
    """

    def __init__(
        self,
        basis: PauliXZ,
        fn_det_basis: Callable[[DemTargetWithCoords], PauliXZ] = _det_basis_coord,
        **decoder_kwargs: object,
    ) -> None:
        """
        args:
            basis: PauliXZ
                CSS decoding sub-problem basis
            fn_det_basis: Callable[[DemTargetWithCoords], PauliXZ]
                For general CSS codes, determining a detector's basis is seemingly non-trivial.
                This function extracts this from the detector coordinates, assuming some convention.
                The default convention is based on first coordinate (1 = X, 2 = Z)
                NOTE: This function needs to be defined at the top-level of a module (i.e. not in a jupyter notebook cell) to work with Sinter
            kwargs: Any
                keyword arguments to be passed to qldpc.decoders.get_decoder when compiling a detector error model

        """
        self.basis = basis
        self.fn_det_basis = fn_det_basis
        self.decoder_kwargs = decoder_kwargs

    def compile_decoder_for_dem(self, dem: DetectorErrorModel) -> sinter.CompiledDecoder:
        z_check_matrices, x_check_matrices = detector_error_model_to_css_checks(
            dem, self.fn_det_basis
        )
        if self.basis is Pauli.Z:
            check_matrices = z_check_matrices
        elif self.basis is Pauli.X:
            check_matrices = x_check_matrices
        else:
            raise ValueError(f"Invalid basis: {self.basis}")

        decoder = get_decoder(
            check_matrices.check_matrix,
            error_channel=list(check_matrices.priors),
            **self.decoder_kwargs,
        )
        return CompiledSinterDecoder(check_matrices, decoder)

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
        raise NotImplementedError()
