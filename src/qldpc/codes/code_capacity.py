# SPDX-License-Identifier: Apache-2.0

"""Decoder and detector-error-model helpers for code-capacity experiments.

Code-capacity sampling only ever asks a decoder which observables an error flips.  These helpers
build that observable-decoder view from code syndrome and observable maps.  An error decoder is used
by converting the physical errors that it infers into observable values.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import cast

import galois
import numpy as np
import numpy.typing as npt
import scipy.sparse
import stim

from qldpc import decoders
from qldpc.decoders.adapters.observables import (
    BitPackedObservableDecoder,
    ErrorsToFieldObservablesDecoder,
    validate_decoder_output,
)
from qldpc.decoders.capabilities import constructs_observable_decoder
from qldpc.decoders.custom import PLACEHOLDER_ERROR_RATE


def get_code_capacity_dem(
    syndrome_matrix: galois.FieldArray,
    observable_matrix: galois.FieldArray | None,
    dem_errors: galois.FieldArray | None = None,
    *,
    symplectic_errors: bool = False,
    error_probs: npt.NDArray[np.floating] | float = PLACEHOLDER_ERROR_RATE,
) -> stim.DetectorErrorModel:
    """Build the detector error model of one sector of a code-capacity experiment.

    Error mechanism j of the model is the error in column j of dem_errors.  By default, each error
    location is an error mechanism.  With symplectic_errors, the mechanisms are instead the
    single-qudit X, Z, and Y errors, in that order.  An error mechanism flips the detectors of its
    syndrome ``syndrome_matrix @ error``, and the observables of its observable values
    ``observable_matrix @ error``.  If observable_matrix is None, every error location is itself an
    observable.

    The error probabilities must be fixed while the returned code-capacity estimator is evaluated
    at different physical error rates.  A code-capacity estimate reuses the decoding outcomes of
    fixed-weight errors at every physical error rate, so it requires such a decoder.

    Raises:
        ValueError: If the matrices are not binary, since Stim detector error models are binary.
    """
    field = type(syndrome_matrix)
    if getattr(field, "order", 2) != 2:
        raise ValueError(
            "A Sinter-style decoder is compiled for a Stim detector error model, which is binary, so"
            f" it cannot decode a code over {field.name}.  Pass decoder settings such as"
            " decoders.guf(), or a prebuilt observable decoder (such as an ObservableLookupDecoder)"
            " built for this code, instead"
        )
    if symplectic_errors:
        if dem_errors is not None:
            raise ValueError("dem_errors and symplectic_errors cannot both be specified")
        if observable_matrix is None:
            raise ValueError("symplectic_errors requires an observable_matrix")
        detector_flip_matrix = _get_single_qudit_error_effects(syndrome_matrix)
        observable_flip_matrix = _get_single_qudit_error_effects(observable_matrix)
    elif dem_errors is not None:
        detector_flip_matrix = syndrome_matrix @ dem_errors
        observable_flip_matrix = (
            dem_errors if observable_matrix is None else observable_matrix @ dem_errors
        )
    else:
        detector_flip_matrix = syndrome_matrix
        observable_flip_matrix = (
            observable_matrix
            if observable_matrix is not None
            else scipy.sparse.identity(syndrome_matrix.shape[1], dtype=np.uint8, format="csc")
        )
    dem_arrays = decoders.DetectorErrorModelArrays.from_arrays(
        np.asarray(detector_flip_matrix, dtype=np.uint8),
        observable_flip_matrix,
        error_probs,
    )
    return dem_arrays.to_dem()


def _get_single_qudit_error_effects(matrix: galois.FieldArray) -> galois.FieldArray:
    """Apply a symplectic map to single-qudit X, Z, and Y errors without forming those errors."""
    components_x, components_z = np.hsplit(matrix, 2)
    return np.hstack([components_x, components_z, components_x + components_z]).view(type(matrix))


@dataclasses.dataclass(frozen=True)
class CodeCapacityDecoder:
    """Observable decoder for one sector of a code-capacity experiment.

    A code-capacity experiment samples errors, and decodes the syndrome ``syndrome_matrix @ error``
    of each error to predict the values ``observable_matrix @ error`` of its observables.  Decoding
    fails if the prediction differs from these values, and is discarded if the decoder signals
    erasure by setting any of the num_erasure_flags flags that it appends to each prediction.
    Matrices are arrays over the field of the code; None denotes an identity observable map.

    Build a CodeCapacityDecoder with get_code_capacity_decoder.
    """

    # predicts the values of the observables of an error from its syndrome
    decoder: decoders.ObservableDecoder
    # maps an error to the syndrome that the decoder decodes
    syndrome_matrix: galois.FieldArray
    # maps an error to the values of its observables
    observable_matrix: galois.FieldArray | None
    # number of erasure flags that the decoder appends to each prediction
    num_erasure_flags: int = 0

    @property
    def field(self) -> type[galois.FieldArray]:
        """The field of the syndromes and observables of this decoder."""
        return type(self.syndrome_matrix)

    @property
    def num_observables(self) -> int:
        """The number of observables that this decoder predicts."""
        return (
            self.syndrome_matrix.shape[1]
            if self.observable_matrix is None
            else len(self.observable_matrix)
        )

    @property
    def can_discard(self) -> bool:
        """Whether this decoder can signal erasure, and thereby discard a sample."""
        return self.num_erasure_flags > 0

    @staticmethod
    def from_error_decoder(
        error_decoder: decoders.ErrorDecoder,
        syndrome_matrix: galois.FieldArray,
        observable_matrix: galois.FieldArray | None,
    ) -> CodeCapacityDecoder:
        """Predict observables by converting the errors that an error decoder infers."""
        observable_decoder = ErrorsToFieldObservablesDecoder(
            error_decoder,
            observable_matrix,
            type(syndrome_matrix),
            syndrome_matrix.shape[1],
        )
        return CodeCapacityDecoder(
            observable_decoder,
            syndrome_matrix,
            observable_matrix,
            int(observable_decoder.has_erasure_bit),
        )

    def reuse_for(
        self,
        syndrome_matrix: galois.FieldArray,
        observable_matrix: galois.FieldArray | None,
    ) -> CodeCapacityDecoder | None:
        """Reuse this decoder for another sector with the same syndrome matrix, if possible.

        The decoder is reused as is if the observable matrices are also equal.  An error decoder is
        reused with a different observable matrix, since the errors that it infers do not depend on
        observables.  Otherwise, return None: an observable decoder only predicts the observables
        that it was built for.
        """
        if not np.array_equal(syndrome_matrix, self.syndrome_matrix):
            return None
        if _observable_matrices_equal(observable_matrix, self.observable_matrix):
            return self
        if isinstance(self.decoder, ErrorsToFieldObservablesDecoder):
            return CodeCapacityDecoder.from_error_decoder(
                self.decoder.error_decoder, syndrome_matrix, observable_matrix
            )
        return None

    def decode(self, syndrome: galois.FieldArray) -> tuple[galois.FieldArray, bool]:
        """Predict the observable values of an error from its syndrome, and whether it was erased.

        Raises:
            ValueError: If the decoder returns a prediction of the wrong shape, with entries that
                are not elements of the field of the code, or with erasure flags other than 0 or 1.
        """
        prediction = np.asarray(self.decoder.decode_observables(syndrome.view(np.ndarray)))
        validate_decoder_output(
            prediction,
            self.num_observables,
            self.num_erasure_flags,
            self.field,
            "An observable decoder predicted observable values",
        )
        observables = self.field(prediction[: self.num_observables].astype(int))
        return observables, bool(np.any(prediction[self.num_observables :]))

    def get_failure_and_erasure(self, error: galois.FieldArray) -> tuple[bool, bool]:
        """Decode the syndrome of an error, and report whether decoding failed or was erased.

        Decoding fails if the predicted observable values differ from those of the error.  An erased
        sample is not a failure.
        """
        predicted_observables, erased = self.decode(self.syndrome_matrix @ error)
        if erased:
            return False, True
        actual_observables = (
            error if self.observable_matrix is None else self.observable_matrix @ error
        )
        return bool(np.any(predicted_observables != actual_observables)), False


def _observable_matrices_equal(
    matrix_a: galois.FieldArray | None, matrix_b: galois.FieldArray | None
) -> bool:
    """Whether two observable maps, where None denotes the identity, are equal."""
    if matrix_a is None or matrix_b is None:
        return matrix_a is matrix_b
    return bool(np.array_equal(matrix_a, matrix_b))


def get_code_capacity_decoder(
    syndrome_matrix: galois.FieldArray,
    observable_matrix: galois.FieldArray | None,
    decoder: decoders.ErrorDecoderInput | decoders.ObservableDecoderInput,
    decoder_args: Mapping[str, object] | None = None,
    *,
    dem_errors: galois.FieldArray | None = None,
    symplectic_dem_errors: bool = False,
    dem_error_weights: npt.NDArray[np.floating] | None = None,
    prebuilt_rejection_reason: str | None = None,
    warn_deprecated: bool = True,
) -> CodeCapacityDecoder:
    """Build an observable decoder for one sector of a code-capacity experiment.

    Code-capacity sampling decodes the syndrome ``syndrome_matrix @ error`` of each sampled error to
    predict its observable values ``observable_matrix @ error``.  The decoder input is resolved into
    an observable decoder as follows:

    - A Sinter-style decoder (an object with a compile_decoder_for_dem method, such as a
      decoders.SinterDecoder), or a constructor explicitly declared to return an observable decoder,
      is built for the detector error model that get_code_capacity_dem constructs.  This requires
      the matrices to be binary.
    - A prebuilt observable decoder (an object with a decode_observables method, or a compiled
      Sinter decoder with a decode_shots_bit_packed method, that is not also an error decoder) is
      used as is.  It must predict the observable values ``observable_matrix @ error`` from the
      syndrome ``syndrome_matrix @ error``.
    - Anything else (None, decoder settings, a constructor, or a prebuilt error decoder, together
      with any deprecated decoder_args) builds an error decoder for syndrome_matrix, exactly as
      qldpc.decoders.resolve_decoder does.  The observable values of the errors that it infers are
      its predictions.  A decoder that is both an error decoder and an observable decoder, such as a
      RelayBPDecoder, is used as an error decoder.

    This intentionally differs from qldpc.decoders.resolve_observable_decoder, which may ask decoder
    settings to build a native observable decoder.  Code-capacity settings retain their historical
    error-decoder semantics; direct observable decoding must be requested explicitly.

    Args:
        syndrome_matrix: The matrix that maps an error to its syndrome.
        observable_matrix: The matrix that maps an error to its observable values, or None if every
            error location is itself an observable.
        decoder: The decoder input.
        decoder_args: Deprecated keyword-based decoder options, which build an error decoder.
        dem_errors: The errors of the error mechanisms of the detector error model for which a
            Sinter-style decoder is compiled, as columns of a matrix.  Defaults to the identity
            matrix, making each error location an error mechanism.
        symplectic_dem_errors: Whether the detector error model has one X, Z, and Y mechanism per
            qudit.  Cannot be combined with dem_errors.
        dem_error_weights: Relative probabilities for the detector error model's error mechanisms.
            These are scaled by a fixed placeholder error rate, so decoder decisions do not change
            when the returned estimator is evaluated at different physical error rates.
        prebuilt_rejection_reason: If not None, reject a prebuilt (error or observable) decoder,
            with this reason; see help(qldpc.decoders.reject_prebuilt_decoder).  A Sinter-style
            decoder is compiled here, so it is not rejected.
        warn_deprecated: Whether to warn when decoder_args is nonempty.

    Returns:
        A CodeCapacityDecoder.
    """
    decoder_args = decoder_args or {}
    dem_error_probs: npt.NDArray[np.floating] | float = PLACEHOLDER_ERROR_RATE
    if dem_error_weights is not None:
        dem_error_probs = PLACEHOLDER_ERROR_RATE * np.asarray(dem_error_weights, dtype=float)
    if not decoder_args and decoders.compiles_for_dem(decoder):
        dem = get_code_capacity_dem(
            syndrome_matrix,
            observable_matrix,
            dem_errors,
            symplectic_errors=symplectic_dem_errors,
            error_probs=dem_error_probs,
        )
        compiled_decoder = decoder.compile_decoder_for_dem(dem=dem)  # type:ignore[union-attr]
        return _get_observable_code_capacity_decoder(
            compiled_decoder,
            syndrome_matrix,
            observable_matrix,
            "A decoder compiled by compile_decoder_for_dem",
            require_dimensions=False,
        )

    if prebuilt_rejection_reason is not None:
        decoders.reject_prebuilt_decoder(decoder, prebuilt_rejection_reason)

    if not decoder_args and decoders.is_prebuilt_observable_decoder(decoder):
        return _get_observable_code_capacity_decoder(
            decoder,
            syndrome_matrix,
            observable_matrix,
            "A prebuilt observable decoder",
            require_dimensions=True,
        )

    if not decoder_args and constructs_observable_decoder(decoder):
        dem = get_code_capacity_dem(
            syndrome_matrix,
            observable_matrix,
            dem_errors,
            symplectic_errors=symplectic_dem_errors,
            error_probs=dem_error_probs,
        )
        constructor = cast(decoders.ObservableDecoderConstructor, decoder)
        return _get_observable_code_capacity_decoder(
            constructor(dem),
            syndrome_matrix,
            observable_matrix,
            "An observable decoder constructor",
            require_dimensions=False,
        )

    error_decoder = decoders.resolve_decoder(
        syndrome_matrix,
        decoder,  # type:ignore[arg-type]
        decoder_args,
        warn_deprecated=warn_deprecated,
    )
    return CodeCapacityDecoder.from_error_decoder(error_decoder, syndrome_matrix, observable_matrix)


def _get_observable_code_capacity_decoder(
    decoder: object,
    syndrome_matrix: galois.FieldArray,
    observable_matrix: galois.FieldArray | None,
    source: str,
    *,
    require_dimensions: bool,
) -> CodeCapacityDecoder:
    """Wrap a prebuilt or compiled observable decoder, checking that it fits the given matrices."""
    field = type(syndrome_matrix)
    num_detectors = len(syndrome_matrix)
    num_observables = (
        syndrome_matrix.shape[1] if observable_matrix is None else len(observable_matrix)
    )
    observable_decoder: decoders.ObservableDecoder
    if isinstance(decoder, decoders.CompiledSinterDecoder):
        if field.order != 2:
            raise ValueError(f"{source} is binary, so it cannot decode a code over {field.name}")
        observable_decoder, num_erasure_flags = decoder, decoder.num_erasure_bits
    elif isinstance(decoder, decoders.ObservableDecoder):
        num_erasure_flags = int(bool(getattr(decoder, "has_erasure_bit", False)))
        observable_decoder = decoder
    elif callable(getattr(decoder, "decode_shots_bit_packed", None)):
        if field.order != 2:
            raise ValueError(f"{source} is binary, so it cannot decode a code over {field.name}")
        observable_decoder = BitPackedObservableDecoder(decoder, num_observables)
        num_erasure_flags = 1
    else:
        raise TypeError(
            f"{source} must provide a decode_observables or decode_shots_bit_packed method"
        )

    dimensions = [("num_detectors", num_detectors), ("num_observables", num_observables)]
    for name, expected in dimensions:
        value = getattr(decoder, name, None)
        if (
            require_dimensions
            and not isinstance(decoder, decoders.ObservableDecoder)
            and not isinstance(value, (int, np.integer))
        ):
            raise ValueError(
                f"{source} does not declare {name}, so its compatibility with this code-capacity"
                " sector cannot be validated"
            )
        if isinstance(value, (int, np.integer)) and value != expected:
            raise ValueError(
                f"{source} has {name}={value}, but this code-capacity sector has {expected}.  An"
                " observable decoder must be built for the syndromes and observables of the sector"
                " that it decodes"
            )
    decoder_field = getattr(decoder, "field", None)
    if (
        isinstance(decoder_field, type)
        and issubclass(decoder_field, galois.FieldArray)
        and decoder_field is not field
    ):
        raise ValueError(
            f"{source} is built over {decoder_field.name}, but this code-capacity sector is over"
            f" {field.name}"
        )
    return CodeCapacityDecoder(
        observable_decoder, syndrome_matrix, observable_matrix, num_erasure_flags
    )
