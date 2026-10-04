# SPDX-License-Identifier: Apache-2.0

"""Decoder and detector-error-model helpers for code-capacity experiments.

Code-capacity sampling only ever asks a decoder which observables an error flips.  These helpers
build that observable-decoder view from code syndrome and observable maps.  An error decoder is used
by converting the physical errors that it infers into observable values.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Mapping

import galois
import numpy as np
import numpy.typing as npt
import scipy.sparse
import stim

from qldpc._util import get_external_caller_stacklevel
from qldpc.decoders.adapters.observable_decoders import (
    BitPackedObservableDecoder,
    ErrorsToFieldObservablesDecoder,
    validate_decoder_output,
)
from qldpc.decoders.capabilities import (
    compiles_for_dem,
    is_prebuilt_observable_decoder,
)
from qldpc.decoders.common import PLACEHOLDER_ERROR_RATE, ObservableDecodingFallbackWarning
from qldpc.decoders.construction.factories import DEMDecoderFactory
from qldpc.decoders.construction.resolution import _resolve_error_decoder, reject_prebuilt_decoder
from qldpc.decoders.construction.specs import (
    DecoderInput,
    DecoderSpec,
)
from qldpc.decoders.dems import DetectorErrorModelArrays
from qldpc.decoders.protocols import ErrorDecoder, ObservableDecoder
from qldpc.decoders.sinter.core import CompiledSinterDecoder

_MAX_SPEC_REPR_LENGTH = 80  # truncate long option values in warnings


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
            "An observable decoder built from a Stim detector error model requires a binary code, so"
            f" it cannot decode a code over {field.name}.  Pass a decoder specification such as"
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
    dem_arrays = DetectorErrorModelArrays.from_arrays(
        np.asarray(detector_flip_matrix, dtype=np.uint8),
        observable_flip_matrix,
        error_probs,
    )
    return dem_arrays.to_dem()


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
    decoder: ObservableDecoder
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
        error_decoder: ErrorDecoder,
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


def get_code_capacity_decoder(
    syndrome_matrix: galois.FieldArray,
    observable_matrix: galois.FieldArray | None,
    decoder: DecoderInput,
    decoder_args: Mapping[str, object] | None = None,
    *,
    dem_errors: galois.FieldArray | None = None,
    symplectic_dem_errors: bool = False,
    dem_error_weights: npt.NDArray[np.floating] | None = None,
    prebuilt_rejection_reason: str | None = None,
    warn_deprecated: bool = True,
) -> CodeCapacityDecoder:
    """Build an observable decoder for one sector of a code-capacity experiment.

    Code-capacity sampling predicts ``observable_matrix @ error`` from the syndrome
    ``syndrome_matrix @ error``.  Pass a decoder specification such as ``decoders.bp_osd(...)``
    as ``decoder=``.

    For a binary code, a specification with native observable support is first built as an
    observable decoder for the code-capacity detector error model.  If that build fails, for example
    because the specification sets matrix-only options or the backend rejects the model's structure,
    an ObservableDecodingFallbackWarning is emitted and an error decoder is built for
    syndrome_matrix instead.  Wrap a specification as ``decoders.from_matrix(spec.build)`` to
    select error decoding explicitly.

    Args:
        syndrome_matrix: The matrix that maps an error to its syndrome.
        observable_matrix: The matrix that maps an error to its observable values, or None if every
            error location is itself an observable.
        decoder: A decoder specification for this sector, or None for the default decoder.
        decoder_args: Deprecated keyword-based decoder options, which build an error decoder.
        dem_errors: The errors of the error mechanisms of the detector error model for which a
            Sinter-style decoder is compiled, as columns of a matrix.  Defaults to the identity
            matrix, making each error location an error mechanism.
        symplectic_dem_errors: Whether the detector error model has one X, Z, and Y mechanism per
            qudit.  Cannot be combined with dem_errors.
        dem_error_weights: Relative probabilities for the detector error model's error mechanisms
            when building a native observable decoder or compiling a Sinter-style decoder.  These
            are scaled by a fixed placeholder error rate, so decoder decisions do not change when
            the returned estimator is evaluated at different physical error rates.
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
    if not decoder_args and compiles_for_dem(decoder):
        dem = get_code_capacity_dem(
            syndrome_matrix,
            observable_matrix,
            dem_errors,
            symplectic_errors=symplectic_dem_errors,
            error_probs=dem_error_probs,
        )
        compiled_decoder = decoder.compile_decoder_for_dem(dem=dem)
        return _get_observable_code_capacity_decoder(
            compiled_decoder,
            syndrome_matrix,
            observable_matrix,
            "A decoder compiled by compile_decoder_for_dem",
            require_dimensions=False,
        )

    if prebuilt_rejection_reason is not None:
        reject_prebuilt_decoder(decoder, prebuilt_rejection_reason)

    if not decoder_args and is_prebuilt_observable_decoder(decoder):
        return _get_observable_code_capacity_decoder(
            decoder,
            syndrome_matrix,
            observable_matrix,
            "A prebuilt observable decoder",
            require_dimensions=True,
        )

    if not decoder_args and isinstance(decoder, DEMDecoderFactory):
        dem = get_code_capacity_dem(
            syndrome_matrix,
            observable_matrix,
            dem_errors,
            symplectic_errors=symplectic_dem_errors,
            error_probs=dem_error_probs,
        )
        return _get_observable_code_capacity_decoder(
            decoder.build(dem),
            syndrome_matrix,
            observable_matrix,
            "A from_dem factory",
            require_dimensions=False,
        )

    native_error: Exception | None = None
    if not decoder_args and isinstance(decoder, DecoderSpec):
        is_binary = getattr(type(syndrome_matrix), "order", 2) == 2
        if not decoder.infers_errors or (is_binary and decoder.predicts_observables_natively):
            dem = get_code_capacity_dem(
                syndrome_matrix,
                observable_matrix,
                dem_errors,
                symplectic_errors=symplectic_dem_errors,
                error_probs=dem_error_probs,
            )
            if not decoder.infers_errors:
                # an observable-only specification has nothing to fall back to
                observable_decoder = decoder.build_observable_decoder(dem)
            else:
                try:
                    observable_decoder = decoder.build_observable_decoder(dem)
                except (ValueError, TypeError) as error:
                    native_error = error
            if native_error is None:
                return _get_observable_code_capacity_decoder(
                    observable_decoder,
                    syndrome_matrix,
                    observable_matrix,
                    "A decoder specification",
                    require_dimensions=False,
                )
            spec_repr = repr(decoder)
            if len(spec_repr) > _MAX_SPEC_REPR_LENGTH:
                spec_repr = spec_repr[: _MAX_SPEC_REPR_LENGTH - 4] + "...)"
            warnings.warn(
                f"{spec_repr} could not build an observable decoder for this code-capacity"
                f" detector error model ({type(native_error).__name__}: {native_error}), so qLDPC"
                " is decoding errors with the syndrome matrix instead.  Pass"
                " decoder=decoders.from_matrix(spec.build) to select error decoding explicitly",
                ObservableDecodingFallbackWarning,
                stacklevel=get_external_caller_stacklevel(),
            )

    try:
        error_decoder = _resolve_error_decoder(
            syndrome_matrix,
            decoder,  # type:ignore[arg-type]
            decoder_args,
            warn_deprecated=warn_deprecated,
        )
    except (ValueError, TypeError) as error:
        if native_error is not None:
            raise error from native_error
        raise
    return CodeCapacityDecoder.from_error_decoder(error_decoder, syndrome_matrix, observable_matrix)


# Private helpers


def _get_single_qudit_error_effects(matrix: galois.FieldArray) -> galois.FieldArray:
    """Apply a symplectic map to single-qudit X, Z, and Y errors without forming those errors."""
    components_x, components_z = np.hsplit(matrix, 2)
    return np.hstack([components_x, components_z, components_x + components_z]).view(type(matrix))


def _observable_matrices_equal(
    matrix_a: galois.FieldArray | None, matrix_b: galois.FieldArray | None
) -> bool:
    """Whether two observable maps, where None denotes the identity, are equal."""
    if matrix_a is None or matrix_b is None:
        return matrix_a is matrix_b
    return bool(np.array_equal(matrix_a, matrix_b))


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
    observable_decoder: ObservableDecoder
    if isinstance(decoder, CompiledSinterDecoder):
        if field.order != 2:
            raise ValueError(f"{source} is binary, so it cannot decode a code over {field.name}")
        observable_decoder, num_erasure_flags = decoder, decoder.num_erasure_bits
    elif isinstance(decoder, ObservableDecoder):
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
            and not isinstance(decoder, ObservableDecoder)
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
