# SPDX-License-Identifier: Apache-2.0

"""Relay-BP decoder adapter and builders."""

from __future__ import annotations

import functools
import warnings
from collections.abc import Sequence
from typing import Any

import galois
import numpy as np
import numpy.typing as npt
import scipy.sparse
import stim

from qldpc.math import IntegerArray

from ..common import PLACEHOLDER_ERROR_RATE, _erasure_bit_support, with_erasure_bits
from ..dems import DetectorErrorModelArrays
from ..protocols import BatchErrorDecoder

# Public decoder and builders


class RelayBPDecoder(BatchErrorDecoder):
    """Wrapper class for Relay-BP decoders, introduced in arXiv:2506.01779.

    Requires ``relay_bp`` to be installed, for example via ``pip install 'qldpc[relay-bp]'``.

    This class first constructs a ``relay_bp.decoder.DynDecoder`` decoder by class name, such as
    ``RelayDecoderF32``; see ``help(relay_bp)`` for more options.  To enable parallelized decoding,
    which as of ``relay-bp==0.2.1`` is only implemented for the
    ``relay_bp.ObservableDecoderRunner`` class, ``RelayBPDecoder`` wraps the
    ``relay_bp.decoder.DynDecoder`` in a ``relay_bp.ObservableDecoderRunner`` at initialization
    time.

    A RelayBPDecoder is both an error decoder and an observable decoder: ``.decode_errors`` (or its
    alias ``.decode``) returns an inferred error, and ``.decode_observables`` returns predicted
    observable flips.  Predicting
    observable flips requires an ``observable_error_matrix``, which a detector error model provides.

    .. important::
        Relay-BP has two integration constraints:

        1. ``relay_bp.ObservableDecoderRunner`` expects an ``observable_error_matrix`` when
           initialized.  If a ``RelayBPDecoder`` is initialized without one, this matrix is set to
           ``np.empty((0, 0), dtype=np.uint8)``.  All observable-related methods of the decoder will
           subsequently fail.
        2. ``RelayBPDecoder`` "wants" to be a subclass of ``relay_bp.ObservableDecoderRunner``.
           However, the latter does not allow subclassing because it is implemented in Rust and
           exposed to Python via bindings.  As a workaround, if a ``RelayBPDecoder`` is asked for a
           method or attribute it does not recognize, such as
           ``decoder.decode_observables_batch(detectors, parallel=True)`` or
           ``decoder.decode_detailed(detectors)``, it passes all arguments to an identically named
           method of ``relay_bp.ObservableDecoderRunner``.  Consequently, most methods recognized
           by ``RelayBPDecoder`` in practice do not appear in its documentation.  See
           ``help(relay_bp.ObservableDecoderRunner)`` for a complete list.

    For details about Relay-BP decoders, see:

    - Documentation: https://pypi.org/project/relay-bp
    - Reference: https://arxiv.org/abs/2506.01779

    If initialized with ``add_erasure_bit=True``, this decoder appends a bit to all decoded errors,
    set to 1 when the error Relay-BP settles on does not reproduce the syndrome and to 0 otherwise.
    """

    def __init__(
        self,
        pcm_or_dem: IntegerArray | stim.DetectorErrorModel,
        error_priors: npt.NDArray[np.floating] | Sequence[float] | None = None,
        *,
        name: str = "RelayDecoderF32",
        observable_error_matrix: IntegerArray | None = None,
        include_decode_result: bool = False,
        add_erasure_bit: bool = False,
        **decoder_args: object,
    ) -> None:
        """Initialize a RelayBP decoder from the relay_bp package.

        Args:
            pcm_or_dem: A parity check matrix or detector error model (DEM).
            error_priors: Prior probabilities for each error, or None.  If ``error_priors is None``
                and ``pcm_or_dem`` is a DEM, these are set to the error probabilities in the DEM by
                default.
            name: The name of the RelayBP decoder to instantiate.  Must be one of the classes listed
                under ``help(relay_bp.bp)``.
            observable_error_matrix: A binary matrix whose rows specify which error mechanisms
                flip which observables, or None.  If ``pcm_or_dem`` is a DEM, this matrix is
                extracted from the DEM.  If ``pcm_or_dem`` is a matrix and
                ``observable_error_matrix is None``, the constructed ``RelayBPDecoder`` will not be
                able to predict observable flips (or logical error rates).
            include_decode_result: Argument passed to ``relay_bp.ObservableDecoderRunner``.
            add_erasure_bit: Whether to append a bit to all decoded errors, set to 1 when the
                error Relay-BP settles on does not reproduce the syndrome and to 0 otherwise.
                Without that bit, such a shot is reported as an ordinary inferred error.
            **decoder_args: Arguments passed to the "inner" (syndrome -> error) decoder from
                relay_bp.  See help(relay_bp.RelayDecoderF32) or https://pypi.org/project/relay-bp/
                for the options (alpha, alpha_iteration_scaling_factor, gamma0, etc.).
        """
        try:
            import relay_bp
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Failed to import relay-bp.  Try installing 'qldpc[relay-bp]'"
            )
        if not isinstance(name, str) or not hasattr(relay_bp, name):
            raise ValueError(
                f"Relay-BP decoder name not recognized: {name}\n"
                "See 'import relay_bp; help(relay_bp.bp)' for available Relay-BP decoders"
            )
        if isinstance(pcm_or_dem, str):
            raise TypeError(
                "I think you provided a Relay-BP decoder decoder name in place of a parity check"
                " matrix.  There was breaking change to this API.  See"
                " help(qldpc.decoders.RelayBPDecoder)"
            )

        if isinstance(pcm_or_dem, stim.DetectorErrorModel):
            if observable_error_matrix is not None:
                raise ValueError(
                    "Cannot specify an observable_error_matrix when providing a detector error"
                    " model"
                )
            dem_arrays = DetectorErrorModelArrays(pcm_or_dem)
            pcm = dem_arrays.detector_flip_matrix
            observable_error_matrix = dem_arrays.observable_flip_matrix
            if error_priors is None:
                error_priors = dem_arrays.error_probs
            else:
                warnings.warn(
                    "Explicitly provided error_priors will override the error probabilities of the "
                    "provided detector error model",
                    stacklevel=2,
                )
        else:
            pcm = pcm_or_dem
            if error_priors is None:
                error_priors = [PLACEHOLDER_ERROR_RATE] * pcm.shape[1]

        if isinstance(pcm, galois.FieldArray):
            pcm = pcm.view(np.ndarray)
        elif isinstance(pcm, scipy.sparse.spmatrix):
            pcm = pcm.tocsc()
            pcm.sort_indices()
        self.has_observable_error_matrix = observable_error_matrix is not None
        if observable_error_matrix is None:
            observable_error_matrix = np.empty((0, 0), dtype=np.uint8)

        self.has_erasure_bit = add_erasure_bit
        self.pcm_transposed = scipy.sparse.csr_matrix(pcm, dtype=np.uint8).T.tocsr()
        self.observable_error_matrix_transposed = scipy.sparse.csr_matrix(
            observable_error_matrix, dtype=np.uint8
        ).T.tocsr()
        self.decoder = relay_bp.ObservableDecoderRunner(
            getattr(relay_bp, name)(pcm, np.asarray(error_priors), **decoder_args),
            observable_error_matrix,
            include_decode_result,
        )

    def decode_errors(self, /, detectors: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode an error syndrome and return an inferred error.

        Typecast detectors to np.uint8 for compatibility with the relay_bp package.
        """
        detectors = np.asarray(detectors, dtype=np.uint8)
        error = self.decoder.decode(detectors)
        if not self.has_erasure_bit:
            return error
        erased = ~self._reproduces_syndrome(np.asarray(error)[None, :], detectors[None, :])
        return with_erasure_bits(error, erased[0])

    def decode(self, /, detectors: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode an error syndrome and return an inferred error (alias for decode_errors)."""
        return self.decode_errors(detectors)

    def decode_errors_batch(
        self,
        /,
        detectors: npt.NDArray[np.int_],
        parallel: bool = False,
        progress_bar: bool = True,
        leave_progress_bar_on_finish: bool = False,
    ) -> npt.NDArray[np.int_]:
        """Decode a batch of error syndromes and return inferred errors.

        Typecast detectors to np.uint8 for compatibility with the relay_bp package.
        """
        detectors = np.asarray(detectors, dtype=np.uint8)
        if len(detectors) == 0:
            num_errors = self.pcm_transposed.shape[0]
            return np.zeros((0, num_errors + self.has_erasure_bit), dtype=np.uint8)
        errors = self.decoder.decode_batch(
            detectors, parallel, progress_bar, leave_progress_bar_on_finish
        )
        if not self.has_erasure_bit:
            return errors
        erased = ~self._reproduces_syndrome(np.asarray(errors), detectors)
        return with_erasure_bits(errors, erased)

    def decode_batch(
        self,
        /,
        detectors: npt.NDArray[np.int_],
        parallel: bool = False,
        progress_bar: bool = True,
        leave_progress_bar_on_finish: bool = False,
    ) -> npt.NDArray[np.int_]:
        """Decode a batch of error syndromes (alias for decode_errors_batch)."""
        return self.decode_errors_batch(
            detectors, parallel, progress_bar, leave_progress_bar_on_finish
        )

    def decode_observables(self, /, detectors: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode an error syndrome and return predicted observable flips.

        If initialized with ``add_erasure_bit=True``, append an erasure bit that is set to 1 when
        the inferred error does not reproduce the syndrome.
        """
        self._require_observables()
        if not self.has_erasure_bit:
            return np.asarray(
                self.decoder.decode_observables(np.asarray(detectors, dtype=np.uint8))
            )
        return self._errors_to_observable_flips(self.decode_errors(detectors)[None, :])[0]

    def decode_observables_batch(
        self,
        /,
        detectors: npt.NDArray[np.int_],
        parallel: bool = False,
        progress_bar: bool = False,
        leave_progress_bar_on_finish: bool = False,
    ) -> npt.NDArray[np.int_]:
        """Decode a batch of error syndromes, one per row, and return predicted observable flips.

        If initialized with ``add_erasure_bit=True``, append an erasure bit to each prediction that
        is set to 1 when the inferred error does not reproduce the syndrome.

        Unlike relay_bp.ObservableDecoderRunner.decode_observables_batch, this method shows no
        progress bar by default, since Sinter decoders call it for every batch of shots.
        """
        self._require_observables()
        if len(detectors) == 0:
            num_observables = self.observable_error_matrix_transposed.shape[1]
            return np.zeros((0, num_observables + self.has_erasure_bit), dtype=np.uint8)
        if not self.has_erasure_bit:
            return np.asarray(
                self.decoder.decode_observables_batch(
                    np.asarray(detectors, dtype=np.uint8),
                    parallel,
                    progress_bar,
                    leave_progress_bar_on_finish,
                )
            )
        errors = self.decode_errors_batch(
            detectors, parallel, progress_bar, leave_progress_bar_on_finish
        )
        return self._errors_to_observable_flips(errors)

    def _require_observables(self) -> None:
        """Raise an error if this decoder was not given observables to predict."""
        if not self.has_observable_error_matrix:
            raise ValueError(
                "Predicting observable flips with a RelayBPDecoder requires an"
                " observable_error_matrix, or a detector error model with observables"
            )

    def _errors_to_observable_flips(
        self, errors_and_erasure_bits: npt.NDArray[np.int_]
    ) -> npt.NDArray[np.int_]:
        """Convert inferred errors, each with an erasure bit appended, into observable flips."""
        errors = np.asarray(errors_and_erasure_bits[:, :-1], dtype=np.uint8)
        flips = np.asarray(errors @ self.observable_error_matrix_transposed) & 1
        return np.hstack([flips, errors_and_erasure_bits[:, -1:]]).astype(np.uint8)

    def _reproduces_syndrome(
        self, errors: npt.NDArray[np.int_], detectors: npt.NDArray[np.int_]
    ) -> npt.NDArray[np.bool_]:
        """Whether each inferred error reproduces the syndrome it was inferred from.

        Relay-BP settles on a best guess whether or not it converges, so checking that guess is
        what separates a syndrome it explained from one it could not.

        The parity accumulates in uint8 and overflows for a check that many error mechanisms
        address.  That is harmless: overflow reduces modulo 256, and only the low bit is read.
        """
        residuals = np.asarray(errors.astype(np.uint8, copy=False) @ self.pcm_transposed) & 1
        return np.all(residuals == detectors, axis=1)

    def __getattr__(self, name: str) -> Any:
        """Inherit all methods of self.decoder: relay_bp.ObservableDecoderRunner.

        Typecast the first argument, if there is one, to np.uint8 for compatibility with the
        relay_bp package.
        """
        if name == "decoder":
            raise AttributeError(name)
        inner_func = getattr(self.decoder, name)

        @functools.wraps(inner_func)
        def outer_func(*args: object, **kwargs: object) -> Any:
            if args:
                args = (np.asarray(args[0], dtype=np.uint8), *args[1:])
            return inner_func(*args, **kwargs)

        return outer_func


@_erasure_bit_support("RBP", supported=True)
def get_decoder_rbp(
    pcm_or_dem: IntegerArray | stim.DetectorErrorModel,
    error_priors: npt.NDArray[np.floating] | Sequence[float] | None = None,
    **decoder_args: object,
) -> RelayBPDecoder:
    """Build a Relay-BP decoder.

    Args:
        pcm_or_dem: A parity-check matrix or detector error model (DEM) to decode.
        error_priors: Prior probabilities for each error.  A DEM supplies these by default.
        **decoder_args: Arguments passed to :class:`RelayBPDecoder`, including the backend class
            ``name``, observable matrix, and ``add_erasure_bit``.

    Returns:
        A :class:`RelayBPDecoder`, which can infer errors and, when observable metadata is
        available, predict observable flips.

    With ``add_erasure_bit=True``, the decoder appends a flag set when the inferred error does not
    reproduce the syndrome.

    See the `relay-bp package documentation <https://pypi.org/project/relay-bp>`_ and
    `arXiv:2506.01779 <https://arxiv.org/abs/2506.01779>`_.
    """
    return RelayBPDecoder(pcm_or_dem, error_priors, **decoder_args)  # type: ignore[arg-type]


def get_relay_bp_decoder(
    pcm_or_dem: IntegerArray | stim.DetectorErrorModel, **decoder_args: Any
) -> RelayBPDecoder:
    """Build the ``RelayDecoder`` backend selected by a ``relay_bp`` :class:`DecoderSpec`.

    The specification supplies a ``precision`` suffix and forwards all other options to
    :func:`get_decoder_rbp`.  This public builder exists so deferred specifications have a stable,
    pickleable construction path.
    """
    return _get_relay_decoder(pcm_or_dem, decoder_class_prefix="RelayDecoder", **decoder_args)


def get_min_sum_bp_decoder(
    pcm_or_dem: IntegerArray | stim.DetectorErrorModel, **decoder_args: Any
) -> RelayBPDecoder:
    """Build the ``MinSumBPDecoder`` backend selected by a ``min_sum_bp`` :class:`DecoderSpec`.

    The specification supplies a ``precision`` suffix and forwards all other options to
    :func:`get_decoder_rbp`.  This public builder exists so deferred specifications have a stable,
    pickleable construction path.
    """
    return _get_relay_decoder(pcm_or_dem, decoder_class_prefix="MinSumBPDecoder", **decoder_args)


# Private builder helpers


def _get_relay_decoder(
    pcm_or_dem: IntegerArray | stim.DetectorErrorModel,
    *,
    decoder_class_prefix: str,
    precision: str,
    **decoder_args: Any,
) -> RelayBPDecoder:
    """Build a RelayBPDecoder from a class-name prefix and precision."""
    return get_decoder_rbp(pcm_or_dem, name=f"{decoder_class_prefix}{precision}", **decoder_args)
