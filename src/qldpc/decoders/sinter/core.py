# SPDX-License-Identifier: Apache-2.0

"""Core decoders for using qLDPC decoders with Sinter."""

from __future__ import annotations

import pathlib
import warnings
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import sinter
import stim

from qldpc._util import get_external_caller_stacklevel

from ..adapters.error_decoders import ErrorsToObservablesDecoder
from ..construction.legacy import (
    get_legacy_decoder_migration_message,
    reject_removed_decoder_args,
    resolve_observable_decoder,
)
from ..construction.resolution import reject_prebuilt_decoder
from ..construction.specs import DecoderSpec, DeferredDecoderInput
from ..dems import DetectorErrorModelArrays
from ..protocols import ErrorDecoder, ObservableDecoder, as_error_decoder

# sinter does not ship type information, so mypy treats sinter.Decoder and sinter.CompiledDecoder as
# Any.  A subclass of Any is assumed to have every attribute, so it would structurally satisfy the
# ErrorDecoder protocol, and mypy would accept an observable decoder wherever an error decoder is
# required.  Empty stand-in base classes let mypy check observable decoders by their own attributes.
if TYPE_CHECKING:

    class _SinterDecoder: ...

    class _SinterCompiledDecoder: ...

else:
    _SinterDecoder = sinter.Decoder
    _SinterCompiledDecoder = sinter.CompiledDecoder


class SinterDecoder(_SinterDecoder):
    """Sinter-compatible configuration that builds observable decoders.

    A SinterDecoder stores settings for an inner decoder.  When Sinter compiles a SinterDecoder for
    a detector error model, the SinterDecoder builds the inner decoder for that model, and returns a
    CompiledSinterDecoder that predicts observable flips.  If the inner decoder can predict
    observable flips natively (as Frontier, MWPM, Relay-BP, and lookup-table decoders can), it is
    built in that mode.  Otherwise, it is built as an error decoder, and the compiled decoder
    converts the errors that it infers into observable flips.
    """

    decode_is_defunct = True

    # completes the error message "A prebuilt decoder cannot be passed as decoder= here because ..."
    _prebuilt_decoder_rejection_reason = (
        "a SinterDecoder builds a new decoder for each (simplified) detector error model that it is"
        " compiled for"
    )

    def __init__(
        self,
        *,
        simplify: bool = True,
        decompose_errors: bool = False,
        decoder: DeferredDecoderInput = None,
        **decoder_kwargs: object,
    ) -> None:
        """Initialize a SinterDecoder.

        A SinterDecoder is used by Sinter to decode events from a detector error model and predict
        observable flips.  See help(sinter.Decoder) for additional information.

        Args:
            simplify: Whether to merge equivalent errors in a DEM when compiling a decoder for
                that DEM.
            decompose_errors: Whether to decompose errors according to their suggested decomposition
                when compiling a decoder for a DEM.
            decoder: Settings for the inner decoder, such as ``decoders.mwpm(...)``, a constructor
                that builds an error decoder or an observable decoder from a detector error model,
                an observable-decoder compiler such as another SinterDecoder, or None to select the
                default decoder.  A prebuilt decoder is rejected, because the inner decoder is built
                for each (simplified) detector error model.  Settings build a native observable
                decoder where they support one, and an error decoder is wrapped so that the
                observable flips of the errors that it infers become its predictions.
            **decoder_kwargs: Deprecated arguments to pass to qldpc.decoders.get_decoder.
        """
        reject_removed_decoder_args(decoder_kwargs)
        reject_prebuilt_decoder(decoder, self._prebuilt_decoder_rejection_reason)
        if (
            decompose_errors
            and isinstance(decoder, DecoderSpec)
            and decoder.options.get("enable_correlations")
        ):
            raise ValueError(
                "Correlated matching (enable_correlations=True) uses the decompositions that a"
                " detector error model suggests for its errors, which decompose_errors=True"
                " discards.  Leave decompose_errors=False to decode with correlated matching"
            )
        self.simplify = simplify
        self.decompose_errors = decompose_errors
        self.decoder_input = decoder
        self.decoder_kwargs = decoder_kwargs
        if decoder_kwargs:
            warnings.warn(
                get_legacy_decoder_migration_message(None, decoder_kwargs),
                DeprecationWarning,
                stacklevel=get_external_caller_stacklevel(),
            )
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
        observable_decoder = resolve_observable_decoder(
            dem_arrays.to_dem(),
            self.decoder_input,
            self.decoder_kwargs.copy(),
            warn_deprecated=False,
        )
        return CompiledSinterDecoder(dem_arrays, observable_decoder)

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

    # Defunct compatibility method
    if TYPE_CHECKING:
        # Hide this method from mypy, so that a SinterDecoder does not satisfy ErrorDecoder.
        decode: None
    else:

        def decode(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            """Reject a defunct direct-decoding call."""
            raise ValueError(
                "SinterDecoder.decode is DEFUNCT.  Compile the SinterDecoder for a detector error"
                " model, then call decode_observables or decode_shots on the compiled decoder."
                "\nIf you need this method restored, please open an issue at"
                " https://github.com/qLDPCOrg/qLDPC/issues"
            )


class CompiledSinterDecoder(_SinterCompiledDecoder, ObservableDecoder):
    """Observable decoder compiled to a specific detector error model.

    Instances are constructed by SinterDecoder.compile_decoder_for_dem.  A CompiledSinterDecoder
    predicts observable flips with an inner observable decoder, which may be an error decoder whose
    inferred errors are converted into observable flips.

    When the decoder being wrapped signals erasure with an erasure bit, .decode_shots appends one
    erasure bit to the observable flips of every shot, and .decode_shots_bit_packed reports those
    erasure bits in one whole byte added past the packed observable flips.  Sinter reads that added
    byte as a request to discard the shot, so an erasure becomes a discarded shot rather than a
    predicted flip of an observable that the sampled circuit does not have.

    If the inner observable decoder has a decode_shots_bit_packed method, .decode_shots_bit_packed
    uses it directly, unless a subclass customizes how shots are unpacked, decoded, or packed.
    """

    num_detectors: int
    num_observables: int
    num_erasure_bits: int = 0

    decode_is_defunct = True

    def __init__(
        self, dem_arrays: DetectorErrorModelArrays, decoder: ErrorDecoder | ObservableDecoder
    ) -> None:
        """Initialize a decoder compiled to a specific detector error model.

        Args:
            dem_arrays: The detector error model to decode.
            decoder: A decoder for that detector error model.  A decoder with a decode_observables
                method predicts observable flips natively.  Otherwise, the decoder is an error
                decoder, whose inferred errors are converted into observable flips.

        The decoder is exposed as the .decoder attribute, and the observable decoder that predicts
        observable flips as the .observable_decoder attribute.  These coincide for a decoder that
        predicts observable flips natively.
        """
        self.dem_arrays = dem_arrays
        self.decoder: ErrorDecoder | ObservableDecoder
        self.observable_decoder: ObservableDecoder
        if isinstance(decoder, ErrorsToObservablesDecoder):
            # expose the error decoder that the converter wraps
            self.decoder, self.observable_decoder = decoder.error_decoder, decoder
        elif isinstance(decoder, ObservableDecoder):
            self.decoder = self.observable_decoder = decoder
        else:
            self.decoder = decoder
            self.observable_decoder = ErrorsToObservablesDecoder(
                as_error_decoder(decoder, "The decoder"), dem_arrays.to_dem()
            )
        self.num_detectors = dem_arrays.num_detectors
        self.num_observables = dem_arrays.num_observables
        self.num_erasure_bits = int(getattr(self.observable_decoder, "has_erasure_bit", False))

    def decode_shots_bit_packed(
        self, bit_packed_detection_event_data: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.uint8]:
        """Predicts observable flips from the given detection events.

        This method accepts and returns bit-packed data.

        See help(sinter.CompiledDecoder) for additional information.
        """
        # Hand bit-packed shots straight to the inner observable decoder if it decodes them itself.
        # That skips the generic unpack/decode_shots/pack path below, so only do it when this class
        # does not override any of those steps (as composite decoders and post-processing subclasses
        # do); otherwise packed and unpacked predictions could disagree.
        overrides_shot_pipeline = any(
            getattr(type(self), name) is not getattr(CompiledSinterDecoder, name)
            for name in ("decode_shots", "unpack_detection_event_data", "pack_observable_flips")
        )
        if not overrides_shot_pipeline and callable(
            decode_shots_bit_packed := getattr(
                self.observable_decoder, "decode_shots_bit_packed", None
            )
        ):
            packed_flips = np.asarray(
                decode_shots_bit_packed(
                    bit_packed_detection_event_data=bit_packed_detection_event_data
                ),
                dtype=np.uint8,
            )
            expected_shape = (
                len(bit_packed_detection_event_data),
                -(-self.num_observables // 8) + self.num_erasure_bits,
            )
            if packed_flips.shape != expected_shape:
                raise ValueError(
                    f"The inner observable decoder predicted bit-packed shots of shape"
                    f" {packed_flips.shape}, but expected {expected_shape}"
                )
            return packed_flips
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
        if hasattr(self.observable_decoder, "decode_observables_batch"):
            observable_flips = self.observable_decoder.decode_observables_batch(
                detection_event_data
            )
        else:
            observable_flips = [
                self.observable_decoder.decode_observables(syndrome)
                for syndrome in detection_event_data
            ]
        return np.asarray(observable_flips, dtype=np.uint8).reshape(
            len(detection_event_data), self.num_observables + self.num_erasure_bits
        )

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

    def decode_observables(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Predict observable flips for one syndrome."""
        syndrome_uint8 = np.asarray(syndrome, dtype=np.uint8)
        return self.decode_shots(syndrome_uint8.reshape(1, *syndrome.shape))[0]

    @property
    def has_erasure_bit(self) -> bool:
        """Whether decode_observables appends an erasure bit to the predicted observable flips."""
        return bool(self.num_erasure_bits)

    # Defunct compatibility method
    if TYPE_CHECKING:
        # Hide this method from mypy, so that a CompiledSinterDecoder does not satisfy ErrorDecoder.
        decode: None
    else:

        def decode(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            """Reject a defunct alias for observable decoding."""
            raise ValueError(
                "CompiledSinterDecoder.decode is DEFUNCT; use decode_observables instead."
                "\nIf you need this method restored, please open an issue at"
                " https://github.com/qLDPCOrg/qLDPC/issues"
            )


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
