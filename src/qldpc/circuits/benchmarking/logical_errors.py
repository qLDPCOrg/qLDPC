# SPDX-License-Identifier: Apache-2.0

"""Sampled logical error rates and discard rates of annotated circuits."""

from __future__ import annotations

from collections.abc import Collection
from typing import cast

import numpy as np
import numpy.typing as npt
import sinter
import stim

from qldpc import decoders
from qldpc.decoders.adapters._observable_decoders import split_discard_byte

from .._post_selection import is_integer

####################################################################################################
# sampled logical error rates


def get_logical_error_and_discard_rate(
    circuit_or_dem: stim.Circuit | stim.DetectorErrorModel,
    decoder: sinter.Decoder | decoders.DeferredDecoderInput,
    num_samples: int,
    *,
    post_select: bool | Collection[int] = False,
    post_select_observables: bool | Collection[int] = False,
    dem_to_decode: stim.DetectorErrorModel | None = None,
) -> tuple[float, float]:
    """Compute a logical error rate and discard rate from samples of the provided circuit.

    Each logical error rate is a fraction of the (possibly post-selected) shots in which observable
    flips are predicted incorrectly by the provided decoder.

    This method is provided for convenience, but if you are doing heavy numerics you should probably
    build a sinter.Task and call sinter.collect.  In this case, circuit_or_dem should just be a
    circuit, and the sinter.Task would be built as follows, after replacing ``post_select=True`` by
    ``range(circuit.num_detectors)``, ``post_select_observables=True`` by
    ``range(circuit.num_observables)``, and False by ``()``::

        postselection_mask = np.zeros(circuit.num_detectors, dtype=int)
        postselection_mask[list(post_select)] = 1
        postselected_observables_mask = np.zeros(circuit.num_observables, dtype=int)
        postselected_observables_mask[list(post_select_observables)] = 1
        task = sinter.Task(
            circuit=circuit,
            detector_error_model=dem_to_decode,
            postselection_mask=np.packbits(postselection_mask, bitorder="little"),
            postselected_observables_mask=np.packbits(
                postselected_observables_mask, bitorder="little"
            ),
        )

    Sampling data would then be collected with::

        stats = sinter.collect(
            tasks=[task],  # or more tasks
            decoders=["custom"],
            custom_decoders={"custom": decoder},
            num_shots=num_samples,
            # other options such as num_workers=os.cpu_count() or max_errors=100
        )

    where a decoder that is not a Sinter decoder, such as ``decoders.bp_osd()``, must first be
    wrapped in ``decoders.SinterDecoder(decoder=decoder)``.

    Args:
        circuit_or_dem: The circuit or detector error model we wish to sample.
        decoder: The circuit-level decoder used to predict observable flips: a Sinter decoder, or a
            qLDPC decoder input such as the specification ``decoders.bp_osd()``, which is wrapped in
            ``decoders.SinterDecoder(decoder=decoder)``.  None selects the default decoder of a
            ``decoders.SinterDecoder``.
        num_samples: The number of times to sample the circuit_or_dem.

    Keyword args:
        post_select: The detectors to post-select on: True for all detectors of circuit_or_dem,
            False (the default) for none, or a collection of detector indices.  Shots in which any
            of these detectors fires are discarded.
        post_select_observables: The observables to post-select on: True for all observables,
            False (the default) for none, or a collection of observable indices.  Any time the
            decoder's prediction for these observable flips do not agree with the actual observable
            flips, the shot is discarded rather than counting as an error.
        dem_to_decode: The detector error model to decode.  This DEM should include _all_ detectors,
            including any that are post-selected on: the decoder receives every detector bit, for
            consistency with how Sinter works.  If ``dem_to_decode is None``, this method decodes
            with the (simplified) DEM sampled from ``circuit_or_dem``.

    Returns:
        A fraction of the retained samples in which at least one observable was decoded incorrectly,
        or NaN if no sample was retained.
        A fraction of samples that were discarded, either by post-selection or at the request of the
        decoder, which signals that a shot should be discarded by predicting observable flips in one
        more byte than the observables of the sampled circuit require.  These are point estimates.
        Use ``sinter.collect`` when counts or confidence intervals are needed.

    Raises:
        TypeError: If ``post_select`` or ``post_select_observables`` is neither a bool nor a
            collection of integer indices.
        ValueError: If ``dem_to_decode`` has a different number of detectors or observables than
            the detector error model to sample, or if ``post_select`` or ``post_select_observables``
            has an index outside the detectors or observables.
    """
    # identify and simplify the DEM to sample
    dem_arrays = decoders.DetectorErrorModelArrays(circuit_or_dem, simplify=True)
    dem = dem_arrays.to_dem()

    if dem_to_decode is not None:
        same_num_observables = dem_to_decode.num_observables == dem.num_observables
        same_num_detectors = dem_to_decode.num_detectors == dem.num_detectors
        if not same_num_observables or not same_num_detectors:
            raise ValueError(
                f"Incompatible detector error models."
                "\n(num_detectors, num_observables) in the DEM to sample:"
                f" {(dem.num_detectors, dem.num_observables)}"
                "\n(num_detectors, num_observables) in the DEM to decode:"
                f" {(dem_to_decode.num_detectors, dem_to_decode.num_observables)}"
            )

    # sample detector and observable flips in the circuit
    sampler = dem.compile_sampler()
    det_data, obs_data, _ = sampler.sample(shots=num_samples, bit_packed=True)

    # identify post-selection masks
    postselection_mask = _get_postselection_mask(
        post_select, circuit_or_dem.num_detectors, "Detector"
    )
    postselected_observables_mask = _get_postselection_mask(
        post_select_observables, dem.num_observables, "Observable"
    )
    num_discards = 0

    # if applicable, post-select on detectors
    if postselection_mask is not None:
        shot_mask = ~np.any(det_data & postselection_mask, axis=1)
        det_data = det_data[shot_mask]
        obs_data = obs_data[shot_mask]
        num_discards += int(np.sum(~shot_mask))

    # decode and identify incorrectly predicted observable flips
    if not hasattr(decoder, "decode_via_files"):
        decoder = decoders.SinterDecoder(decoder=decoder)
    compiled_sinter_decoder = cast(sinter.Decoder, decoder).compile_decoder_for_dem(
        dem=dem if dem_to_decode is None else dem_to_decode
    )
    predicted_flips = compiled_sinter_decoder.decode_shots_bit_packed(
        bit_packed_detection_event_data=det_data
    )

    # the decoder may ask for shots to be discarded in a byte added to its predictions
    predicted_flips, discarded = split_discard_byte(
        predicted_flips, len(det_data), dem.num_observables
    )
    obs_data = obs_data[~discarded]
    predicted_flips = predicted_flips[~discarded]
    num_discards += int(np.sum(discarded))

    incorrectly_predicted_flips = obs_data ^ predicted_flips

    # if applicable, post-select on observables
    if postselected_observables_mask is not None:
        shot_mask = ~np.any(incorrectly_predicted_flips & postselected_observables_mask, axis=1)
        incorrectly_predicted_flips = incorrectly_predicted_flips[shot_mask]
        num_discards += int(np.sum(~shot_mask))

    # compute logical error rate: fraction of shots with incorrectly predicted observable flips
    failures = np.any(incorrectly_predicted_flips, axis=1)
    logical_error_rate = np.sum(failures) / len(failures) if len(failures) else np.nan
    discard_rate = num_discards / num_samples if num_samples else 0.0
    return logical_error_rate, discard_rate


def _get_postselection_mask(
    indices: bool | Collection[int], size: int, name: str
) -> npt.NDArray[np.uint8] | None:
    """Build a bit-packed sinter post-selection mask, or None if there is nothing to select.

    Args:
        indices: The indices to select, which may be negative to count back from the end, or True
            to select all ``size`` indices, or False to select none.
        size: The number of detectors or observables that the mask covers.
        name: The kind of index ("Detector" or "Observable"), for error messages.

    Returns:
        A little-endian bit-packed mask with ``size`` bits, or None if no index is selected.

    Raises:
        TypeError: If ``indices`` is neither a bool nor a collection of integers.
        ValueError: If an index is outside the range [-size, size).
    """
    if isinstance(indices, bool):
        indices = range(size) if indices else ()
    invalid = f"{name} post-selection accepts a bool or a collection of integer indices"
    try:
        indices = list(indices)
    except TypeError as error:
        raise TypeError(f"{invalid}, not {indices!r}") from error
    if not all(is_integer(index) for index in indices):
        raise TypeError(f"{invalid}; requested: {indices}")
    if not indices:
        return None
    if not all(-size <= index < size for index in indices):
        raise ValueError(
            f"{name} post-selection accepts indices in the range [-{size}, {size});"
            f" requested: {indices}"
        )
    mask = np.zeros(size, dtype=np.uint8)
    mask[indices] = 1
    return np.packbits(mask, bitorder="little")
