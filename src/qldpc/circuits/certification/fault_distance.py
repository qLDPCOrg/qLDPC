# SPDX-License-Identifier: Apache-2.0

"""Exact fault distance of annotated circuits."""

from __future__ import annotations

from typing import Literal, get_args

import numpy as np
import stim

from qldpc import decoders
from qldpc._binary import get_kernel
from qldpc.codes.distance import DistanceMethod, NoDistance, get_distance_from_checks

####################################################################################################
# exact circuit fault distance


def get_fault_distance(
    circuit: stim.Circuit, *, method: DistanceMethod = "brouwer_zimmermann"
) -> int | Literal[NoDistance.NAN]:
    r"""Compute the exact fault distance of a noisy circuit.

    The detector error model (DEM) of the circuit, which
    ``circuit.detector_error_model(approximate_disjoint_errors=True)`` returns, lists error
    mechanisms and the detectors and observables that each flips.
    ``decoders.DetectorErrorModelArrays`` merges error mechanisms with identical effects, and
    collects these flips into its ``detector_flip_matrix`` :math:`D` and ``observable_flip_matrix``
    :math:`L`, with one column per error mechanism and one row per detector or observable.

    A fault :math:`e` is a binary vector indicating which error mechanisms have occurred, and its
    fault weight :math:`|e|` is its number of error mechanisms.  It flips the detectors
    :math:`D \cdot e` and the observables :math:`L \cdot e`.  The circuit fault distance is the
    smallest fault weight of an undetected fault that flips an observable:

    .. math::

        d_{\mathrm{fault}} = \min_{D \cdot e=0,\;L \cdot e\ne0}|e|.

    This function computes it with :func:`~qldpc.codes.distance.get_distance_from_checks`, with
    :math:`D` as the checks and the faults that flip no detector or observable excluded.  The exact
    calculation can be expensive for larger circuits.

    Args:
        circuit: A noisy Stim circuit annotated with detectors and logical observables.
        method: The exact-distance method: ``"brouwer_zimmermann"`` (the default, with alias
            ``"information_sets"``), ``"brute_force"``, or ``"connected_cluster"``, which searches
            the sparse detector matrix :math:`D` for undetected faults that are connected through
            its detectors (see :data:`~qldpc.codes.distance.DistanceMethod`).

    Returns:
        The minimum number of DEM error mechanisms causing an undetected observable flip, or
        ``NO_DISTANCE`` (see :class:`~qldpc.codes.distance.NoDistance`) if no such fault exists.

    Raises:
        ValueError: If the method is unknown, or if Stim cannot convert the circuit to a DEM.
    """
    # Check the method before the costly analysis of the circuit.
    if method not in get_args(DistanceMethod):
        raise ValueError(
            f"Unknown distance method {method!r}; choose from 'brouwer_zimmermann' (or its alias"
            " 'information_sets'), 'brute_force', or 'connected_cluster'."
        )
    dem_arrays = decoders.DetectorErrorModelArrays(circuit)
    detectors = dem_arrays.detector_flip_matrix.toarray().astype(np.uint8)
    observables = dem_arrays.observable_flip_matrix.toarray().astype(np.uint8)
    harmless_faults = get_kernel(np.vstack([detectors, observables]))
    return get_distance_from_checks(detectors, harmless_faults, method=method)
