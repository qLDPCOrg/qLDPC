# SPDX-License-Identifier: Apache-2.0

"""Exact fault distance of annotated circuits."""

from __future__ import annotations

from typing import Literal

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

    Consider a noisy Stim circuit annotated with detectors and logical observables.  Calling
    ``circuit.detector_error_model(approximate_disjoint_errors=True)`` converts it into a detector
    error model (DEM).  The DEM lists error mechanisms, their probabilities, and the detectors and
    observables flipped by each mechanism.

    ``decoders.DetectorErrorModelArrays`` collects this data into two binary flip matrices and a
    vector of error probabilities, after merging error mechanisms with identical effects.
    Specifically, its ``detector_flip_matrix`` has one column per error mechanism and one row per
    detector, with each row specifying which error mechanisms flip that detector.  Its
    ``observable_flip_matrix`` similarly specifies how error mechanisms flip observables.

    Denote these matrices by :math:`D` and :math:`L`, respectively.  Let :math:`e` be a fault: a
    binary vector indicating which error mechanisms have occurred.  The detector and observable
    flips caused by :math:`e` are :math:`D \cdot e` and :math:`L \cdot e`.  The circuit fault
    distance is the minimum-weight undetectable logical fault:

    .. math::

        d_{\mathrm{fault}} = \min_{D \cdot e=0,\;L \cdot e\ne0}|e|,

    where :math:`|e|` is the fault weight of :math:`e`, or its number of error mechanisms.

    The undetected faults are the null space of :math:`D`.  Those in the null spaces of both
    :math:`D` and :math:`L` flip no observable, so they are harmless, and every other undetected
    fault flips an observable.  The fault distance is therefore the minimum weight of an undetected
    fault that is not harmless, which :func:`~qldpc.codes.distance.get_distance_from_checks` finds
    with :math:`D` as the checks and a basis of the harmless faults excluded.  The exact calculation
    can be expensive for larger circuits.

    Args:
        circuit: A noisy Stim circuit annotated with detectors and logical observables.
        method: The exact-distance method: ``"brouwer_zimmermann"`` (the default, with alias
            ``"information_sets"``), ``"brute_force"``, or ``"connected_support"``, which searches
            the sparse detector matrix :math:`D` for undetected faults that are connected through
            its detectors (see :data:`~qldpc.codes.distance.DistanceMethod`).

    Returns:
        The minimum number of DEM error mechanisms causing an undetected observable flip, or
        ``NO_DISTANCE`` (see :class:`~qldpc.codes.distance.NoDistance`) if no such fault exists.
    """
    dem_arrays = decoders.DetectorErrorModelArrays(circuit)
    detectors = dem_arrays.detector_flip_matrix.toarray().astype(np.uint8)
    observables = dem_arrays.observable_flip_matrix.toarray().astype(np.uint8)
    harmless_faults = get_kernel(np.vstack([detectors, observables]))
    return get_distance_from_checks(detectors, harmless_faults, method=method)
