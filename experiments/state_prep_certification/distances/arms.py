# SPDX-License-Identifier: Apache-2.0

"""The two arms of the distance experiment: qLDPC's BZ search with everything on, and the prior.

The "on" arm calls the code's ``get_distance_exact`` at qLDPC's defaults, and the "prior" arm
configures qLDPC with only the techniques that some public exact tool shares with it:

* ``q_bz_on``: ``code.get_distance_exact(method="brouwer_zimmermann")``.
* ``q_bz_prior``: the same call inside ``use_tuning(**BZ_PRIOR)``: no overlapping families or
  their coverage certificate, no completion tables, and no automorphism images, with disjoint
  matroid partitioning on (``partition_max_coverage=1``).

The connected-cluster search is not compared: the certification experiment measures it as an
oracle of the certifier.  Both arms make the same code-method call, so they search the same
sectors: only the X sector when a plain ``CSSCode`` has equal check matrices.

Every arm builds a plain code object from the stored matrices inside the timer: ``CSSCode(H_X,
H_Z, is_subsystem_code=False)``, or ``QuditCode(H)``, never a family class.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from collections.abc import Callable
from typing import Any

import numpy.typing as npt

from qldpc import codes
from qldpc.codes.distance import _tuning

CONFIG = json.loads((pathlib.Path(__file__).with_name("config.json")).read_text())
ARMS = ("q_bz_on", "q_bz_prior")


def _tuning_values(changes: dict[str, Any]) -> dict[str, Any]:
    """Tuning changes from ``config.json``, with ``"MAX_VALUE"`` replaced by its value."""
    return {
        key: _tuning.MAX_VALUE if value == "MAX_VALUE" else value for key, value in changes.items()
    }


BZ_PRIOR = _tuning_values(CONFIG["bz_prior"])


def tuning_changes(arm: str) -> dict[str, Any]:
    """The tuning changes of an arm."""
    return BZ_PRIOR if arm == "q_bz_prior" else {}


def distance_call(arrays: dict[str, npt.NDArray[Any]]) -> Callable[[], int]:
    """The measured call of both arms on a stored code: build the plain code, then search.

    The arms differ only in the tuning around the call (``tuning_changes``).
    """

    def call() -> int:
        if "matrix" in arrays:
            code: Any = codes.QuditCode(arrays["matrix"])
        else:
            code = codes.CSSCode(arrays["matrix_x"], arrays["matrix_z"], is_subsystem_code=False)
        return int(code.get_distance_exact(method="brouwer_zimmermann"))

    return call


def active_tuning() -> dict[str, Any]:
    """The fields of the active tuning, as evidence of the settings of a run."""
    return dataclasses.asdict(_tuning.get_tuning())
