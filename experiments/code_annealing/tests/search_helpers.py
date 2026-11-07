# SPDX-License-Identifier: Apache-2.0

"""Small codes that several tests of the search share."""

from __future__ import annotations

from typing import Any

import numpy as np

from qldpc import codes


def counter_from_checks(checks_x: Any, checks_z: Any) -> codes.CSSLogicalCounter:
    """A counter of the CSS code with the given binary X and Z checks."""
    code = codes.CSSCode(np.asarray(checks_x, dtype=int), np.asarray(checks_z, dtype=int))
    return codes.CSSLogicalCounter(code)


def steane_counter() -> codes.CSSLogicalCounter:
    """The Steane code, with a counter."""
    return codes.CSSLogicalCounter(codes.SteaneCode())


def canonical_checks(counter: codes.CSSLogicalCounter) -> tuple[bytes, bytes]:
    """The checks of a code in standard form, which do not depend on the generators' order."""
    normalized = counter_from_checks(counter.checks_x, counter.checks_z)
    return normalized.checks_x.tobytes(), normalized.checks_z.tobytes()
