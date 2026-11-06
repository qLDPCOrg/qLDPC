# SPDX-License-Identifier: Apache-2.0

"""Integrations with external tools: the GAP computer algebra system and the sqetch estimator."""

from typing import TYPE_CHECKING

from qldpc import _deprecations

from . import gap, sqetch

__all__ = ["gap", "sqetch"]

if not TYPE_CHECKING:  # DEPRECATED: see qldpc._deprecations

    def __getattr__(name: str) -> object:
        return _deprecations.get_deprecated_attribute(__name__, name)
