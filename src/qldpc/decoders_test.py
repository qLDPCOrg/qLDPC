# SPDX-License-Identifier: Apache-2.0

"""Tests for the package-root decoder facade."""

from __future__ import annotations

import warnings

from qldpc import decoders
from qldpc.decoders import construction

# Deprecated compatibility


def test_v0_3_3_deprecated_class_aliases() -> None:
    """Deprecated v0.3.3 class aliases warn and resolve to current classes."""
    expected = {
        "BatchDecoder": decoders.BatchErrorDecoder,
        "Decoder": decoders.ErrorDecoder,
        "SequentialSinterDecoder": decoders.SequentialWindowDecoder,
        "SubgraphSinterDecoder": decoders.SubgraphDecoder,
    }
    for name, replacement in expected.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert getattr(decoders, name) is replacement
        assert len(caught) == 1
        assert caught[0].filename == __file__
        assert replacement.__name__ in str(caught[0].message)


def test_v0_3_3_deprecated_builder_aliases() -> None:
    """Uppercase v0.3.3 builders warn and resolve to lowercase builders."""
    for suffix in ["BF", "BP_LSD", "BP_OSD", "GUF", "ILP", "MWPM", "RBP"]:
        replacement = getattr(construction, f"get_decoder_{suffix.lower()}")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert getattr(decoders, f"get_decoder_{suffix}") is replacement
        assert len(caught) == 1
        assert caught[0].filename == __file__
        assert replacement.__name__ in str(caught[0].message)
