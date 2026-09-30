# SPDX-License-Identifier: Apache-2.0

"""Unit tests for adapters/dem.py compatibility."""

from __future__ import annotations

import pickle
import warnings
from typing import Any

import pytest

from qldpc import decoders
from qldpc.decoders import retrieval
from qldpc.decoders.adapters import dem


def test_conversion_root_exports_are_canonical() -> None:
    """Root conversion exports are warning-free and use adapters/dem.py identities."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert decoders.ExpandedErrorDecoder is dem.ExpandedErrorDecoder
        assert decoders.ErrorsToObservablesDecoder is dem.ErrorsToObservablesDecoder
        assert decoders.match_error_decoder_to_dem is dem.match_error_decoder_to_dem


@pytest.mark.parametrize(
    ("old_name", "replacement"),
    [
        ("ExpandedErrorDecoder", dem.ExpandedErrorDecoder),
        ("ErrorsToObservablesDecoder", dem.ErrorsToObservablesDecoder),
        ("match_error_decoder_to_dem", dem.match_error_decoder_to_dem),
    ],
)
def test_deprecated_retrieval_conversion_paths(old_name: str, replacement: Any) -> None:
    """Old retrieval conversion paths warn at the external caller and remain identical."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert getattr(retrieval, old_name) is replacement
    assert len(caught) == 1
    assert caught[0].filename == __file__
    assert replacement.__name__ in str(caught[0].message)


@pytest.mark.parametrize(
    "replacement",
    [
        dem.ExpandedErrorDecoder,
        dem.ErrorsToObservablesDecoder,
        dem.match_error_decoder_to_dem,
    ],
)
def test_old_conversion_pickle_paths(replacement: Any) -> None:
    """Pickles naming conversion objects in retrieval.py remain loadable."""
    name = replacement.__name__
    payload = pickle.dumps(replacement, protocol=0)
    payload = payload.replace(
        f"qldpc.decoders.adapters.dem\n{name}".encode(),
        f"qldpc.decoders.retrieval\n{name}".encode(),
    )
    with pytest.warns(DeprecationWarning, match=name):
        restored = pickle.loads(payload)  # noqa: S301 - deliberately constructed compatibility data
    assert restored is replacement
