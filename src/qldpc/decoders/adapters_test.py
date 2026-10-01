# SPDX-License-Identifier: Apache-2.0

"""Tests for the decoder-adapter package facade."""

from __future__ import annotations

import pickle
import subprocess
import sys

import pytest

from qldpc.decoders import adapters
from qldpc.decoders.adapters import backends, error_decoders, observable_decoders
from qldpc.decoders.protocols import BatchErrorDecoder, ErrorDecoder


def test_adapter_facade_exports() -> None:
    """The facade exposes implementations and protocols from their owning modules."""
    assert adapters.BpOsdDecoder is backends.BpOsdDecoder
    assert adapters.BpLsdDecoder is backends.BpLsdDecoder
    assert adapters.BeliefFindDecoder is backends.BeliefFindDecoder
    assert adapters.Matching is backends.Matching
    assert adapters.ErrorsToObservablesDecoder is error_decoders.ErrorsToObservablesDecoder
    assert adapters.ExpandedErrorDecoder is error_decoders.ExpandedErrorDecoder
    assert adapters.BitPackedObservableDecoder is observable_decoders.BitPackedObservableDecoder
    assert adapters.ErrorDecoder is ErrorDecoder
    assert adapters.BatchErrorDecoder is BatchErrorDecoder

    missing_name = "NotAnAdapter"
    with pytest.raises(AttributeError, match="has no attribute"):
        getattr(adapters, missing_name)


def test_old_pickle_class_lookups() -> None:
    """Pickles naming classes in the old package facade resolve to their canonical classes."""
    for name, adapted_class in [
        ("BpOsdDecoder", adapters.BpOsdDecoder),
        ("BpLsdDecoder", adapters.BpLsdDecoder),
        ("BeliefFindDecoder", adapters.BeliefFindDecoder),
        ("Matching", adapters.Matching),
    ]:
        payload = pickle.dumps(adapted_class, protocol=0)
        payload = payload.replace(
            f"qldpc.decoders.adapters.backends\n{name}".encode(),
            f"qldpc.decoders.adapters\n{name}".encode(),
        )
        assert pickle.loads(payload) is adapted_class  # noqa: S301 - compatibility payload


def test_external_backends_are_loaded_lazily() -> None:
    """Importing qldpc and the adapters facade does not import external backends."""
    code = """
import sys
import qldpc
assert "ldpc" not in sys.modules
assert "pymatching" not in sys.modules
from qldpc.decoders import adapters
assert "ldpc" not in sys.modules
assert "pymatching" not in sys.modules
assert adapters.BpOsdDecoder.__name__ == "BpOsdDecoder"
assert "ldpc" in sys.modules
assert "pymatching" in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)
