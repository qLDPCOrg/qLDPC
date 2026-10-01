# SPDX-License-Identifier: Apache-2.0

"""Tests for the external-integration package facade."""

from __future__ import annotations

import subprocess
import sys

from qldpc import decoders
from qldpc.decoders import construction, external
from qldpc.decoders.external import ldpc, pymatching, relay_bp


def test_external_package_exports() -> None:
    """The facade exports integrations from their canonical owner modules."""
    expected = {
        "MatchingObservableDecoder": pymatching.MatchingObservableDecoder,
        "RelayBPDecoder": relay_bp.RelayBPDecoder,
        "get_decoder_bf": ldpc.get_decoder_bf,
        "get_decoder_bp_lsd": ldpc.get_decoder_bp_lsd,
        "get_decoder_bp_osd": ldpc.get_decoder_bp_osd,
        "get_decoder_mwpm": pymatching.get_decoder_mwpm,
        "get_decoder_rbp": relay_bp.get_decoder_rbp,
        "get_error_decoder_mwpm": pymatching.get_error_decoder_mwpm,
        "get_min_sum_bp_decoder": relay_bp.get_min_sum_bp_decoder,
        "get_observable_decoder_mwpm": pymatching.get_observable_decoder_mwpm,
        "get_relay_bp_decoder": relay_bp.get_relay_bp_decoder,
    }
    assert set(external.__all__) == set(expected)
    for name, value in expected.items():
        assert getattr(external, name) is value
        if hasattr(decoders, name):
            assert getattr(decoders, name) is value
        if hasattr(construction, name):
            assert getattr(construction, name) is value


def test_external_package_import_is_lazy() -> None:
    """Importing the facade does not import any optional decoder dependency."""
    code = (
        "import sys\n"
        "import qldpc.decoders.external\n"
        "assert not {'ldpc', 'pymatching', 'relay_bp'} & sys.modules.keys()\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
