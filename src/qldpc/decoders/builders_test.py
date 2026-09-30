# SPDX-License-Identifier: Apache-2.0

"""Unit tests for builders.py and its public compatibility aliases."""

from __future__ import annotations

import pickle
import subprocess
import sys
import warnings
from typing import Any

import numpy as np
import pytest

from qldpc import decoders
from qldpc.decoders import builders, retrieval


def test_public_builder_identities_and_warnings() -> None:
    """Lowercase builders are canonical; uppercase aliases warn at the external caller."""
    canonical_builders = {
        "BF": builders.get_decoder_bf,
        "BP_LSD": builders.get_decoder_bp_lsd,
        "BP_OSD": builders.get_decoder_bp_osd,
        "GUF": builders.get_decoder_guf,
        "ILP": builders.get_decoder_ilp,
        "MWPM": builders.get_decoder_mwpm,
        "RBP": builders.get_decoder_rbp,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert decoders.get_decoder_lookup is builders.get_decoder_lookup
        assert retrieval.get_decoder_lookup is builders.get_decoder_lookup
        for suffix, builder in canonical_builders.items():
            lowercase_name = f"get_decoder_{suffix.lower()}"
            assert lowercase_name in decoders.__all__
            assert lowercase_name in builders.__all__
            assert getattr(decoders, lowercase_name) is builder

    for suffix, builder in canonical_builders.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert getattr(decoders, f"get_decoder_{suffix}") is builder
        assert len(caught) == 1
        assert caught[0].filename == __file__
        assert builder.__name__ in str(caught[0].message)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert getattr(retrieval, f"get_decoder_{suffix}") is builder
        assert len(caught) == 1
        assert caught[0].filename == __file__
        assert builder.__name__ in str(caught[0].message)


def test_decoder_specs_store_public_builders() -> None:
    """Every callable stored by a DecoderSpec has a stable public builders.py path."""
    specs = [
        decoders.bp_osd(),
        decoders.bp_lsd(),
        decoders.bf(),
        decoders.mwpm(),
        decoders.relay_bp(),
        decoders.min_sum_bp(),
        decoders.lookup_table(1),
        decoders.ilp(),
        decoders.guf(),
    ]
    for spec in specs:
        for builder in (spec._builder, spec._observable_builder):
            if builder is None:
                continue
            assert builder.__module__ == "qldpc.decoders.builders"
            assert not builder.__name__.startswith("_")

        restored = pickle.loads(pickle.dumps(spec))  # noqa: S301 - trusted round trip
        assert restored.options == spec.options
        assert restored._builder is spec._builder
        assert restored._observable_builder is spec._observable_builder


def test_old_builder_pickle_path() -> None:
    """Pickles naming an uppercase retrieval getter still load as the canonical builder."""
    payload = pickle.dumps(builders.get_decoder_bp_osd, protocol=0)
    payload = payload.replace(
        b"qldpc.decoders.builders\nget_decoder_bp_osd",
        b"qldpc.decoders.retrieval\nget_decoder_BP_OSD",
    )
    with pytest.warns(DeprecationWarning, match="get_decoder_bp_osd"):
        restored = pickle.loads(payload)  # noqa: S301 - deliberately constructed compatibility data
    assert restored is builders.get_decoder_bp_osd


@pytest.mark.parametrize(
    ("builder", "old_name"),
    [
        (builders.get_error_decoder_mwpm, "_get_error_decoder_MWPM"),
        (builders.get_observable_decoder_mwpm, "_get_observable_decoder_MWPM"),
        (builders.get_relay_bp_decoder, "_get_relay_bp_decoder"),
        (builders.get_min_sum_bp_decoder, "_get_min_sum_bp_decoder"),
        (builders.get_observable_decoder_lookup, "_get_observable_lookup_decoder"),
    ],
)
def test_old_decoder_spec_builder_pickle_paths(builder: Any, old_name: str) -> None:
    """Pickles naming the former private builders stored by public specs remain loadable."""
    payload = pickle.dumps(builder, protocol=0)
    payload = payload.replace(
        f"qldpc.decoders.builders\n{builder.__name__}".encode(),
        f"qldpc.decoders.retrieval\n{old_name}".encode(),
    )
    with pytest.warns(DeprecationWarning, match=builder.__name__):
        restored = pickle.loads(payload)  # noqa: S301 - deliberately constructed compatibility data
    assert restored is builder


def test_retrieval_wildcard_import_compatibility() -> None:
    """Wildcard imports retain the retrieval module's former public compatibility names."""
    code = """
import warnings
from qldpc import decoders
from qldpc.decoders import builders
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    from qldpc.decoders.retrieval import *
assert caught
assert get_decoder_BP_OSD is builders.get_decoder_bp_osd
assert ErrorsToObservablesDecoder is decoders.ErrorsToObservablesDecoder
assert match_error_decoder_to_dem is decoders.match_error_decoder_to_dem
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_clean_import_order() -> None:
    """Canonical modules can be imported before the package root without warnings."""
    code = """
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("error", DeprecationWarning)
    from qldpc.decoders import builders, retrieval
    from qldpc.decoders.adapters import dem as dem_adapters
    from qldpc import decoders
    assert decoders.get_decoder_bp_osd is builders.get_decoder_bp_osd
    assert decoders.ErrorsToObservablesDecoder is dem_adapters.ErrorsToObservablesDecoder
    assert retrieval.bp_osd()._builder is builders.get_decoder_bp_osd
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_builder_display_name_is_explicit() -> None:
    """Erasure errors use the declared display name, not capitalization of a function name."""

    @builders._erasure_bit_support("Friendly Name", supported=True)
    def unusually_named_builder(
        matrix: np.ndarray, *, add_erasure_bit: bool = False
    ) -> decoders.ErrorDecoder:
        return builders.get_decoder_bp_osd(matrix)

    with pytest.raises(ValueError, match="The Friendly Name decoder cannot signal erasure"):
        unusually_named_builder(np.eye(1, dtype=int), add_erasure_bit=True)
