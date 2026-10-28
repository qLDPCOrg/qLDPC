# SPDX-License-Identifier: Apache-2.0

"""A fixture that keeps the tests of the search from writing to the experiment."""

from __future__ import annotations

import pytest

from ..core import paths


@pytest.fixture(autouse=True)
def _keep_default_results_out_of_the_experiment(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Send any file that a test writes without ``--results-dir`` to a throwaway directory."""
    monkeypatch.setattr(paths, "RESULTS_DIR", tmp_path_factory.mktemp("default_results"))
