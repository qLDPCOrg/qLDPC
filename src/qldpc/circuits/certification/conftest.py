# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures for the tests of certification."""

from __future__ import annotations

# The fault lister lists class codewords on threads, so each test here runs with the check of the
# rules on threads of the distance package ("Threads" in fault_propagation.md).
from qldpc.codes.distance.conftest import check_code_on_threads  # noqa: F401
