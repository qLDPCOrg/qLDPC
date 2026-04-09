"""Unit tests for benchmarking.py

Copyright 2023 The qLDPC Authors and Infleqtion Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import numpy as np
import pytest
import stim

from qldpc import circuits, codes, math
from qldpc.objects import Pauli


def test_state_prep() -> None:
    """State preparation circuit benchmarks."""

    # construct a state prep circuit for the Steane code
    code = codes.SteaneCode()
    state_prep_circuit = stim.Circuit("""
        H 0 2 4
        CX 0 3 2 1 4 5
        CX 0 1 2 6 4 3
        CX 2 5 3 6
        H 7
        CZ 7 1 7 3 7 5
        MX 7
    """)

    noise_model_family = circuits.DepolarizingNoiseModel
    error_rates = np.logspace(-3, -1, 5)

    # obsrevables can be specified by either a symplectic matrix or list of Pauli strings
    observables = code.get_logical_ops(Pauli.Z, symplectic=True)
    string_observables = [math.op_to_string(obs) for obs in observables]

    # post selection is broken in sinter
    with pytest.raises(ValueError, match="bug in sinter"):
        circuits.get_state_prep_diagnostic_tasks(
            code,
            state_prep_circuit,
            error_rates,
            noise_model_family,
            observables=string_observables,
            post_select_on_flags=True,
        )

    # build sinter tasks
    tasks = circuits.get_state_prep_diagnostic_tasks(
        code,
        state_prep_circuit,
        error_rates,
        noise_model_family,
        observables=string_observables,
    )

    print()
    print()
    print()
    for task in tasks:
        print()
        print(task)
    print()
    print()
