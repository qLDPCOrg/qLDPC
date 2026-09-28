# SPDX-License-Identifier: Apache-2.0

"""Shared pytest helpers for circuits tests."""

from __future__ import annotations

import stim
from typing_extensions import Self


class StimCircuitWrapper:
    """A minimal stim-wrapping circuit (like tsim.Circuit) for exercising StimCircuitProtocol.

    It satisfies StimCircuitProtocol (append / __iadd__ / indexing / len) and is
    default-constructible, so circuit-polymorphic functions accept it and return the same type.
    The ``stim_circuit`` attribute is not required by those functions; it just lets tests read back
    the wrapped circuit to check results.
    """

    def __init__(self, stim_circuit: stim.Circuit | None = None) -> None:
        self.stim_circuit = stim.Circuit() if stim_circuit is None else stim_circuit

    def append(self, *args: object, **kwargs: object) -> None:
        self.stim_circuit.append(*args, **kwargs)

    def __iadd__(self, other: stim.Circuit) -> Self:
        self.stim_circuit += other
        return self

    def __getitem__(self, index: int) -> stim.CircuitInstruction | stim.CircuitRepeatBlock:
        return self.stim_circuit[index]

    def __len__(self) -> int:
        return len(self.stim_circuit)
