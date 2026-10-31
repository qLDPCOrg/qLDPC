# SPDX-License-Identifier: Apache-2.0

"""Circuit-to-circuit transformations.

The modules of this subpackage import only ``qldpc.circuits.common`` and
``qldpc.circuits.bookkeeping`` from ``qldpc.circuits``, so that every module that constructs
circuits can use them (see ``qldpc.circuits``).
"""

from .qubit_remapping import remap_qubit_target, with_remapped_qubits

__all__ = [
    "remap_qubit_target",
    "with_remapped_qubits",
]
