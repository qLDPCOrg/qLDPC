# SPDX-License-Identifier: Apache-2.0

"""Noise models for stim circuits (and stim-wrapping circuits such as tsim's).

The main components of this package are:

- PauliChannel: A sparse multi-qubit Pauli channel used to specify multi-qubit noise.
- NoiseRule: Defines how to add noise to individual operations.
- NoiseModel: Defines how noise is added to circuits.
- Built-in noise models: DepolarizingNoiseModel and the superconducting-inspired SI1000NoiseModel.

Examples of basic usage with a predefined noise model::

    import stim
    from qldpc.circuits.noise_model import DepolarizingNoiseModel, NoiseModel, SI1000NoiseModel

    # Create a simple circuit
    circuit = stim.Circuit("H 0 \\n CX 0 1")

    # Apply simple depolarizing noise
    noise_model = DepolarizingNoiseModel(0.001)
    noisy_circuit = noise_model.noisy_circuit(circuit)

    # Apply superconducting-inspired noise
    noise_model = SI1000NoiseModel(0.001)
    noisy_circuit = noise_model.noisy_circuit(circuit)

    # Create a custom noise model
    custom_model = NoiseModel(
        clifford_1q_error=3e-5,
        clifford_2q_error=1e-3,
        readout_error=1e-3,
        reset_error=1e-3,
        idle_error=2e-4,
    )
    noisy_circuit = custom_model.noisy_circuit(circuit)


Noise on multi-qubit Clifford gates (SPP / MPP)::

    from qldpc.circuits.noise_model import NoiseModel, NoiseRule, PauliChannel

    # `clifford_nq_error` maps a qubit count ``k`` to the noise applied after each ``k``-qubit
    # unitary Clifford gate (namely, the Pauli-product Cliffords SPP and SPP_DAG, which are stim's
    # multi-qubit unitary Clifford primitives).  Values may be a float (uniform ``k``-qubit
    # depolarizing channel), a Mapping[str, float] (auto-wrapped as PauliChannel), a PauliChannel,
    # or a full NoiseRule.
    noise_model = NoiseModel(
        clifford_nq_error={
            1: 1e-4,                                 # DEPOLARIZE1(1e-4) after each 1-qubit gate
            2: 1e-3,                                 # DEPOLARIZE2(1e-3) after each 2-qubit gate
            3: PauliChannel.depolarizing(3, 5e-3),   # 3-qubit depolarizing channel, emitted as
                                                     # a chain of CORRELATED_ERROR /
                                                     # ELSE_CORRELATED_ERROR instructions
            4: {"XXXX": 1e-4, "ZZZZ": 2e-4},         # sparse 4-qubit channel (raw dict)
        }
    )

    # Multi-qubit MPP / SPP gates receive ordinary readout_error / clifford_nq_error.  To assign
    # per-basis rules, use "M<paulis>" for MPP (e.g. "MXYZ" for `MPP X*Y*Z`, "MXX" for `MPP X*X`)
    # and "S<paulis>" / "S<paulis>_DAG" for SPP / SPP_DAG (e.g. "SXYZ" for `SPP X*Y*Z`, "SXY_DAG"
    # for `SPP_DAG X*Y`).
    noise_model = NoiseModel(
        readout_error=1e-3,                                # default readout flip probability
        rules={
            "MXYZ": NoiseRule(readout_error=5e-3),         # override for MPP X*Y*Z specifically
            "SXYZ": NoiseRule(after=PauliChannel.depolarizing(3, 1e-2)),  # SPP X*Y*Z override
        },
    )

Multi-qubit Pauli channels with arity three or greater are emitted as correlated ``E`` /
``ELSE_CORRELATED_ERROR`` chains.  Stim's exact detector-error-model conversion rejects these chains
unless ``approximate_disjoint_errors=True`` is passed.  qLDPC's DEM consumers use that explicit
approximation; callers extracting DEMs directly must make the same modeling choice.

Per-gate-application noise via a callback (``rule_func``)::

    from qldpc.circuits.noise_model import NoiseModel, NoiseRule

    # `rule_func` takes top priority over every other rule in a `NoiseModel`.  Broadcast gates such
    # as `H 0 1 2` and `CX 0 1 2 3` are decomposed into individual gates before being passed to
    # `rule_func`, so the `op` argument is always a stim.CircuitInstruction holding exactly one gate
    # application's worth of targets (e.g. two targets for a two-qubit gate, one Pauli product for
    # an SPP/MPP).  Returning `None` falls back to the ordinary `NoiseModel` rules.
    def bad_qubit_noise(op: stim.CircuitInstruction) -> NoiseRule | None:
        # Give any gate touching qubit 7 a heavier depolarizing kick; leave everything else alone.
        targets = [t for t in op.targets_copy() if not t.is_combiner]
        if any(t.qubit_value == 7 for t in targets):
            return NoiseRule(after=PauliChannel.depolarizing(len(targets), 1e-2))
        return None

    noise_model = NoiseModel(
        clifford_1q_error=1e-4,
        clifford_2q_error=1e-3,
        rule_func=bad_qubit_noise,
    )
    noisy_circuit = noise_model.noisy_circuit(circuit)

.. rubric:: Moments and repeat blocks

Noise is added moment by moment, where a moment is the time between two TICKs.  ``REPEAT`` blocks
are treated as though they were flattened, so a moment may span the boundary of a repeat block or of
its iterations.  The noisy circuit is nonetheless compact: once an iteration of a repeat block gets
the same noise as the next, the remaining iterations are kept in a repeat block.  A repeat block
tagged with ``immune_op_tag`` is noiseless and is not merged with neighboring moments, a repeat
block that contains only annotations and noise is kept as-is, and a repeat block that is empty when
flattened is dropped.

.. rubric:: Idling errors

``idle_error`` applies to every qubit that is not operated on during a circuit moment.  When
``idle_error`` is a Pauli channel, consecutive idle errors on the same qubit are combined into one
equivalent error that is emitted before the qubit is next used.  Custom Stim noise fragments are
applied once per idle moment.  ``additional_error_waiting_for_m_or_r`` applies additional noise to
non-collapsing qubits during a moment that contains a measurement or reset; it is always applied
once per such moment.

Explicit identity gates (``I`` and ``II``) are treated as idle for this accounting.  ``rules`` can
append errors after these gates, and ``rule_func`` can select errors based on an identity's tag or
encoded duration.  A rule returned by the callback takes precedence over a matching named rule.
Identity rules and ``idle_error`` stack: when both match, both are applied.

.. rubric:: Attribution and license

This package was originally taken and modified from::

    https://github.com/tqec/tqec/blob/main/src/tqec/utils/noise_model.py

which itself was taken from::

    https://zenodo.org/records/7487893

and licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/legalcode).

The original code was written for the paper at "Inplace Access to the Surface Code Y Basis", at::

    https://quantum-journal.org/papers/q-2024-04-08-1310.

Portions copyright the original authors, licensed under CC BY 4.0 — see above.
"""

from .channels import (
    AbstractPauliChannel,
    PauliChannel,
    PauliChannelSequence,
)
from .noise_model import (
    DEFAULT_IMMUNE_OP_TAG,
    DEFAULT_IMMUNE_QUBIT_TAG,
    DepolarizingNoiseModel,
    NoiseModel,
    SI1000NoiseModel,
    as_noiseless_circuit,
)
from .rules import (
    ErrorSpec,
    NoiseRule,
)

__all__ = [
    "DEFAULT_IMMUNE_OP_TAG",
    "DEFAULT_IMMUNE_QUBIT_TAG",
    "AbstractPauliChannel",
    "DepolarizingNoiseModel",
    "ErrorSpec",
    "NoiseModel",
    "NoiseRule",
    "PauliChannel",
    "PauliChannelSequence",
    "SI1000NoiseModel",
    "as_noiseless_circuit",
]
