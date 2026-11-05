# SPDX-License-Identifier: Apache-2.0

"""Noise models for stim circuits, and for circuits that wrap them, such as tsim's.

The main components of this package are:

- PauliChannel: A sparse Pauli channel on one or more qubits.
- NoiseRule: Defines how to add noise to individual operations.
- NoiseModel: Defines how to add noise to circuits.
- Built-in noise models: DepolarizingNoiseModel and the superconducting-inspired SI1000NoiseModel.
- Scheduler: Places the operations of circuits in moments, which set idle noise.
- Schedule: The moments of the operations of a circuit.
- Built-in schedulers: TickScheduler, ASAPScheduler, and ALAPScheduler.

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


``clifford_nq_error`` sets the noise of unitary Clifford gates by their number of qubits.  Each
Pauli product of an SPP or SPP_DAG gate counts as a gate on its qubits.  :class:`NoiseModel` lists
the forms that this noise can take::

    from qldpc.circuits.noise_model import NoiseModel, NoiseRule, PauliChannel

    noise_model = NoiseModel(
        clifford_nq_error={
            1: 1e-4,                                 # DEPOLARIZE1(1e-4) after one-qubit gates
            2: 1e-3,                                 # DEPOLARIZE2(1e-3) after two-qubit gates
            3: PauliChannel.depolarizing(3, 5e-3),   # a chain of E / ELSE_CORRELATED_ERROR
            4: {"XXXX": 1e-4, "ZZZZ": 2e-4},         # a sparse four-qubit channel
        }
    )

A rule can target one Pauli product of an MPP, SPP, or SPP_DAG gate.  Its key names the product,
as in ``"MXYZ"`` for ``MPP X*Y*Z`` and ``"SXY_DAG"`` for ``SPP_DAG X*Y``::

    from qldpc.circuits.noise_model import NoiseModel, NoiseRule, PauliChannel

    noise_model = NoiseModel(
        readout_error=1e-3,
        rules={
            "MXYZ": NoiseRule(readout_error=5e-3),                        # for MPP X*Y*Z
            "SXYZ": NoiseRule(after=PauliChannel.depolarizing(3, 1e-2)),  # for SPP X*Y*Z
        },
    )

Stim needs ``approximate_disjoint_errors=True`` to turn some Pauli channels into a detector error
model, such as the ``E`` / ``ELSE_CORRELATED_ERROR`` chains that encode channels on three or more
qubits.  qLDPC passes this option where it makes detector error models, and callers who make their
own may need to pass it too.

``rule_func`` sets noise that depends on the qubits of each gate.  It takes precedence over all
other rules, and returns None to fall back on them::

    import stim
    from qldpc.circuits.noise_model import NoiseModel, NoiseRule, PauliChannel

    def bad_qubit_noise(op: stim.CircuitInstruction) -> NoiseRule | None:
        # add strong depolarizing noise after every gate on qubit 7
        targets = [t for t in op.targets_copy() if not t.is_combiner]
        if any(t.qubit_value == 7 for t in targets):
            return NoiseRule(after=PauliChannel.depolarizing(len(targets), 1e-2))
        return None

    noise_model = NoiseModel(
        clifford_1q_error=1e-4,
        clifford_2q_error=1e-3,
        rule_func=bad_qubit_noise,
    )
    circuit = stim.Circuit("H 7 \\n CX 0 1 2 7")
    noisy_circuit = noise_model.noisy_circuit(circuit)

.. rubric:: Idle noise

The idle errors of a noise model depend on how long qubits idle.  Its ``scheduler`` places the
operations of a circuit in moments to find this.  The default ``ASAPScheduler()`` places gates as
early as possible, and then resets as late as the other gates allow.  ``SI1000NoiseModel`` uses a
``TickScheduler``, which takes the moments of a circuit from its TICKs.  See :class:`NoiseModel` for
idle runs, :class:`Schedule` for moments, and the :doc:`noise-model notebook
</examples/noise_models>` for examples.

TICKs can set the timing of circuits in two ways.  With ``scheduler=TickScheduler()``, the TICKs of
each circuit set its moments.  With ``scheduler=schedule``, for a schedule of a circuit with TICKs,
a noise model takes the moments of each circuit from that schedule.  This circuit may order
operations differently, for example to measure qubits earlier, but it must have the same operations
on each qubit, in the same order, apart from their measurement-record targets.  The noisy circuit
keeps the order of its input::

    from qldpc.circuits.noise_model import NoiseModel, TickScheduler

    # the TICKs of each circuit set its moments
    noise_model = NoiseModel(idle_error=1e-4, scheduler=TickScheduler())
    noisy_circuit = noise_model.noisy_circuit(circuit_with_ticks)

    # the TICKs of timed_circuit set the moments of circuit
    schedule = TickScheduler().get_schedule(timed_circuit)
    noise_model = NoiseModel(idle_error=1e-4, scheduler=schedule)
    noisy_circuit = noise_model.noisy_circuit(circuit)

``quadratic_idle_error`` models slow noise, such as a qubit frequency that changes from one shot to
the next.  Over an idle run, such noise rotates a qubit about a fixed axis.  The angle grows
linearly, at a random rate with a Gaussian distribution.  Averaged over this rate, the Pauli twirl
of the rotation has an error probability that grows quadratically at first and saturates at 1/2.
For a channel with several Paulis, the axis may change from one idle run to the next, for example
as a compiler changes the frame of a qubit.  The channel then sets the mean squared components of
the axis.  The model treats idle runs as independent, and all moments as equally long.

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
from .scheduling import (
    ALAPScheduler,
    ASAPScheduler,
    Schedule,
    Scheduler,
    TickScheduler,
)

__all__ = [
    "DEFAULT_IMMUNE_OP_TAG",
    "DEFAULT_IMMUNE_QUBIT_TAG",
    "ALAPScheduler",
    "ASAPScheduler",
    "AbstractPauliChannel",
    "DepolarizingNoiseModel",
    "ErrorSpec",
    "NoiseModel",
    "NoiseRule",
    "PauliChannel",
    "PauliChannelSequence",
    "SI1000NoiseModel",
    "Schedule",
    "Scheduler",
    "TickScheduler",
    "as_noiseless_circuit",
]
