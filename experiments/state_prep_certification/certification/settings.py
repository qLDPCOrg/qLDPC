# SPDX-License-Identifier: Apache-2.0

"""The settings, paths, and certify options that the modules of the certification experiment share.

``CONFIG`` is the experiment's ``config.json``.  ``DATA`` is its data directory,
``circuit_data/``, which Git ignores, and ``INPUTS`` the directory of the certification inputs in
it, with their record ``circuits.json`` (see ``circuits.py``).  The families of circuits say which
circuits are verified and which take their design t from the distance of their code.
``certify_options``, ``noisy_circuit``, and ``noisy_input`` give the noise and the options with
which a circuit of ``circuits.json`` is certified.
"""

from __future__ import annotations

import importlib
import json
import pathlib
from typing import Any

import stim

HERE = pathlib.Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text())
DATA = HERE / "circuit_data"
INPUTS = DATA / "inputs"
CIRCUITS_JSON = INPUTS / "circuits.json"

# The families of synthesized circuits: the prefix of their ids, and the folder of their circuits.
SYNTHESES = {
    "D-synth": ("dsynth", DATA / "synthesized"),
    "D-synth (MQT 1.9.0)": ("dsynth190", DATA / "synthesized_mqt_1.9.0"),
}

# The families of verified circuits: each gets a weakened variant, a code file, and deletion tasks.
VERIFIED_FAMILIES = ("A-ft", "C-goto", *SYNTHESES)

# The families whose design t is (d - 1) // 2 for the distance d of their code
# (``circuits._sources``).
T_FROM_DISTANCE = ("A-ft", "A-unv", *SYNTHESES)


def certify_options(options: dict[str, Any]) -> dict[str, Any]:
    """The keyword options of ``certify_bounded_fault_propagation`` in a circuit's options.

    ``options`` are the ``options`` of the circuit's record in ``circuits.json``, with its noise;
    the keyword options include K.
    """
    keys = ("add_flags", "post_select", "output_qubits", "css_weight", "max_weight")
    return {key: options[key] for key in keys}


def noisy_circuit(circuit: stim.Circuit, noise: dict[str, Any]) -> stim.Circuit:
    """The given noiseless circuit with the noise of ``noise``.

    ``noise["model"]`` is the import path of a noise model of ``qldpc.circuits``, such as
    ``qldpc.circuits.DepolarizingNoiseModel``, and ``noise["p"]`` its probability.
    """
    module, name = noise["model"].rsplit(".", 1)
    model = getattr(importlib.import_module(module), name)
    return model(noise["p"]).noisy_circuit(circuit)


def noisy_input(entry: dict[str, Any]) -> stim.Circuit:
    """The circuit's frozen noiseless input, with the noise of ``circuits.json``."""
    circuit = stim.Circuit((INPUTS / f"{entry['id']}.stim").read_text())
    return noisy_circuit(circuit, entry["options"]["noise"])
