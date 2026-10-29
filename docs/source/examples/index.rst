Examples
========

These notebooks live in the `examples/ directory <https://github.com/qLDPCOrg/qLDPC/tree/main/examples>`_ of the repository.
They serve three purposes: an introduction to using ``qLDPC``, pedagogical material for learning about quantum error correction, and code you can copy and adapt for your own use case.

Each notebook runs independently from a fresh kernel.
Setup cells install qLDPC and any additional packages needed by that notebook into the active kernel.
The decoder introduction uses an already installed checkout and runs offline.
Workflow cells use ordinary Python, but a script must place calls to ``sinter.collect`` or TQEC's Sinter runner under an ``if __name__ == "__main__":`` guard because Sinter starts worker processes with Python's spawn method.
The stored simulations use bounded sample budgets for documentation; they are not publication-quality benchmarks.

Getting started
---------------

Start with :doc:`basics` for matrix-defined classical and CSS codes, logical operators, Tanner graphs, and subsystem-code terminology.
The other notebooks in this group are standalone technical examples.

.. toctree::
   :maxdepth: 1

   basics
   decoders
   bivariate_bicycle_codes
   noise_models
   transversal_gates

Logical error rates
--------------------

The numbered route starts with sampled logical error rates and then introduces exact structural checks on circuits.
Each notebook answers a different scientific question, and none relies on files or state produced by an earlier notebook.

* :doc:`Code capacity <logical_error_rates/1_code_capacity>` isolates the code and decoder with perfect checks.
* :doc:`Fixed-basis memory <logical_error_rates/2_quantum_memory_x_or_z>` includes circuit-level faults and tracks one logical basis at a time.
* :doc:`Combined-basis memory <logical_error_rates/3_quantum_memory_combined>` uses ideal reference qubits to track both logical bases in one diagnostic experiment.
* :doc:`Memory fault distance <logical_error_rates/4_memory_fault_distance>` computes the minimum number of modeled mechanisms causing an undetected logical flip.
* :doc:`State preparation <logical_error_rates/5_state_preparation>` brings sampled quality, discard rates, and bounded-propagation certificates together for a flagged preparation circuit.
* :doc:`Steane-swap QEC <logical_error_rates/6_steane_qec>` uses fresh encoded blocks in a complete correction workflow.

For a memory study, start with code capacity and follow the first four notebooks.
For preparation protocols, start directly with state preparation and continue to Steane-swap QEC; its ideal-reference construction is explained in the combined-basis memory notebook.
These are suggested reading routes, not execution prerequisites.
The Sinter examples on this route use one worker by default and keep collected statistics separate from plotting.

.. toctree::
   :maxdepth: 1

   logical_error_rates/1_code_capacity
   logical_error_rates/2_quantum_memory_x_or_z
   logical_error_rates/3_quantum_memory_combined
   logical_error_rates/4_memory_fault_distance
   logical_error_rates/5_state_preparation
   logical_error_rates/6_steane_qec

Specialized topics
------------------

Self-contained logical-error-rate examples on narrower topics that fall outside the progressive series above.
Use :doc:`individual-observable rates <logical_error_rates/specialized/per_observable_error_rates>` to unpack aggregate memory failures, and :doc:`sliding-window decoding <logical_error_rates/specialized/sliding_window_decoding>` for longer detector histories.
The :doc:`AlphaSyndrome example <logical_error_rates/specialized/alpha_syndrome>` explores extraction schedules, while the :doc:`Clifford preparation proxy <logical_error_rates/specialized/clifford_state_preparation_proxy>` applies preparation diagnostics to a circuit inspired by a non-Clifford protocol without simulating that protocol.
The :doc:`TQEC example <logical_error_rates/specialized/decoding_tqec_circuits>` connects externally generated lattice-surgery circuits to qLDPC decoders.
The TQEC example requires Python 3.11--3.13 because its pinned ``tqec==0.2.0`` dependency does not support Python 3.14.

.. toctree::
   :maxdepth: 1

   logical_error_rates/specialized/alpha_syndrome
   logical_error_rates/specialized/clifford_state_preparation_proxy
   logical_error_rates/specialized/decoding_tqec_circuits
   logical_error_rates/specialized/per_observable_error_rates
   logical_error_rates/specialized/sliding_window_decoding

Experimental
------------

Examples built on ``qldpc.experimental``, whose public API is unstable and may change without notice.
The lattice-surgery notebook is an advanced, bounded tutorial with separate scripts for publication-scale data collection.
The notebook on benchmarking logical state preparation estimates a decoder-free figure of merit for comparing noisy preparation circuits.

.. toctree::
   :maxdepth: 1

   lattice_surgery
   experimental/logical_state_preparation
