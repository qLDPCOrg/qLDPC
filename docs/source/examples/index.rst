Examples
========

These notebooks live in the `examples/ directory
<https://github.com/qLDPCOrg/qLDPC/tree/main/examples>`_ of the repository.  They serve three
purposes: an introduction to using ``qLDPC``, pedagogical material for learning about quantum error
correction, and code you can copy and adapt for your own use case.

Each notebook runs independently from a fresh kernel.
Setup sections include an executable ``%pip`` cell for installing their dependencies into the
active notebook kernel.
Workflow cells use ordinary Python, but a script must place calls to ``sinter.collect`` or TQEC's
Sinter runner under an ``if __name__ == "__main__":`` guard because Sinter starts worker processes
with Python's spawn method.
The stored simulations use bounded sample budgets for documentation; they are not
publication-quality benchmarks.

Getting started
---------------

Start with :doc:`basics` for matrix-defined classical and CSS codes, logical operators, Tanner
graphs, and subsystem-code terminology.
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

A progressive series on estimating logical error rates, from the code-capacity model up to
circuit-level simulations with Sinter.
The numbered notebooks build conceptually on one another, but none relies on files or state produced
by an earlier notebook.

.. toctree::
   :maxdepth: 1

   logical_error_rates/1_code_capacity
   logical_error_rates/2_quantum_memory_x_or_z
   logical_error_rates/3_quantum_memory_combined
   logical_error_rates/4_sliding_window_decoding
   logical_error_rates/5_state_preparation
   logical_error_rates/6_knill_qec
   logical_error_rates/7_individual_observables

Miscellaneous
-------------

Further logical-error-rate examples that fall outside the progressive series above.
The TQEC example also requires the external ``tqec`` package.

.. toctree::
   :maxdepth: 1

   logical_error_rates/misc/alpha_syndrome
   logical_error_rates/misc/decoding_tqec_circuits

Experimental
------------

Examples built on ``qldpc.experimental``, whose public API is unstable and may change without
notice.
The lattice-surgery notebook is an advanced, bounded tutorial with separate scripts for
publication-scale data collection.

.. toctree::
   :maxdepth: 1

   lattice_surgery
