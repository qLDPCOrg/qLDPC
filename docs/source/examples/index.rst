Examples
========

These notebooks live in the `examples/ directory
<https://github.com/qLDPCOrg/qLDPC/tree/main/examples>`_ of the repository.  They serve three
purposes: an introduction to using ``qLDPC``, pedagogical material for learning about quantum error
correction, and code you can copy and adapt for your own use case.

Getting started
---------------

.. toctree::
   :maxdepth: 1

   basics
   bivariate_bicycle_codes
   noise_models
   transversal_gates

Logical error rates
--------------------

A progressive series on estimating logical error rates, from the code-capacity model up to
circuit-level simulations with Sinter.

.. toctree::
   :maxdepth: 1

   logical_error_rates/1_code_capacity
   logical_error_rates/2_quantum_memory_x_or_z
   logical_error_rates/3_quantum_memory_combined
   logical_error_rates/4_sliding_window_decoding
   logical_error_rates/5_state_preparation
   logical_error_rates/6_alpha_syndrome
   logical_error_rates/7_knill_qec
   logical_error_rates/8_decoding_tqec_circuits
