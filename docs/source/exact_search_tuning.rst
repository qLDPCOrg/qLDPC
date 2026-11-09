Exact-search performance and tuning
===================================

.. currentmodule:: qldpc.codes.distance._tuning

Exact distance searches and logical-operator counts choose between routes, such as exhaustive enumeration and searches of overlapping information sets, by cost models, and they bound their batches, memory, and caches.
This page leads from symptoms and routes to the values that control them, and renders the documentation of each from the private module ``qldpc.codes.distance._tuning``.

That module is an unsupported escape hatch for contributors, benchmarks, and measured investigations of very costly searches, not part of the API of qLDPC.
Its names, fields, defaults, and constants can change in any release, without a deprecation period or any other warning, so code that uses it should pin the exact version of qLDPC that it was written for.
No value changes an exact result; the defaults suit typical searches.

Every value that changes only how long a search takes or how much memory it uses is a field of :class:`SearchTuning`, which :func:`use_tuning` changes for the searches that start within a ``with`` block.
The arguments of the public search functions say instead what to compute and by which method, such as ``cutoff``, ``max_weight``, and ``method``.
Each field says what it decides, what raising or lowering it does, and how to retune it.
The constants, :data:`MAX_VALUE` and the sizes of caches, are fixed when qLDPC is imported.
The `Brouwer--Zimmermann guide <https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/codes/distance/brouwer_zimmermann.md>`_ and the `connected-support guide <https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/codes/distance/connected_support.md>`_ explain the methods.

Start from a symptom
--------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Symptom
     - What to try first, then measure
   * - A search of a small problem, or of a long code of low dimension, spends its time on setup
     - Raise :attr:`~SearchTuning.enumeration_max_dimension` or :attr:`~SearchTuning.enumeration_setup_cost_factor` to enumerate every word instead.
   * - A hard distance search spends its time in its last, costliest layers
     - Lower :attr:`~SearchTuning.completion_bound_minimum_layer_work` or raise :attr:`~SearchTuning.completion_bound_max_cost_percent` to use completion tables in more layers; raise :attr:`~SearchTuning.max_information_sets`; lower :attr:`~SearchTuning.lp_minimum_work` or :attr:`~SearchTuning.automorphism_minimum_layer_work`.
   * - A cheap search spends its time building families of information sets
     - Raise :attr:`~SearchTuning.overlapping_minimum_work` or :attr:`~SearchTuning.partition_work_ratio`, or lower :attr:`~SearchTuning.max_information_sets`.
   * - A search or count uses too much memory
     - For distance searches, lower :attr:`~SearchTuning.completion_bound_max_table_bytes`, which applies to each problem of a joint search, such as each sector of a CSS code.
       For exhaustive enumeration, lower :attr:`~SearchTuning.enumeration_batch_exponent`.
       For ``CosetLeaderSearch``, lower :attr:`~SearchTuning.max_css_single_pairs`; for the caches, edit their constants.
       No value limits the memory of a whole process.
   * - A keyboard interrupt takes long to take effect
     - Lower :attr:`~SearchTuning.enumeration_batch_exponent`, :attr:`~SearchTuning.information_set_batch_exponent`, :attr:`~SearchTuning.scan_chunk_size`, :attr:`~SearchTuning.scan_chunks_per_call`, or :attr:`~SearchTuning.macwilliams_chunk_size`.
   * - Completion tables are used, or declined, against measured savings
     - Refit the fields of their cost model, as :ref:`Retuning <tuning-retuning>` describes.

Routes and the names that control them
--------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 45 15 15

   * - Route, or where time goes
     - Fields of :class:`SearchTuning`
     - Constants
     - Guide
   * - Exhaustive enumeration of small problems, instead of building information sets
     - :attr:`SearchTuning.allow_enumeration_by_size`, :attr:`SearchTuning.allow_enumeration_by_work`, :attr:`SearchTuning.enumeration_max_dimension`, :attr:`SearchTuning.enumeration_setup_cost_factor`, :attr:`SearchTuning.enumeration_batch_exponent`
     -
     - `Section 3.6 <https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/codes/distance/brouwer_zimmermann.md#36-exhaustive-enumeration-as-a-fallback>`__
   * - Estimating the work of searches of information sets, which chooses between their routes
     -
     - :data:`BINOMIAL_CACHE_SIZE`
     - `Section 3.6 <https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/codes/distance/brouwer_zimmermann.md#36-exhaustive-enumeration-as-a-fallback>`__
   * - Packing the columns into disjoint information sets
     - :attr:`SearchTuning.disjoint_random_column_orders`
     -
     - `Section 4.4 <https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/codes/distance/brouwer_zimmermann.md#44-heuristics-and-cost-models>`__
   * - Building families of overlapping information sets
     - :attr:`SearchTuning.max_information_sets`, :attr:`SearchTuning.overlapping_minimum_work`, :attr:`SearchTuning.overlapping_attempts_per_set`
     -
     - `Section 3.1 <https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/codes/distance/brouwer_zimmermann.md#31-overlapping-search-bases>`__
   * - Building families by matroid partitioning
     - :attr:`SearchTuning.max_information_sets`, :attr:`SearchTuning.partition_work_ratio`, :attr:`SearchTuning.max_partition_coverage`, :attr:`SearchTuning.partition_cost_factor`
     -
     - `Section 3.1 <https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/codes/distance/brouwer_zimmermann.md#31-overlapping-search-bases>`__
   * - Ending a search of overlapping sets sooner with a certificate from linear programming
     - :attr:`SearchTuning.lp_minimum_work`, :attr:`SearchTuning.lp_dual_max_denominator`
     -
     - `Section 3.1 <https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/codes/distance/brouwer_zimmermann.md#31-overlapping-search-bases>`__
   * - Searching the images of one information set under automorphisms
     - :attr:`SearchTuning.automorphism_minimum_layer_work`
     -
     - `Section 3.7 <https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/codes/distance/brouwer_zimmermann.md#37-images-of-an-information-set-under-automorphisms>`__
   * - Searching from one root for each orbit of the automorphisms of a code, in connected-support searches
     - :attr:`SearchTuning.automorphism_minimum_states`
     -
     - `Section 3.5 <https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/codes/distance/connected_support.md#35-symmetries>`__
   * - Whether a layer of one information set uses completion tables to skip hopeless supports
     - :attr:`SearchTuning.completion_bound_minimum_layer_work`, :attr:`SearchTuning.completion_bound_max_cost_percent`
     -
     - `Section 3.9 <https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/codes/distance/brouwer_zimmermann.md#39-skipping-completions-that-cannot-lower-u>`__
   * - The memory and the blocks of completion tables
     - :attr:`SearchTuning.completion_bound_max_table_bytes`, :attr:`SearchTuning.completion_bound_block_bits`, :attr:`SearchTuning.completion_bound_fallback_block_bits`
     -
     -
   * - Estimating how much completion tables would skip
     - :attr:`SearchTuning.completion_bound_samples`, :attr:`SearchTuning.completion_bound_min_samples`, :attr:`SearchTuning.completion_bound_sampling_work`, :attr:`SearchTuning.completion_bound_confidence_z`
     -
     -
   * - The cost model of completion tables
     - :attr:`SearchTuning.completion_bound_lookup_block_cost`, :attr:`SearchTuning.completion_bound_lookup_block_cost_per_mib`, :attr:`SearchTuning.completion_bound_kept_first_row_ratio`, :attr:`SearchTuning.completion_bound_table_byte_cost`, :attr:`SearchTuning.completion_bound_build_step_cost`
     -
     -
   * - Scans of information sets by the Brouwer--Zimmermann method: time before a keyboard interrupt takes effect, and the overhead of returning to Python
     - :attr:`SearchTuning.information_set_batch_exponent`
     -
     - `Section 4.5 <https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/codes/distance/brouwer_zimmermann.md#45-behavior-spanning-files>`__
   * - Counts by the Brouwer--Zimmermann or brute-force method: time before a keyboard interrupt takes effect
     - :attr:`SearchTuning.scan_chunk_size`, :attr:`SearchTuning.scan_chunks_per_call`
     -
     -
   * - Counts by the MacWilliams method
     - :attr:`SearchTuning.macwilliams_shared_walk_rank`, :attr:`SearchTuning.macwilliams_chunk_size`
     - :data:`KRAWTCHOUK_CACHE_SIZE`
     -
   * - Searches of the lightest Pauli errors by CSS weight on a CSS state
     - :attr:`SearchTuning.disjoint_random_column_orders`
     - :data:`CSS_SEARCH_CACHE_SIZE`
     -
   * - Searches of the lightest Pauli errors by CSS weight on another stabilizer state
     - :attr:`SearchTuning.disjoint_random_column_orders`, :attr:`SearchTuning.max_css_single_pairs`
     -
     -
   * - Counts of deformed codes
     -
     - :data:`LOCAL_CLIFFORD_CACHE_SIZE`
     -

Settings
--------

.. autoclass:: SearchTuning
   :members:
   :undoc-members:
   :member-order: bysource

.. autofunction:: get_tuning

.. autofunction:: use_tuning

Constants
---------

.. autodata:: MAX_VALUE
.. autodata:: BINOMIAL_CACHE_SIZE
.. autodata:: CSS_SEARCH_CACHE_SIZE
.. autodata:: KRAWTCHOUK_CACHE_SIZE
.. autodata:: LOCAL_CLIFFORD_CACHE_SIZE

The module
----------

.. automodule:: qldpc.codes.distance._tuning
