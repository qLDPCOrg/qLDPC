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
       At the default :attr:`~SearchTuning.enumeration_batch_exponent`, enumeration handles rows of more than 960 bits one word at a time, so time it against information sets on such codes.
   * - A hard distance search spends its time in its last, costliest layers
     - Raise :attr:`~SearchTuning.completion_bound_max_cost_percent` or lower :attr:`~SearchTuning.completion_bound_minimum_layer_work` to use completion tables in more layers; raise :attr:`~SearchTuning.max_information_sets`; lower :attr:`~SearchTuning.lp_minimum_work` or :attr:`~SearchTuning.automorphism_minimum_layer_work`.
   * - A cheap search spends its time building families of information sets
     - Raise :attr:`~SearchTuning.overlapping_minimum_work` or :attr:`~SearchTuning.partition_work_ratio`, or lower :attr:`~SearchTuning.max_information_sets`.
   * - Queries of ``CosetLeaderSearch`` by the Brouwer--Zimmermann method at moderate maximum weights, or its searches for minimum weights up to them in residual-weight estimates, are slow
     - Lower :attr:`~SearchTuning.overlapping_minimum_work`, so that cheaper queries try a family of overlapping information sets, or a disjoint family with more information sets or a larger partial set.
       In the BZ searches by physical weight, and in the searches of the X and Z parts of errors for the CSS weight on CSS states, a query also weighs words without the pivot columns of each set, and skips completions by tables on blocks of :attr:`~SearchTuning.coset_completion_block_bits` columns, if its predicted examinations with the family that it takes reach the threshold too.
       A search for minimum weights up to a maximum weight takes the family of a query at that weight, or at the number of qubits if that is smaller.
       The oracle builds these families once for each cached search, and their tails and tables once for each family (see :data:`CSS_SEARCH_CACHE_SIZE` and :data:`PAULI_SEARCH_CACHE_SIZE`).
       For larger overlapping families, raise :attr:`~SearchTuning.max_information_sets` or :attr:`~SearchTuning.overlapping_attempts_per_set`.
   * - The costliest queries of ``CosetLeaderSearch`` by the Brouwer--Zimmermann method, which weigh words without pivot columns (see the row above), spend their time building completion tables
     - Lower :attr:`~SearchTuning.coset_completion_block_bits`, or set :attr:`~SearchTuning.completion_bound_max_table_bytes` to zero to build none.
   * - A search or count uses too much memory
     - For distance searches and counts, lower :attr:`~SearchTuning.completion_bound_max_table_bytes`, which applies to each problem of a joint search, such as each sector of a CSS code, and to the scan of each sector of a count.
       For exhaustive enumeration, lower :attr:`~SearchTuning.enumeration_batch_exponent`.
       For the BZ engine of ``CosetLeaderSearch``, lower :attr:`~SearchTuning.max_css_single_pairs`.
       For its completion tables, lower :attr:`~SearchTuning.coset_completion_block_bits`, or :attr:`~SearchTuning.completion_bound_max_table_bytes`, which applies to each set of a family.
       For the caches, edit their constants.
       No value limits the memory of a whole process.
   * - A keyboard interrupt takes long to take effect
     - Lower :attr:`~SearchTuning.enumeration_batch_exponent`, :attr:`~SearchTuning.information_set_batch_exponent`, :attr:`~SearchTuning.scan_chunk_size`, :attr:`~SearchTuning.scan_chunks_per_call`, or :attr:`~SearchTuning.macwilliams_chunk_size`.
       A connected-support search takes an interrupt only between its weight limits, or between coset queries, and no value changes that.
   * - A cheap connected-support search spends its time finding automorphisms
     - Raise :attr:`~SearchTuning.automorphism_minimum_states`.
       A search whose first weight limit a cutoff raises asks for them unless the field is ``MAX_VALUE``.
   * - A ``CosetLeaderSearch`` with the connected-support method takes long to build, or its queries are slow
     - Lower :attr:`~SearchTuning.max_sparsifying_rounds` or :attr:`~SearchTuning.sparsifying_rounds_without_improvement` to build it sooner, or raise them for sparser checks and faster queries.
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
   * - Making the checks of the connected-support coset searches of ``CosetLeaderSearch`` sparser
     - :attr:`SearchTuning.max_sparsifying_rounds`, :attr:`SearchTuning.sparsifying_rounds_without_improvement`
     -
     - `Section 3.8 <https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/codes/distance/connected_support.md#38-vectors-with-a-given-syndrome>`__
   * - Whether an information set in a layer of a distance search, or a step of a count, uses completion tables to skip hopeless supports
     - :attr:`SearchTuning.completion_bound_minimum_layer_work`, :attr:`SearchTuning.completion_bound_max_cost_percent`
     -
     - `Section 3.9 <https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/codes/distance/brouwer_zimmermann.md#39-skipping-completions-by-completion-tables>`__
   * - The memory and the blocks of completion tables
     - :attr:`SearchTuning.completion_bound_max_table_bytes`, :attr:`SearchTuning.completion_bound_block_bits`, :attr:`SearchTuning.completion_bound_fallback_block_bits`, and in ``CosetLeaderSearch`` :attr:`SearchTuning.coset_completion_block_bits`
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
   * - Searches of the lightest Pauli errors by CSS weight on a CSS state, by the Brouwer--Zimmermann method
     - :attr:`SearchTuning.disjoint_random_column_orders`, and for costly queries :attr:`SearchTuning.overlapping_minimum_work`, :attr:`SearchTuning.max_information_sets`, :attr:`SearchTuning.overlapping_attempts_per_set`, :attr:`SearchTuning.coset_completion_block_bits`, :attr:`SearchTuning.completion_bound_max_table_bytes`
     - :data:`CSS_SEARCH_CACHE_SIZE`
     - `Section 4.4 <https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/codes/distance/brouwer_zimmermann.md#44-heuristics-and-cost-models>`__
   * - Searches of the lightest Pauli errors by physical weight, or by CSS weight on another stabilizer state, by the Brouwer--Zimmermann method
     - :attr:`SearchTuning.disjoint_random_column_orders`, :attr:`SearchTuning.max_css_single_pairs`, and for costly queries :attr:`SearchTuning.overlapping_minimum_work`, :attr:`SearchTuning.max_information_sets`, :attr:`SearchTuning.overlapping_attempts_per_set`, and by physical weight :attr:`SearchTuning.coset_completion_block_bits`, :attr:`SearchTuning.completion_bound_max_table_bytes`
     - :data:`PAULI_SEARCH_CACHE_SIZE`
     - `Section 4.4 <https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/codes/distance/brouwer_zimmermann.md#44-heuristics-and-cost-models>`__
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
.. autodata:: PAULI_SEARCH_CACHE_SIZE

The module
----------

.. automodule:: qldpc.codes.distance._tuning
