Exact-search performance and tuning
===================================

.. currentmodule:: qldpc.codes.distance._tuning

This page leads from the symptoms of slow or memory-hungry exact distance searches and logical-operator counts, and from their routes, to the values of the private module ``qldpc.codes.distance._tuning``.
Before you change a value, read :ref:`the documentation of that module <tuning-module>` below: what its privacy means, what tuning can and cannot change, how to change the tuning, and how the defaults are chosen.
The `Brouwer-Zimmermann guide <https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/codes/distance/brouwer_zimmermann.md>`_ and the `connected-cluster guide <https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/codes/distance/connected_cluster.md>`_ explain the methods.

Start from a symptom
--------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Symptom
     - What to try first, then measure
   * - A distance search of a classical code of low dimension, whose distance is a large fraction of its length, is slow
     - Pass ``method="brute_force"`` to enumerate every word instead.
   * - A distance search of a quantum LDPC code by ``method="brouwer_zimmermann"`` is slow
     - Pass ``method="connected_cluster"``, which is often much faster on such codes, though not on all of them (`Section 2 <https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/codes/distance/connected_cluster.md#2-what-the-cost-of-the-search-depends-on>`__ of the connected-cluster guide), or ``method="auto"``, the default, which starts with that search and may switch to the Brouwer-Zimmermann method.
   * - A distance search by ``method="auto"``, the default, is slow
     - Time the search with ``method="brouwer_zimmermann"`` and with ``method="connected_cluster"``, and pass the faster method.
       With a large ``cutoff``, ``"auto"`` cannot switch to the Brouwer-Zimmermann method before it has searched past the cutoff, and with one near the distance it may never switch, so pass ``method="brouwer_zimmermann"``.
       To make ``"auto"`` stay longer with a connected-cluster search, lower :attr:`~SearchTuning.connected_cluster_state_evaluations`, and to make it switch to the Brouwer-Zimmermann method sooner, raise that field.
       Raise :attr:`~SearchTuning.auto_free_work` to decide later, with better estimates of growth, or lower it to decide sooner (`Choosing a method <https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/codes/distance/README.md#choosing-a-method>`__ in the guide to the distance package).
       Raise :attr:`~SearchTuning.auto_probe_work` to let the short Brouwer-Zimmermann search that ``"auto"`` runs before it decides, its probe, finish more searches, or find a lighter word for the connected-cluster search to stop at, at a higher fixed cost.
       Lower :attr:`~SearchTuning.auto_brute_force_max_words` if ``"auto"`` enumerates every word for too long.
   * - A hard distance search spends its time in its last, costliest layers
     - Raise :attr:`~SearchTuning.completion_bound_max_cost_percent` or lower :attr:`~SearchTuning.completion_bound_minimum_layer_work` to use completion tables in more layers.
       For a basis too large for its tables, raise :attr:`~SearchTuning.completion_bound_max_table_bytes` or lower :attr:`~SearchTuning.completion_bound_block_bits`.
       Also raise :attr:`~SearchTuning.max_information_sets`, or lower :attr:`~SearchTuning.lp_minimum_work` or :attr:`~SearchTuning.automorphism_minimum_layer_work`.
   * - A cheap search spends its time building families of information sets
     - Raise :attr:`~SearchTuning.overlapping_minimum_work` or :attr:`~SearchTuning.partition_work_ratio`, or lower :attr:`~SearchTuning.max_information_sets`.
   * - Queries of :class:`~qldpc.codes.distance.CosetLeaderSearch` by the Brouwer-Zimmermann method at moderate maximum weights are slow, as in its searches for minimum weights in residual-weight estimates
     - Lower :attr:`~SearchTuning.coset_route_minimum_work`, so that cheaper queries try a maximum disjoint family and an overlapping family (`Section 4.4 <https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/codes/distance/brouwer_zimmermann.md#44-heuristics-and-cost-models>`__ of the Brouwer-Zimmermann guide).
       In searches by Hamming weight (by physical weight, or of X and Z parts of errors on CSS states), queries that stay costly with the family that they take also skip completions by tables on blocks of :attr:`~SearchTuning.coset_completion_block_bits` columns.
       For larger overlapping families, raise :attr:`~SearchTuning.max_information_sets` or :attr:`~SearchTuning.overlapping_attempts_per_set`.
   * - The costliest queries of :class:`~qldpc.codes.distance.CosetLeaderSearch` by the Brouwer-Zimmermann method spend their time building completion tables
     - Lower :attr:`~SearchTuning.coset_completion_block_bits`, or set :attr:`~SearchTuning.completion_bound_max_table_bytes` to zero to build none.
   * - A search or count uses too much memory
     - For distance searches and counts, lower :attr:`~SearchTuning.completion_bound_max_table_bytes`, which applies to each sector of a search, such as the X-type logical operators of a CSS code, and to each sector of a count, besides the tables that it is building.
       For exhaustive enumeration, lower :attr:`~SearchTuning.enumeration_batch_exponent`.
       For the Brouwer-Zimmermann method of :class:`~qldpc.codes.distance.CosetLeaderSearch`, lower :attr:`~SearchTuning.max_css_single_pairs`.
       For its completion tables, lower :attr:`~SearchTuning.coset_completion_block_bits`, or :attr:`~SearchTuning.completion_bound_max_table_bytes`, which applies to each set of a family, or raise :attr:`~SearchTuning.coset_route_minimum_work`, so that fewer queries build extra families and tables.
       For caches, edit the constants that size them.
       No value limits the memory of a whole process.
   * - A keyboard interrupt takes long to take effect
     - Lower :attr:`~SearchTuning.enumeration_batch_exponent`, :attr:`~SearchTuning.information_set_batch_exponent`, :attr:`~SearchTuning.scan_chunk_size`, :attr:`~SearchTuning.scan_chunks_per_call`, or :attr:`~SearchTuning.macwilliams_chunk_size`.
       A connected-cluster search takes an interrupt only between its weight limits, or between coset queries, and no value changes that.
       ``method="auto"``, the default, runs one for the distances of quantum codes and the fault distances of circuits; pass ``method="brouwer_zimmermann"`` for prompt interrupts.
   * - A cheap connected-cluster search spends its time finding automorphisms
     - Raise :attr:`~SearchTuning.automorphism_minimum_states`.
       Searches ask for them before a first limit that ``cutoff`` sets, unless the field is ``MAX_VALUE``.
   * - Many cheap distance searches of large CSS codes spend their time looking for a qubit permutation that maps the X sector onto the Z sector
     - Set :attr:`~SearchTuning.sector_maps` to ``False``.
   * - A :class:`~qldpc.codes.distance.CosetLeaderSearch` with the connected-cluster method takes long to build, or its queries are slow
     - Lower :attr:`~SearchTuning.max_sparsifying_rounds`, or :attr:`~SearchTuning.sparsifying_rounds_without_improvement` below it, for a quicker build; raise :attr:`~SearchTuning.max_sparsifying_rounds` for sparser checks and faster queries.
   * - A connected-cluster distance search of a subsystem code is slow to start, or slow to search
     - Lower :attr:`~SearchTuning.max_distance_sparsifying_rounds`, or :attr:`~SearchTuning.sparsifying_rounds_without_improvement` below it, to start sooner; raise :attr:`~SearchTuning.max_distance_sparsifying_rounds` for sparser checks and a faster search.
   * - Completion tables are used where they cost more than they save, or declined where they would save
     - For your searches, lower :attr:`~SearchTuning.completion_bound_max_cost_percent` to use them in fewer layers, or raise it to use them in more.

Routes and the names that control them
--------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 45 15 15

   * - Route, or where time goes
     - Fields of :class:`SearchTuning`
     - Constants
     - Guide
   * - Exhaustive enumeration, which ``method="brute_force"`` runs
     - :attr:`SearchTuning.enumeration_batch_exponent`
     -
     - `Section 3.6 <https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/codes/distance/brouwer_zimmermann.md#36-work-estimates-and-exhaustive-enumeration>`__
   * - Searching one sector of a CSS code whose sectors a qubit permutation maps onto each other
     - :attr:`SearchTuning.sector_maps`
     -
     - `Searching one sector of a CSS code <https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/codes/distance/README.md#searching-one-sector-of-a-css-code>`__ in the guide to the distance package
   * - Choosing a method, which ``method="auto"`` does for each distance search
     - :attr:`SearchTuning.auto_brute_force_max_words`, :attr:`SearchTuning.auto_free_work`, :attr:`SearchTuning.auto_probe_work`, :attr:`SearchTuning.connected_cluster_state_evaluations`
     -
     - `Choosing a method <https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/codes/distance/README.md#choosing-a-method>`__ in the guide to the distance package
   * - Estimating the work of searches of information sets, which chooses between their routes
     -
     - :data:`BINOMIAL_CACHE_SIZE`
     - `Section 3.6 <https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/codes/distance/brouwer_zimmermann.md#36-work-estimates-and-exhaustive-enumeration>`__
   * - Packing the columns into disjoint information sets
     - :attr:`SearchTuning.disjoint_random_column_orders`
     -
     - `Section 4.4 <https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/codes/distance/brouwer_zimmermann.md#44-heuristics-and-cost-models>`__
   * - Building families of overlapping information sets
     - :attr:`SearchTuning.max_information_sets`, :attr:`SearchTuning.overlapping_minimum_work`, :attr:`SearchTuning.overlapping_attempts_per_set`
     -
     - `Section 3.1 <https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/codes/distance/brouwer_zimmermann.md#31-overlapping-search-bases>`__
   * - Building families by matroid partitioning
     - :attr:`SearchTuning.max_information_sets`, :attr:`SearchTuning.partition_work_ratio`, :attr:`SearchTuning.partition_max_coverage`
     -
     - `Section 4.4 <https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/codes/distance/brouwer_zimmermann.md#44-heuristics-and-cost-models>`__
   * - Ending a search of overlapping sets sooner with a certificate from linear programming
     - :attr:`SearchTuning.lp_minimum_work`, :attr:`SearchTuning.lp_dual_max_denominator`
     -
     - `Section 3.1 <https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/codes/distance/brouwer_zimmermann.md#31-overlapping-search-bases>`__
   * - Searching the images of one information set under automorphisms
     - :attr:`SearchTuning.automorphism_minimum_layer_work`
     -
     - `Section 3.7 <https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/codes/distance/brouwer_zimmermann.md#37-images-of-an-information-set-under-automorphisms>`__
   * - Searching from one root for each orbit of the automorphisms of a code, in connected-cluster searches
     - :attr:`SearchTuning.automorphism_minimum_states`
     -
     - `Section 3.5 <https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/codes/distance/connected_cluster.md#35-symmetries>`__
   * - Restricting the subtrees of a connected-cluster search to viable symbols, in distance searches and in the coset searches of :class:`~qldpc.codes.distance.CosetLeaderSearch`
     - :attr:`SearchTuning.restrict_viable_symbols`
     -
     - `Section 3.4 <https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/codes/distance/connected_cluster.md#34-pruning>`__
   * - Making the checks of the connected-cluster coset searches of :class:`~qldpc.codes.distance.CosetLeaderSearch` sparser
     - :attr:`SearchTuning.max_sparsifying_rounds`, :attr:`SearchTuning.sparsifying_rounds_without_improvement`
     -
     - `Section 3.8 <https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/codes/distance/connected_cluster.md#38-vectors-with-a-given-syndrome>`__
   * - Making the stabilizers that connected-cluster distance searches derive for subsystem codes sparser
     - :attr:`SearchTuning.max_distance_sparsifying_rounds`, :attr:`SearchTuning.sparsifying_rounds_without_improvement`
     -
     - `Section 3.9 <https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/codes/distance/connected_cluster.md#39-sparser-stabilizers-of-subsystem-codes>`__
   * - Whether an information set in a layer of a distance search, or a step of a count, uses completion tables to skip hopeless supports
     - :attr:`SearchTuning.completion_bound_minimum_layer_work`, :attr:`SearchTuning.completion_bound_max_cost_percent`
     -
     - `Section 3.9 <https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/codes/distance/brouwer_zimmermann.md#39-skipping-completions-by-completion-tables>`__
   * - The memory and the blocks of completion tables
     - :attr:`SearchTuning.completion_bound_max_table_bytes`, :attr:`SearchTuning.completion_bound_block_bits`, and in :class:`~qldpc.codes.distance.CosetLeaderSearch` :attr:`SearchTuning.coset_completion_block_bits`
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
   * - Distance searches by the Brouwer-Zimmermann method: time before a keyboard interrupt takes effect, and the overhead of returning to Python
     - :attr:`SearchTuning.information_set_batch_exponent`
     -
     - `Section 4.5 <https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/codes/distance/brouwer_zimmermann.md#45-behavior-spanning-files>`__
   * - Counts by the Brouwer-Zimmermann or brute-force method: time before a keyboard interrupt takes effect
     - :attr:`SearchTuning.scan_chunk_size`, :attr:`SearchTuning.scan_chunks_per_call`
     -
     -
   * - Counts by the MacWilliams method
     - :attr:`SearchTuning.macwilliams_shared_walk_rank`, :attr:`SearchTuning.macwilliams_chunk_size`
     - :data:`KRAWTCHOUK_CACHE_SIZE`
     -
   * - Searches of the lightest Pauli errors by CSS weight on a CSS state, by the Brouwer-Zimmermann method
     - :attr:`SearchTuning.disjoint_random_column_orders`, and for costly queries :attr:`SearchTuning.coset_route_minimum_work`, :attr:`SearchTuning.max_information_sets`, :attr:`SearchTuning.overlapping_attempts_per_set`, :attr:`SearchTuning.coset_completion_block_bits`, :attr:`SearchTuning.completion_bound_max_table_bytes`
     - :data:`CSS_SEARCH_CACHE_SIZE`
     - `Section 4.4 <https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/codes/distance/brouwer_zimmermann.md#44-heuristics-and-cost-models>`__
   * - Searches of the lightest Pauli errors by physical weight, or by CSS weight on another stabilizer state, by the Brouwer-Zimmermann method
     - :attr:`SearchTuning.disjoint_random_column_orders`, by CSS weight :attr:`SearchTuning.max_css_single_pairs`, and for costly queries :attr:`SearchTuning.coset_route_minimum_work`, :attr:`SearchTuning.max_information_sets`, :attr:`SearchTuning.overlapping_attempts_per_set`, and by physical weight :attr:`SearchTuning.coset_completion_block_bits`, :attr:`SearchTuning.completion_bound_max_table_bytes`
     - :data:`PAULI_SEARCH_CACHE_SIZE`
     - `Section 4.4 <https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/codes/distance/brouwer_zimmermann.md#44-heuristics-and-cost-models>`__
   * - Counts of deformed codes
     -
     - :data:`LOCAL_CLIFFORD_CACHE_SIZE`
     -

.. _tuning-module:

The tuning module
-----------------

.. automodule:: qldpc.codes.distance._tuning

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
