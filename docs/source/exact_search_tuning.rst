Exact-search performance and tuning
===================================

.. currentmodule:: qldpc.codes.distance._tuning

This page suggests what to try first when exact distance searches or logical-operator counts are slow or memory-hungry, often by changing values of the private module ``qldpc.codes.distance._tuning``.
The :ref:`table of routes <tuning-routes>` below leads from the routes that searches and counts take, such as which optional steps they run, to the values that control them.
Before you change a value, read :ref:`the documentation of that module <tuning-module>` below: what its privacy means, what tuning can and cannot change, how to change the tuning, and how the defaults are chosen.
The :guide:`Brouwer-Zimmermann guide <codes/distance/brouwer_zimmermann.md>` and the `connected-cluster guide <https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/codes/distance/connected_cluster.md>`_ explain the methods.
The parts of the Brouwer-Zimmermann guide on the source code of the method, its heuristics, and its correctness requirements are in :guide:`brouwer_zimmermann_code.md <codes/distance/brouwer_zimmermann_code.md>`.

.. _tuning-symptoms:

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
     - Pass ``method="connected_cluster"``, which is often much faster on such codes, though not on all of them (:guide:`Section 2 <codes/distance/connected_cluster.md#sec-cost>` of the connected-cluster guide), or ``method="auto"``, the default, which, unless enumerating is cheap, starts with that search and may switch to the Brouwer-Zimmermann method.
   * - A distance search by ``method="auto"``, the default, is slow
     - Time the search with ``method="brouwer_zimmermann"`` and with ``method="connected_cluster"``, and pass the faster method.
       With a large ``cutoff``, ``"auto"`` cannot switch to the Brouwer-Zimmermann method before it has searched past the cutoff, and with one near the distance it may never switch, so pass ``method="brouwer_zimmermann"``.
       To make ``"auto"`` stay longer with a connected-cluster search, lower :attr:`~SearchTuning.connected_cluster_state_evaluations`, and to make it switch to the Brouwer-Zimmermann method sooner, raise that field.
       Raise :attr:`~SearchTuning.auto_free_work` to decide later, with better estimates of growth, or lower it to decide sooner (`Choosing a method <https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/codes/distance/problems.md#choosing-a-method>`__ in ``problems.md`` of the distance package).
       Before it decides, ``"auto"`` runs a short Brouwer-Zimmermann search, its probe.
       Raise :attr:`~SearchTuning.auto_probe_work` so that the probe finishes more often, which gives the answer.
       A probe that does not finish still caps the connected-cluster search at the probe's upper bound on the distance.
       A longer probe can lower that bound, but costs more.
       Raise :attr:`~SearchTuning.auto_brute_force_max_words` if ``"auto"`` takes the Brouwer-Zimmermann method for a classical code of high relative distance whose words it could enumerate faster.
       Lower it if ``"auto"`` enumerates every word for too long.
   * - A hard distance search spends its time in its last, costliest layers
     - Raise :attr:`~SearchTuning.completion_bound_max_cost_percent` or lower :attr:`~SearchTuning.completion_bound_minimum_layer_work` to use completion tables in more layers.
       For a basis too large for its tables, raise :attr:`~SearchTuning.completion_bound_max_table_bytes` or lower :attr:`~SearchTuning.completion_bound_block_bits`.
       Also raise :attr:`~SearchTuning.max_information_sets`, or lower :attr:`~SearchTuning.automorphism_minimum_layer_work`.
   * - A cheap search spends its time building families of information sets
     - Raise :attr:`~SearchTuning.overlapping_minimum_work` or :attr:`~SearchTuning.partition_work_ratio`, or lower :attr:`~SearchTuning.max_information_sets`.
   * - Queries of :class:`~qldpc.codes.distance.CosetLeaderSearch` by the Brouwer-Zimmermann method at moderate maximum weights are slow
     - Lower :attr:`~SearchTuning.coset_route_minimum_work`, so that cheaper queries try a maximum disjoint family and an overlapping family (:guide:`Heuristics and cost models <codes/distance/brouwer_zimmermann_code.md#sec-heuristics>` of the Brouwer-Zimmermann guide).
       In searches of :class:`~qldpc.codes.distance.CosetLeaderSearch` by physical weight, and by CSS weight on CSS states, lowering :attr:`~SearchTuning.coset_route_minimum_work` also lets more queries use completion tables, which skip groups of words that are all too heavy.
       For larger overlapping families, raise :attr:`~SearchTuning.max_information_sets` or :attr:`~SearchTuning.overlapping_attempts_per_set`.
   * - The costliest queries of :class:`~qldpc.codes.distance.CosetLeaderSearch` by the Brouwer-Zimmermann method spend their time building completion tables
     - Lower :attr:`~SearchTuning.coset_completion_block_bits` or :attr:`~SearchTuning.coset_completion_max_table_bytes`, or set the latter to zero to build none.
   * - A search or count uses too much memory
     - For distance searches and counts, lower :attr:`~SearchTuning.completion_bound_max_table_bytes`.
       For exhaustive enumeration, lower :attr:`~SearchTuning.enumeration_batch_exponent`.
       For the Brouwer-Zimmermann method of :class:`~qldpc.codes.distance.CosetLeaderSearch` by CSS weight, on a stabilizer state that is not CSS, lower :attr:`~SearchTuning.max_css_single_pairs`.
       For the completion tables of its other searches, by physical weight or on a CSS state, lower :attr:`~SearchTuning.coset_completion_block_bits` or :attr:`~SearchTuning.coset_completion_max_table_bytes`.
       The latter applies to each set of each family that has tables: the disjoint family, and the maximum disjoint and overlapping families that :guide:`Heuristics and cost models <codes/distance/brouwer_zimmermann_code.md#sec-heuristics>` describes.
       For every search of :class:`~qldpc.codes.distance.CosetLeaderSearch` by the Brouwer-Zimmermann method, raise :attr:`~SearchTuning.coset_route_minimum_work`, so that a search builds its extra families later or not at all, and, as a rule, in its searches by physical weight or on a CSS state, fewer queries build completion tables.
       For caches, edit the constants that size them.
       No value limits the memory of a whole process.
   * - A search, count, or certification uses too many or too few cores, as when several processes run searches at once
     - This applies to distance searches by the Brouwer-Zimmermann and connected-cluster methods, batch queries of :class:`~qldpc.codes.distance.CosetLeaderSearch`, and certifications of bounded fault propagation (:func:`~qldpc.circuits.certification.fault_propagation.certify_bounded_fault_propagation`).
       It applies to a count of logical operators by ``method="brouwer_zimmermann"`` or ``"auto"`` only when the count is given no ``weight``, since the count may then first search for the distance.
       By default, a search or certification uses ``QLDPC_NUM_THREADS`` threads if that environment variable was set when qLDPC was imported, or else ``numba.config.NUMBA_NUM_THREADS``.
       That is one for each CPU that numba counts, unless the environment variable ``NUMBA_NUM_THREADS`` sets another number.
       To change it, set ``QLDPC_NUM_THREADS`` before Python starts, or set :attr:`~SearchTuning.num_threads`.
       A distance search or a certification reads ``num_threads`` during the call that runs it, and a :class:`~qldpc.codes.distance.CosetLeaderSearch` reads it when it is built.
       Functions that run coset-leader searches, such as :func:`~qldpc.circuits.certification.fault_propagation.certify_bounded_fault_propagation` and ``estimate_residual_weight_histogram``, build them during the call, so calling such a function within ``use_tuning(num_threads=1)`` keeps its searches on one thread.
       When several processes or threads run searches at once, as in a pool of workers, set ``QLDPC_NUM_THREADS=1`` before Python starts, which reaches every thread and worker process.
       Without that variable, enter ``use_tuning(num_threads=1)`` in the function that each worker runs, since new threads and spawned processes normally start with the default tuning.
       A ``CosetLeaderSearch`` keeps the thread count of the tuning that was active when it was built, so build one that the workers share within ``use_tuning(num_threads=1)`` as well.
       Numba does not read the CPU quotas of containers, so in a container, set ``QLDPC_NUM_THREADS`` to the container's CPU quota rounded down, or to 1 if the quota is below 1.
       The section `Threads <https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/codes/distance/README.md#threads>`__ of the guide to the distance package says when ``QLDPC_NUM_THREADS`` also sets numba's own thread count, and which other variables limit BLAS threads.
   * - A keyboard interrupt takes long to take effect
     - Lower :attr:`~SearchTuning.enumeration_batch_exponent`, :attr:`~SearchTuning.search_chunk_exponent`, :attr:`~SearchTuning.scan_chunk_size`, :attr:`~SearchTuning.scan_chunks_per_call`, or :attr:`~SearchTuning.macwilliams_chunk_size`.
       Lowering :attr:`~SearchTuning.search_chunk_exponent` also lowers the amounts of work at which searches and batch queries move to threads (see its description below).
       A Brouwer-Zimmermann distance search on several threads scans chunks :attr:`~SearchTuning.num_threads` times as large, so the wait grows with the number of threads.
       A connected-cluster distance search runs chunks of at most ``2**search_chunk_exponent`` states on each thread, and takes an interrupt once the running chunks end.
       Before a weight limit splits across threads, the calling thread divides the work of the limit into items that the threads then claim.
       It does so in passes, and each pass is one compiled call that is not split into chunks, so an interrupt waits for the current pass (:guide:`Section 3.6 <codes/distance/connected_cluster.md#sec-search-to-distance>` of ``connected_cluster.md``).
       A connected-cluster coset query takes an interrupt only after the query ends.
       On one thread (``num_threads=1``, or ``QLDPC_NUM_THREADS=1``), a batch query of :class:`~qldpc.codes.distance.CosetLeaderSearch` by the Brouwer-Zimmermann method takes an interrupt only after one compiled call over the whole batch ends.
       Once a batch query by either method has moved to threads, the query takes an interrupt only when each thread has ended its current block of queries, and a smaller batch has smaller blocks.
       So, on one thread or several, split a long batch into smaller ones.
   * - A cheap connected-cluster search spends its time finding automorphisms
     - Raise :attr:`~SearchTuning.automorphism_minimum_states`.
       Searches ask for them before a first limit that ``cutoff`` sets, unless the field is ``MAX_VALUE``.
   * - Many cheap distance searches of large CSS codes spend their time looking for a qubit permutation that maps the X sector onto the Z sector
     - Set :attr:`~SearchTuning.sector_maps` to ``False``.
   * - A program that makes very many counts of similar codes, such as of many deformed codes, is slow
     - Call :func:`~qldpc.codes.distance.compile_search_setup` once, at the start of the program (see its docstring).
   * - A :class:`~qldpc.codes.distance.CosetLeaderSearch` with the connected-cluster method takes long to build, or its queries are slow
     - Lower :attr:`~SearchTuning.max_coset_sparsifying_rounds`, or :attr:`~SearchTuning.sparsifying_rounds_without_improvement` below it, for a quicker build; raise :attr:`~SearchTuning.max_coset_sparsifying_rounds` for sparser checks and faster queries.
   * - A connected-cluster distance search of a subsystem code is slow to start, or slow to search
     - Lower :attr:`~SearchTuning.max_distance_sparsifying_rounds`, or :attr:`~SearchTuning.sparsifying_rounds_without_improvement` below it, to start sooner; raise :attr:`~SearchTuning.max_distance_sparsifying_rounds` for sparser checks and a faster search.
   * - Completion tables are used where they cost more than they save, or declined where they would save
     - Lower :attr:`~SearchTuning.completion_bound_max_cost_percent` to use them in fewer layers; to use them in more, see the row on hard distance searches above.

.. _tuning-routes:

Routes and the names that control them
--------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 45 15 15

   * - Route, or where time goes
     - Fields of :class:`SearchTuning`
     - Constants
     - Guide
   * - Exhaustive enumeration, which ``method="brute_force"`` runs, and ``"auto"`` when it is cheap
     - :attr:`SearchTuning.enumeration_batch_exponent`
     -
     - :guide:`Work estimates and exhaustive enumeration <codes/distance/brouwer_zimmermann.md#sec-work-estimates>`
   * - Searching one sector of a CSS code whose sectors a qubit permutation maps onto each other
     - :attr:`SearchTuning.sector_maps`
     -
     - `Searching one sector of a CSS code <https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/codes/distance/problems.md#searching-one-sector-of-a-css-code>`__ in ``problems.md`` of the distance package
   * - Choosing a method, which ``method="auto"`` does for each distance search
     - :attr:`SearchTuning.auto_brute_force_max_words`, :attr:`SearchTuning.auto_free_work`, :attr:`SearchTuning.auto_probe_work`, :attr:`SearchTuning.connected_cluster_state_evaluations`
     -
     - `Choosing a method <https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/codes/distance/problems.md#choosing-a-method>`__ in ``problems.md`` of the distance package
   * - Estimating the work of searches of information sets, which chooses between their routes
     -
     - :data:`BINOMIAL_CACHE_SIZE`
     - :guide:`Work estimates and exhaustive enumeration <codes/distance/brouwer_zimmermann.md#sec-work-estimates>`
   * - Packing the columns into disjoint information sets
     - :attr:`SearchTuning.disjoint_random_column_orders`
     -
     - :guide:`Heuristics and cost models <codes/distance/brouwer_zimmermann_code.md#sec-heuristics>`
   * - Building families of overlapping information sets
     - :attr:`SearchTuning.max_information_sets`, :attr:`SearchTuning.overlapping_minimum_work`, :attr:`SearchTuning.overlapping_attempts_per_set`
     -
     - :guide:`Overlapping search bases <codes/distance/brouwer_zimmermann.md#sec-overlapping-bases>`
   * - Building families by matroid partitioning
     - :attr:`SearchTuning.max_information_sets`, :attr:`SearchTuning.partition_work_ratio`, :attr:`SearchTuning.partition_max_coverage`
     -
     - :guide:`Heuristics and cost models <codes/distance/brouwer_zimmermann_code.md#sec-heuristics>`
   * - Searching the images of one information set under permutation automorphisms
     - :attr:`SearchTuning.automorphism_minimum_layer_work`
     -
     - :guide:`Images of an information set under automorphisms <codes/distance/brouwer_zimmermann.md#sec-automorphisms>`
   * - Searching from one root for each orbit of the permutation automorphisms of a code, in connected-cluster searches
     - :attr:`SearchTuning.automorphism_minimum_states`
     -
     - :guide:`Section 3.5 <codes/distance/connected_cluster.md#sec-symmetries>`
   * - Restricting the subtrees of a connected-cluster search to viable symbols, in distance searches and in the coset searches of :class:`~qldpc.codes.distance.CosetLeaderSearch`
     - :attr:`SearchTuning.restrict_viable_symbols`
     -
     - :guide:`Section 3.4 <codes/distance/connected_cluster.md#sec-pruning>`
   * - Forbidding symbols that would make a support meet a known word, such as a stabilizer or gauge generator, in more than about half of it, in connected-cluster distance searches and in the coset searches of :class:`~qldpc.codes.distance.CosetLeaderSearch`
     - :attr:`SearchTuning.stabilizer_overlap_cuts`
     -
     - :guide:`Section 3.4 <codes/distance/connected_cluster.md#sec-pruning>`
   * - Adding light sums of pairs of checks, and of pairs of known words, as redundant checks and known words, in connected-cluster distance searches
     - :attr:`SearchTuning.pair_sum_checks`
     -
     - :guide:`Section 3.4 <codes/distance/connected_cluster.md#sec-pruning>`
   * - Making the checks of the connected-cluster coset searches of :class:`~qldpc.codes.distance.CosetLeaderSearch` sparser
     - :attr:`SearchTuning.max_coset_sparsifying_rounds`, :attr:`SearchTuning.sparsifying_rounds_without_improvement`
     -
     - :guide:`Section 3.8 <codes/distance/connected_cluster.md#sec-syndromes>`
   * - Making the stabilizers that connected-cluster distance searches derive for subsystem codes sparser
     - :attr:`SearchTuning.max_distance_sparsifying_rounds`, :attr:`SearchTuning.sparsifying_rounds_without_improvement`
     -
     - :guide:`Section 3.9 <codes/distance/connected_cluster.md#sec-sparser-stabilizers>`
   * - Whether a search basis in a layer of a distance search, or a step of a count, uses completion tables to skip hopeless supports
     - :attr:`SearchTuning.completion_bound_minimum_layer_work`, :attr:`SearchTuning.completion_bound_max_cost_percent`
     -
     - :guide:`Skipping completions by completion tables <codes/distance/brouwer_zimmermann.md#sec-completion-tables>`
   * - The memory and the blocks of completion tables
     - :attr:`SearchTuning.completion_bound_max_table_bytes`, :attr:`SearchTuning.completion_bound_block_bits`, and in :class:`~qldpc.codes.distance.CosetLeaderSearch` :attr:`SearchTuning.coset_completion_max_table_bytes` and :attr:`SearchTuning.coset_completion_block_bits`
     -
     - :guide:`Skipping completions by completion tables <codes/distance/brouwer_zimmermann.md#sec-completion-tables>`
   * - Estimating how much completion tables would skip
     - :attr:`SearchTuning.completion_bound_samples`, :attr:`SearchTuning.completion_bound_min_samples`, :attr:`SearchTuning.completion_bound_sampling_work`, :attr:`SearchTuning.completion_bound_confidence_z`
     -
     - :guide:`Heuristics and cost models <codes/distance/brouwer_zimmermann_code.md#sec-heuristics>`
   * - The cost model of completion tables
     - :attr:`SearchTuning.completion_bound_lookup_block_cost`, :attr:`SearchTuning.completion_bound_lookup_block_cost_per_mib`, :attr:`SearchTuning.completion_bound_kept_first_row_ratio`, :attr:`SearchTuning.completion_bound_table_byte_cost`, :attr:`SearchTuning.completion_bound_build_step_cost`
     -
     - :guide:`Heuristics and cost models <codes/distance/brouwer_zimmermann_code.md#sec-heuristics>`
   * - Distance searches by the Brouwer-Zimmermann or connected-cluster method: time before a keyboard interrupt takes effect, and the overhead of returning to Python
     - :attr:`SearchTuning.search_chunk_exponent`
     -
     - :guide:`Behavior spanning files <codes/distance/brouwer_zimmermann_code.md#sec-cross-file-behavior>` of ``brouwer_zimmermann_code.md``, and :guide:`Section 3.6 <codes/distance/connected_cluster.md#sec-search-to-distance>` of ``connected_cluster.md``
   * - Distance searches by the Brouwer-Zimmermann method: the threads that scan a costly layer of a set
     - :attr:`SearchTuning.num_threads`, :attr:`SearchTuning.search_chunk_exponent`
     -
     - :guide:`Behavior spanning files <codes/distance/brouwer_zimmermann_code.md#sec-cross-file-behavior>`
   * - Batch queries of :class:`~qldpc.codes.distance.CosetLeaderSearch`: when a costly batch moves to threads, and how many
     - :attr:`SearchTuning.num_threads`, :attr:`SearchTuning.search_chunk_exponent`
     -
     - :guide:`Behavior spanning files <codes/distance/brouwer_zimmermann_code.md#sec-cross-file-behavior>` of ``brouwer_zimmermann_code.md``, and :guide:`Section 3.11 <codes/distance/connected_cluster.md#sec-threads>` of ``connected_cluster.md`` ("Syndrome queries")
   * - Distance searches by the connected-cluster method: the threads that search a costly weight limit
     - :attr:`SearchTuning.num_threads`, :attr:`SearchTuning.search_chunk_exponent`, :attr:`SearchTuning.connected_cluster_split_shift`
     -
     - :guide:`Section 3.11 <codes/distance/connected_cluster.md#sec-threads>`
   * - Certifications of bounded fault propagation: when a costly fault listing moves to threads, and how many
     - :attr:`SearchTuning.num_threads`
     -
     - :guide:`Threads <circuits/certification/fault_propagation.md#sec-threads>` of ``fault_propagation.md``
   * - Counts by the Brouwer-Zimmermann or brute-force method: time before a keyboard interrupt takes effect
     - :attr:`SearchTuning.scan_chunk_size`, :attr:`SearchTuning.scan_chunks_per_call`
     -
     -
   * - Counts by the MacWilliams method
     - :attr:`SearchTuning.macwilliams_shared_walk_rank`, :attr:`SearchTuning.macwilliams_chunk_size`
     - :data:`KRAWTCHOUK_CACHE_SIZE`
     - `The MacWilliams method <https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/codes/distance/problems.md#the-macwilliams-method>`__ in ``problems.md`` of the distance package
   * - Searches of the lightest Pauli errors by CSS weight on a CSS state, by the Brouwer-Zimmermann method
     - :attr:`SearchTuning.disjoint_random_column_orders`, and for costly queries :attr:`SearchTuning.coset_route_minimum_work`, :attr:`SearchTuning.max_information_sets`, :attr:`SearchTuning.overlapping_attempts_per_set`, :attr:`SearchTuning.coset_completion_block_bits`, and :attr:`SearchTuning.coset_completion_max_table_bytes`
     - :data:`CSS_SEARCH_CACHE_SIZE`
     - :guide:`Heuristics and cost models <codes/distance/brouwer_zimmermann_code.md#sec-heuristics>`
   * - Searches of the lightest Pauli errors by physical weight, or by CSS weight on another stabilizer state, by the Brouwer-Zimmermann method
     - :attr:`SearchTuning.disjoint_random_column_orders`, by CSS weight :attr:`SearchTuning.max_css_single_pairs`, and for costly queries :attr:`SearchTuning.coset_route_minimum_work`, :attr:`SearchTuning.max_information_sets`, :attr:`SearchTuning.overlapping_attempts_per_set`, and by physical weight :attr:`SearchTuning.coset_completion_block_bits` and :attr:`SearchTuning.coset_completion_max_table_bytes`
     - :data:`PAULI_SEARCH_CACHE_SIZE`
     - :guide:`Heuristics and cost models <codes/distance/brouwer_zimmermann_code.md#sec-heuristics>`
   * - Counts of deformed codes
     -
     - :data:`LOCAL_CLIFFORD_CACHE_SIZE`
     - :guide:`Deformed counters <codes/distance/brouwer_zimmermann.md#sec-deformed-counters>` in "Counting logical operators" of ``brouwer_zimmermann.md``

.. _tuning-module:

The tuning module
-----------------

.. automodule:: qldpc.codes.distance._tuning
   :members:
   :member-order: bysource
