Library map
===========

This page explains how qLDPC represents codes, what each package does, and how the packages work together.

Overview and which page to read
-------------------------------

* Read this page to learn the data model, how the packages connect, and what qLDPC supports.
* Read :doc:`Choosing a decoder <decoders>` to select, build, and use decoders, including with Sinter.
* Follow the :doc:`examples <examples/index>` for complete workflows that you can run and adapt.
* Use the :doc:`API reference <autoapi/index>` for exact signatures, every public name, and construction-specific literature.
* Read the `agent and contributor guide <https://github.com/qLDPCOrg/qLDPC/blob/main/AGENTS.md>`_ before changing qLDPC itself.

Most user-facing names are exported from their subpackage, as in ``qldpc.codes.SurfaceCode``.
Concrete decoder backends live in ``qldpc.decoders.custom`` and ``qldpc.decoders.external``, and tool integrations live in ``qldpc.external.gap`` and ``qldpc.external.sqetch``.
Modules and names that start with an underscore are internal, and everything under ``qldpc.experimental`` can change without notice.

Packages at a glance
--------------------

.. list-table::
   :header-rows: 1
   :widths: 16 32 32 20

   * - Package
     - What it does
     - Entry points
     - Start here
   * - ``qldpc.codes``
     - Classical, quantum, CSS, and subsystem codes; built-in families; distances; code-capacity logical error rates.
     - ``ClassicalCode``, ``QuditCode``, ``CSSCode``, families such as ``SurfaceCode`` and ``BBCode``, and the methods ``get_code_params``, ``get_distance``, and ``get_logical_error_rate_func``.
     - :doc:`qLDPC basics <examples/basics>`
   * - ``qldpc.decoders``
     - Decoder specifications, error and observable decoders, detector-error-model arrays, and Sinter and windowed decoding.
     - Specifications such as ``bp_osd``, ``mwpm``, ``guf``, and ``tesseract``; their ``build`` and ``build_observable_decoder`` methods; ``SinterDecoder``.
     - :doc:`Choosing a decoder <decoders>` and the :doc:`decoders notebook <examples/decoders>`
   * - ``qldpc.circuits``
     - Qubit-only Stim circuits: memory experiments, noise models, encoders, state-preparation diagnostics, fault-tolerance certificates, and transversal gates.
     - ``get_memory_experiment``, ``DepolarizingNoiseModel``, ``SI1000NoiseModel``, ``get_encoding_circuit``, ``get_fault_distance``, ``certify_bounded_fault_propagation``, and ``get_transversal_ops``.
     - :doc:`Noise models <examples/noise_models>`, :doc:`fault-tolerance certificates <examples/logical_error_rates/5_fault_tolerance_certificates>`, and :doc:`transversal gates <examples/transversal_gates>`
   * - ``qldpc.abstract``
     - Finite groups, group rings, matrices over group algebras, and semisimple-ring linear algebra.
     - ``Group``, ``GroupRing``, and ``RingArray``.
     - :doc:`abstract API <autoapi/qldpc/abstract/index>`
   * - ``qldpc.objects`` and ``qldpc.math``
     - Pauli labels, Tanner-graph nodes, Cayley and chain complexes, and symplectic and finite-field array helpers.
     - ``Pauli``, ``CayleyComplex``, ``ChainComplex``, and ``symplectic_conjugate``.
     - :doc:`objects API <autoapi/qldpc/objects/index>` and :doc:`math API <autoapi/qldpc/math/index>`
   * - ``qldpc.external`` and ``qldpc.cache``
     - Access to GAP (with GUAVA and QDistRnd) and the optional ``sqetch`` distance estimator, plus persistent disk caches.
     - ``external.gap``, ``external.sqetch``, and ``cache.clear_entry``.
     - `External tools, side effects, and caching`_
   * - ``qldpc.experimental``
     - Unstable research code: lattice surgery, and decoder-free benchmarks of state-preparation circuits.
     - ``experimental.surgery`` and ``experimental.circuits.estimate_residual_weight_histogram``.
     - :doc:`Lattice surgery <examples/lattice_surgery>` and :doc:`benchmarking state preparation <examples/experimental/logical_state_preparation>`

The arrows below show which objects pass between packages.

.. code-block:: text

   qldpc.abstract, qldpc.objects  <-- groups --  qldpc.external
   Group, RingArray, CayleyComplex               GAP (GUAVA, QDistRnd), sqetch
                |                                          |
                | construction inputs                      | named codes,
                v                                          v distance bounds
   +---------------------------- qldpc.codes ----------------------------+
   |                 ClassicalCode, QuditCode, CSSCode                   |
   +---------------------------------------------------------------------+
                |                                          |
                | qubit QuditCode                          | parity-check matrices
                v                                          v
         qldpc.circuits                              qldpc.decoders
                |                                          ^
                | stim.Circuit --> stim.DetectorErrorModel |
                +------------------------------------------+

Code methods call decoders for code-capacity estimates and decoder-based distance bounds.
Circuit workflows pass detector error models to decoders directly, or through a ``SinterDecoder`` and Sinter.
``qldpc.cache`` stores the results of slow external lookups on disk.

Data model
----------

Fields
~~~~~~

Code matrices are ``galois.FieldArray`` objects.
Constructors use :math:`GF(2)` by default, and accept a field order or a ``galois`` field class as ``field=``.
Extension-field values are field elements, not integers modulo the field order, so preserve the array's field type when transforming a matrix.

Classical and quantum codes
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A :class:`~qldpc.codes.common.ClassicalCode` is defined by a parity-check matrix :math:`H`, and a word :math:`x` is in the code exactly when :math:`Hx = 0`.
A :class:`~qldpc.codes.common.QuditCode` stores each Pauli check as a symplectic row :math:`[P_X\mid P_Z]`.
A :class:`~qldpc.codes.common.CSSCode` stores X-type and Z-type checks as separate matrices :math:`H_X` and :math:`H_Z`, which commute when :math:`H_X H_Z^T = 0`.
Built-in families and codes built from custom matrices share these classes and their methods.

By default, qLDPC infers whether a code is a subsystem code from whether its checks commute.
If they do not, the checks generate a gauge group, and the stabilizers are derived from its center.
Check ``code.is_subsystem_code`` when you meant to build a stabilizer code.
The :doc:`basics notebook <examples/basics>` builds classical, CSS, and subsystem codes from matrices.

Known parameters and promises
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some built-in families and online lookups record parameters known from their construction or source, such as a distance.
Some constructor arguments are promises, such as ``CSSCode(..., promise_equal_distance_xz=True)``, which lets qLDPC compute the distance from one sector alone.
qLDPC trusts recorded parameters and promises rather than verifying them, so a false promise gives wrong distances.

Algebraic inputs of lifted and geometric families
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Lifted-product families such as :class:`~qldpc.codes.quantum.LPCode` take :class:`~qldpc.abstract.ring_array.RingArray` matrices, whose entries belong to a :class:`~qldpc.abstract.rings.GroupRing` over a finite field.
Each entry lifts to a block matrix over that field, and ``RingArray.lift()`` returns the resulting ``galois.FieldArray``.
Every ``RingArray`` has one base ring, and arithmetic between arrays over different rings raises an error.
:class:`~qldpc.codes.quantum.QTCode` builds a :class:`~qldpc.objects.CayleyComplex` from two subsets of a group, and families such as :class:`~qldpc.codes.quantum.T4Code` (built from a 4×4 integer lattice matrix) build a :class:`~qldpc.objects.ChainComplex` internally and expose it as ``code.chain``.
The API reference documents each family's conventions and literature.

Distances across packages
-------------------------

``get_distance()`` computes an exact distance, and ``get_distance_bound`` computes a randomized upper bound; CSS codes also have separate X and Z distances.
Exact distances of binary codes use an optimized `Brouwer--Zimmermann search <https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/codes/distance/brouwer_zimmermann.md>`_, and nonbinary codes use exhaustive enumeration; both can take exponential time.
``get_distance_exact(cutoff=c)`` stops once it finds a nontrivial word of weight at most ``c``.
A result of at most ``c`` is therefore only an upper bound, while a larger result, or any result with ``cutoff <= 1`` (the default is 1), is exact.
Distance-bound backends depend on the class: ``ClassicalCode`` uses a decoder, a non-CSS ``QuditCode`` uses GAP's QDistRnd, and ``CSSCode`` supports ``sqetch`` (binary only), GAP, and a decoder.
For ``CSSCode``, the default ``backend="auto"`` tries those three in order and uses GAP only if a GAP binding or executable can be called, so it never falls back to the manual copy-and-paste workflow.
It can still ask on standard input before installing a missing GUAVA or QDistRnd package.
Codes remember the best distance bound that they have computed or been given, and CSS codes remember X and Z bounds separately.
With ``num_trials=None``, ``get_distance_bound`` returns the exact distance or best known bound if there is one, and otherwise computes one bound.
An explicit ``num_trials`` runs that many trials, and never returns a worse bound than the best known one.
``forget_distance()`` discards remembered distances and bounds.
:class:`~qldpc.codes.distance.logical_counts.CSSLogicalCounter` and :func:`~qldpc.codes.distance.logical_counts.count_logical_ops_css` count the logical operators of a given weight in a binary CSS code.

From codes to logical error rates
---------------------------------

An error decoder infers an error from a syndrome with ``decode_errors``, while an observable decoder predicts observable flips with ``decode_observables``.
A decoder specification such as ``decoders.mwpm()`` builds an error decoder for a parity-check matrix with ``.build(matrix)``, or an observable decoder for a Stim detector error model with ``.build_observable_decoder(dem)``.
:doc:`Choosing a decoder <decoders>` explains both kinds, the available specifications, and custom decoders.

In the code-capacity model, each qudit fails independently and checks are measured perfectly.
:meth:`~qldpc.codes.common.CSSCode.get_logical_error_rate_func` returns a function from a physical error rate to an estimated logical error rate and its uncertainty:

.. code-block:: python

   from qldpc import codes, decoders

   code = codes.SurfaceCode(3)
   estimate = code.get_logical_error_rate_func(num_samples=1000, decoder=decoders.mwpm())
   logical_error_rate, uncertainty = estimate(0.01)

CSS codes also accept separate ``decoder_x=`` and ``decoder_z=`` specifications.
Without a specification, the default decoder is BP+OSD for binary codes and GUF otherwise.

For circuit-level noise, ``qldpc.circuits`` builds a memory experiment as a ``stim.Circuit``, with errors inserted by a noise model.
Stim converts the circuit into a ``stim.DetectorErrorModel`` with ``circuit.detector_error_model()``, and a :class:`~qldpc.decoders.sinter.core.SinterDecoder` builds an observable decoder for each detector error model that Sinter samples:

.. code-block:: python

   import sinter

   from qldpc import circuits, codes, decoders
   from qldpc.objects import Pauli

   code = codes.SurfaceCode(3)
   noise_model = circuits.DepolarizingNoiseModel(1e-3)
   circuit = circuits.get_memory_experiment(
       code, basis=Pauli.Z, num_rounds=3, noise_model=noise_model
   )
   sinter_decoder = decoders.SinterDecoder(decoder=decoders.mwpm())
   (stats,) = sinter.collect(
       tasks=[sinter.Task(circuit=circuit, decoder="mwpm")],
       custom_decoders={"mwpm": sinter_decoder},
       num_workers=1,
       max_shots=1000,
   )
   logical_error_rate = stats.errors / stats.shots

The :doc:`code-capacity <examples/logical_error_rates/1_code_capacity>`, :doc:`quantum-memory <examples/logical_error_rates/2_quantum_memory_x_or_z>`, and :doc:`sliding-window <examples/logical_error_rates/specialized/sliding_window_decoding>` notebooks develop these workflows, and the :doc:`noise-model notebook <examples/noise_models>` explains the noise models.

Certify fault tolerance
~~~~~~~~~~~~~~~~~~~~~~~

:func:`qldpc.circuits.certify_bounded_fault_propagation <qldpc.circuits.certification.fault_propagation.certify_bounded_fault_propagation>` checks whether a noisy Clifford circuit that prepares a stabilizer state has bounded fault propagation.
Bounded fault propagation is a sufficient criterion for fault-tolerant state preparation; see Definition 3 in Section 4.2 of `arXiv:0904.2557 <https://arxiv.org/abs/0904.2557>`_.
An error mechanism is an error instruction of the circuit's detector error model, and a fault is a set of error mechanisms.
A fault leaves a Pauli error on the prepared state.
The residual weight of the fault is the smallest weight of that Pauli error after multiplying it by any stabilizer of the state.
Fault propagation is bounded if the residual weight of every fault is at most its number of error mechanisms.
For circuits that post-select on detectors such as measurement flags, only faults that pass post-selection count.
The docstring of ``certify_bounded_fault_propagation`` defines these terms precisely and states the limitations of its check, and the :doc:`fault-tolerance certificates notebook <examples/logical_error_rates/5_fault_tolerance_certificates>` applies it to a flagged preparation circuit.
For annotated memory circuits, :func:`~qldpc.circuits.certification.fault_distance.get_fault_distance` instead computes the minimum number of detector-error-model mechanisms causing an undetected logical-observable flip.
The same notebook illustrates why this circuit property need not equal the code distance.

Bounded fault propagation is a yes-or-no criterion.
To compare preparation circuits, including those that satisfy it, the experimental :func:`qldpc.experimental.circuits.estimate_residual_weight_histogram <qldpc.experimental.circuits.benchmarking.residual_weights.estimate_residual_weight_histogram>` estimates, over the shots that pass post-selection, the probability distribution of the residual weight of the fault that occurs in a shot.
The residual weight needs no decoder, and it accounts for Pauli errors that a chosen logical readout cannot detect, such as Z errors on a logical :math:`|0\rangle` state.

Support and limitations
-----------------------

The following table summarizes support by field; the subsystem column applies to subsystem codes over any field.

===============  ===========================  ==============================  ===================================  ========================================
Capability       :math:`GF(2)`                Prime field :math:`GF(p)`       Extension field :math:`GF(p^m)`      Subsystem codes
===============  ===========================  ==============================  ===================================  ========================================
Codes            Yes                          Yes                             Yes                                  Yes, inferred from non-commuting checks
Exact distance   Brouwer--Zimmermann          Exhaustive enumeration          Exhaustive enumeration               Yes
Distance bounds  Decoder, GAP, or ``sqetch``  Decoder or GAP                  Decoder or GAP                       CSS codes only
Decoders         All built-in decoders        GUF (default), ILP, and lookup  GUF (default) and lookup             Yes, in code-capacity estimates
Circuits         Yes                          No                              No                                   No memory experiments
===============  ===========================  ==============================  ===================================  ========================================

Other limitations:

* Circuit and tableau tools work only with qubit codes, and memory experiments for non-CSS codes need ``basis=None``.
* :func:`~qldpc.circuits.encoding.get_encoding_circuit` builds a valid encoder that is not fault-tolerant.
* Transversal-gate searches enumerate code automorphisms, can take exponential time, and may run GAP or prompt on standard input.
* Windowed decoders such as :class:`~qldpc.decoders.sinter.window.SlidingWindowDecoder` infer each detector's time from its coordinates by a heuristic; pass ``detector_to_time`` when a detector error model uses another convention.

What qLDPC computes exactly, bounds, and trusts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Exact: ``get_distance_exact`` with ``cutoff <= 1`` (the default is 1), or whenever its result exceeds the cutoff, which ``get_distance`` calls unless it is given ``bound=``; logical-operator counts; circuit fault distances from ``get_fault_distance``; and bounded-propagation certificates up to their ``max_weight``, if one is given.
* Upper bounds: ``get_distance_bound`` with every backend.
* Estimates: code-capacity and circuit-level logical error rates are sampled.
  A code-capacity estimate treats unsampled heavy errors as failures, so it can be high by up to the amount that its ``truncation_error_bound`` method reports.
* Trusted, not verified: recorded parameters of built-in families and online lookups, promises such as ``promise_equal_distance_xz``, an explicit ``is_subsystem_code=True`` hint to ``QuditCode`` or ``CSSCode``, a code-capacity ``min_error_weight`` below which errors are assumed to be corrected, and an explicit ``detector_to_time`` mapping.
  An explicit ``is_subsystem_code=False`` is verified instead, and raises a ``ValueError`` if some checks do not commute.
  Omit ``is_subsystem_code`` to have it computed when first needed.

External tools, side effects, and caching
-----------------------------------------

GAP backends
~~~~~~~~~~~~

qLDPC calls GAP in process through the ``passagemath-gap`` libgap binding when it is installed, with ``pip install 'qldpc[gap]'`` on Python 3.13 outside Windows.
Otherwise, qLDPC runs a ``gap`` executable on ``PATH`` in a subprocess.
If neither backend is available, some functions offer a manual workflow that prints GAP commands, copies them to the system clipboard, and reads GAP's output from standard input.
When a needed GAP package such as GUAVA or QDistRnd is missing, qLDPC asks on standard input before installing it: with GAP's PackageManager for libgap, or by cloning it with Git into qLDPC's user cache directory for the executable.

Functions that can use GAP include :meth:`ClassicalCode.from_name <qldpc.codes.common.ClassicalCode.from_name>`, :meth:`ClassicalCode.get_automorphism_group <qldpc.codes.common.ClassicalCode.get_automorphism_group>`, :meth:`abstract.Group.from_name <qldpc.abstract.groups.Group.from_name>`, distance bounds with the GAP backend, and transversal-gate searches such as :func:`~qldpc.circuits.transversal.get_transversal_ops`.

Code and group lookups
~~~~~~~~~~~~~~~~~~~~~~

* :meth:`QuditCode.from_qecdb_id <qldpc.codes.common.QuditCode.from_qecdb_id>` and :meth:`QuditCode.from_qldpc_challenge_id <qldpc.codes.common.QuditCode.from_qldpc_challenge_id>` download code data on their first call for each ID, and later calls reuse the copy cached on disk.
* ``ClassicalCode.from_name`` runs GAP and GUAVA rather than an online lookup, and also caches its result on disk.
* ``Group.from_name`` tries a small local database, then GAP if a GAP binding or executable can be called (GAP may ask before installing a missing package), then GroupNames.org for a ``SmallGroup``, and finally the manual GAP workflow.

Pass ``refresh=True`` to ``ClassicalCode.from_name``, ``QuditCode.from_qecdb_id``, or ``QuditCode.from_qldpc_challenge_id`` to bypass and replace its cached result.

``sqetch``
~~~~~~~~~~

The optional ``sqetch`` distance-bound backend for binary CSS codes requires a CUDA GPU and is not published on PyPI.
See :ref:`GPU distance estimation <installation-sqetch>` for the tested installation command.

Disk caches
~~~~~~~~~~~

qLDPC caches the results of slow or interactive lookups, such as named codes, groups, downloaded code data, and pasted GAP output, in a ``qldpc`` directory under the user's platform cache directory.
Unless ``NUMBA_CACHE_DIR`` is set, code that numba compiles is cached in its ``numba/qldpc-<hash>`` subdirectory, where the hash is of qLDPC's source files.
Each import of qLDPC deletes the outdated directories of its own source location, compiled from earlier versions of its source files and unused for more than 7 days, and keeps those of other checkouts or installations.
A function decorated with :func:`qldpc.cache.use_disk_cache` has a ``refresh`` method, which takes the same arguments, recomputes the result, and overwrites the cached value.
:func:`qldpc.cache.clear_entry(cache_name, key) <qldpc.cache.clear_entry>` removes a single cached entry.
Any process that has imported ``pytest`` bypasses the result caches of ``use_disk_cache``, so every call recomputes its result and saves nothing; compiled code is still cached.
