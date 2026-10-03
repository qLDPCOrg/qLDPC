Library map
===========

This page explains how qLDPC represents codes, what each package does, and how common tasks fit together.
It fills the gap between the :doc:`example notebooks <examples/index>` and the :doc:`API reference <autoapi/index>`.

Choose the right documentation
------------------------------

* Use this page to understand how the library fits together and which part owns a task.
* Use the :doc:`examples <examples/index>` for complete workflows that can be executed and adapted.
* Use the :doc:`API reference <autoapi/index>` for exact signatures, details about each class and function, and construction-specific literature.
* Use the `agent and contributor guide <https://github.com/qLDPCOrg/qLDPC/blob/main/AGENTS.md>`_ when changing the repository.

Public APIs and experimental features
-------------------------------------

The top-level ``qldpc`` package contains subpackages such as ``qldpc.codes``, ``qldpc.decoders``, and ``qldpc.circuits``.
Their public classes and functions are available directly from those subpackages; for example, construct a surface code as ``qldpc.codes.SurfaceCode(...)`` rather than importing its defining module.

Imports from these ordinary subpackages are public API and should remain compatible.
Everything under ``qldpc.experimental`` is different: it is under active development and can change without notice or deprecation.
Pin the qLDPC version and validate results independently before depending on an experimental feature.

Core representations
--------------------

Finite fields
~~~~~~~~~~~~~

Code matrices are ``galois.FieldArray`` objects.
Constructors use :math:`GF(2)` by default and accept either a field order or a ``galois`` field class where nonbinary behavior is supported.
Extension-field values are field elements, not ordinary integers modulo the field order; preserve the array's field type when transforming a matrix.

Classical codes
~~~~~~~~~~~~~~~

A ``qldpc.codes.ClassicalCode`` is defined by a parity-check matrix :math:`H`.
A word :math:`x` is in the code exactly when :math:`Hx = 0`.
Built-in classical families and custom matrices share this representation, so generators, Tanner graphs, distances, puncturing, shortening, and code products use the same core interface.

Quantum and subsystem codes
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ``qldpc.codes.QuditCode`` stores each Pauli check as a symplectic row :math:`[P_X\mid P_Z]`.
For a stabilizer code these rows commute and generate the stabilizer group.
For a subsystem code they can fail to commute and instead generate the gauge group; stabilizers are derived from its center.

A ``qldpc.codes.CSSCode`` keeps separate matrices :math:`H_X` and :math:`H_Z`.
When :math:`H_X H_Z^T = 0`, their rows define commuting X- and Z-type stabilizers.
A nonzero product is valid only when the object is intentionally a subsystem code.

Some built-in families carry known parameters from their construction or the literature.
Some methods also accept explicit promises, such as equal X and Z distances.
These values and promises are trusted rather than recomputed automatically; exact distance calculations can be exponential.

Exact distances
~~~~~~~~~~~~~~~

Binary exact-distance calculations use Brouwer--Zimmermann enumeration by default.
Pass ``method="brute_force"`` to enumerate every generator combination instead.
Both methods return once an observed upper bound is at most ``cutoff``; the default cutoff of one gives an exact result for valid codes.

Classical distance is the minimum Hamming weight of a nonzero codeword.
CSS codes support separate X and Z distances as well as their minimum.
Stabilizer and subsystem codes use the minimum Pauli weight of a nontrivial logical operator.
The implementations follow `Algorithm 994 <https://arxiv.org/abs/1603.06757>`_ and its `quantum-code adaptation <https://arxiv.org/abs/2408.10743>`_.

Distance bounds
~~~~~~~~~~~~~~~

``CSSCode.get_distance_bound`` accepts a typed ``backend`` selector.
Set ``backend="gap"`` to use GAP/QDistRnd, ``backend="sqetch"`` to use the optional GPU-accelerated random-ISD estimator for binary CSS codes, or ``backend="decoder"`` to use qLDPC's decoder-based estimator.
By default, ``backend="auto"`` chooses the first applicable backend in this order: an installed ``sqetch`` for binary CSS codes, available GAP/QDistRnd, then the decoder.
``sqetch`` is not published on PyPI; install the version that qLDPC is tested against with:

.. code-block:: bash

   python -m pip install 'sqetch[gpu] @ git+https://github.com/a7b/yarn.git@e9ce9d0fcecc973988558bfee27fab6d6b8d7f97#subdirectory=sqetch'

The backend requires a CUDA-enabled PyTorch installation and a visible CUDA GPU.

``sqetch`` estimates the X- and Z-distance sectors separately.
The lowest logical weight it observes is an upper bound on the corresponding distance.

Codes remember the best upper bound on distance that they have found or been given, such as the witness-certified bounds loaded by ``QuditCode.from_qldpc_challenge_id``.
CSS codes remember X and Z bounds separately.
With the default ``num_trials=None``, ``get_distance_bound`` returns the exact distance or best known bound if there is one, and otherwise computes a single bound.
An explicit ``num_trials`` always runs that many trials and never returns a worse bound than the best known one.

Graphs, complexes, and lifted matrices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Code matrices can be converted to Tanner graphs whose nodes distinguish data and checks.
The ``qldpc.objects`` module also provides Pauli labels, Cayley complexes, and chain complexes used by geometric and product constructions.

The ``qldpc.abstract`` package provides finite groups, group rings, semisimple-ring linear algebra, and ``RingArray``.
A ``RingArray`` is a matrix over a group algebra whose entries can be lifted to blocks over a finite field; lifted-product and related code families build on this representation.

What each package does
----------------------

.. list-table::
   :header-rows: 1
   :widths: 18 39 43

   * - Package
     - What it does
     - Start here
   * - ``qldpc.codes``
     - Core classical, quantum, CSS, and subsystem code models; built-in constructions; distance and code-capacity estimation.
       Fixed-weight statistics and allocation live separately from code-capacity detector-error-model and decoder orchestration.
     - :doc:`qLDPC basics <examples/basics>` and :doc:`codes API <autoapi/qldpc/codes/index>`.
   * - ``qldpc.decoders``
     - Decoder protocol and adapters; lookup, ILP, BP-family, Frontier, MWPM, Relay-BP, detector-error-model, Sinter, and windowed decoding support.
     - :doc:`logical-error-rate examples <examples/index>` and :doc:`decoders API <autoapi/qldpc/decoders/index>`.
   * - ``qldpc.circuits``
     - Qubit-only Stim circuits, memory experiments, state-preparation diagnostics, encoders, noise models, scheduling, and transversal operations.
     - :doc:`noise models <examples/noise_models>`, :doc:`transversal gates <examples/transversal_gates>`, and :doc:`circuits API <autoapi/qldpc/circuits/index>`.
   * - ``qldpc.abstract``
     - Groups, group rings, lifted matrices, Howell-form linear algebra, and Wedderburn--Artin tools.
     - :doc:`abstract API <autoapi/qldpc/abstract/index>`.
   * - ``qldpc.math`` and ``qldpc.objects``
     - Finite-field and symplectic helpers, Pauli labels, graph nodes, Cayley complexes, and chain complexes.
     - :doc:`math API <autoapi/qldpc/math/index>` and :doc:`objects API <autoapi/qldpc/objects/index>`.
   * - ``qldpc.external`` and ``qldpc.cache``
     - Optional GAP, GUAVA, QDistRnd, GroupNames, and code-database integration, plus persistent caches for expensive results.
     - :doc:`external API <autoapi/qldpc/external/index>` and :doc:`cache API <autoapi/qldpc/cache/index>`.
   * - ``qldpc.experimental``
     - Unstable constructions under active development, currently including lattice-surgery tools.
     - :doc:`lattice surgery <examples/lattice_surgery>` and :doc:`experimental API <autoapi/qldpc/experimental/index>`.

How the packages work together
------------------------------

The packages call one another rather than following a strict one-way hierarchy.
For example, code-level estimators use decoders, decoder adapters understand code and detector-error-model objects, and circuit workflows combine both.

.. code-block:: text

   math / abstract / objects
              |
              v
            codes <----------> decoders
              \                  /
               \                /
                    circuits
                       |
                Stim circuits and
             detector error models

   external integrations and disk caches support expensive algebra,
   construction, and distance workflows across these packages.

Common workflows
----------------

Construct and inspect a code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Built-in and custom codes share the same basic interface:

.. code-block:: python

   from qldpc import codes

   code = codes.SurfaceCode(5)
   print(code.get_code_params())
   logical_ops = code.get_logical_ops()

Use ``code.matrix`` for a classical or general quantum check matrix, and ``code.matrix_x`` / ``code.matrix_z`` for the two CSS sectors.
Use ``forget_distance()`` deliberately when a cached or construction-supplied distance (or distance bound) should be discarded before recomputation.

Build a custom CSS code
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np

   from qldpc import codes

   matrix_x = np.array([[1, 1, 0]])
   matrix_z = np.array([[0, 0, 1]])
   code = codes.CSSCode(matrix_x, matrix_z)

For a stabilizer CSS code, verify ``matrix_x @ matrix_z.T == 0`` over the intended field.
If the checks do not commute, make sure you intend to build a subsystem code and test that behavior.

Choose or supply a decoder
~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a decoder specification with a typed helper such as ``decoders.bp_lsd(...)`` or ``decoders.mwpm(...)``, and build a decoder for a parity-check matrix or Stim detector error model with ``.build(...)``, or pass the specification as ``decoder=`` to a method that decodes.
Some specifications, such as ``decoders.frontier(...)``, predict observable flips but cannot infer errors, so they are accepted only where observable flips are wanted.
The default decoder is GUF for a nonbinary field array and BP+OSD otherwise.
``.build_observable_decoder(dem)`` builds a decoder that predicts the observable flips of a detector error model instead, and ``qldpc.decoders.SinterDecoder`` does so for Sinter.
Code-capacity estimators accept either kind of decoder.
See :doc:`Choosing a decoder <decoders>` for the difference between error and observable decoders, custom decoders, and per-sector CSS choices.

.. code-block:: python

   import numpy as np

   from qldpc import codes, decoders

   code = codes.RepetitionCode(5)
   syndrome = np.array([1, 0, 0, 0])
   decoder = decoders.bp_osd().build(code.matrix)
   correction = decoder.decode_errors(syndrome)

Only some decoders can signal an erasure.
Those decoders append the erasure flag as the last entry of each inferred error.

Build and simulate a memory circuit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from qldpc import circuits, codes
   from qldpc.objects import Pauli

   code = codes.SurfaceCode(3)
   circuit = circuits.get_memory_experiment(code, basis=Pauli.Z, num_rounds=3)

Circuit utilities reject nonbinary codes.
Encoding circuits are not fault-tolerant, and transversal-gate searches can be exponential.
For sliding-window decoding, detector coordinates are only a heuristic source of time; pass ``detector_to_time`` when a model uses another convention.

Build a lifted construction
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Lifted-product and related families begin with groups and ``RingArray`` objects from ``qldpc.abstract``.
Keep every array in one base ring, lift only when converting to finite-field matrices, and use the construction's docstring and tests for its precise transpose and orientation conventions.
The :doc:`API reference <autoapi/index>` carries the literature links for each family.

Optional integrations and side effects
--------------------------------------

* Install ``qldpc[gap]`` for the optional ``passagemath-gap`` libgap binding.
  The binding requires non-Windows Python 3.13.
  If neither option is available, some paths offer a manual copy/paste workflow that reads standard input and uses the system clipboard.
* When a GAP package is unavailable to libgap, qLDPC asks for permission to install it with GAP's PackageManager so that the active libgap runtime can discover it.
  If that attempt fails and a separate GAP executable is available, qLDPC prints manual installation instructions before using the executable instead.
  Packages missing from the executable backend can trigger an installation prompt and ``git clone`` into qLDPC's GAP package directory.
* Group and code lookups may use online resources when local data and GAP cannot answer the request.
* Expensive results can be cached under the user's platform cache directory.
  A cached function ``func`` also has a ``func.refresh(...)`` method, which takes the same arguments, recomputes the result, and overwrites the cached value.
  Disk caching is intentionally bypassed while pytest is running.
* The test configuration disables network sockets.
  Tests for optional integrations use mocks rather than live services.

Where to go next
----------------

* Follow the :doc:`examples <examples/index>` for runnable workflows.
* Browse the :doc:`API reference <autoapi/index>` for exact behavior and literature.
* Read `AGENTS.md <https://github.com/qLDPCOrg/qLDPC/blob/main/AGENTS.md>`_ before modifying the implementation.
