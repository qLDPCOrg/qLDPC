Library map
===========

This page explains qLDPC's core representations, package responsibilities, and common workflows.
It is the conceptual layer between the :doc:`example notebooks <examples/index>` and the
:doc:`exhaustive API reference <autoapi/index>`.

Choose the right documentation layer
------------------------------------

* Use this page to understand how the library fits together and which boundary owns a task.
* Use the :doc:`examples <examples/index>` for complete workflows that can be executed and adapted.
* Use the :doc:`API reference <autoapi/index>` for exact signatures, per-object contracts, and
  construction-specific literature.
* Use the `agent and contributor guide
  <https://github.com/qLDPCOrg/qLDPC/blob/main/AGENTS.md>`_ when changing the repository.

Public surface and stability
----------------------------

The top-level ``qldpc`` package exposes subpackages such as ``qldpc.codes``, ``qldpc.decoders``, and
``qldpc.circuits``. Public classes and functions are re-exported from those subpackages; for
example, construct a surface code as ``qldpc.codes.SurfaceCode(...)`` rather than importing its
defining module directly.

The ordinary subpackage exports are compatibility-preserving public API. Everything under
``qldpc.experimental`` is different: it is under active development and can change without notice
or deprecation. Pin the qLDPC version and validate results independently before depending on an
experimental workflow.

Core representations
--------------------

Finite fields
~~~~~~~~~~~~~

Code matrices are ``galois.FieldArray`` objects. Constructors use :math:`GF(2)` by default and
accept either a field order or a ``galois`` field class where nonbinary behavior is supported.
Extension-field values are field elements, not ordinary integers modulo the field order; preserve
the array's field type when transforming a matrix.

Classical codes
~~~~~~~~~~~~~~~

A ``qldpc.codes.ClassicalCode`` is defined by a parity-check matrix :math:`H`. A word :math:`x` is
in the code exactly when :math:`Hx = 0`. Built-in classical families and custom matrices share this
representation, so generators, Tanner graphs, distances, puncturing, shortening, and code products
use the same core interface.

Quantum and subsystem codes
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ``qldpc.codes.QuditCode`` stores each Pauli check as a symplectic row
:math:`[P_X\mid P_Z]`. For a stabilizer code these rows commute and generate the stabilizer group.
For a subsystem code they can fail to commute and instead generate the gauge group; stabilizers are
derived from its center.

A ``qldpc.codes.CSSCode`` keeps separate matrices :math:`H_X` and :math:`H_Z`. When
:math:`H_X H_Z^T = 0`, their rows define commuting X- and Z-type stabilizers. A nonzero product is
valid only when the object is intentionally a subsystem code.

Some built-in families carry known parameters from their construction or the literature. Some
methods also accept explicit promises, such as equal X and Z distances. These values and promises
are trusted rather than recomputed automatically; exact distance calculations can be exponential.

Graphs, complexes, and lifted matrices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Code matrices can be converted to Tanner graphs whose nodes distinguish data and checks. The
``qldpc.objects`` module also provides Pauli labels, Cayley complexes, and chain complexes used by
geometric and product constructions.

The ``qldpc.abstract`` package provides finite groups, group rings, semisimple-ring linear algebra,
and ``RingArray``. A ``RingArray`` is a matrix over a group algebra whose entries can be lifted to
blocks over a finite field; lifted-product and related code families build on this representation.

Package responsibilities
------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 39 43

   * - Package
     - Responsibility
     - Start here
   * - ``qldpc.codes``
     - Core classical, quantum, CSS, and subsystem code models; built-in constructions; distance and
       code-capacity estimation.
     - :doc:`qLDPC basics <examples/basics>` and
       :doc:`codes API <autoapi/qldpc/codes/index>`.
   * - ``qldpc.decoders``
     - Decoder protocol and adapters; lookup, ILP, BP-family, MWPM, Relay-BP, detector-error-model,
       Sinter, and windowed decoding support.
     - :doc:`logical-error-rate examples <examples/index>` and
       :doc:`decoders API <autoapi/qldpc/decoders/index>`.
   * - ``qldpc.circuits``
     - Qubit-only Stim circuits, memory experiments, state-preparation diagnostics, encoders, noise
       models, scheduling, and transversal operations.
     - :doc:`noise models <examples/noise_models>`,
       :doc:`transversal gates <examples/transversal_gates>`, and
       :doc:`circuits API <autoapi/qldpc/circuits/index>`.
   * - ``qldpc.abstract``
     - Groups, group rings, lifted matrices, Howell-form linear algebra, and Wedderburn--Artin tools.
     - :doc:`abstract API <autoapi/qldpc/abstract/index>`.
   * - ``qldpc.math`` and ``qldpc.objects``
     - Finite-field and symplectic helpers, Pauli labels, graph nodes, Cayley complexes, and chain
       complexes.
     - :doc:`math API <autoapi/qldpc/math/index>` and
       :doc:`objects API <autoapi/qldpc/objects/index>`.
   * - ``qldpc.external`` and ``qldpc.cache``
     - Optional GAP, GUAVA, QDistRnd, GroupNames, and code-database integration, plus persistent
       caches for expensive results.
     - :doc:`external API <autoapi/qldpc/external/index>` and
       :doc:`cache API <autoapi/qldpc/cache/index>`.
   * - ``qldpc.experimental``
     - Unstable constructions under active development, currently including lattice-surgery tools.
     - :doc:`lattice surgery <examples/lattice_surgery>` and
       :doc:`experimental API <autoapi/qldpc/experimental/index>`.

How the packages collaborate
----------------------------

This is a responsibility and workflow map, not a strict dependency graph. Code-level estimators use
decoders, decoder adapters understand code and detector-error-model objects, and circuit workflows
combine both.

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

Use ``code.matrix`` for a classical or general quantum check matrix, and ``code.matrix_x`` /
``code.matrix_z`` for the two CSS sectors. Use ``forget_distance()`` deliberately when a cached or
construction-supplied distance should be discarded before recomputation.

Build a custom CSS code
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np

   from qldpc import codes

   matrix_x = np.array([[1, 1, 0]])
   matrix_z = np.array([[0, 0, 1]])
   code = codes.CSSCode(matrix_x, matrix_z)

For a stabilizer CSS code, verify ``matrix_x @ matrix_z.T == 0`` over the intended field. If the
checks do not commute, declare and test the intended subsystem semantics instead of treating the
matrix as an ordinary stabilizer code.

Choose or inject a decoder
~~~~~~~~~~~~~~~~~~~~~~~~~~

``qldpc.decoders.get_decoder`` accepts a parity-check matrix or Stim detector error model. It uses
GUF by default for a nonbinary field array and BP+OSD otherwise. Select one named decoder with its
``with_<NAME>`` option, pass a ``decoder_constructor``, or supply a ``static_decoder``.

.. code-block:: python

   import numpy as np

   from qldpc import codes, decoders

   code = codes.RepetitionCode(5)
   syndrome = np.array([1, 0, 0, 0])
   decoder = decoders.get_decoder(code.matrix)
   correction = decoder.decode(syndrome)

Erasure signaling is an explicit decoder capability. Decoders that support it append the erasure
flag as the last entry of each inferred error.

Build and simulate a memory circuit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from qldpc import circuits, codes
   from qldpc.objects import Pauli

   code = codes.SurfaceCode(3)
   circuit = circuits.get_memory_experiment(code, basis=Pauli.Z, num_rounds=3)

Circuit utilities reject nonbinary codes. Encoding circuits are not fault-tolerant, and
transversal-gate searches can be exponential. For sliding-window decoding, detector coordinates are
only a heuristic source of time; pass ``detector_to_time`` when a model uses another convention.

Build a lifted construction
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Lifted-product and related families begin with groups and ``RingArray`` objects from
``qldpc.abstract``. Keep every array in one base ring, lift only at the finite-field boundary, and
use the construction's docstring and tests for its precise transpose and orientation conventions.
The :doc:`API reference <autoapi/index>` carries the literature links for each family.

Optional integrations and execution boundaries
----------------------------------------------

* GAP-backed features can run subprocesses. If GAP is unavailable, some paths offer a manual
  copy/paste workflow that reads standard input and uses the system clipboard.
* Missing GAP packages can trigger an installation prompt and ``git clone`` into GAP's package
  directory.
* Group and code lookup can access external resources when local data and GAP do not answer the
  request.
* Expensive results can be cached under the user's platform cache directory. Disk caching is
  intentionally bypassed while pytest is running.
* The test configuration disables network sockets. Tests for optional integrations use controlled
  doubles rather than live services.

Where to go next
----------------

* Follow the :doc:`examples <examples/index>` for runnable workflows.
* Browse the :doc:`API reference <autoapi/index>` for exact contracts and literature.
* Read `AGENTS.md <https://github.com/qLDPCOrg/qLDPC/blob/main/AGENTS.md>`_ before modifying the
  implementation.
