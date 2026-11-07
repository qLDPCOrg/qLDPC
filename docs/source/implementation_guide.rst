Implementation guide
====================

This guide lists requirements for changing qLDPC's algebra, decoders, circuits, and external integrations, and recipes for common kinds of change.

Implementation requirements
---------------------------

Use the :doc:`library map <library_map>` and source docstrings for the data model and scientific conventions.
The requirements here concern changes to their implementation.

Algebra and code state
~~~~~~~~~~~~~~~~~~~~~~

* Preserve concrete ``galois.FieldArray`` types when copying or transforming code data.
  Do not coerce extension-field values with ``% field.order``.
* Reject incompatible group rings in NumPy operations, including arrays supplied through keyword arguments such as ``out=``.
* Use ``qldpc.math.symplectic_conjugate`` rather than duplicating sign and half-order conventions.
* Do not route nonbinary codes through the binary-only ``codes/distance/`` routines.
* Preserve stabilizer-based syndrome construction in subsystem decoding, and include a focused subsystem regression test when changing that path.

Decoder integration
~~~~~~~~~~~~~~~~~~~

Follow the contracts and maintainer requirements in the :doc:`adding a decoder guide <adding_decoders>`.
Use its backend-contribution checklist for a maintained integration, and its shared-plumbing requirements when changing decoder construction or resolution.
Reuse the existing construction and resolution helpers rather than duplicating capability dispatch.

Circuits
~~~~~~~~

* Use the ``_restrict_to_qubits`` guard from ``circuits/common.py`` in circuit and tableau constructors.
* Keep data, check, reference, and ancilla roles explicit through ``QubitIDs`` and bookkeeping types.
  Do not infer roles from coincident integer indices.
* Preserve memory-circuit detector-coordinate semantics; a monotone check index must not be interpreted as time.
* Noise-model operation immunity and qubit immunity are separate controls; preserve both.
* Encode T gates and other π/4 rotations in Clifford+T tools as ``RotatedPauliString`` documents, following tsim.

External systems and caching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Document subprocess, standard-input, clipboard, network, and package-installation side effects in the public entry points that can trigger them, and in error messages.
* Expensive reusable computations should use ``qldpc.cache.use_disk_cache(cache_name)``.
  Do not hide errors by returning a default value that looks successful.
* Numba code uses ``always_compiled`` or ``compiled_on_request`` from ``qldpc/_compilation.py``, never ``njit(cache=True)``; that module's docstring says which to use, and ``_compilation_test.py`` enforces its rules.

Change recipes
--------------

Before starting one of the following kinds of change, read its recipe.
For a decoder, follow :doc:`Adding a decoder <adding_decoders>`.
For documentation and examples, follow the :doc:`writing guide <writing_guide>`.

Add or change a code construction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Put a classical family in ``codes/classical.py``, next to its closest base or sibling.
   Put a quantum or CSS family in the module of ``codes/quantum/`` for its construction, such as ``two_block.py`` or ``hypergraph_product.py``, or in a new module named after a new construction.
   Name modules by construction, not by properties that several constructions share, such as being quasi-cyclic.
   Put a named code with a fixed length and dimension, such as ``SteaneCode``, in ``codes/quantum/named.py``.
2. Re-export public names from ``codes/__init__.py``, and quantum ones from ``codes/quantum/__init__.py`` as well.
3. Add a linked literature reference and document the field, subsystem, distance, and validation assumptions in the class docstring.
4. Add co-located tests with independent invariants: rank-derived dimension, CSS orthogonality, symplectic commutation, row-space identities, known small distances, or exact cross-family equivalence.
   Do not validate construction-supplied or cached parameters by reading those same values back.
5. If the construction of a binary quantum code comes with a group of qubit permutations that preserve the code, consider overriding ``_get_qudit_automorphisms`` to return generators of that group.
   Exact distance searches use them to search less when the optional ``pybliss`` package does not find such permutations.
   ``MBCode._get_qudit_automorphisms`` is an example, which uses ``_get_block_translations`` for a group that translates every block of qudits at once.
   For codes built from lifts of a group algebra, ``_get_abelian_block_translations`` returns these translations if the group is abelian and its generators lift to permutation matrices, and None otherwise.
   Prove in the docstring that each generator maps the row space of the parity check matrix onto itself, and test it.
   For a CSS code, a generator does so if it maps the row spaces of both ``matrix_x`` and ``matrix_z`` onto themselves.

Change code-core algebra
~~~~~~~~~~~~~~~~~~~~~~~~

1. Read the cached-state interactions in ``codes/common.py`` before changing standard form, logicals, stabilizers, gauges, dimension, or distance.
2. Test both CSS and non-CSS paths, stabilizer and subsystem paths, ``k=0``, and at least one odd characteristic when signs matter.
3. Preserve public imports and invalidate or transfer cached values deliberately.

Change a circuit workflow
~~~~~~~~~~~~~~~~~~~~~~~~~

1. Keep matrix, qubit-role, measurement-record, detector-record, and Stim-circuit bookkeeping in sync.
2. Test noiseless detector determinism and observable behavior before adding noise.
3. Exercise repeated rounds and seam behavior; a locally valid circuit fragment can still become wrong when initialization, QEC cycles, and readout are joined.
4. State computational and fault-tolerance limitations in the relevant function or class docstring.

Change an external integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Separate parsing/encoding logic from process, network, clipboard, and prompt mechanics.
2. Test both the callable-tool path and the absent/manual/error paths without live resources.
3. For finite fields, round-trip values through the external system's actual element encoding; do not assume integer storage conventions match.
4. Report nonzero return codes, standard error, malformed output, and missing data clearly.
