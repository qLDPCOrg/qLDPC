Implementation standard
=======================

This standard lists requirements for changing qLDPC's code constructions, algebra, circuits, and external integrations.
Use the :doc:`library map </library_map>` and source docstrings for the data model and scientific conventions.
The :doc:`decoder standard <decoders>` covers decoders.

.. _implementation-code-constructions:

Code constructions
------------------

* Put a classical family in ``codes/classical.py``, next to its closest base or sibling.
  Put a quantum or CSS family in the module of ``codes/quantum/`` for its construction, such as ``two_block.py`` or ``hypergraph_product.py``, or in a new module named after a new construction.
  Name modules by construction, not by properties that several constructions share, such as being quasi-cyclic.
  Put a named code with a fixed length and dimension, such as ``SteaneCode``, in ``codes/quantum/named.py``.
* Re-export public names from ``codes/__init__.py``, and quantum ones from ``codes/quantum/__init__.py`` as well.
* Add a linked literature reference and document the field, subsystem, distance, and validation assumptions in the class docstring.
* Add co-located tests with independent invariants: rank-derived dimension, CSS orthogonality, symplectic commutation, row-space identities, known small distances, or exact cross-family equivalence.
  Do not validate construction-supplied or cached parameters by reading those same values back.
* If the construction of a binary quantum code comes with a group of qubit permutations that preserve the code, consider overriding ``_get_qudit_automorphisms`` to return generators of that group.
  Exact distance searches use them to search less when the optional ``pybliss`` package does not find such permutations.
  ``MBCode._get_qudit_automorphisms`` is an example, which uses ``_get_block_translations`` for a group that translates every block of qudits at once.
  For codes built from lifts of a group algebra, ``_get_abelian_block_translations`` returns these translations if the group is abelian and its generators lift to permutation matrices, and None otherwise.
  Prove in the docstring that each generator maps the row space of the parity check matrix onto itself, and test it.
  For a CSS code, a generator does so if it maps the row spaces of both ``matrix_x`` and ``matrix_z`` onto themselves.

.. _implementation-algebra:

Algebra and code state
----------------------

* Read the cached-state interactions in ``codes/common.py`` before changing standard form, logicals, stabilizers, gauges, dimension, or distance.
* Invalidate or transfer cached values deliberately.
* Preserve concrete ``galois.FieldArray`` types when copying or transforming code data.
  Do not coerce extension-field values with ``% field.order``.
* Reject incompatible group rings in NumPy operations, including arrays supplied through keyword arguments such as ``out=``.
* Use ``qldpc.math.symplectic_conjugate`` rather than duplicating sign and half-order conventions.
* Do not route nonbinary codes through the binary-only ``codes/distance/`` routines.
* Preserve stabilizer-based syndrome construction in subsystem decoding, and include a focused subsystem regression test when changing that path.
* Test both CSS and non-CSS paths, stabilizer and subsystem paths, ``k=0``, and at least one odd characteristic when signs matter.

.. _implementation-circuits:

Circuits
--------

* Use the ``_restrict_to_qubits`` guard from ``circuits/common.py`` in circuit and tableau constructors.
* Keep data, check, reference, and ancilla roles explicit through ``QubitIDs`` and bookkeeping types.
  Do not infer roles from coincident integer indices.
* Preserve memory-circuit detector-coordinate semantics; a monotone check index must not be interpreted as time.
* Noise-model operation immunity and qubit immunity are separate controls; preserve both.
* Encode T gates and other π/4 rotations in Clifford+T tools as ``RotatedPauliString`` documents, following tsim.
* State computational and fault-tolerance limitations in the relevant function or class docstring.
* In a circuit workflow, such as a memory experiment, keep matrix, qubit-role, measurement-record, detector-record, and Stim-circuit bookkeeping in sync.
  Test noiseless detector determinism and observable behavior before adding noise.
  Exercise repeated rounds and seam behavior; a locally valid circuit fragment can still become wrong when initialization, QEC cycles, and readout are joined.

.. _implementation-external-systems:

External systems and caching
----------------------------

* Document subprocess, standard-input, clipboard, network, and package-installation side effects in the public entry points that can trigger them, and in error messages.
* Separate parsing/encoding logic from process, network, clipboard, and prompt mechanics.
* For finite fields, round-trip values through the external system's actual element encoding; do not assume integer storage conventions match.
* Report nonzero return codes, standard error, malformed output, and missing data clearly.
  Do not hide errors by returning a default value that looks successful.
* For an external tool, test both the callable-tool path and the absent/manual/error paths without live resources.
* Expensive reusable computations should use ``qldpc.cache.use_disk_cache(cache_name)``.
