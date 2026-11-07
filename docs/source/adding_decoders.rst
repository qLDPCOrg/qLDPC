Adding a decoder
================

You can use a decoder of your own in qLDPC workflows, or add a decoder to qLDPC itself.
To **use your own decoder**, implement one decoding method and pass its constructor as ``decoder=``.
To **ship a decoder with qLDPC**, add an implementation, a public specification helper, and tests.
No registration, specification helper, or change to qLDPC itself is needed for an experiment.
To configure an existing backend instead, see :doc:`Choosing a decoder <decoders>`.

Use your own decoder
--------------------

An error decoder returns a physical error with ``decode_errors(syndrome)``.
The result has one entry per matrix column, over the code's field.
Protocol subclassing is optional; the object only needs the method required by the workflow.
Inheriting ``ErrorDecoder`` also supplies the ``decode`` alias.
This small decoder works only for an identity parity-check matrix, for which the syndrome is the error itself:

.. code-block:: python

   import galois
   import numpy as np
   import numpy.typing as npt

   from qldpc import codes, decoders

   class IdentityDecoder(decoders.ErrorDecoder):
       def __init__(self, matrix: galois.FieldArray) -> None:
           if not np.array_equal(matrix, type(matrix).Identity(matrix.shape[1])):
               raise ValueError("IdentityDecoder requires an identity parity-check matrix")

       def decode_errors(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
           return np.asarray(syndrome, dtype=int)

   matrix = galois.GF2.Identity(2)
   syndrome = np.array([1, 0], dtype=int)
   correction = IdentityDecoder(matrix).decode_errors(syndrome)
   assert np.array_equal(matrix @ type(matrix)(correction), syndrome)

   estimator = codes.ClassicalCode(matrix).get_logical_error_rate_func(
       3, decoder=IdentityDecoder
   )

The code-capacity estimator calls ``IdentityDecoder(matrix)`` when it builds its decoder.
For configurable construction, pass a factory such as ``functools.partial`` as ``decoder=``.
The factory receives the parity-check matrix.
Wrapping it with ``decoders.from_matrix(factory)`` is optional and makes its matrix-only contract explicit.

Choose the workflow contract
----------------------------

The workflow determines what input the factory receives and what prediction it must return:

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Workflow
     - Factory input
     - Decoder method
   * - Code capacity, ordinary factory
     - Parity-check matrix
     - ``decode_errors``
   * - Binary code capacity, ``decoders.from_dem(factory)``
     - Detector error model (DEM)
     - ``decode_observables``
   * - Ordinary ``SinterDecoder``
     - DEM
     - ``decode_errors`` or ``decode_observables``
   * - Sliding-window decoder
     - Each window's DEM
     - ``decode_errors``

For observable predictions, implement ``decode_observables(syndrome)``.
It returns one value per observable.
An ordinary Sinter decoder supplies a Stim detector error model (DEM) to its constructor:

.. code-block:: python

   import numpy as np
   import numpy.typing as npt
   import stim

   from qldpc import decoders

   class NoFlipDecoder:
       def __init__(self, dem: stim.DetectorErrorModel) -> None:
           self.num_observables = dem.num_observables

       def decode_observables(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
           return np.zeros(self.num_observables, dtype=int)

   sinter_decoder = decoders.SinterDecoder(decoder=NoFlipDecoder)
   dem = stim.DetectorErrorModel("error(0.1) D0 L0")
   assert sinter_decoder.compile_decoder_for_dem(dem).decode_observables(
       np.array([1], dtype=int)
   ).tolist() == [0]

This toy decoder illustrates the output contract by always predicting no flips.
For a **binary** code-capacity sector, pass ``decoder=decoders.from_dem(NoFlipDecoder)`` to supply the DEM to this constructor.
The :doc:`decoder guide <decoders>` and :doc:`example notebook <examples/decoders>` show code-capacity and Sinter usage.
Do not pass ``from_dem`` observable factories to workflows that require inferred errors.

Optional capabilities
---------------------

Add these only when the experiment or integration needs them:

* **Batch decoding:** implement ``decode_errors_batch`` or ``decode_observables_batch`` with one syndrome per row and one prediction per row.
* **Erasure signaling:** set ``has_erasure_bit = True`` and append one binary flag at the end of each prediction.
  Only advertise this capability when it is implemented; unsupported decoders must reject ``add_erasure_bit=True``.
* **Nonbinary inputs:** preserve the matrix's concrete field and reject unsupported fields explicitly.
* **Sinter multiprocessing:** define classes and builders in an importable module so Sinter can pickle them.

Ship a decoder with qLDPC
-------------------------

A decoder that ships with qLDPC needs an implementation, a public specification helper, and tests.
The :doc:`decoder standard </contributing/standards/decoders>` lists the steps and requirements.

Run focused co-located tests while developing.
Replace the GUF test path below with your own test file:

.. code-block:: bash

   python -m pytest src/qldpc/decoders/custom/guf_test.py -q

Use the :ref:`validation commands <validation-commands>` for targeted formatting, typing, coverage, and documentation checks.
Run the strict documentation build to check guide and API references; it does not execute the examples.
