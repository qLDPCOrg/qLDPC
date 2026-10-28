Adding a decoder
================

There are two paths.
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
This identity-matrix example demonstrates the interface, not a useful code or a decoder benchmark.

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

Add a public decoder to qLDPC
-----------------------------

1. **Choose the contract first.**
   Decide whether the decoder infers errors, predicts observables, or both.
   Decide separately which inputs it accepts: parity-check matrices, binary DEMs, and nonbinary field arrays.
   Validate the input in the builder and raise a clear error for unsupported inputs.
   Window decoders need inferred errors; code-capacity and ordinary Sinter decoders can use observable predictions.
2. **Put implementation and builder together.**
   Use :mod:`qldpc.decoders.custom` for a qLDPC-owned algorithm or :mod:`qldpc.decoders.external` for a third-party integration.
   Implement :class:`~qldpc.decoders.protocols.ErrorDecoder` (``decode_errors``) and/or :class:`~qldpc.decoders.protocols.ObservableDecoder` (``decode_observables``).
   Keep a module-level builder with a typed signature: its first input is a matrix or DEM, followed by the options for this algorithm.
   The builder stays private, with a name that starts with an underscore, and the public specification helper of the next step is its entry point.
   For an optional backend, import its dependency inside the builder and report how to install it.
   Preserve the backend's own option names rather than introducing a generic translation layer.
3. **Expose a specification.**
   Use :func:`~qldpc.decoders.construction.specs.decoder_spec` with an error builder, and optionally a DEM-based native observable builder.
   An observable-only backend uses :func:`~qldpc.decoders.construction.specs.observable_decoder_spec`.
   These helpers take their option signature from the typed builder (or from ``signature_source=DecoderClass`` if that builder forwards ``**kwargs``).
   Provide a written ``doc=`` string with the helper's options and return contract.
   Give the builder a one-line docstring that names this helper, rather than repeating its options.
   A :class:`~qldpc.decoders.construction.specs.DecoderSpec` holds the builder and options; call ``build`` or ``build_observable_decoder`` to construct a decoder.
4. **Export, document, and test.**
   Export the backend class at its qualified ``custom`` or ``external`` package path.
   If you add a public specification helper, export it from ``qldpc.decoders`` and add it to ``__all__``.
   Add its ``.. autofunction:: qldpc.decoders.<name>`` entry to the :doc:`decoder guide <decoders>`: AutoAPI hides generated helpers because it cannot inspect their signatures statically.
   The helper test discovers root-exported generated helpers and checks their signature, written documentation, and guide entry automatically.
   Add co-located ``*_test.py`` tests that independently check ``H @ inferred_error == syndrome``, observable predictions against DEM targets, unsupported inputs, erasure width, and batch shapes as applicable.
   Test pickling if the decoder is used by Sinter.

For an optional published backend, add an extra under ``optional-dependencies`` in ``pyproject.toml`` with any required platform markers.
Existing integrations such as :mod:`qldpc.decoders.external.tesseract` show a backend with both output contracts.
Do not change generic decoder resolution just to add an ordinary backend.

Run focused co-located tests while developing.
Replace the GUF test path below with your own test file:

.. code-block:: bash

   python -m pytest src/qldpc/decoders/custom/guf_test.py -q

Use the :doc:`contributor reference <contributing>` for targeted formatting, typing, coverage, and documentation checks.
The repository still requires full per-module statement coverage and offline tests.
Run the strict documentation build to check guide and API references; it does not execute the examples.

Changing shared decoder plumbing
--------------------------------

These requirements apply when changing existing integration paths, not merely implementing a custom decoder:

* Coerce user-supplied error decoders with ``_as_error_decoder`` from ``decoders/protocols.py`` inside ``qldpc.decoders``, or ``build_error_decoder`` from ``decoders/construction/_resolution.py`` outside it, and call ``decode_errors``.
* Reject prebuilt decoders with ``reject_prebuilt_decoder`` from ``decoders/construction/_resolution.py`` when a method constructs the matrix that it decodes.
* Keep ``build_error_decoder`` specific to error decoders.
  Dispatch on explicit capabilities such as ``compile_decoder_for_dem``, ``decode_observables``, and ``ErrorDecoder``, never on output length.
* In code-capacity callers under ``qldpc.codes``, resolve decoder inputs with ``get_code_capacity_decoder`` from ``codes/_code_capacity.py``.
  It always provides an observable decoder.
* When changing DEM decomposition or simplification, test malformed components and index remaps after cancellation.
* Preserve the sliding-window detector-time heuristic in ``decoders/sinter/window.py``.
  Use an explicit ``detector_to_time`` mapping for other layouts rather than changing the fallback.
