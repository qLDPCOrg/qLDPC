Adding a decoder
================

There are two paths.
To **use your own decoder**, implement one decoding method and configure a decoder specification; no registration or library changes are needed.
To **ship a decoder with qLDPC**, add an implementation, a public specification helper, and tests.
There is no plugin registry and no need to edit the generic resolver for a new backend.

Use a custom decoder without changing qLDPC
--------------------------------------------

An error decoder returns a physical error with ``decode_errors(syndrome)``.
The result has one entry per matrix column, over the code's field.
This small decoder works only for an identity parity-check matrix, for which the syndrome is the error itself:

.. code-block:: python

   import galois
   import numpy as np

   from qldpc import codes, decoders
   from qldpc.decoders.construction.specs import decoder_spec

   class IdentityDecoder(decoders.ErrorDecoder):
       def __init__(self, matrix: galois.FieldArray) -> None:
           if not np.array_equal(matrix, type(matrix).Identity(matrix.shape[1])):
               raise ValueError("IdentityDecoder requires an identity parity-check matrix")

       def decode_errors(self, syndrome: np.ndarray) -> np.ndarray:
           return np.asarray(syndrome, dtype=int)

   matrix = galois.GF2.Identity(3)
   identity = decoder_spec("identity", IdentityDecoder)
   identity_spec = identity()
   assert np.array_equal(
       identity_spec.build(matrix).decode_errors(np.array([1, 0, 0])), [1, 0, 0]
   )
   estimator = codes.ClassicalCode(matrix).get_logical_error_rate_func(
       3, decoder=identity_spec
   )
   assert estimator.num_failures[1] == 0

``decoder_spec`` derives the helper's options from the builder's typed signature.
The resulting specification builds the error decoder for the field-valued parity-check matrix provided by the code-capacity estimator and projects inferred errors onto observables.
This builder accepts only a matrix; it does not convert a Stim detector error model (DEM).

To predict observables directly, implement ``decode_observables(syndrome)`` instead.
It returns one value per observable.
For a **binary** code-capacity sector or a Sinter decoder, provide a DEM-based observable builder to ``observable_decoder_spec``:

.. code-block:: python

   import numpy as np
   import stim
   from qldpc import decoders
   from qldpc.decoders.construction.specs import observable_decoder_spec

   class NoFlipDecoder(decoders.ObservableDecoder):
       def __init__(self, dem: stim.DetectorErrorModel) -> None:
           self.num_observables = dem.num_observables

       def decode_observables(self, syndrome: np.ndarray) -> np.ndarray:
           return np.zeros(self.num_observables, dtype=int)

   no_flip = observable_decoder_spec("no_flip", NoFlipDecoder)
   sinter_decoder = decoders.SinterDecoder(decoder=no_flip())
   dem = stim.DetectorErrorModel("error(0.1) D0 L0")
   assert sinter_decoder.compile_decoder_for_dem(dem).decode_observables(
       np.array([1], dtype=int)
   ).tolist() == [0]

This toy decoder always predicts no flips; it only illustrates the output contract.
A DEM is binary; a decoder for a nonbinary code must use field-valued matrix inputs.
Define builders in an importable module, not as local functions or lambdas, when Sinter needs to pickle them for worker processes.
To signal erasure, set ``has_erasure_bit = True`` and append **one binary flag at the end** of each prediction.
Do not append it unless the decoder declares erasure support.
Batch methods are optional; if provided, accept one syndrome per row and return one prediction per row.
See the :doc:`decoder guide <decoders>` for code-capacity and CSS sector choices.

Add a public decoder to qLDPC
-----------------------------

1. **Choose the contract first.**
   Decide whether the decoder infers errors, predicts observables, or both.
   Decide separately which inputs it accepts: parity-check matrices, binary DEMs, and nonbinary field arrays.
   Reject unsupported inputs with a clear error in the builder; do not rely on implicit matrix/DEM conversion.
   Window decoders need inferred errors; code-capacity and ordinary Sinter decoders can use observable predictions.
2. **Put implementation and builder together.**
   Use :mod:`qldpc.decoders.custom` for a qLDPC-owned algorithm or :mod:`qldpc.decoders.external` for a third-party integration.
   Implement :class:`~qldpc.decoders.protocols.ErrorDecoder` (``decode_errors``) and/or :class:`~qldpc.decoders.protocols.ObservableDecoder` (``decode_observables``).
   Keep a module-level builder with a typed signature: its first input is a matrix or DEM, followed by the options for this algorithm.
   If the backend is optional, import it when building, not when importing ``qldpc.decoders``, and report how to install it.
3. **Expose a specification.**
   Use :func:`~qldpc.decoders.construction.specs.decoder_spec` with an error builder, and optionally a DEM-based native observable builder.
   An observable-only backend uses :func:`~qldpc.decoders.construction.specs.observable_decoder_spec`.
   These helpers take their option signature from the typed builder (or from ``signature_source=DecoderClass`` if that builder forwards ``**kwargs``).
   Provide a written ``doc=`` string with the helper's options and return contract.
   A :class:`~qldpc.decoders.construction.specs.DecoderSpec` holds the builder and options; it does not build a decoder until ``build`` or ``build_observable_decoder`` is called.
   No registry entry or new ``construction/resolution.py`` branch is needed for an ordinary backend.
4. **Export, document, and test.**
   Export the backend class at its qualified ``custom`` or ``external`` package path.
   If you add a public specification helper, export it from ``qldpc.decoders`` and add it to ``__all__``.
   Add its ``.. autofunction:: qldpc.decoders.<name>`` entry to the :doc:`decoder guide <decoders>`: AutoAPI hides generated helpers because it cannot inspect their signatures statically.
   The helper test discovers root-exported generated helpers and checks their signature, written documentation, and guide entry automatically.
   Add co-located ``*_test.py`` tests that independently check ``H @ inferred_error == syndrome``, observable predictions against DEM targets, unsupported inputs, erasure width, and batch shapes as applicable.
   Test pickling if the decoder is used by Sinter.

For an optional published backend, add an extra under ``optional-dependencies`` in ``pyproject.toml`` with any required platform markers; do not assume every backend can be installed through the ``decoders`` extra.
Existing integrations such as :mod:`qldpc.decoders.external.tesseract` show a backend with both output contracts.

Run the relevant co-located tests while developing, then check the full change with the existing repository commands.
Replace the GUF test path below with your own test file:

.. code-block:: bash

   python -m pytest src/qldpc/decoders/custom/guf_test.py -q
   python checks/mypy_.py
   python checks/lint_.py
   python checks/format_.py --check
   python checks/coverage_.py
   python checks/build_docs.py

The repository requires 100% statement coverage; tests cannot depend on live network services.
The strict docs build also verifies the new page and API references.
