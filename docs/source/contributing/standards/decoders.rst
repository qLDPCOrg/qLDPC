Decoder standard
================

This standard covers decoders that ship with qLDPC and the shared plumbing that builds and resolves decoders.
:doc:`Adding a decoder </adding_decoders>` explains the decoder contracts and how to use a decoder of your own.

Public decoders
---------------

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
   Add its ``.. autofunction:: qldpc.decoders.<name>`` entry to the :doc:`decoder guide </decoders>`: AutoAPI hides generated helpers because it cannot inspect their signatures statically.
   The helper test discovers root-exported generated helpers and checks their signature, written documentation, and guide entry automatically.
   Add co-located ``*_test.py`` tests that independently check ``H @ inferred_error == syndrome``, observable predictions against DEM targets, unsupported inputs, erasure width, and batch shapes as applicable.
   Test pickling if the decoder is used by Sinter.

For an optional published backend, add an extra under ``optional-dependencies`` in ``pyproject.toml`` with any required platform markers.
Also add its import name to the backends that ``qldpc/import_test.py`` checks importing qLDPC does not import.
Existing integrations such as :mod:`qldpc.decoders.external.tesseract` show a backend with both output contracts.
Do not change generic decoder resolution just to add an ordinary backend.

Shared decoder plumbing
-----------------------

These requirements apply when changing existing integration paths, not merely implementing a custom decoder:

* Reuse the existing construction and resolution helpers rather than duplicating capability dispatch.
* Coerce user-supplied error decoders with ``_as_error_decoder`` from ``decoders/protocols.py`` inside ``qldpc.decoders``, or ``build_error_decoder`` from ``decoders/_resolution.py`` outside it, and call ``decode_errors``.
* Reject prebuilt decoders with ``reject_prebuilt_decoder`` from ``decoders/_resolution.py`` when a method constructs the matrix that it decodes.
* Keep the import layers of ``qldpc.decoders`` that ``_LAYERS`` in ``decoders/layering_test.py`` lists, and add a new top-level module or subpackage to a layer there; the :doc:`code standard <code>` describes the rule and its exceptions.
* Keep ``build_error_decoder`` specific to error decoders.
  Dispatch on explicit capabilities such as ``compile_decoder_for_dem``, ``decode_observables``, and ``ErrorDecoder``, never on output length.
* In code-capacity callers under ``qldpc.codes``, resolve decoder inputs with ``get_code_capacity_decoder`` from ``codes/_code_capacity.py``.
  It always provides an observable decoder.
* When changing DEM decomposition or simplification, test malformed components and index remaps after cancellation.
* Preserve the sliding-window detector-time heuristic in ``decoders/sinter/window.py``.
  Use an explicit ``detector_to_time`` mapping for other layouts rather than changing the fallback.
