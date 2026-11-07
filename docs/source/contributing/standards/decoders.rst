Decoder standard
================

This standard covers decoders that ship with qLDPC and the shared plumbing that builds and resolves decoders.
:doc:`Using your own decoder </decoders/custom>` explains the workflow contracts that decoders implement.
:mod:`qldpc.decoders.external.tesseract` is an example of a backend with both output contracts.

* Reuse the existing construction and resolution helpers rather than duplicating capability dispatch.
* Do not change generic decoder resolution just to add an ordinary backend.

Public decoders
---------------

Contracts and inputs
~~~~~~~~~~~~~~~~~~~~

* Choose the output contract before implementing: whether the decoder infers errors, predicts observables, or both.
* Choose separately which inputs it accepts: parity-check matrices, binary DEMs, and nonbinary field arrays.
  Validate the input in the builder and raise a clear error for unsupported inputs.
* Advertise erasure signaling only when it is implemented; decoders without it must reject ``add_erasure_bit=True``.

Implementation and builder
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Put the implementation and its builder together, in :mod:`qldpc.decoders.custom` for a qLDPC-owned algorithm or :mod:`qldpc.decoders.external` for a third-party integration.
* Implement :class:`~qldpc.decoders.protocols.ErrorDecoder` (``decode_errors``) and/or :class:`~qldpc.decoders.protocols.ObservableDecoder` (``decode_observables``).
* Keep a module-level builder with a typed signature: its first input is a matrix or DEM, followed by the options for this algorithm.
  The builder stays private, with a name that starts with an underscore, and the public specification helper is its entry point.
* For an optional backend, import its dependency inside the builder and report how to install it.
* Preserve the backend's own option names rather than introducing a generic translation layer.

Specification helpers
~~~~~~~~~~~~~~~~~~~~~

* Expose a specification with :func:`~qldpc.decoders.construction.specs.decoder_spec`, given an error builder and optionally a DEM-based native observable builder.
  An observable-only backend uses :func:`~qldpc.decoders.construction.specs.observable_decoder_spec`.
* These helpers take their option signature from the typed builder.
  If that builder forwards ``**kwargs``, pass ``signature_source=DecoderClass``.
* Provide a written ``doc=`` string with the helper's options, return contract, and limitations.
* Give the builder a one-line docstring that names this helper, rather than repeating its options.

Exports, documentation, and tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Export the backend class at its qualified ``custom`` or ``external`` package path.
* If you add a public specification helper, export it from ``qldpc.decoders`` and add it to ``__all__``.
* Add its ``.. autofunction:: qldpc.decoders.<name>`` entry to the :doc:`decoder reference </decoders/reference>`: AutoAPI hides generated helpers because it cannot inspect their signatures statically.
  An existing test checks the signature, written documentation, and reference entry of every generated helper that ``qldpc.decoders`` exports.
* Add :ref:`backend notes <backend-notes>` to the decoder reference only for what a docstring cannot carry, such as installation outside PyPI or behavior across workflows.
* Add co-located ``*_test.py`` tests that independently check ``H @ inferred_error == syndrome``, observable predictions against DEM targets, unsupported inputs, erasure width, and batch shapes as applicable.
* Test pickling if the decoder is used by Sinter.

Optional backends
~~~~~~~~~~~~~~~~~

* For an optional published backend, add an extra under ``optional-dependencies`` in ``pyproject.toml`` with any required platform markers.
* Add its import name to the backends that ``qldpc/import_test.py`` checks importing qLDPC does not import.

Shared decoder plumbing
-----------------------

These requirements apply when changing how qLDPC builds, coerces, or resolves decoders:

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
