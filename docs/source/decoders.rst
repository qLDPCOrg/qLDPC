Choosing a decoder
==================

qLDPC distinguishes two kinds of decoders:

* an :class:`decoders.ErrorDecoder <qldpc.decoders.protocols.ErrorDecoder>` maps a syndrome to an inferred physical error, with a ``decode_errors`` method, or its alias ``decode``; and
* an :class:`decoders.ObservableDecoder <qldpc.decoders.protocols.ObservableDecoder>` maps a syndrome (detection events) to predicted observable flips, with a ``decode_observables`` method.

The distinction matters when composing decoders.
Decoder-based distance bounds, logical-operator reduction, and sliding-window decoders all work with physical errors, so they need error decoders.
Circuit-level simulations only need to know which observables flipped, so they use observable decoders.
Code-capacity estimates only need to know whether decoding changed the logical state, so they accept either kind (see `Code-capacity estimates`_).
Some decoders are both: :class:`decoders.RelayBPDecoder <qldpc.decoders.external.relay_bp.RelayBPDecoder>` and :class:`decoders.TesseractDecoder <qldpc.decoders.external.tesseract.TesseractDecoder>` infer errors and predict observable flips.
Exact distance calculations do not need a decoder.

A :class:`decoders.SinterDecoder <qldpc.decoders.sinter.core.SinterDecoder>` stores decoder settings for Sinter and builds a compiled observable decoder for each detector error model.

The :doc:`decoders example notebook <examples/decoders>` walks through the workflows on this page.

Configuring decoders
--------------------

Use a typed helper to configure a decoder, and build the decoder for a parity-check matrix or detector error model:

.. code-block:: python

   import numpy as np

   from qldpc import codes, decoders

   code = codes.RepetitionCode(5)
   syndrome = np.array([1, 0, 0, 0])

   error_decoder = decoders.bp_lsd(max_iter=30, bp_method="minimum_sum").build(code.matrix)
   correction = error_decoder.decode_errors(syndrome)

Methods that decode, such as the code-capacity estimators below, instead accept the settings as ``decoder=`` and build the decoder themselves.

The helpers are available directly under ``qldpc.decoders``:

* :func:`decoders.bp_osd <qldpc.decoders.construction.specs.bp_osd>`
* :func:`decoders.bp_lsd <qldpc.decoders.construction.specs.bp_lsd>`
* :func:`decoders.bf <qldpc.decoders.construction.specs.bf>`
* :func:`decoders.mwpm <qldpc.decoders.construction.specs.mwpm>`
* :func:`decoders.relay_bp <qldpc.decoders.construction.specs.relay_bp>`
* :func:`decoders.min_sum_bp <qldpc.decoders.construction.specs.min_sum_bp>`
* :func:`decoders.tesseract <qldpc.decoders.construction.specs.tesseract>`
* :func:`decoders.lookup_table <qldpc.decoders.construction.specs.lookup_table>`
* :func:`decoders.ilp <qldpc.decoders.construction.specs.ilp>`
* :func:`decoders.guf <qldpc.decoders.construction.specs.guf>`

Each helper returns a :class:`decoders.DecoderSpec <qldpc.decoders.construction.specs.DecoderSpec>`, which only stores settings.
The signature of a helper lists the options of its decoder explicitly, so they are visible to autocomplete and static-analysis tools, and a misspelled option raises a ``TypeError``.
The exception is ``ilp``, which forwards additional options to ``cvxpy.Problem.solve``.
Passing ``decoder=None`` uses qLDPC's default: BP+OSD for binary inputs and generalized union-find for nonbinary field arrays.

Building decoders immediately
-----------------------------

The typed helpers above store validated settings; they do not build a decoder until ``.build(...)`` or a higher-level API supplies a matrix or detector error model.
To construct one immediately, use the lowercase builders owned by their implementation modules.
They are exported from ``qldpc.decoders`` and ``qldpc.decoders.construction``:

* :func:`decoders.get_decoder_bp_osd <qldpc.decoders.external.ldpc.get_decoder_bp_osd>`
* :func:`decoders.get_decoder_bp_lsd <qldpc.decoders.external.ldpc.get_decoder_bp_lsd>`
* :func:`decoders.get_decoder_bf <qldpc.decoders.external.ldpc.get_decoder_bf>`
* :func:`decoders.get_decoder_mwpm <qldpc.decoders.external.pymatching.get_decoder_mwpm>`
* :func:`decoders.get_decoder_rbp <qldpc.decoders.external.relay_bp.get_decoder_rbp>`
* :func:`decoders.get_decoder_tesseract <qldpc.decoders.external.tesseract.get_decoder_tesseract>`
* :func:`decoders.get_decoder_lookup <qldpc.decoders.custom.lookup.get_decoder_lookup>`
* :func:`decoders.get_decoder_ilp <qldpc.decoders.custom.ilp.get_decoder_ilp>`
* :func:`decoders.get_decoder_guf <qldpc.decoders.custom.guf.get_decoder_guf>`

For example, ``decoders.get_decoder_bp_lsd(code.matrix, max_iter=30)`` builds immediately, whereas ``decoders.bp_lsd(max_iter=30)`` returns reusable typed settings.

Tesseract is an optional binary decoder.
Install qLDPC with the ``tesseract`` extra to use it: ``pip install 'qldpc[tesseract]'``.
The pinned upstream release is distributed only as wheels for CPython 3.12--3.14 on macOS arm64 and recent glibc-based Linux x86-64 systems.
A parity-check matrix is converted to a detector error model with probabilities supplied by ``error_rate`` or ``error_channel``; a detector error model supplies its own probabilities and observable definitions.
By default, Tesseract merges the interchangeable error mechanisms of a detector error model, but not identical columns of a parity-check matrix, for which merging could report a column other than the most likely one.
Set ``merge_errors`` to override this choice.
Tesseract reports a low-confidence result if its search does not converge within its configured beam or priority-queue limits.
Set ``add_erasure_bit=True`` to expose that result as qLDPC's appended erasure flag.

Higher-level APIs accept the same settings:

.. code-block:: python

   estimator = code.get_logical_error_rate_func(
       num_samples=10_000,
       decoder=decoders.mwpm(),
   )

   css_code = codes.SurfaceCode(3)
   estimator = css_code.get_logical_error_rate_func(
       num_samples=10_000,
       decoder_x=decoders.bp_lsd(max_iter=30),
       decoder_z=decoders.mwpm(),
   )

A CSS code decodes X-type and Z-type errors independently.
``decoder_x`` infers X-type errors from their syndrome with respect to the Z-type stabilizers, and ``decoder_z`` infers Z-type errors from their syndrome with respect to the X-type stabilizers.
A sector that is not configured explicitly falls back to the shared ``decoder=`` argument.

Code-capacity estimates
-----------------------

A code-capacity estimate samples errors, decodes their syndromes, and counts a failure whenever the decoder mispredicts the logical action of a sampled error.
It therefore only ever asks which logical operators (observables) an error flips, and its ``decoder=``, ``decoder_x=``, and ``decoder_z=`` arguments each accept either kind of decoder:

* An error decoder (decoder settings, a constructor, or, where accepted, a prebuilt error decoder) infers a physical error, and the logical operators flipped by that error are its prediction.
  This is the default, and ``decoder=None`` selects BP+OSD or GUF.
  Decoder settings build an error decoder here, even if the configured decoder could predict observable flips natively.
* A Sinter-style decoder, such as ``decoders.SinterDecoder(decoder=decoders.lookup_table(max_weight=2))``, is compiled for a code-capacity detector error model whose detectors are the stabilizers (or parity checks) of the code and whose observables are its logical operators (or, for a classical code, its bits).
  A shared Sinter-style decoder is compiled separately for each CSS sector.
  Stim detector error models are binary, so such a decoder is rejected for a code over another field.
  A callable explicitly annotated to return an observable decoder is treated as an observable-decoder constructor and is built from the same detector error model.
* A prebuilt observable decoder, such as an :class:`decoders.ObservableLookupDecoder <qldpc.decoders.custom.lookup.ObservableLookupDecoder>` built with the stabilizers and logical operators of a CSS sector, predicts logical flips directly, over any field.
  Detector, observable, and field metadata is validated when a decoder exposes it.
  Built-in observable decoders expose this metadata; a raw precompiled decoder that only provides Sinter's bit-packed interface must do so as well.

The low-level detector-error-model and decoder orchestration is in :mod:`qldpc.codes.code_capacity`; the fixed-weight sampling statistics are in :mod:`qldpc.codes.monte_carlo`.

The two kinds can be mixed across CSS sectors:

.. code-block:: python

   estimator = css_code.get_logical_error_rate_func(
       num_samples=10_000,
       decoder_x=decoders.bp_lsd(max_iter=30),
       decoder_z=decoders.SinterDecoder(decoder=decoders.lookup_table(max_weight=2)),
   )

Direct observable decoding can lower the estimated logical error rate of a degenerate code, because the most likely logical class of an error need not contain the most likely individual error.
The predictions of every decoder are validated: an inferred error or predicted observable vector with the wrong length, entries outside the field of the code, or malformed erasure flags raises an error, as does a prebuilt or compiled decoder built for a different number of detectors or observables.
Decoder-based distance bounds still require error decoders, since they inspect the weights of the errors that a decoder infers.

Predicting observable flips
---------------------------

:func:`decoders.get_observable_decoder <qldpc.decoders.construction.resolution.get_observable_decoder>` builds an observable decoder for a detector error model, and :func:`decoders.decode_observables <qldpc.decoders.construction.resolution.decode_observables>` predicts the observable flips of one syndrome:

.. code-block:: python

   observable_decoder = decoders.get_observable_decoder(dem, decoder=decoders.mwpm())
   predicted_flips = observable_decoder.decode_observables(syndrome)

Some decoders can predict observable flips natively, without first inferring an error.
Their settings build a native observable decoder wherever observable flips are wanted:

* ``mwpm`` builds a PyMatching decoder that tracks observables along matched paths;
* ``relay_bp`` and ``min_sum_bp`` build a :class:`decoders.RelayBPDecoder <qldpc.decoders.external.relay_bp.RelayBPDecoder>`;
* ``tesseract`` builds a :class:`decoders.TesseractDecoder <qldpc.decoders.external.tesseract.TesseractDecoder>` that natively predicts the observables of its detector error model; and
* ``lookup_table`` builds an :class:`decoders.ObservableLookupDecoder <qldpc.decoders.custom.lookup.ObservableLookupDecoder>`, which maps each syndrome directly to its most likely observable flip.

The settings of any other decoder build an error decoder, whose inferred errors are converted into observable flips.
``DecoderSpec.predicts_observables_natively`` reports which of these applies.

For Sinter, wrap decoder settings (or a constructor) in a :class:`decoders.SinterDecoder <qldpc.decoders.sinter.core.SinterDecoder>`, or in one of its subclasses:

.. code-block:: python

   sinter_decoder = decoders.SinterDecoder(
       decoder=decoders.mwpm(),
       decompose_errors=True,
   )

   sliding_decoder = decoders.SlidingWindowDecoder(
       window_size=5,
       stride=2,
       decoder=decoders.bp_lsd(max_iter=30),
   )

A ``SinterDecoder`` is compiled for a detector error model before it predicts flips.
Its compiled form, a :class:`decoders.CompiledSinterDecoder <qldpc.decoders.sinter.core.CompiledSinterDecoder>`, exposes ``decode_observables`` for one shot and ``decode_shots`` for a batch.
A ``SinterDecoder`` and a :class:`decoders.SubgraphDecoder <qldpc.decoders.sinter.subgraph.SubgraphDecoder>` use native observable decoders where possible.
Window decoders (:class:`decoders.SequentialWindowDecoder <qldpc.decoders.sinter.window.SequentialWindowDecoder>` and :class:`decoders.SlidingWindowDecoder <qldpc.decoders.sinter.window.SlidingWindowDecoder>`) commit the errors that they infer in each window, so they always use error decoders.
Sinter passes decoders to its worker processes by pickling them, so a custom decoder that is used with several workers should be defined in an importable module.

Custom and prebuilt decoders
----------------------------

A custom error decoder subclasses :class:`decoders.ErrorDecoder <qldpc.decoders.protocols.ErrorDecoder>` and implements ``decode_errors``, which maps a syndrome to an inferred error.
The subclass inherits ``decode`` as an alias for ``decode_errors``.
A custom observable decoder implements ``decode_observables``, which maps a syndrome to predicted observable flips, to satisfy the :class:`decoders.ObservableDecoder <qldpc.decoders.protocols.ObservableDecoder>` protocol.
A custom decoder may also define:

* ``decode_errors_batch`` or ``decode_observables_batch``, which decode a two-dimensional array of syndromes (one per row), to satisfy :class:`decoders.BatchErrorDecoder <qldpc.decoders.protocols.BatchErrorDecoder>` or :class:`decoders.BatchObservableDecoder <qldpc.decoders.protocols.BatchObservableDecoder>`, so that Sinter decoders decode shots in batches; and
* ``has_erasure_bit = True``, to declare that it appends an erasure flag to each inferred error or predicted observable flip.

Methods that use an error decoder also accept any object whose ``decode`` method returns an inferred error, such as a decoder built directly with the ldpc package, and wrap it in a :class:`decoders.WrappedErrorDecoder <qldpc.decoders.protocols.WrappedErrorDecoder>`.
The immediate builders of library decoders, such as :func:`decoders.get_decoder_bp_osd <qldpc.decoders.external.ldpc.get_decoder_bp_osd>`, return subclasses of the library's decoder classes defined by their integration modules; for example, ``get_decoder_bp_osd`` returns an ``ldpc.BpOsdDecoder`` that is also an ``ErrorDecoder``.

External integration classes have canonical paths such as :class:`decoders.RelayBPDecoder <qldpc.decoders.external.relay_bp.RelayBPDecoder>` and :class:`decoders.TesseractDecoder <qldpc.decoders.external.tesseract.TesseractDecoder>`.
qLDPC's own implementation classes have canonical paths at :class:`decoders.ILPDecoder <qldpc.decoders.custom.ilp.ILPDecoder>`, :class:`decoders.GUFDecoder <qldpc.decoders.custom.guf.GUFDecoder>`, :class:`decoders.CompositeDecoder <qldpc.decoders.custom.composition.CompositeDecoder>`, and :class:`decoders.DirectDecoder <qldpc.decoders.custom.composition.DirectDecoder>`.
:class:`decoders.ErrorsToObservablesDecoder <qldpc.decoders.adapters.error_decoders.ErrorsToObservablesDecoder>` and :class:`decoders.ExpandedErrorDecoder <qldpc.decoders.adapters.error_decoders.ExpandedErrorDecoder>` are the conversion adapters.
These classes are also exported from ``qldpc.decoders``.

Besides a ``DecoderSpec``, the ``decoder=`` argument accepts:

* a constructor, such as a decoder class, or any other callable that builds a decoder from a parity-check matrix or detector error model;
* where observable flips are predicted for a detector error model, an observable-decoder compiler (see :class:`decoders.ObservableDecoderCompiler <qldpc.decoders.construction.specs.ObservableDecoderCompiler>`), such as a ``SinterDecoder``, which is compiled for that model; or
* a prebuilt decoder, which is used as is.

A prebuilt decoder is tied to the matrix used to construct it, so it is only accepted where the caller knows the matrix being decoded: by ``decoders.get_error_decoder``, ``decoders.decode_observables``, and ``decoders.get_observable_decoder``; by the code-capacity estimators of classical codes; by ``ClassicalCode.get_distance_bound`` when given a ``vector`` (whose syndrome is computed with the parity check matrix of the code); and per sector (as ``decoder_x=`` and ``decoder_z=``) by the code-capacity estimators of CSS codes.
A shared prebuilt ``decoder=`` for a CSS code is rejected unless its two stabilizer matrices are equal (and, for a prebuilt observable decoder, so are its two sets of logical operators).

Some methods decode a matrix that they construct internally, and therefore reject prebuilt decoders: decoder-based distance bounds of codes (other than a classical distance bound to a ``vector``), logical-operator reduction, the code-capacity estimators of non-CSS codes, and Sinter decoders, which build a new decoder for every (simplified) detector error model, window, or subgraph that they decode.
These methods accept a ``DecoderSpec`` or a constructor.
A constructor can fix custom options with ``functools.partial`` or a ``lambda``.
A ``SinterDecoder``, a ``SubgraphDecoder``, and the code-capacity estimators also accept an observable-decoder compiler, which they compile for each detector error model that they decode.

Lookup-table outputs
--------------------

:class:`decoders.LookupDecoder <qldpc.decoders.custom.lookup.LookupDecoder>` and :class:`decoders.WeightedLookupDecoder <qldpc.decoders.custom.lookup.WeightedLookupDecoder>` are error decoders.
They may use an observable-flip matrix to group candidate errors by logical effect, but their output is a representative physical error.

:class:`decoders.ObservableLookupDecoder <qldpc.decoders.custom.lookup.ObservableLookupDecoder>` and :class:`decoders.WeightedObservableLookupDecoder <qldpc.decoders.custom.lookup.WeightedObservableLookupDecoder>` are observable decoders, which return the observable flip itself:

.. code-block:: python

   observable_lookup = decoders.ObservableLookupDecoder(dem, max_weight=2)
   predicted_flips = observable_lookup.decode_observables(syndrome)

An observable lookup decoder built from a parity-check matrix, rather than a detector error model, requires an ``observable_flip_matrix`` that specifies which errors flip which observables.

Erasure-aware decoders append their erasure flag after the inferred error or observable vector.
Compiled Sinter decoders translate that flag into a discarded shot.

Migrating from qLDPC 0.3.3
--------------------------

This section describes how the decoder API differs from that of ``qldpc==0.3.3``, and how to update code written for it.

Breaking changes
~~~~~~~~~~~~~~~~

The following changes take effect without a deprecation period:

* The ``static_decoder`` argument has been removed, except from the deprecated ``decoders.get_decoder`` and ``decoders.decode``.
  Pass a prebuilt decoder as ``decoder=`` instead, where a prebuilt decoder is accepted (see above), and otherwise pass decoder settings or a constructor.
* When the deprecated ``decoder_x_kwargs`` or ``decoder_z_kwargs`` of a CSS code set the same option as its shared keyword arguments, the sector-specific value now takes precedence, just as ``decoder_x=`` and ``decoder_z=`` take precedence over ``decoder=``.
* A ``SinterDecoder`` whose settings support native observable prediction (``mwpm``, ``relay_bp``, ``min_sum_bp``, and ``lookup_table``) now uses it.
  The predicted observable flips are unchanged.
  For ``mwpm`` and ``lookup_table`` settings, however, the ``decoder`` attribute of the resulting ``CompiledSinterDecoder`` is now a decoder that predicts observable flips rather than errors.
  For other settings, the ``decoder`` attribute remains the decoder that the settings build.

Deprecated usage
~~~~~~~~~~~~~~~~

The keyword-based decoder API of ``qldpc==0.3.3`` remains available during a deprecation period, and each use emits a ``DeprecationWarning`` that names its replacement.
Usage marked as defunct instead raises an error that names its replacement.
Compatibility is provided for the names exported from the package root, ``qldpc.decoders``, in that release; internal module paths are not part of this guarantee.
In particular, ``decoders.get_decoder`` and ``decoders.decode`` behave as they did in ``qldpc==0.3.3``:

.. list-table::
   :header-rows: 1

   * - Deprecated usage
     - Replacement
   * - ``decoders.get_decoder(pcm_or_dem, with_BP_LSD=True, max_iter=30)``
     - ``decoders.bp_lsd(max_iter=30).build(pcm_or_dem)``
   * - ``decoders.decode(pcm_or_dem, syndrome, with_BP_LSD=True, max_iter=30)``
     - ``decoders.bp_lsd(max_iter=30).build(pcm_or_dem).decode(syndrome)``
   * - ``with_BP_LSD=True, max_iter=30``
     - ``decoder=decoders.bp_lsd(max_iter=30)``
   * - free-form decoder options without ``with_<NAME>``
     - ``decoder=decoders.bp_osd(...)``, or ``decoders.guf(...)`` for nonbinary codes
   * - ``decoder_constructor=MyDecoder``
     - ``decoder=MyDecoder``
   * - ``decoders.get_decoder_BP_OSD`` (and the other uppercase builder names)
     - ``decoders.get_decoder_bp_osd`` (and the corresponding lowercase builder)
   * - ``decoder_x_kwargs={...}`` and ``decoder_z_kwargs={...}``
     - ``decoder_x=...`` and ``decoder_z=...``
   * - ``LookupDecoder(..., predict_observable_flips=True)``
     - ``ObservableLookupDecoder(...)`` and its ``decode_observables`` method
   * - ``WeightedLookupDecoder(..., predict_observable_flips=True)``
     - ``WeightedObservableLookupDecoder(...)`` and its ``decode_observables`` method
   * - ``SinterDecoder.decode`` and ``CompiledSinterDecoder.decode`` (defunct)
     - ``CompiledSinterDecoder.decode_observables``
   * - ``SubgraphSinterDecoder`` and ``SequentialSinterDecoder``
     - ``SubgraphDecoder`` and ``SequentialWindowDecoder``
   * - ``Decoder`` and ``BatchDecoder``
     - ``ErrorDecoder`` and ``BatchErrorDecoder``
