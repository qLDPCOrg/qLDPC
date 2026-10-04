Choosing a decoder
==================

qLDPC distinguishes two kinds of decoders:

* an :class:`decoders.ErrorDecoder <qldpc.decoders.protocols.ErrorDecoder>` maps a syndrome to an inferred physical error, with a ``decode_errors`` method, or its alias ``decode``; and
* an :class:`decoders.ObservableDecoder <qldpc.decoders.protocols.ObservableDecoder>` maps a syndrome (detection events) to predicted observable flips, with a ``decode_observables`` method.

The distinction matters when composing decoders.
Decoder-based distance bounds, logical-operator reduction, and sliding-window decoders all work with physical errors, so they need error decoders.
Circuit-level simulations only need to know which observables flipped, so they use observable decoders.
Code-capacity estimates only need to know whether decoding changed the logical state, so they accept either kind (see `Code-capacity estimates`_).
Some decoders are both: :class:`RelayBPDecoder <qldpc.decoders.external.relay_bp.RelayBPDecoder>` and :class:`TesseractDecoder <qldpc.decoders.external.tesseract.TesseractDecoder>` infer errors and predict observable flips.
Exact distance calculations do not need a decoder.

A :class:`decoders.SinterDecoder <qldpc.decoders.sinter.core.SinterDecoder>` accepts a decoder specification or constructor and builds a compiled observable decoder for each detector error model.

The :doc:`decoders example notebook <examples/decoders>` walks through the workflows on this page.

Configuring decoders
--------------------

Use a typed helper to create a decoder specification, then build a decoder for a parity-check matrix or detector error model:

.. code-block:: python

   import numpy as np

   from qldpc import codes, decoders

   code = codes.RepetitionCode(5)
   syndrome = np.array([1, 0, 0, 0])

   error_decoder = decoders.bp_lsd(max_iter=30, bp_method="minimum_sum").build(code.matrix)
   correction = error_decoder.decode_errors(syndrome)

Methods that decode, such as the code-capacity estimators below, instead accept a specification as ``decoder=`` and build the decoder themselves.

The helpers are available directly under ``qldpc.decoders``:

* :func:`decoders.bp_osd <qldpc.decoders.bp_osd>`
* :func:`decoders.bp_lsd <qldpc.decoders.bp_lsd>`
* :func:`decoders.bf <qldpc.decoders.bf>`
* :func:`decoders.mwpm <qldpc.decoders.mwpm>`
* :func:`decoders.frontier <qldpc.decoders.frontier>`
* :func:`decoders.relay_bp <qldpc.decoders.relay_bp>`
* :func:`decoders.min_sum_bp <qldpc.decoders.min_sum_bp>`
* :func:`decoders.tesseract <qldpc.decoders.tesseract>`
* :func:`decoders.tesseract_preset <qldpc.decoders.tesseract_preset>`
* :func:`decoders.lookup <qldpc.decoders.lookup>`
* :func:`decoders.ilp <qldpc.decoders.ilp>`
* :func:`decoders.guf <qldpc.decoders.guf>`

Each helper returns a :class:`decoder specification <qldpc.decoders.construction.specs.DecoderSpec>`, which stores construction options but does not yet build a decoder.
See `Decoder specification reference`_ for the options of each helper.
The signatures list known options explicitly for autocomplete and static analysis, and a misspelled option raises a ``TypeError`` when the specification is created.
The helpers for ldpc, PyMatching, and Relay-BP also accept a ``backend_options`` mapping, which forwards options that they do not list to the backend unchecked, as in ``decoders.bp_osd(max_iter=30, backend_options={"input_vector_type": "syndrome"})``.
An option that a helper lists must be passed by name rather than in ``backend_options``.
The ldpc and Relay-BP backends reject unsupported names when the decoder is built.
PyMatching and ldpc's BP+LSD decoder would instead silently ignore them, so qLDPC rejects names absent from PyMatching's signature, and warns about BP+LSD options that it does not know.
The ``ilp`` helper forwards additional keyword options to ``cvxpy.Problem.solve``.
Passing ``decoder=None`` uses qLDPC's default: BP+OSD for binary inputs and generalized union-find for nonbinary field arrays.
Install every optional decoder available on the current platform with ``pip install 'qldpc[decoders]'``.
PyMatching is included in the base qLDPC installation.

Building decoders
-----------------

The typed helpers above return decoder specifications, which do not build a decoder until a matrix or detector error model is supplied.
A specification with ``infers_errors=True`` builds an error decoder with ``.build(pcm_or_dem)``, and any specification builds an observable decoder for a detector error model with ``.build_observable_decoder(dem)``.
For example, ``decoders.bp_lsd(max_iter=30)`` returns a reusable decoder specification, and ``decoders.bp_lsd(max_iter=30).build(code.matrix)`` builds a decoder for ``code``.
Higher-level APIs that accept ``decoder=`` call these methods themselves.

Tesseract is an optional binary decoder.
Install qLDPC with the ``tesseract`` extra to use it: ``pip install 'qldpc[tesseract]'``.
The pinned upstream release provides wheels for CPython 3.12--3.14 on macOS arm64 and Linux x86-64.
A parity-check matrix is converted to a detector error model with probabilities supplied by ``error_channel``; a detector error model supplies its own probabilities and observable definitions.
By default, Tesseract merges the interchangeable error mechanisms of a detector error model, but not identical columns of a parity-check matrix, for which merging could report a column other than the most likely one.
Set ``merge_errors`` to override this choice.
Tesseract reports a low-confidence result if its search does not converge within its configured beam or priority-queue limits.
Set ``add_erasure_bit=True`` to expose that result as qLDPC's appended erasure flag.
For upstream's named Sinter configurations, use ``decoders.tesseract_preset(...)``:

.. code-block:: python

   long_beam = decoders.tesseract_preset()
   short_beam = decoders.tesseract_preset("short-beam")
   surface_like = decoders.tesseract_preset("long-beam", sparsify="surface-code-like")
   color_like = decoders.tesseract_preset("short-beam", sparsify="color-code-like")

The default long-beam family matches upstream's named ``tesseract`` alias and performs a stronger,
more expensive search than short beam.
Use ``decoders.tesseract(...)`` to specify custom options.

Higher-level APIs accept decoder specifications:

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

* An error-decoder constructor, a specification without native observable prediction (such as ``decoders.bp_osd(...)``), or, where accepted, a prebuilt error decoder infers a physical error; the logical operators flipped by that error are its prediction.
  This is also the default for ``decoder=None`` (BP+OSD for binary codes, GUF for nonbinary ones).
  On nonbinary codes, a specification that can infer errors uses this path even if it also supports native binary observable decoding.
  Prebuilt decoders with both methods also use this path: their observable map may not match the sector being decoded.
* A specification with native observable prediction, such as ``decoders.lookup(max_weight=2)``, builds an observable decoder from the code-capacity detector error model, even if it can also infer physical errors.
  An observable-only specification, such as ``decoders.frontier(...)``, requires this model.
* A Sinter-style decoder, such as ``decoders.SinterDecoder(decoder=decoders.lookup(max_weight=2))``, is compiled for a code-capacity detector error model whose detectors are the stabilizers (or parity checks) of the code and whose observables are its logical operators (or, for a classical code, its bits).
  A shared Sinter-style decoder is compiled separately for each CSS sector.
  Stim detector error models are binary, so such a decoder is rejected for a code over another field.
  A factory wrapped with ``decoders.from_dem(factory)`` is built from the same detector error model.
  Bare callables are treated as error-decoder constructors and receive the parity-check matrix.
* A prebuilt observable decoder built for a CSS sector's stabilizers and logical operators predicts logical flips directly, over any field.
  Detector, observable, and field metadata is validated when a decoder exposes it.
  Built-in observable decoders expose this metadata; a raw precompiled decoder that only provides Sinter's bit-packed interface must do so as well.

The low-level detector-error-model and decoder orchestration is in :mod:`qldpc.codes.code_capacity`; the fixed-weight sampling statistics are in :mod:`qldpc.codes.monte_carlo`.

The two kinds can be mixed across CSS sectors:

.. code-block:: python

   estimator = css_code.get_logical_error_rate_func(
       num_samples=10_000,
       decoder_x=decoders.bp_lsd(max_iter=30),
       decoder_z=decoders.SinterDecoder(decoder=decoders.lookup(max_weight=2)),
   )

Direct observable decoding can lower the estimated logical error rate of a degenerate code, because the most likely logical class of an error need not contain the most likely individual error.
The predictions of every decoder are validated: an inferred error or predicted observable vector with the wrong length, entries outside the field of the code, or malformed erasure flags raises an error, as does a prebuilt or compiled decoder built for a different number of detectors or observables.
Decoder-based distance bounds still require error decoders, since they inspect the weights of the errors that a decoder infers.

Predicting observable flips
---------------------------

:meth:`DecoderSpec.build_observable_decoder <qldpc.decoders.construction.specs.DecoderSpec.build_observable_decoder>` builds an observable decoder for a detector error model:

.. code-block:: python

   observable_decoder = decoders.mwpm().build_observable_decoder(dem)
   predicted_flips = observable_decoder.decode_observables(syndrome)

Some decoders can predict observable flips natively, without first inferring an error.
Their specifications build a native observable decoder wherever observable flips are wanted:

* ``mwpm`` builds a PyMatching decoder that tracks observables along matched paths;
* ``frontier`` builds a :class:`FrontierObservableDecoder <qldpc.decoders.external.frontier.FrontierObservableDecoder>`, described below;
* ``relay_bp`` and ``min_sum_bp`` build a :class:`RelayBPDecoder <qldpc.decoders.external.relay_bp.RelayBPDecoder>`;
* ``tesseract`` builds a :class:`TesseractDecoder <qldpc.decoders.external.tesseract.TesseractDecoder>` that natively predicts the observables of its detector error model; and
* ``lookup`` builds an :class:`ObservableLookupDecoder <qldpc.decoders.custom.lookup.ObservableLookupDecoder>`, which maps each syndrome directly to its most likely observable flip.

Specifications for other decoders build an error decoder, whose inferred errors are converted into observable flips.
``DecoderSpec.predicts_observables_natively`` reports which of these applies.
Conversely, ``DecoderSpec.infers_errors`` reports whether a specification can build an error decoder at all.

Frontier
~~~~~~~~

`Frontier <https://github.com/aleverrier/frontier>`_ is an approximate maximum-likelihood decoder that scans the error mechanisms of a detector error model, and uses pruned dynamic programming to predict the most likely observable flips.
Frontier cannot infer errors, so ``decoders.frontier(...).build(...)`` raises a ``TypeError``, as do methods that require an error decoder.
Its specification builds an observable decoder with ``.build_observable_decoder(dem)``, and is accepted wherever observable flips are wanted, such as by ``SinterDecoder`` and code-capacity estimators.

.. code-block:: python

   frontier_spec = decoders.frontier(K=512, Delta=12, committee=True, add_erasure_bit=True)
   observable_decoder = frontier_spec.build_observable_decoder(dem)
   predicted_flips = observable_decoder.decode_observables(syndrome)

   sinter_decoder = decoders.SinterDecoder(decoder=frontier_spec)

``K`` and ``Delta`` control how aggressively Frontier prunes; larger values are slower and more accurate.
By default, Frontier reorders error mechanisms so that detectors are resolved early; ``column_order="time_order"`` keeps the order of the detector error model.
``committee=True`` also scans the error mechanisms in reverse order, and keeps the better-supported of the two predictions.
If no error that Frontier keeps is consistent with a syndrome, it predicts no observable flips; with ``add_erasure_bit=True``, it also sets an erasure flag.
See :func:`decoders.frontier <qldpc.decoders.frontier>` for all options.

Frontier is not published on PyPI, so it is not a qLDPC dependency.
If it is missing, building a Frontier decoder raises an error that shows how to install the version that qLDPC is tested against.

For Sinter, wrap a decoder specification (or a constructor) in a :class:`decoders.SinterDecoder <qldpc.decoders.sinter.core.SinterDecoder>`, or in one of its subclasses:

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

To write a custom decoder or contribute a new backend, follow :doc:`Adding a decoder <adding_decoders>`.

A custom error decoder subclasses :class:`decoders.ErrorDecoder <qldpc.decoders.protocols.ErrorDecoder>` and implements ``decode_errors``, which maps a syndrome to an inferred error.
The subclass inherits ``decode`` as an alias for ``decode_errors``.
A custom observable decoder implements ``decode_observables``, which maps a syndrome to predicted observable flips, to satisfy the :class:`decoders.ObservableDecoder <qldpc.decoders.protocols.ObservableDecoder>` protocol.
A custom decoder may also define:

* ``decode_errors_batch`` or ``decode_observables_batch``, which decode a two-dimensional array of syndromes (one per row), to satisfy :class:`decoders.BatchErrorDecoder <qldpc.decoders.protocols.BatchErrorDecoder>` or :class:`decoders.BatchObservableDecoder <qldpc.decoders.protocols.BatchObservableDecoder>`, so that Sinter decoders decode shots in batches; and
* ``has_erasure_bit = True``, to declare that it appends an erasure flag to each inferred error or predicted observable flip.

Methods that use an error decoder also accept any object whose ``decode`` method returns an inferred error, such as a decoder built directly with the ldpc package, and wrap it in a :class:`decoders.WrappedErrorDecoder <qldpc.decoders.protocols.WrappedErrorDecoder>`.
Decoder specifications build instances of the backend decoder classes, defined by their integration modules; for example, ``decoders.bp_osd().build(pcm)`` returns an ``ldpc.BpOsdDecoder`` that is also an ``ErrorDecoder``.

External integration classes have qualified paths such as :class:`RelayBPDecoder <qldpc.decoders.external.relay_bp.RelayBPDecoder>` and :class:`TesseractDecoder <qldpc.decoders.external.tesseract.TesseractDecoder>`.
qLDPC's own implementation classes live in the :mod:`qldpc.decoders.custom` package, such as :class:`ILPDecoder <qldpc.decoders.custom.ilp.ILPDecoder>`, :class:`GUFDecoder <qldpc.decoders.custom.guf.GUFDecoder>`, :class:`CompositeDecoder <qldpc.decoders.custom.composition.CompositeDecoder>`, and :class:`DirectDecoder <qldpc.decoders.custom.composition.DirectDecoder>`.
:class:`ErrorsToObservablesDecoder <qldpc.decoders.adapters.error_decoders.ErrorsToObservablesDecoder>` and :class:`ExpandedErrorDecoder <qldpc.decoders.adapters.error_decoders.ExpandedErrorDecoder>` are the conversion adapters.

Besides a ``DecoderSpec``, the ``decoder=`` argument accepts:

* a constructor, such as a decoder class, or any other callable that builds a decoder from a parity-check matrix or detector error model;
* ``decoders.from_matrix(factory)`` for a custom error decoder that requires a parity-check matrix, or ``decoders.from_dem(factory)`` for a custom observable decoder that requires a binary detector error model (including in code-capacity estimators);
* where observable flips are predicted for a detector error model, an observable-decoder compiler (see :class:`decoders.ObservableDecoderCompiler <qldpc.decoders.construction.specs.ObservableDecoderCompiler>`), such as a ``SinterDecoder``, which is compiled for that model; or
* a prebuilt decoder, which is used as is.

A prebuilt decoder is tied to the matrix used to construct it, so it is only accepted where the caller knows the matrix being decoded: by the code-capacity estimators of classical codes; by ``ClassicalCode.get_distance_bound`` when given a ``vector`` (whose syndrome is computed with the parity check matrix of the code); and per sector (as ``decoder_x=`` and ``decoder_z=``) by the code-capacity estimators of CSS codes.
A shared prebuilt ``decoder=`` for a CSS code is rejected unless its two stabilizer matrices are equal (and, for a prebuilt observable decoder, so are its two sets of logical operators).

Some methods decode a matrix that they construct internally, and therefore reject prebuilt decoders: decoder-based distance bounds of codes (other than a classical distance bound to a ``vector``), logical-operator reduction, the code-capacity estimators of non-CSS codes, and Sinter decoders, which build a new decoder for every (simplified) detector error model, window, or subgraph that they decode.
These methods accept a ``DecoderSpec`` or a constructor.
A constructor can fix custom options with ``functools.partial`` or a ``lambda``.
A ``SinterDecoder``, a ``SubgraphDecoder``, and the code-capacity estimators also accept an observable-decoder compiler, which they compile for each detector error model that they decode.
The two factory wrappers declare the expected input explicitly; they do not convert between matrices and detector error models.

Detailed decode results
-----------------------

Decoders that expose per-shot diagnostics have ``decode_errors_detailed`` or ``decode_observables_detailed`` methods.
These methods return a detailed result for one syndrome, while ``decode_errors`` and ``decode_observables`` return hard prediction arrays without building diagnostics:

.. code-block:: python

   decoder = decoders.frontier(K=512).build_observable_decoder(dem)
   assert isinstance(decoder, decoders.DetailedObservableDecoder)
   result = decoder.decode_observables_detailed(syndrome)
   print(result.observable_flips, result.erasure, result.diagnostics)

Relay-BP and PyMatching decoders also have ``decode_errors_detailed_batch`` and ``decode_observables_detailed_batch`` methods, which use their native batch decoding and return a tuple of detailed results in syndrome order.

An :class:`decoders.ErrorDecodeResult <qldpc.decoders.protocols.ErrorDecodeResult>` contains ``error``.
An :class:`decoders.ObservableDecodeResult <qldpc.decoders.protocols.ObservableDecodeResult>` contains ``observable_flips``.
Both also contain an ``erasure`` boolean and a namespaced ``diagnostics`` dictionary.
Each detailed-result prediction is the array that the corresponding hard method returns, which ends with an erasure bit only if erasure signaling is enabled.
The ``erasure`` field reports whether the decoder signals erasure, whether or not erasure signaling is enabled.

The available diagnostics depend on the decoder:

* Frontier reports its status, terminal logical-class log masses, log evidence, top-mass gap, search statistics, and committee results when provided by the backend.
* Relay-BP reports convergence, iterations, decoded detectors, and per-variable posterior ratios.
* PyMatching reports the matching objective weight for ordinary matching; correlated matching does not provide that weight through its batch API.
* Tesseract reports its low-confidence flag as ``erasure``.
* ldpc BP-family decoders report available convergence, iteration, and log-probability-ratio state.
* ILP reports solver status and a finite objective value when one is available.

An observable decoder built from an error decoder, for example by ``decoders.bp_osd().build_observable_decoder(dem)``, forwards the diagnostics of a detailed error decoder.
These values are deliberately not normalized into a common ``confidence`` field because posterior ratios, logical-class gaps, convergence flags, and optimization costs have different meanings and calibration.
A custom decoder can implement the optional :class:`decoders.DetailedErrorDecoder <qldpc.decoders.protocols.DetailedErrorDecoder>` or :class:`decoders.DetailedObservableDecoder <qldpc.decoders.protocols.DetailedObservableDecoder>` protocol.
Sinter's compiled decoder interface returns observable arrays.

Lookup-table outputs
--------------------

Use a lookup specification to build an error decoder that returns a representative physical error, or an observable decoder that returns the most likely observable flip for each syndrome.
With observable information, an error decoder may group candidate errors by logical effect before selecting a representative error.
For observable predictions from a detector error model:

.. code-block:: python

   observable_lookup = decoders.lookup(max_weight=2).build_observable_decoder(dem)
   predicted_flips = observable_lookup.decode_observables(syndrome)

Erasure-aware decoders append their erasure flag after the inferred error or observable vector.
Compiled Sinter decoders translate that flag into a discarded shot.

Migrating from qLDPC 0.3.3
--------------------------

This section describes how the decoder API differs from that of ``qldpc==0.3.3``, and how to update code written for it.

Breaking changes
~~~~~~~~~~~~~~~~

The following changes take effect without a deprecation period:

* The ``static_decoder`` argument has been removed, except from the deprecated ``decoders.get_decoder`` and ``decoders.decode``.
  Pass a prebuilt decoder as ``decoder=`` instead, where a prebuilt decoder is accepted (see above), and otherwise pass a decoder specification or a constructor.
* When the deprecated ``decoder_x_kwargs`` or ``decoder_z_kwargs`` of a CSS code set the same option as its shared keyword arguments, the sector-specific value now takes precedence, just as ``decoder_x=`` and ``decoder_z=`` take precedence over ``decoder=``.
* A ``SinterDecoder`` whose specification supports native observable prediction (``mwpm``, ``relay_bp``, ``min_sum_bp``, and ``lookup``) now uses it.
  The predicted observable flips are unchanged.
  For ``mwpm`` and ``lookup`` specifications, however, the ``decoder`` attribute of the resulting ``CompiledSinterDecoder`` is now a decoder that predicts observable flips rather than errors.
  For other specifications, the ``decoder`` attribute remains the decoder that the specification builds.

Deprecated usage
~~~~~~~~~~~~~~~~

The keyword-based decoder API of ``qldpc==0.3.3`` remains available during a deprecation period, and each use emits a ``DeprecationWarning`` that names its replacement.
Defunct Sinter ``decode`` methods have been removed; use ``decode_observables`` on a compiled decoder.
Compatibility is provided for the names exported from the package root, ``qldpc.decoders``, in that release; internal module paths are not part of this guarantee.
In particular, ``decoders.get_decoder`` and ``decoders.decode`` behave as they did in ``qldpc==0.3.3``, except that ``error_rate`` and ``error_channel`` are rejected for a detector error model (see `Migrating from qLDPC 0.4.0`_):

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
   * - ``decoders.get_decoder_BP_OSD(pcm_or_dem, ...)`` (and the other uppercase builder names)
     - ``decoders.bp_osd(...).build(pcm_or_dem)`` (and the corresponding specification helper)
   * - ``decoder_x_kwargs={...}`` and ``decoder_z_kwargs={...}``
     - ``decoder_x=...`` and ``decoder_z=...``
   * - ``LookupDecoder(..., predict_observable_flips=True)``
     - ``decoders.lookup(...).build_observable_decoder(dem)`` for a DEM.  For a matrix with an
       ``observable_flip_matrix`` and ``error_channel``, use
       ``qldpc.decoders.custom.lookup.ObservableLookupDecoder`` and ``decode_observables``;
       the specification's observable builder requires a DEM.
   * - ``WeightedLookupDecoder(..., predict_observable_flips=True)``
     - ``qldpc.decoders.custom.lookup.WeightedObservableLookupDecoder(...)`` and its
       ``decode_observables`` method for late-bound penalty functions; the ordinary lookup
       specification does not support choosing a new penalty at decode time.
   * - ``LookupDecoder(..., penalty_func=penalty)``
     - ``decoders.lookup(max_weight=..., error_channel=log_probability)`` with a callable
       returning a normalized log probability.  An arbitrary penalty cannot be substituted
       without first turning it into a valid probability distribution.
   * - ``decoders.bp_osd(error_rate=p)`` (also ``bp_lsd``, ``bf``, and ``tesseract``)
     - ``decoders.bp_osd(error_channel=p)`` (or the corresponding specification helper)
   * - ``SinterDecoder.decode`` and ``CompiledSinterDecoder.decode`` (removed)
     - Compile the ``SinterDecoder``, then call ``CompiledSinterDecoder.decode_observables``
   * - ``SubgraphSinterDecoder`` and ``SequentialSinterDecoder``
     - ``SubgraphDecoder`` and ``SequentialWindowDecoder``
   * - ``Decoder`` and ``BatchDecoder``
     - ``ErrorDecoder`` and ``BatchErrorDecoder``

Migrating from qLDPC 0.4.0
--------------------------

This section describes how the decoder API differs from that of ``qldpc==0.4.0``.

The following changes take effect without a deprecation period:

* The subpackages ``qldpc.decoders.construction``, ``qldpc.decoders.custom``, and ``qldpc.decoders.external``, and the module ``qldpc.decoders.construction.specs``, no longer re-export decoder builders, specification helpers, or resolution functions.
  Import these names from ``qldpc.decoders`` instead.
* Builders that were not exported from ``qldpc.decoders``, such as ``get_error_decoder_mwpm``, ``get_observable_decoder_mwpm``, ``get_relay_bp_decoder``, ``get_min_sum_bp_decoder``, and ``get_observable_decoder_lookup``, have been removed.
  Use decoder specifications and their ``build`` and ``build_observable_decoder`` methods instead.
* A detector error model supplies its own error probabilities, so passing ``error_channel`` or ``error_rate`` together with a detector error model raises a ``ValueError``.
  Previously, the ldpc decoders (BP+OSD, BP+LSD, and BF) ignored ``error_rate`` for a detector error model, and let ``error_channel`` override its probabilities.
  This affects every path that builds such a decoder for a detector error model, including the deprecated keyword arguments: for example, ``decoders.SinterDecoder(error_rate=p)``, ``decoders.SinterDecoder(decoder=decoders.bp_osd(error_rate=p))``, and ``decoders.get_decoder(dem, with_BP_OSD=True, error_rate=p)`` now fail when the decoder is built for a detector error model.
  Remove the option to use the probabilities of the detector error model, as a ``SinterDecoder`` should.
  To override them, decode the detector-flip matrix of the model instead, as in ``decoders.bp_osd(error_channel=p).build(decoders.DetectorErrorModelArrays(dem).detector_flip_matrix)``.
* The deprecated builder ``get_decoder_mwpm`` and ``get_decoder(..., with_MWPM=True)`` now reject options that PyMatching would silently ignore, and reserve ``faults_matrix`` for ``decoders.mwpm(...).build_observable_decoder(dem)``.
  The corresponding paths for BP+LSD warn about options that ldpc's BP+LSD decoder would silently ignore.

Specification helpers accept the options that they list, and helpers for ldpc, PyMatching, and Relay-BP accept other backend options in a ``backend_options`` mapping (see `Configuring decoders`_).
Deprecated builders such as ``get_decoder_bp_osd(pcm_or_dem, **options)`` still accept backend options as keyword arguments.

The following usage remains available during a deprecation period, and each use emits a ``DeprecationWarning`` that names its replacement:

.. list-table::
   :header-rows: 1

   * - Deprecated usage
     - Replacement
   * - ``error_rate=p``
     - ``error_channel=p``
   * - ``decoders.get_decoder_bp_osd(pcm_or_dem, ...)`` (and the other ``get_decoder_<NAME>`` builders)
     - ``decoders.bp_osd(...).build(pcm_or_dem)`` (and the corresponding specification helper)
   * - ``decoders.get_decoder_rbp(pcm_or_dem, ...)``
     - ``decoders.relay_bp(...).build(pcm_or_dem)`` or ``decoders.min_sum_bp(...).build(pcm_or_dem)``
   * - ``decoders.get_error_decoder(pcm_or_dem, decoder=spec)``
     - ``spec.build(pcm_or_dem)``
   * - ``decoders.get_observable_decoder(dem, decoder=spec)``
     - ``spec.build_observable_decoder(dem)``
   * - ``decoders.get_observable_decoder(dem, decoder=error_decoder)``
     - ``decoders.ErrorsToObservablesDecoder(error_decoder, dem)``
   * - ``decoders.decode_observables(dem, syndrome, decoder=spec)``
     - ``spec.build_observable_decoder(dem).decode_observables(syndrome)``
   * - ``decoders.lookup_table(...)``
     - ``decoders.lookup(...)``

Migrating custom code-capacity factories
----------------------------------------

Custom code-capacity factories that relied on a return annotation to receive a detector error model
must now be wrapped with ``decoders.from_dem(factory)``.  Bare factories receive the parity-check
matrix and must build error decoders; use ``decoders.from_matrix(factory)`` to declare a matrix-only
factory explicitly.  Annotations no longer control this routing, without a deprecation period.

Migrating backend class imports
-------------------------------

Backend implementation classes formerly exported from ``qldpc.decoders``, such as
``LookupDecoder`` and ``RelayBPDecoder``, remain importable at the package root, including
through star imports, but now warn on access.  Workflow classes such as ``SinterDecoder``
remain direct root exports.  Import backend classes from their defining modules instead:
``from qldpc.decoders.custom.lookup import LookupDecoder`` or
``from qldpc.decoders.external.relay_bp import RelayBPDecoder``.  The objects and their
constructor signatures are unchanged.

Decoder specification reference
-------------------------------

.. autofunction:: qldpc.decoders.bp_osd
.. autofunction:: qldpc.decoders.bp_lsd
.. autofunction:: qldpc.decoders.bf
.. autofunction:: qldpc.decoders.mwpm
.. autofunction:: qldpc.decoders.frontier
.. autofunction:: qldpc.decoders.relay_bp
.. autofunction:: qldpc.decoders.min_sum_bp
.. autofunction:: qldpc.decoders.tesseract
.. autofunction:: qldpc.decoders.tesseract_preset
.. autofunction:: qldpc.decoders.lookup
.. autofunction:: qldpc.decoders.ilp
.. autofunction:: qldpc.decoders.guf
