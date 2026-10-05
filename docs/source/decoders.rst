Choosing a decoder
==================

Decoder-based distance bounds and logical error rate estimates require a decoder.
qLDPC lets you choose a decoder by creating a :class:`decoder specification <qldpc.decoders.construction.specs.DecoderSpec>`, such as ``qldpc.decoders.bp_osd(...)``, which defines a decoding algorithm and its options.
Pass that specification as ``decoder=`` to let a method build its decoder, or build one yourself for a fixed parity check matrix or detector error model (see below).

The :doc:`decoders example notebook <examples/decoders>` walks through the workflows on this page.

Configuring decoders
--------------------

For example, build a decoder from a specification for a parity-check matrix:

.. code-block:: python

   import numpy as np
   import stim

   from qldpc import codes, decoders

   code = codes.RepetitionCode(5)
   syndrome = np.array([1, 0, 0, 0])

   error_decoder = decoders.bp_lsd(max_iter=30, bp_method="minimum_sum").build(code.matrix)
   correction = error_decoder.decode_errors(syndrome)

See `Decoder specification reference`_ for available helpers and their options, including backend-specific options.
Passing ``decoder=None`` uses qLDPC's default (subject to change in the future): BP+OSD for binary codes and generalized union-find for codes over nonbinary finite fields.
``pip install 'qldpc[decoders]'`` installs all optional decoders published as qLDPC extras for your platform.
You can install only one extra with, for example, ``pip install 'qldpc[relay-bp]'``.
Frontier must be installed separately; see `Backend limitations`_.

Building decoders
-----------------

Use an error-inference specification with ``.build(matrix)`` to infer errors from a parity-check matrix, or ``.build_observable_decoder(dem)`` to predict observable flips from a detector error model.

.. code-block:: python

   dem = stim.DetectorErrorModel("error(0.1) D0 D1 L0\n error(0.1) D1")
   observable_decoder = decoders.mwpm().build_observable_decoder(dem)
   events = np.array([1, 0])
   predicted_flips = observable_decoder.decode_observables(events)

A specification can be reused for compatible inputs; qLDPC methods with a ``decoder=`` argument accept a decoder specification, and build a decoder as necessary.
A detector error model supplies its own error probabilities and observable definitions, so a decoder specification must not override its ``error_channel``.
For a parity-check matrix, ``error_channel`` sets the matrix-column error probabilities when the decoder supports them.

Code-capacity estimates
-----------------------

A code-capacity estimate samples errors and counts a failure when the decoder mispredicts their logical effect.
Pass a decoder specification such as ``decoders.mwpm()`` as ``decoder=``.
For a binary code, a specification that predicts observable flips natively is built for the code-capacity detector error model.
If that build raises a ``ValueError``, a ``decoders.ObservableDecodingFallbackWarning`` is emitted and the specification instead builds an error decoder for the syndrome matrix.
Pass ``decoder=decoders.from_matrix(spec.build)`` to select error decoding explicitly, or ``decoder=decoders.from_dem(factory)`` for a custom observable decoder.

.. code-block:: python

   code = codes.SurfaceCode(3)
   estimator = code.get_logical_error_rate_func(
       num_samples=10_000,
       decoder=decoders.mwpm(),
   )
   logical_error_rate, stderr = estimator(np.logspace(-4, -2, 5))

Predicting observable flips
---------------------------

An :class:`error decoder <qldpc.decoders.protocols.ErrorDecoder>` infers a physical error with ``decode_errors``; an :class:`observable decoder <qldpc.decoders.protocols.ObservableDecoder>` predicts flips with ``decode_observables``.
Specifications such as ``decoders.lookup(...)``, ``decoders.mwpm()``, ``decoders.relay_bp()``, and ``decoders.tesseract()`` can predict observable flips natively.
Other specifications infer errors and project them onto the observables defined by the detector error model.
``DecoderSpec.predicts_observables_natively`` identifies native observable prediction support; ``DecoderSpec.infers_errors`` identifies specifications that can build error decoders.

Using decoders with Sinter
--------------------------

For Sinter, pass a decoder specification to a :class:`decoders.SinterDecoder <qldpc.decoders.sinter.core.SinterDecoder>` or a subclass such as the :class:`decoders.SlidingWindowDecoder <qldpc.decoders.sinter.window.SlidingWindowDecoder>`:

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
   assert isinstance(sliding_decoder, decoders.SinterDecoder)

Pass the configured decoder to Sinter under the same name used by the task:

.. code-block:: python

   import sinter
   import stim

   circuit = stim.Circuit.generated(
       "repetition_code:memory", distance=3, rounds=3, after_clifford_depolarization=0.01
   )
   stats = sinter.collect(
       tasks=[sinter.Task(circuit=circuit, decoder="qldpc-mwpm")],
       custom_decoders={"qldpc-mwpm": sinter_decoder},
       num_workers=1,
       max_shots=100,
       print_progress=False,
   )

A ``SinterDecoder`` is compiled for a detector error model before it predicts flips.
Its compiled form, which ``compile_decoder_for_dem`` returns and which is not meant to be constructed directly, is a :class:`~qldpc.decoders.sinter.core.CompiledSinterDecoder` that exposes ``decode_observables`` for one shot and ``decode_shots`` for a batch.
Window decoders require a specification that can infer errors because they commit physical corrections in each window.

Backend limitations
-------------------

Frontier
~~~~~~~~

`Frontier <https://github.com/aleverrier/frontier>`_ is an approximate maximum-likelihood decoder that scans the error mechanisms of a detector error model, and uses pruned dynamic programming to predict the most likely observable flips.
Frontier cannot infer errors, so ``decoders.frontier(...).build(...)`` raises a ``TypeError``, as do methods that require an error decoder.
Its specification builds an observable decoder with ``.build_observable_decoder(dem)``, and is accepted wherever observable flips are wanted, including Sinter and code-capacity estimators.

.. code-block:: python

   frontier_spec = decoders.frontier(K=512, Delta=12, committee=True, add_erasure_bit=True)
   observable_decoder = frontier_spec.build_observable_decoder(dem)
   predicted_flips = observable_decoder.decode_observables(events)

``K`` and ``Delta`` control how aggressively Frontier prunes; larger values are slower and more accurate.
By default, Frontier reorders error mechanisms so that detectors are resolved early; ``column_order="time_order"`` keeps the order of the detector error model.
``committee=True`` also scans the error mechanisms in reverse order, and keeps the better-supported of the two predictions.
If no error that Frontier keeps is consistent with a syndrome, it predicts no observable flips; with ``add_erasure_bit=True``, it also sets an erasure flag.
See :func:`decoders.frontier <qldpc.decoders.frontier>` for all options.

Frontier is not published on PyPI, so it is not a qLDPC dependency or an installable extra.
If it is missing, building a Frontier decoder raises an error that shows how to install the version that qLDPC is tested against.

Tesseract
~~~~~~~~~

Tesseract is an optional binary decoder; install qLDPC with the ``tesseract`` extra to use it: ``pip install 'qldpc[tesseract]'``.
The pinned upstream release provides wheels for CPython 3.12--3.14 on macOS arm64 and Linux x86-64.
A parity-check matrix uses probabilities supplied by ``error_channel``; a detector error model supplies its own probabilities and observable definitions.
By default, Tesseract merges interchangeable error mechanisms of a detector error model, but not identical columns of a parity-check matrix, for which merging could report a column other than the most likely one.
Set ``merge_errors`` to override this choice.
Tesseract reports a low-confidence result if its search does not converge within its configured beam or priority-queue limits; set ``add_erasure_bit=True`` to expose it as qLDPC's appended erasure flag.
Upstream's named Sinter configurations are available as specifications:

.. code-block:: python

   long_beam = decoders.tesseract_preset()
   short_beam = decoders.tesseract_preset("short-beam")

The default long-beam family matches upstream's named ``tesseract`` alias and performs a stronger, more expensive search than short beam.
Use ``decoders.tesseract(...)`` for custom options.

Adding your own decoder specification
-------------------------------------

To use a custom decoder or contribute a backend, define its construction signature and wrap it in a decoder specification.
The same ``.build(...)``, ``.build_observable_decoder(dem)``, and ``decoder=spec`` workflow then applies.
The :doc:`adding a decoder guide <adding_decoders>` covers the error and observable contracts, optional batch and erasure support, and a tested example.
The generated :doc:`API reference <autoapi/index>` documents the concrete backend classes and low-level adapters.

Detailed decode results
-----------------------

Decoders that expose per-shot diagnostics have ``decode_errors_detailed`` or ``decode_observables_detailed`` methods.
These methods return a detailed result for one syndrome, while ``decode_errors`` and ``decode_observables`` return hard prediction arrays without building diagnostics:

.. code-block:: python

   decoder = decoders.frontier(K=512).build_observable_decoder(dem)
   assert isinstance(decoder, decoders.DetailedObservableDecoder)
   result = decoder.decode_observables_detailed(events)
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
   predicted_flips = observable_lookup.decode_observables(events)

Erasure-aware decoders append their erasure flag after the inferred error or observable vector.
Compiled Sinter decoders translate that flag into a discarded shot.

Migrating from qLDPC 0.3.3
--------------------------

This section describes how the decoder API differs from that of ``qldpc==0.3.3``, and how to update code written for it.

Breaking changes
~~~~~~~~~~~~~~~~

The following changes take effect without a deprecation period:

* The ``static_decoder`` argument has been removed, except from the deprecated ``decoders.get_decoder`` and ``decoders.decode``.
  Pass a decoder specification as ``decoder=`` instead.
  Prebuilt decoders remain accepted where the caller knows the matrix being decoded, but cannot be rebuilt for internally constructed matrices or Sinter detector error models.
* When the deprecated ``decoder_x_kwargs`` or ``decoder_z_kwargs`` of a CSS code set the same option as its shared keyword arguments, the sector-specific value now takes precedence, just as ``decoder_x=`` and ``decoder_z=`` take precedence over ``decoder=``.
* A ``SinterDecoder`` whose specification supports native observable prediction (``mwpm``, ``relay_bp``, ``min_sum_bp``, and ``lookup``) now uses it.
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
     - ``decoders.lookup(...).build_observable_decoder(dem)`` for a DEM.
       For a matrix with an ``observable_flip_matrix`` and ``error_channel``, use ``qldpc.decoders.custom.lookup.ObservableLookupDecoder`` and ``decode_observables``; the specification's observable builder requires a DEM.
   * - ``WeightedLookupDecoder(..., predict_observable_flips=True)``
     - ``qldpc.decoders.custom.lookup.WeightedObservableLookupDecoder(...)`` and its ``decode_observables`` method for late-bound penalty functions; the ordinary lookup specification does not support choosing a new penalty at decode time.
   * - ``LookupDecoder(..., penalty_func=penalty)``
     - ``decoders.lookup(max_weight=..., error_channel=log_probability)`` with a callable returning a normalized log probability.
       An arbitrary penalty cannot be substituted without first turning it into a valid probability distribution.
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

Specification helpers accept the options that they list, and helpers for ldpc, PyMatching, and Relay-BP accept other backend options in a ``backend_options`` mapping (see `Decoder specification reference`_).
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

Custom code-capacity factories that relied on a return annotation to receive a detector error model must now be wrapped with ``decoders.from_dem(factory)``, which returns a :class:`decoders.DEMDecoderFactory <qldpc.decoders.construction.factories.DEMDecoderFactory>`.
Bare factories receive the parity-check matrix and must build error decoders; use ``decoders.from_matrix(factory)``, which returns a :class:`decoders.MatrixDecoderFactory <qldpc.decoders.construction.factories.MatrixDecoderFactory>`, to declare a matrix-only factory explicitly.
Annotations no longer control this routing, without a deprecation period.

Migrating backend class imports
-------------------------------

Backend implementation classes formerly exported from ``qldpc.decoders``, such as ``LookupDecoder`` and ``RelayBPDecoder``, remain importable at the package root, including through star imports, but now warn on access.
Workflow classes such as ``SinterDecoder`` remain direct root exports.
The compiled Sinter decoders, such as ``CompiledSinterDecoder``, also warn at the package root; they are returned by ``compile_decoder_for_dem`` rather than constructed directly, and remain available from ``qldpc.decoders.sinter``.
Import backend classes from their defining modules instead: ``from qldpc.decoders.custom.lookup import LookupDecoder`` or ``from qldpc.decoders.external.relay_bp import RelayBPDecoder``.
The objects and their constructor signatures are unchanged.

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
