Choosing a decoder
==================

Decoder-based distance bounds and logical error rate estimates require a decoder.
qLDPC lets you choose a decoder by creating a :class:`decoder specification <qldpc.decoders.construction.specs.DecoderSpec>`, such as ``spec = qldpc.decoders.bp_osd(...)``, which defines a decoding algorithm and its options.
Any method with a ``decoder=`` argument, such as :meth:`qldpc.codes.CSSCode.get_logical_error_rate_func <qldpc.codes.common.CSSCode.get_logical_error_rate_func>` (see, for example, the :doc:`code-capacity notebook <examples/logical_error_rates/1_code_capacity>`), accepts and uses a decoder specification to internally compile a decoder for a fixed parity check matrix or detector error model.
You can also pass your own decoder class or factory as ``decoder=``; see `Code-capacity logical error rate estimates`_ and `Using decoders with Sinter`_ for the inputs each workflow supplies.

The :doc:`decoders example notebook <examples/decoders>` walks through the workflows on this page.

Built-in decoders
-----------------

qLDPC provides the following methods to construct decoder specifications:

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

See, for example, ``help(qldpc.decoders.bp_osd)`` for configuration options.
Specification helpers accept the options that they list, and the helpers for ldpc, PyMatching, and Relay-BP decoders accept other backend options in a ``backend_options`` mapping.
``pip install 'qldpc[decoders]'`` installs all optional decoders published as qLDPC extras for your platform.
You can install individual extras with, for example, ``pip install 'qldpc[relay-bp]'``.
The ILP decoder needs the ``ilp`` extra, which installs ``cvxpy``: ``pip install 'qldpc[ilp]'``.
The Frontier decoder is not available on PyPI, and must be installed manually; see `Backend limitations`_.

The default decoder in qLDPC (with an argument of ``decoder=None``, where applicable), is BP+OSD for binary codes, and generalized union-find for codes over nonbinary finite fields.

Inferring errors
~~~~~~~~~~~~~~~~

An :class:`error decoder <qldpc.decoders.protocols.ErrorDecoder>` inferrs errors from a syndrome with ``.decode_errors``.
You can compile an error decoder for a parity check matrix with ``.build(parity_check_matrix)``:

.. code-block:: python

   import numpy as np

   from qldpc import codes, decoders

   code = codes.RepetitionCode(5)
   syndrome = np.array([1, 0, 0, 0])

   decoder_spec = decoders.bp_lsd(max_iter=30, bp_method="minimum_sum")
   error_decoder = decoder_spec.build(code.matrix)
   correction = error_decoder.decode_errors(syndrome)

Predicting observable flips
~~~~~~~~~~~~~~~~~~~~~~~~~~~

An :class:`observable decoder <qldpc.decoders.protocols.ObservableDecoder>` predicts observable flips with ``.decode_observables``.
Use ``.build_observable_decoder(dem)`` to compile an observable decoder for a detector error model:

.. code-block:: python

   import numpy as np
   import stim

   from qldpc import decoders

   dem = stim.DetectorErrorModel("error(0.1) D0 D1 L0\n error(0.1) D1")
   observable_decoder = decoders.mwpm().build_observable_decoder(dem)
   events = np.array([1, 0])
   predicted_flips = observable_decoder.decode_observables(events)

Specifications such as ``decoders.lookup(...)``, ``decoders.mwpm()``, ``decoders.relay_bp()``, and ``decoders.tesseract()`` can predict observable flips directly.
Other specifications infer errors under the hood, and return the induced observable flips.
Some decoders (such as Frontier) support observable predictions but not error inference.
``DecoderSpec.predicts_observables_natively`` identifies native observable prediction support; ``DecoderSpec.infers_errors`` identifies specifications that can build error decoders.

A detector error model supplies its own error probabilities, so passing ``error_channel`` together with a detector error model raises a ``ValueError``.
To override these probabilities, decode the detector-flip matrix of the model instead, as in ``decoders.bp_osd(error_channel=p).build(decoders.DetectorErrorModelArrays(dem).detector_flip_matrix)``.

Code-capacity logical error rate estimates
------------------------------------------

A code-capacity estimate samples bit/qubit/qudit errors and counts a failure when a decoder mispredicts their logical effect.
To use a built-in decoder family, you can pass a specification such as ``decoders.mwpm()``:

.. code-block:: python

   code = codes.SurfaceCode(3)
   estimator = code.get_logical_error_rate_func(
       num_samples=10_000,
       decoder=qldpc.decoders.mwpm(),
       seed=0,
   )
   logical_error_rate, stderr = estimator(np.logspace(-4, -2, 5))

The ``seed=`` keyword accepts an integer or a NumPy ``Generator``.
An integer starts a repeatable local stream of physical errors, while a supplied generator is consumed and advanced.
Explicitly seeded sampling is independent of NumPy's global random state and of decoder early exits, including erasure decisions.
With ``seed=None`` (the default), sampling retains its existing global-NumPy behavior, so ``np.random.seed(...)`` still applies.
The seed controls error sampling, not randomness inside a decoder; exact replay also requires deterministic decoder settings and the same software versions and experiment configuration.
Sampling happens when the estimator is constructed; evaluating it at different physical error rates reuses those samples.

To use your own decoder, pass a class or factory directly as ``decoder=MyErrorDecoder``.
This factory must accept a parity-check matrix and return an error decoder, with a ``.decode_errors`` method (as in `Inferring errors`_).
To instead build a custom observable decoder from a binary code-capacity detector error model (DEM), pass ``decoder=qldpc.decoders.from_dem(MyObservableDecoder)``.
Wrapping a factory with ``decoders.from_matrix(factory)`` is optional, and makes its matrix-only contract explicit; return annotations do not affect which input a factory receives.
Prebuilt decoders are accepted where the caller knows the matrix being decoded, but not where a method decodes a matrix that it constructs internally, or by a ``SinterDecoder``, which builds a decoder for each detector error model.

Using decoders with Sinter
--------------------------

Pass a built-in specification to a :class:`decoders.SinterDecoder <qldpc.decoders.sinter.core.SinterDecoder>` or a subclass such as the :class:`decoders.SlidingWindowDecoder <qldpc.decoders.sinter.window.SlidingWindowDecoder>`:

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
       tasks=[sinter.Task(circuit=circuit, decoder="mwpm")],
       custom_decoders={"mwpm": sinter_decoder},
       num_workers=1,
       max_shots=100,
       print_progress=False,
   )

When Sinter compiles a ``SinterDecoder``, its ``decoder=`` class or factory receives a ``stim.DetectorErrorModel``.
A ``SinterDecoder`` uses native observable prediction when its specification supports it, so the ``decoder`` attribute of the compiled decoder predicts observable flips for specifications such as ``decoders.mwpm()`` and ``decoders.lookup(...)``.
Pass your own observable-decoder class or factory directly as ``decoders.SinterDecoder(decoder=MyObservableDecoder)``.
See the :doc:`quantum-memory notebook <examples/logical_error_rates/2_quantum_memory_x_or_z>` for a circuit-level sampling example.

Note that the ``SlidingWindowDecoder`` must infer and commit physical corrections in each window, so an observable-only decoder such as Frontier will not work.
Use a built-in error-decoding specification or a custom error-decoder factory that accepts each window's DEM.

The :doc:`adding a decoder guide <adding_decoders>` covers custom decoder contracts, optional batch and erasure support, and tested examples.
The generated :doc:`API reference <autoapi/index>` documents concrete backend classes and low-level adapters.
Import backend classes from ``qldpc.decoders.custom`` or ``qldpc.decoders.external``, as in ``from qldpc.decoders.custom import ErrorLookupDecoder``, and compiled Sinter decoders from ``qldpc.decoders.sinter``.

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
By default, Tesseract merges interchangeable error mechanisms of a detector error model, but not identical columns of a parity-check matrix, for which merging could report a column other than the most likely one.
Set ``merge_errors`` to override this choice.
Tesseract reports a low-confidence result if its search does not converge within its configured beam or priority-queue limits; set ``add_erasure_bit=True`` to expose it as qLDPC's appended erasure flag.
Named Sinter configurations of ``tessarect`` are available as named decoder specifications:

.. code-block:: python

   long_beam = decoders.tesseract_preset()
   short_beam = decoders.tesseract_preset("short-beam")

The default long-beam family matches upstream's named ``tesseract`` alias and performs a stronger, more expensive search than short beam.
Use ``decoders.tesseract(...)`` for more custom options.

Detailed decoding results
-------------------------

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
With observable information, an error decoder may group candidate errors by logical equivalence class before selecting a representative error.
For observable predictions from a detector error model:

.. code-block:: python

   observable_lookup = decoders.lookup(max_weight=2).build_observable_decoder(dem)
   predicted_flips = observable_lookup.decode_observables(events)

The observable builder of a lookup specification requires a detector error model.
For a parity-check matrix with an observable flip matrix, use ``qldpc.decoders.custom.ObservableLookupDecoder``, or ``qldpc.decoders.custom.WeightedObservableLookupDecoder`` to choose a penalty function at decode time.
To weight candidate errors otherwise, pass ``decoders.lookup(..., error_channel=log_probability)`` a callable that returns a normalized log probability.

Erasure-aware decoders append their erasure flag after the inferred error or observable vector.
Compiled Sinter decoders translate that flag into a discarded shot.
