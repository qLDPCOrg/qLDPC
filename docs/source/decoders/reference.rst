Decoder reference
=================

The examples on this page reuse ``dem`` and ``events`` from the observable-decoding example in the :doc:`decoder guide <index>`.
The generated :doc:`API reference </autoapi/index>` documents concrete backend classes and low-level adapters.
Import backend classes from ``qldpc.decoders.custom`` or ``qldpc.decoders.external``, as in ``from qldpc.decoders.custom import ErrorLookupDecoder``.
Import compiled Sinter decoders from ``qldpc.decoders.sinter``.

.. _specification-helpers:

Specification helpers
---------------------

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

.. _backend-notes:

Backend notes
-------------

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
Upstream's named Sinter configurations of Tesseract are available as decoder specifications:

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

A :class:`decoders.ErrorDecodeResult <qldpc.decoders.protocols.ErrorDecodeResult>` contains ``error``.
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
