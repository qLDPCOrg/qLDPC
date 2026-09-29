Choosing a decoder
==================

qLDPC separates two decoding tasks:

* an :class:`~qldpc.decoders.custom.ErrorDecoder` maps a parity-check syndrome to an inferred
  physical error; and
* an :class:`~qldpc.decoders.sinter.ObservableDecoder` compiles against a Stim detector error model
  and maps detection events to predicted observable flips.

This distinction matters when composing decoders. qLDPC's current code-capacity estimators consume
an error decoder, while Sinter simulations consume an observable decoder, which usually wraps an
error decoder and performs the error-to-observable conversion after decoding. Decoder-based
randomized distance bounds specifically require an error decoder because they operate on physical
candidate errors; exact distance calculations do not require a decoder.

Configuring error decoders
--------------------------

Use a typed helper to defer construction until the consuming method knows its parity-check matrix or
detector error model:

.. code-block:: python

   import numpy as np

   from qldpc import codes, decoders

   code = codes.RepetitionCode(5)
   syndrome = np.array([1, 0, 0, 0])

   settings = decoders.bp_lsd(max_iter=30, bp_method="ms")
   correction = decoders.decode(code.matrix, syndrome, decoder=settings)

The helpers are available directly under ``qldpc.decoders``:

* :func:`~qldpc.decoders.retrieval.bp_osd`
* :func:`~qldpc.decoders.retrieval.bp_lsd`
* :func:`~qldpc.decoders.retrieval.bf`
* :func:`~qldpc.decoders.retrieval.mwpm`
* :func:`~qldpc.decoders.retrieval.relay_bp`
* :func:`~qldpc.decoders.retrieval.lookup_table`
* :func:`~qldpc.decoders.retrieval.ilp`
* :func:`~qldpc.decoders.retrieval.guf`

Their signatures are the source of truth for algorithm-specific options and are visible to
autocomplete and static-analysis tools. Passing ``decoder=None`` retains qLDPC's default: BP+OSD
for binary inputs and generalized union-find for nonbinary field arrays.

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

A prebuilt error decoder or a custom callable that accepts the matrix or detector error model can
also be passed as ``decoder=``. A prebuilt decoder is tied to the matrix used to construct it. For CSS
methods that decode both sectors, pass separate prebuilt instances as ``decoder_x=`` and
``decoder_z=``; a shared ``decoder=`` must be deferred settings or a constructor unless the two
sector matrices are equal. Likewise, when bounding both CSS distances at once, use deferred settings
or a constructor; with prebuilt instances, call ``get_distance_bound`` separately for each
``pauli=`` sector.

Predicting observable flips
---------------------------

Wrap error-decoder settings in an observable decoder for Sinter:

.. code-block:: python

   observable_decoder = decoders.ObservableDecoder(
       decoder=decoders.mwpm(),
       decompose_errors=True,
   )

   sliding_decoder = decoders.SlidingWindowDecoder(
       window_size=5,
       stride=2,
       decoder=decoders.bp_lsd(max_iter=30),
   )

An :class:`~qldpc.decoders.sinter.ObservableDecoder` is compiled for a detector error model before
it predicts flips. Its compiled form exposes ``decode_observables`` for one shot and ``decode_shots``
for a batch. Window and subgraph decoders use the same explicit ``decoder=`` argument for their
inner error decoder.

Lookup-table outputs
--------------------

:class:`~qldpc.decoders.lookup.LookupDecoder` and
:class:`~qldpc.decoders.lookup.WeightedLookupDecoder` always expose error-decoding APIs. They may
still use an observable-flip matrix to group candidate errors by logical effect, but their output is
a representative physical error.

Use :class:`~qldpc.decoders.lookup.ObservableLookupDecoder` or
:class:`~qldpc.decoders.lookup.WeightedObservableLookupDecoder` when the desired output is the
observable flip itself:

.. code-block:: python

   observable_lookup = decoders.ObservableLookupDecoder(dem, max_weight=2)
   predicted_flips = observable_lookup.decode_observables(syndrome)

Erasure-aware decoders append their erasure flag after the inferred error or observable vector.
Sinter-compatible compiled decoders translate that flag into a discarded shot.
