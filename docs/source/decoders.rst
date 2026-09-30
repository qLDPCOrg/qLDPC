Choosing a decoder
==================

qLDPC separates two decoding tasks:

* an :class:`~qldpc.decoders.custom.ErrorDecoder` maps a parity-check syndrome to an inferred
  physical error; and
* an :class:`~qldpc.decoders.sinter.ObservableDecoder` compiles against a Stim detector error model
  and maps detection events to predicted observable flips.

This distinction matters when composing decoders. qLDPC's code-capacity estimators consume an error
decoder, while Sinter simulations consume an observable decoder, which wraps an error decoder and
performs the error-to-observable conversion after decoding. Decoder-based randomized distance bounds
specifically require an error decoder because they operate on physical candidate errors; exact
distance calculations do not require a decoder.

The :doc:`decoders example notebook <examples/decoders>` walks through the workflows on this page.

Configuring error decoders
--------------------------

Use a typed helper to defer construction until the consuming method knows its parity-check matrix or
detector error model:

.. code-block:: python

   import numpy as np

   from qldpc import codes, decoders

   code = codes.RepetitionCode(5)
   syndrome = np.array([1, 0, 0, 0])

   settings = decoders.bp_lsd(max_iter=30, bp_method="minimum_sum")
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

Each helper returns a :class:`~qldpc.decoders.retrieval.DecoderSpec`, which only stores settings.
The signature of a helper lists its options explicitly, so they are visible to autocomplete and
static-analysis tools, and a misspelled option raises a ``TypeError``. The exceptions are
``relay_bp`` and ``ilp``, which forward additional options to the chosen Relay-BP decoder class and
to ``cvxpy.Problem.solve``, respectively. The ``mwpm`` helper omits PyMatching options, such as
``faults_matrix``, that would make the decoder return observable flips rather than errors. Passing
``decoder=None`` retains qLDPC's default: BP+OSD for binary inputs and generalized union-find for
nonbinary field arrays.

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

A CSS code decodes X-type and Z-type errors independently. ``decoder_x`` infers X-type errors from
their syndrome with respect to the Z-type stabilizers, and ``decoder_z`` infers Z-type errors from
their syndrome with respect to the X-type stabilizers. A sector that is not configured explicitly
falls back to the shared ``decoder=`` argument.

Custom and prebuilt decoders
----------------------------

Any object with a ``decode`` method that maps a syndrome to an inferred error satisfies the
:class:`~qldpc.decoders.custom.ErrorDecoder` protocol. A custom decoder may also define:

* ``decode_batch``, which decodes a two-dimensional array of syndromes (one per row) and satisfies
  :class:`~qldpc.decoders.custom.BatchErrorDecoder`, so that observable decoders decode shots in
  batches;
* ``has_erasure_bit = True``, to declare that it appends an erasure flag to each inferred error; and
* ``decodes_observables = True``, to declare that its output is observable flips rather than
  errors, so that it is rejected where an error decoder is required.

Besides a ``DecoderSpec``, the ``decoder=`` argument accepts:

* a constructor, such as a decoder class, or any other callable that builds an error decoder from a
  parity-check matrix or detector error model; or
* a prebuilt error decoder, which is used as is.

A prebuilt decoder is tied to the matrix used to construct it, so it is only accepted where the
caller knows the matrix being decoded: by ``decoders.decode`` and ``decoders.get_decoder``, by the
code-capacity estimators of classical codes, by ``ClassicalCode.get_distance_bound`` when given a
``vector`` (whose syndrome is computed with the parity check matrix of the code), and per sector (as
``decoder_x=`` and ``decoder_z=``) by the code-capacity estimators of CSS codes. A shared prebuilt
``decoder=`` for a CSS code is rejected unless its two stabilizer matrices are equal.

Some methods decode a matrix that they construct internally, and therefore reject prebuilt decoders:
decoder-based distance bounds of codes (other than a classical distance bound to a ``vector``),
logical-operator reduction, the code-capacity estimators of non-CSS codes, and observable decoders,
which build a new error decoder for every detector error model, window, or subgraph that they
decode. These methods accept a ``DecoderSpec`` or a constructor. A
constructor can fix custom options with ``functools.partial`` or a ``lambda``.

Predicting observable flips
---------------------------

Wrap error-decoder settings or a constructor in an observable decoder for Sinter:

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
inner error decoder. Sinter passes decoders to its worker processes by pickling them, so a custom
error decoder that is used with several workers should be defined in an importable module.

Lookup-table outputs
--------------------

:class:`~qldpc.decoders.lookup.LookupDecoder` and
:class:`~qldpc.decoders.lookup.WeightedLookupDecoder` are error decoders. They may use an
observable-flip matrix to group candidate errors by logical effect, but their output is a
representative physical error.

Use :class:`~qldpc.decoders.lookup.ObservableLookupDecoder` or
:class:`~qldpc.decoders.lookup.WeightedObservableLookupDecoder` when the desired output is the
observable flip itself:

.. code-block:: python

   observable_lookup = decoders.ObservableLookupDecoder(dem, max_weight=2)
   predicted_flips = observable_lookup.decode_observables(syndrome)

Erasure-aware decoders append their erasure flag after the inferred error or observable vector.
Sinter-compatible compiled decoders translate that flag into a discarded shot.

Migrating from the previous API
-------------------------------

Breaking changes
~~~~~~~~~~~~~~~~

The following changes take effect without a deprecation period:

* The ``static_decoder`` argument has been removed. Pass a prebuilt decoder as ``decoder=`` instead,
  where a prebuilt decoder is accepted (see above), and otherwise pass decoder settings or a
  constructor.
* ``qldpc.circuits.memory.alpha_syndrome.DEFAULT_SINTER_DECODER`` has been renamed to
  ``DEFAULT_OBSERVABLE_DECODER``.
* When the deprecated ``decoder_x_kwargs`` or ``decoder_z_kwargs`` of a CSS code set the same option
  as its shared keyword arguments, the sector-specific value now takes precedence, just as
  ``decoder_x=`` and ``decoder_z=`` take precedence over ``decoder=``.

Deprecated usage
~~~~~~~~~~~~~~~~

The keyword-based decoder API remains available during a deprecation period, and each use emits a
``DeprecationWarning`` that names its replacement:

.. list-table::
   :header-rows: 1

   * - Deprecated usage
     - Replacement
   * - ``with_BP_LSD=True, max_iter=30``
     - ``decoder=decoders.bp_lsd(max_iter=30)``
   * - free-form decoder options without ``with_<NAME>``
     - ``decoder=decoders.bp_osd(...)``, or ``decoders.guf(...)`` for nonbinary codes
   * - ``decoder_constructor=MyDecoder``
     - ``decoder=MyDecoder``
   * - ``decoder_x_kwargs={...}`` and ``decoder_z_kwargs={...}``
     - ``decoder_x=...`` and ``decoder_z=...``
   * - ``LookupDecoder(..., predict_observable_flips=True)``
     - ``ObservableLookupDecoder(...)`` and its ``decode_observables`` method
   * - ``WeightedLookupDecoder(..., predict_observable_flips=True)``
     - ``WeightedObservableLookupDecoder(...)`` and its ``decode_observables`` method
   * - ``SinterDecoder`` and ``CompiledSinterDecoder``
     - ``ObservableDecoder`` and ``CompiledObservableDecoder``
   * - ``CompiledObservableDecoder.decode``
     - ``CompiledObservableDecoder.decode_observables``
   * - ``SubgraphSinterDecoder`` and ``SequentialSinterDecoder``
     - ``SubgraphDecoder`` and ``SequentialWindowDecoder``
   * - ``Decoder`` and ``BatchDecoder``
     - ``ErrorDecoder`` and ``BatchErrorDecoder``
