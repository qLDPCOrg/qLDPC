Choosing a decoder
==================

qLDPC distinguishes two kinds of decoders:

* an :class:`~qldpc.decoders.custom.ErrorDecoder` maps a syndrome to an inferred physical error,
  with a ``decode`` method; and
* an :class:`~qldpc.decoders.custom.ObservableDecoder` maps a syndrome (detection events) to
  predicted observable flips, with a ``decode_observables`` method.

The distinction matters when composing decoders. Code-capacity estimates, decoder-based distance
bounds, logical-operator reduction, and sliding-window decoders all work with physical errors, so
they need error decoders. Circuit-level simulations only need to know which observables flipped,
so they use observable decoders. Some decoders are both: a
:class:`~qldpc.decoders.custom.RelayBPDecoder` infers errors and predicts observable flips. Exact
distance calculations do not need a decoder.

A :class:`~qldpc.decoders.sinter.SinterDecoder` is the observable decoder that Sinter uses. It
stores decoder settings, and builds an observable decoder for each detector error model that Sinter
compiles it for.

The :doc:`decoders example notebook <examples/decoders>` walks through the workflows on this page.

Configuring decoders
--------------------

Use a typed helper to configure a decoder. The consuming method builds the decoder once it knows
which parity-check matrix or detector error model to decode:

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
* :func:`~qldpc.decoders.retrieval.min_sum_bp`
* :func:`~qldpc.decoders.retrieval.lookup_table`
* :func:`~qldpc.decoders.retrieval.ilp`
* :func:`~qldpc.decoders.retrieval.guf`

Each helper returns a :class:`~qldpc.decoders.retrieval.DecoderSpec`, which only stores settings.
The signature of a helper lists the options of its decoder explicitly, so they are visible to
autocomplete and static-analysis tools, and a misspelled option raises a ``TypeError``. The
exception is ``ilp``, which forwards additional options to ``cvxpy.Problem.solve``. Passing
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

Predicting observable flips
---------------------------

:func:`~qldpc.decoders.retrieval.get_observable_decoder` builds an observable decoder for a
detector error model, and :func:`~qldpc.decoders.retrieval.decode_observables` predicts the
observable flips of one syndrome:

.. code-block:: python

   observable_decoder = decoders.get_observable_decoder(dem, decoder=decoders.mwpm())
   predicted_flips = observable_decoder.decode_observables(syndrome)

Some decoders can predict observable flips natively, without first inferring an error. Their
settings build a native observable decoder wherever observable flips are wanted:

* ``mwpm`` builds a PyMatching decoder that tracks observables along matched paths;
* ``relay_bp`` and ``min_sum_bp`` build a :class:`~qldpc.decoders.custom.RelayBPDecoder`; and
* ``lookup_table`` builds an :class:`~qldpc.decoders.lookup.ObservableLookupDecoder`, which maps each
  syndrome directly to its most likely observable flip.

The settings of any other decoder build an error decoder, whose inferred errors are converted into
observable flips. ``DecoderSpec.predicts_observables_natively`` reports which of these applies.

For Sinter, wrap decoder settings (or a constructor) in a
:class:`~qldpc.decoders.sinter.SinterDecoder`, or in one of its subclasses:

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

A ``SinterDecoder`` is compiled for a detector error model before it predicts flips. Its compiled
form, a :class:`~qldpc.decoders.sinter.CompiledSinterDecoder`, exposes ``decode_observables`` for
one shot and ``decode_shots`` for a batch. A ``SinterDecoder`` and a
:class:`~qldpc.decoders.sinter.SubgraphDecoder` use native observable decoders where possible.
Window decoders (:class:`~qldpc.decoders.sinter.SequentialWindowDecoder` and
:class:`~qldpc.decoders.sinter.SlidingWindowDecoder`) commit the errors that they infer in each
window, so they always use error decoders. Sinter passes decoders to its worker processes by
pickling them, so a custom decoder that is used with several workers should be defined in an
importable module.

Custom and prebuilt decoders
----------------------------

Any object with a ``decode`` method that maps a syndrome to an inferred error satisfies the
:class:`~qldpc.decoders.custom.ErrorDecoder` protocol, and any object with a ``decode_observables``
method that maps a syndrome to predicted observable flips satisfies the
:class:`~qldpc.decoders.custom.ObservableDecoder` protocol. A custom decoder may also define:

* ``decode_batch`` or ``decode_observables_batch``, which decode a two-dimensional array of
  syndromes (one per row), to satisfy :class:`~qldpc.decoders.custom.BatchErrorDecoder` or
  :class:`~qldpc.decoders.custom.BatchObservableDecoder`, so that Sinter decoders decode shots in
  batches; and
* ``has_erasure_bit = True``, to declare that it appends an erasure flag to each inferred error or
  predicted observable flip.

Besides a ``DecoderSpec``, the ``decoder=`` argument accepts:

* a constructor, such as a decoder class, or any other callable that builds a decoder from a
  parity-check matrix or detector error model; or
* a prebuilt decoder, which is used as is.

A prebuilt decoder is tied to the matrix used to construct it, so it is only accepted where the
caller knows the matrix being decoded: by ``decoders.decode``, ``decoders.get_decoder``,
``decoders.decode_observables``, and ``decoders.get_observable_decoder``; by the code-capacity
estimators of classical codes; by ``ClassicalCode.get_distance_bound`` when given a ``vector`` (whose
syndrome is computed with the parity check matrix of the code); and per sector (as ``decoder_x=``
and ``decoder_z=``) by the code-capacity estimators of CSS codes. A shared prebuilt ``decoder=`` for
a CSS code is rejected unless its two stabilizer matrices are equal.

Some methods decode a matrix that they construct internally, and therefore reject prebuilt decoders:
decoder-based distance bounds of codes (other than a classical distance bound to a ``vector``),
logical-operator reduction, the code-capacity estimators of non-CSS codes, and Sinter decoders,
which build a new decoder for every (simplified) detector error model, window, or subgraph that they
decode. These methods accept a ``DecoderSpec`` or a constructor. A constructor can fix custom
options with ``functools.partial`` or a ``lambda``.

Lookup-table outputs
--------------------

:class:`~qldpc.decoders.lookup.LookupDecoder` and
:class:`~qldpc.decoders.lookup.WeightedLookupDecoder` are error decoders. They may use an
observable-flip matrix to group candidate errors by logical effect, but their output is a
representative physical error.

:class:`~qldpc.decoders.lookup.ObservableLookupDecoder` and
:class:`~qldpc.decoders.lookup.WeightedObservableLookupDecoder` are observable decoders, which
return the observable flip itself:

.. code-block:: python

   observable_lookup = decoders.ObservableLookupDecoder(dem, max_weight=2)
   predicted_flips = observable_lookup.decode_observables(syndrome)

Erasure-aware decoders append their erasure flag after the inferred error or observable vector.
Compiled Sinter decoders translate that flag into a discarded shot.

Migrating from qLDPC 0.3.3
--------------------------

This section describes how the decoder API differs from that of ``qldpc==0.3.3``, and how to update
code written for it.

Breaking changes
~~~~~~~~~~~~~~~~

The following changes take effect without a deprecation period:

* The ``static_decoder`` argument has been removed. Pass a prebuilt decoder as ``decoder=`` instead,
  where a prebuilt decoder is accepted (see above), and otherwise pass decoder settings or a
  constructor.
* When the deprecated ``decoder_x_kwargs`` or ``decoder_z_kwargs`` of a CSS code set the same option
  as its shared keyword arguments, the sector-specific value now takes precedence, just as
  ``decoder_x=`` and ``decoder_z=`` take precedence over ``decoder=``.
* A ``SinterDecoder`` whose settings support native observable prediction (``mwpm``, ``relay_bp``,
  ``min_sum_bp``, and ``lookup_table``) now uses it. The predicted observable flips are unchanged.
  For ``mwpm`` and ``lookup_table`` settings, however, the ``decoder`` attribute of the resulting
  ``CompiledSinterDecoder`` is now a decoder that predicts observable flips rather than errors.
  For other settings, the ``decoder`` attribute remains the decoder that the settings build.

Deprecated usage
~~~~~~~~~~~~~~~~

The keyword-based decoder API of ``qldpc==0.3.3`` remains available during a deprecation period, and
each use emits a ``DeprecationWarning`` that names its replacement:

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
   * - ``CompiledSinterDecoder.decode``
     - ``CompiledSinterDecoder.decode_observables``
   * - ``SubgraphSinterDecoder`` and ``SequentialSinterDecoder``
     - ``SubgraphDecoder`` and ``SequentialWindowDecoder``
   * - ``Decoder`` and ``BatchDecoder``
     - ``ErrorDecoder`` and ``BatchErrorDecoder``
