Decoding workflows
==================

qLDPC uses decoders in code-capacity estimates and in Sinter circuit sampling, which supports sliding-window decoding.

Code-capacity logical error rate estimates
------------------------------------------

A code-capacity estimate samples bit/qubit/qudit errors and counts a failure when a decoder mispredicts their logical effect.
To use a built-in decoder family, you can pass a specification such as ``decoders.mwpm()``:

.. code-block:: python

   import numpy as np

   from qldpc import codes, decoders

   code = codes.SurfaceCode(3)
   estimator = code.get_logical_error_rate_func(
       num_samples=10_000,
       decoder=decoders.mwpm(),
       seed=0,
   )
   logical_error_rate, stderr = estimator(np.logspace(-4, -2, 5))

An integer ``seed`` starts a repeatable local sampling stream; a supplied NumPy ``Generator`` is advanced.
With ``seed=None``, sampling uses NumPy's global RNG; decoder randomness is configured separately.

To use your own decoder, see :doc:`custom`.

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

A ``SinterDecoder`` uses native observable prediction when its specification supports it, so the ``decoder`` attribute of the compiled decoder predicts observable flips for specifications such as ``decoders.mwpm()`` and ``decoders.lookup(...)``.
See the :doc:`quantum-memory notebook </examples/logical_error_rates/2_quantum_memory_x_or_z>` for a circuit-level sampling example.

Note that the ``SlidingWindowDecoder`` must infer and commit physical corrections in each window, so an observable-only decoder such as Frontier will not work.
Use a built-in error-decoding specification or a custom error-decoder factory that accepts each window's DEM.
