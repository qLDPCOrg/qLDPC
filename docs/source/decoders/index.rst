Decoders
========

A decoder interprets syndromes or detection events to infer errors or predict their effect on logical observables.
qLDPC uses decoders to estimate logical error rates and to bound code distances.
qLDPC lets you choose a decoder by creating a :class:`decoder specification <qldpc.decoders.construction.specs.DecoderSpec>`, such as ``spec = qldpc.decoders.bp_osd(...)``, which defines a decoding algorithm and its options.
Any method with a ``decoder=`` argument, such as :meth:`qldpc.codes.CSSCode.get_logical_error_rate_func <qldpc.codes.common.CSSCode.get_logical_error_rate_func>` (see, for example, the :doc:`code-capacity notebook </examples/logical_error_rates/1_code_capacity>`), accepts and uses a decoder specification to internally compile a decoder for a fixed parity check matrix or detector error model.
You can also pass your own decoder class or factory as ``decoder=``.
The :doc:`decoders example notebook </examples/decoders>` walks through these workflows.

* :doc:`Decoding workflows <workflows>` covers code-capacity estimates and Sinter, including sliding-window decoding.
* :doc:`Using your own decoder <custom>` covers the contract that a decoder of your own implements in each workflow.
* :doc:`Decoder reference <reference>` documents every specification helper, backend notes, detailed decoding results, and lookup-table outputs.

Built-in decoders
-----------------

qLDPC provides specification helpers, such as ``qldpc.decoders.bp_osd``, for its built-in decoders, and the :ref:`decoder reference <specification-helpers>` documents each of them.
See, for example, ``help(qldpc.decoders.bp_osd)`` for configuration options.
Specification helpers accept the options that they list, and the helpers for ldpc, PyMatching, and Relay-BP decoders accept other backend options in a ``backend_options`` mapping.
``pip install 'qldpc[decoders]'`` installs all optional decoders published as qLDPC extras for your platform.
You can install individual extras with, for example, ``pip install 'qldpc[relay-bp]'``.
The ILP decoder needs the ``ilp`` extra, which installs ``cvxpy``: ``pip install 'qldpc[ilp]'``.
The Frontier decoder is not available on PyPI, and must be installed manually; see its :ref:`backend notes <backend-notes>`.

The default decoder in qLDPC (with an argument of ``decoder=None``, where applicable) is BP+OSD for binary codes, and generalized union-find for codes over nonbinary finite fields.

Inferring errors
~~~~~~~~~~~~~~~~

An :class:`error decoder <qldpc.decoders.protocols.ErrorDecoder>` infers errors from a syndrome with ``.decode_errors``.
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

.. toctree::
   :hidden:

   workflows
   custom
   reference
