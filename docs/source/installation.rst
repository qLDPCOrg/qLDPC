Installation
============

Start with the standard installation and add only the optional tools needed for the task.
The commands below assume that ``python`` selects the intended Python environment.
Optional tools can have narrower Python and platform requirements than qLDPC itself.

Standard installation
---------------------

qLDPC requires Python 3.11 or later:

.. code-block:: bash

   python -m pip install qldpc

This includes code construction and analysis tools, Stim circuit tools, and the ``ldpc`` and ``pymatching`` decoder integrations.
A small check of the installation is:

.. code-block:: bash

   python -c "from qldpc import codes; print(codes.RepetitionCode(3).get_code_params())"

The expected output is ``(3, 1, 3)``.

Choose tools for the task
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Task
     - Setup
   * - Construct and inspect codes, build Stim circuits, or use the standard decoders
     - Standard installation.
   * - Use an additional decoder backend
     - :ref:`Additional decoders <installation-decoders>`.
   * - Use algebra tools, named-code lookups, or distance estimates that require GAP
     - :ref:`GAP integration <installation-gap>`.
   * - Estimate binary CSS code distances with the GPU-based ``sqetch`` backend
     - :ref:`GPU distance estimation <installation-sqetch>`.
   * - Modify qLDPC itself
     - :ref:`Development installation <development-installation>` and the :doc:`contributor reference <contributing>`.

.. _installation-decoders:

Additional decoders
-------------------

Install the optional decoder integrations published as qLDPC extras for the platform with:

.. code-block:: bash

   python -m pip install 'qldpc[decoders]'

Alternatively, install only the extra needed for a specific decoder:

.. code-block:: bash

   python -m pip install 'qldpc[ilp]'
   python -m pip install 'qldpc[relay-bp]'
   python -m pip install 'qldpc[tesseract]'

The `Tesseract <https://github.com/quantumlib/tesseract-decoder>`_ decoder requires CPython 3.12-3.14 on macOS arm64 or Linux x86-64.
`Frontier <https://github.com/aleverrier/frontier>`_ is not published on PyPI and is not included in an extra.
See :doc:`Choosing a decoder <decoders>` for backend capabilities and Frontier's manual build instructions.

.. _installation-gap:

GAP integration
---------------

Some algebra tools, code lookups, and distance estimates require `GAP <https://www.gap-system.org>`_.
For the recommended in-process integration, install qLDPC with its maintained libgap extra:

.. code-block:: bash

   python -m pip install 'qldpc[gap]'

qLDPC uses ``passagemath-gap`` when available, so supported GAP operations do not launch a separate process.
The in-process binding requires Python 3.13 on a non-Windows platform.
Without that binding, qLDPC can call a GAP executable on ``PATH`` (possibly obtainable with ``conda install -c conda-forge gap``), or offer a manual copy/paste fallback for operations that support it.
GAP integration on Windows remains limited; see `issue #294 <https://github.com/qLDPCOrg/qLDPC/issues/294>`_.
See the :doc:`library map <library_map>` for GAP calls, prompts, other side effects, and caching.

.. _installation-sqetch:

GPU distance estimation
-----------------------

The optional ``sqetch`` distance-estimation backend works with binary CSS codes and requires a CUDA GPU.
It is not published on PyPI and is not available as a qLDPC extra.
Install the upstream version that qLDPC is tested against, with GPU support:

.. code-block:: bash

   python -m pip install 'sqetch[gpu] @ git+https://github.com/a7b/yarn.git@e9ce9d0fcecc973988558bfee27fab6d6b8d7f97#subdirectory=sqetch'

Select it with ``code.get_distance_bound(backend="sqetch")``.
The standard installation supports CPU-based distance calculations without this backend.

.. _installation-troubleshooting:

Troubleshooting
---------------

On macOS, it may be necessary to install ``cvxpy``, which the ILP decoder requires, manually by following its `installation instructions <https://www.cvxpy.org/install>`_ before installing ``qldpc[ilp]`` or ``qldpc[decoders]``.
For a Conda-managed environment, install ``cvxpy`` with:

.. code-block:: bash

   conda install -c conda-forge cvxpy

Working on qLDPC
----------------

Follow the :ref:`development installation <development-installation>` instructions in the contributor reference for an editable checkout and the project's validation tools.
