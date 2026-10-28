qLDPC documentation
===================

``qLDPC`` is a Python toolkit for constructing and analyzing finite-size `quantum low-density parity-check codes <https://errorcorrectionzoo.org/c/qldpc>`_, as well as `stabilizer <https://errorcorrectionzoo.org/c/stabilizer>`_ and `subsystem <https://errorcorrectionzoo.org/c/oecc>`_ codes more broadly.
It provides shared implementations and scientific conventions that researchers and their AI assistants can inspect, combine, and extend.
The `project goals <https://github.com/qLDPCOrg/qLDPC/blob/main/README.md#goals>`_ guide its support for exploration, scientific clarity, checking and reproduction, and learning.

Use the :doc:`library map <library_map>` to understand qLDPC's data model and package structure, the :doc:`examples <examples/index>` to learn concepts and adapt complete workflows, and the :doc:`API reference <autoapi/index>` for exact signatures and construction-specific literature.

Installation
------------

``qLDPC`` requires Python 3.11 or later and can be installed from the Python Package Index (PyPI):

.. code-block:: bash

   python -m pip install qldpc

See the :doc:`installation guide <installation>` for optional decoders, GAP integration, GPU distance estimation, and troubleshooting.

Quickstart
----------

.. code-block:: python

   from sympy.abc import x, y

   from qldpc import codes

   # build the bivariate bicycle "gross code" with parameters [[144, 12, 12]]
   code = codes.BBCode({x: 12, y: 6}, x**3 + y + y**2, y**3 + x + x**2)

   print(code)                              # a human-readable summary of the code
   print("physical qubits:", len(code))     # number of physical qubits, n
   print("logical qubits:", code.dimension) # number of logical qubits, k

The gross code is also available as the named construction, ``codes.GrossCode()``.

Where to go next
----------------

* :doc:`Installation <installation>` explains the standard setup and optional tools for specific tasks.
* :doc:`Library map <library_map>` explains how qLDPC represents codes, what each package does, common tasks, and current limitations.
* :doc:`Choosing a decoder <decoders>` explains error and observable decoding APIs.
* :doc:`Adding a decoder <adding_decoders>` explains the shortest custom-decoder path and how to contribute a backend.
* :doc:`Examples <examples/index>` walk from code construction through code-capacity and circuit-level simulations.
* :doc:`API reference <autoapi/index>` documents every public class and function.
* Start with the `agent and contributor guide <https://github.com/qLDPCOrg/qLDPC/blob/main/AGENTS.md>`_ before changing qLDPC, and use the :doc:`contributor reference <contributing>` for detailed rules and procedures.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: User guide

   installation
   library_map
   decoders

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Contributor guides

   contributing
   adding_decoders

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Examples

   examples/index

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: API Reference

   autoapi/index
