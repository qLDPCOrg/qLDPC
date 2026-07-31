qLDPC documentation
===================

``qLDPC`` is a library of tools for constructing and analyzing finite-size
`quantum low density parity check (qLDPC) codes <https://errorcorrectionzoo.org/c/qldpc>`_.
Its primary purpose is to make developments in the quantum error correction literature accessible
to a broad audience of physicists, computer scientists, and tinkerers.  The hope is that these
tools facilitate the discovery and design of practical, finite-size codes for near-term (or at
least medium-term) quantum computers.

In practice the tools here work just as well for
`stabilizer <https://errorcorrectionzoo.org/c/stabilizer>`_ and
`subsystem <https://errorcorrectionzoo.org/c/oecc>`_ codes more broadly.

See the `README on GitHub <https://github.com/qLDPCOrg/qLDPC>`_ for a high-level overview, and the
`examples directory <https://github.com/qLDPCOrg/qLDPC/tree/main/examples>`_ for demonstrations and
use cases.

Installation
------------

``qLDPC`` requires Python 3.10 or later and can be installed from the Python Package Index (PyPI):

.. code-block:: bash

   pip install qldpc

Some features additionally require the `GAP <https://www.gap-system.org>`_ computer algebra system;
see the `README <https://github.com/qLDPCOrg/qLDPC#-installation>`_ for details.

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

The :doc:`examples <examples/index>` walk through common workflows, and the
:doc:`API reference <autoapi/index>` documents every public class and function.

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
