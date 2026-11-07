Standards
=========

Standards state the rules that a change to qLDPC must satisfy, one topic per standard.
Only a maintainer can approve a change to a standard or to the :doc:`review protocol </contributing/reviewing>`; agents propose such changes rather than making them.
The :ref:`kinds-of-change table <kinds-of-change>` lists which standards apply to each kind of change, and a :doc:`standard review </contributing/reviewing>` checks a change against them.

* :doc:`Code standard <code>`: public and private names, code style and organization, and simplicity and clarity.
* :doc:`Writing standard <writing>`: prose style, where documentation belongs, and criteria for example notebooks.
* :doc:`Testing standard <testing>`: what tests guard and how they run.
* :doc:`Implementation standard <implementation>`: code constructions, algebra and code state, circuits, and external systems.
* :doc:`Decoder standard <decoders>`: decoders that ship with qLDPC, and shared decoder plumbing.
* :doc:`Deprecation standard <deprecations>`: deprecations and future changes to released public APIs.
* :doc:`Pull request standard <pull_requests>`: pull request summaries.

.. toctree::
   :hidden:

   code
   writing
   testing
   implementation
   decoders
   deprecations
   pull_requests
