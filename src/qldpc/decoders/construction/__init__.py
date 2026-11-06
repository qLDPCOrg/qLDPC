# SPDX-License-Identifier: Apache-2.0

"""Typed decoder specifications and explicit input contracts for custom decoder factories.

The specs module defines DecoderSpec, which helpers such as qldpc.decoders.bp_osd return, the types
of the inputs from which specifications build decoders, and the decoder_spec and
observable_decoder_spec functions that create such helpers.  The factories module wraps custom
decoder factories that take a parity-check matrix or a detector error model.  The package
qldpc.decoders re-exports the public names of both modules except decoder_spec and
observable_decoder_spec, which authors of decoder backends import from the specs module.
"""
