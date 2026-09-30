# SPDX-License-Identifier: Apache-2.0

"""Decoders from the ldpc and pymatching packages, as qLDPC error decoders.

Each class here subclasses a decoder class from the ldpc or pymatching package, and ErrorDecoder (or
BatchErrorDecoder), from which it inherits decode_errors (and decode_errors_batch) as an alias for
the decode (and decode_batch) method of the original class.  An instance of one of these classes is
therefore also an instance of the original class.

qldpc.decoders imports this module, and with it ldpc and pymatching, only once it is used.
"""

from __future__ import annotations

import ldpc
import ldpc.bplsd_decoder
import pymatching

from .protocols import BatchErrorDecoder, ErrorDecoder


class BpOsdDecoder(ldpc.BpOsdDecoder, ErrorDecoder):
    """An ldpc.BpOsdDecoder that is also an ErrorDecoder.  See help(ldpc.BpOsdDecoder)."""


class BpLsdDecoder(ldpc.bplsd_decoder.BpLsdDecoder, ErrorDecoder):
    """An ldpc.bplsd_decoder.BpLsdDecoder that is also an ErrorDecoder.

    See help(ldpc.bplsd_decoder.BpLsdDecoder).
    """


class BeliefFindDecoder(ldpc.BeliefFindDecoder, ErrorDecoder):
    """An ldpc.BeliefFindDecoder that is also an ErrorDecoder.  See help(ldpc.BeliefFindDecoder)."""


class Matching(pymatching.Matching, BatchErrorDecoder):
    """A pymatching.Matching that is also a BatchErrorDecoder.  See help(pymatching.Matching).

    A Matching is only an error decoder if it has no faults_matrix, with which its decode method
    returns the "faults" (such as observables) that an error flips, rather than the error.
    """
