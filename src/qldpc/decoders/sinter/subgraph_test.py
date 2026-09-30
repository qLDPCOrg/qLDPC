# SPDX-License-Identifier: Apache-2.0

"""Subgraph Sinter decoder tests."""

from .. import sinter_test

test_compiled_decoder_input_validation = sinter_test.test_compiled_decoder_input_validation
test_native_observable_decoders_on_subgraphs = (
    sinter_test.test_native_observable_decoders_on_subgraphs
)
test_observable_decoders_reject_prebuilt_decoders = (
    sinter_test.test_observable_decoders_reject_prebuilt_decoders
)
test_subgraph_decoder_with_erasure = sinter_test.test_subgraph_decoder_with_erasure
test_subgraph_decoding = sinter_test.test_subgraph_decoding
test_subgraph_partition_warnings = sinter_test.test_subgraph_partition_warnings
