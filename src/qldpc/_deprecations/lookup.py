# SPDX-License-Identifier: Apache-2.0

"""Deprecated direct construction of base lookup decoders, and legacy observable lookup decoders.

LookupDecoder and WeightedLookupDecoder are base classes, and calling them directly is deprecated:
such a call builds an ErrorLookupDecoder or a WeightedErrorLookupDecoder, respectively.  With the
deprecated predict_observable_flips=True option, such a call instead builds an observable lookup
decoder of this module, which adds the interface of qLDPC 0.4.1 to the observable lookup decoder
that replaces it: its decode methods are aliases of its decode_observables methods, and it has the
deprecated predict_observable_flips and decode_returns_observables attributes.  This module imports
qldpc.decoders when it is imported, so qLDPC modules import it only within the functions that use
it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import stim

from qldpc import _deprecations
from qldpc._utils import get_external_caller_stacklevel
from qldpc.decoders.custom.lookup import (
    ErrorLookupDecoder,
    LookupDecoder,
    ObservableLookupDecoder,
    WeightedErrorLookupDecoder,
    WeightedLookupDecoder,
    WeightedObservableLookupDecoder,
)

from .decoders import _MATRIX_OBSERVABLE_LOOKUP_ADVICE

if TYPE_CHECKING:
    from qldpc.math import IntegerArray

####################################################################################################
# deprecated observable lookup decoders


class LegacyObservableLookupDecoder(ObservableLookupDecoder):
    """An ObservableLookupDecoder with the decode methods and attributes of qLDPC 0.4.1.

    LookupDecoder(..., predict_observable_flips=True) builds this decoder.  Its decode,
    decode_errors, decode_batch, and decode_errors_batch methods predict observable flips.
    """

    decode = decode_errors = ObservableLookupDecoder.decode_observables
    decode_batch = decode_errors_batch = ObservableLookupDecoder.decode_observables_batch

    @property
    def predict_observable_flips(self) -> bool:
        """Deprecated attribute that reports the deprecated predict_observable_flips option."""
        return _read_deprecated_attribute("LookupDecoder.predict_observable_flips")

    @property
    def decode_returns_observables(self) -> bool:
        """Deprecated attribute that reports that the decode methods return observable flips."""
        return _read_deprecated_attribute("LookupDecoder.decode_returns_observables")


class LegacyWeightedObservableLookupDecoder(WeightedObservableLookupDecoder):
    """A WeightedObservableLookupDecoder with the decode methods and attributes of qLDPC 0.4.1.

    WeightedLookupDecoder(..., predict_observable_flips=True) builds this decoder.  Its decode,
    decode_errors, decode_batch, and decode_errors_batch methods predict observable flips, and take
    the penalty_func argument of decode_observables.
    """

    decode = decode_errors = WeightedObservableLookupDecoder.decode_observables
    decode_batch = decode_errors_batch = WeightedObservableLookupDecoder.decode_observables_batch

    @property
    def predict_observable_flips(self) -> bool:
        """Deprecated attribute that reports the deprecated predict_observable_flips option."""
        return _read_deprecated_attribute("WeightedLookupDecoder.predict_observable_flips")

    @property
    def decode_returns_observables(self) -> bool:
        """Deprecated attribute that reports that the decode methods return observable flips."""
        return _read_deprecated_attribute("WeightedLookupDecoder.decode_returns_observables")


def _read_deprecated_attribute(name: str) -> bool:
    """Warn that an attribute of a legacy observable lookup decoder is deprecated; return True."""
    _deprecations.warn_deprecated(
        f"{name} is DEPRECATED and will be removed from the public API",
        stacklevel=get_external_caller_stacklevel(),
    )
    return True


####################################################################################################
# deprecated direct construction of lookup decoders


def initialize_lookup_decoder(
    decoder: LookupDecoder,
    pcm_or_dem: IntegerArray | stim.DetectorErrorModel,
    max_weight: int,
    /,
    **options: object,
) -> None:
    """Warn about a direct call of LookupDecoder or WeightedLookupDecoder, and initialize it.

    The decoder, whose class is LookupDecoder or WeightedLookupDecoder, becomes an
    ErrorLookupDecoder or a WeightedErrorLookupDecoder, respectively, or with the deprecated
    predict_observable_flips=True option, a LegacyObservableLookupDecoder or a
    LegacyWeightedObservableLookupDecoder, respectively.  It is then initialized with the remaining
    arguments.

    Raises:
        TypeError: If the decoder is an instance of a user-defined subclass of LookupDecoder or
            WeightedLookupDecoder, which should subclass a lookup decoder that can be initialized.
    """
    weighted = isinstance(decoder, WeightedLookupDecoder)
    base = WeightedLookupDecoder if weighted else LookupDecoder
    error_class = WeightedErrorLookupDecoder if weighted else ErrorLookupDecoder
    if type(decoder) is not base:
        observable_class = WeightedObservableLookupDecoder if weighted else ObservableLookupDecoder
        raise TypeError(
            f"A subclass of {base.__name__} cannot be initialized; subclass {error_class.__name__}"
            f" or {observable_class.__name__} instead"
        )

    new_class: type[LookupDecoder]
    if options.pop("predict_observable_flips", False):
        if weighted:
            new_class = LegacyWeightedObservableLookupDecoder
            advice = (
                "use qldpc.decoders.custom.lookup.WeightedObservableLookupDecoder(...)"
                ".decode_observables(syndrome, penalty_func) instead"
            )
        else:
            new_class = LegacyObservableLookupDecoder
            advice = (
                "use decoders.lookup(max_weight=...).build_observable_decoder(dem) instead"
                if isinstance(pcm_or_dem, stim.DetectorErrorModel)
                else _MATRIX_OBSERVABLE_LOOKUP_ADVICE
            )
        message = f"predict_observable_flips=True is deprecated; {advice}"
    else:
        new_class = error_class
        message = (
            f"Constructing {base.__name__} directly is deprecated; use"
            f" {error_class.__name__}(...) instead"
        )
    _deprecations.warn_deprecated(message, stacklevel=get_external_caller_stacklevel())

    decoder.__class__ = new_class
    # each class that a base class becomes accepts the arguments of a direct call of the base class
    initialize = cast("Callable[..., None]", new_class.__init__)
    initialize(decoder, pcm_or_dem, max_weight, **options)
