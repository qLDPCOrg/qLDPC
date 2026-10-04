# SPDX-License-Identifier: Apache-2.0

"""Translation of deprecated decoder probability options at construction boundaries."""

from __future__ import annotations

import warnings

from qldpc._util import get_external_caller_stacklevel


def _deprecate_error_rate_option(
    options: dict[str, object], explicitly_provided: frozenset[str]
) -> dict[str, object]:
    """Replace an explicitly supplied error_rate option with error_channel.

    An explicit error_rate=None is the default value, so it is dropped as if it were omitted.
    """
    if "error_rate" not in explicitly_provided or options.get("error_rate") is None:
        options.pop("error_rate", None)
        return options
    if "error_channel" in explicitly_provided:
        raise ValueError("error_rate and error_channel cannot both be specified")
    error_rate = options["error_rate"]
    warnings.warn(
        f"error_rate={error_rate!r} is deprecated; use error_channel={error_rate!r} instead",
        DeprecationWarning,
        stacklevel=get_external_caller_stacklevel(),
    )
    options.pop("error_rate")
    options["error_channel"] = error_rate
    return options
