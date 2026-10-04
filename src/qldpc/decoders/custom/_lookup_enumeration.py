# SPDX-License-Identifier: Apache-2.0

"""Enumerate bounded-weight errors and their syndromes for lookup decoders."""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from typing import NamedTuple

import galois
import numpy as np
import numpy.typing as npt

from qldpc import math
from qldpc.math import IntegerArray


class _ScoredLocalError(NamedTuple):
    """A local error and its log probability."""

    error: tuple[int, ...]
    log_probability: float


class _ScoredErrorSite(NamedTuple):
    """A physical error site and its possible local errors, sorted from most to least likely."""

    site_index: int
    inactive_log_probability: float
    errors: tuple[_ScoredLocalError, ...]


class _ErrorSelection(NamedTuple):
    """One selection in a linked list of local errors."""

    parent: _ErrorSelection | None
    site_index: int
    error: tuple[int, ...]


def _iter_errors_and_syndromes(
    matrix: IntegerArray,
    max_weight: int,
    syndrome_mask: npt.NDArray[np.bool_] | None,
    symplectic: bool,
    *,
    error_channel: npt.NDArray[np.floating] | None = None,
    probability_cutoff: float = 0,
) -> Iterator[tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]]:
    """Iterate over all errors considered by a lookup decoder, with their syndromes.

    Errors are sorted in decreasing weight (number of bits or qudits addressed nontrivially).  A
    positive probability_cutoff requires an error_channel, and skips errors below the cutoff.

    The syndrome_mask is a boolean mask of syndrome bits to retain, or None to keep all bits.
    When post-selecting, errors whose syndrome is nontrivial on any dropped bit are skipped,
    and dropped bits are omitted from the yielded syndrome.
    """
    from qldpc import codes

    dtype = matrix.dtype
    # rewrite the checks so multiplying by an error produces its syndrome
    code = codes.ClassicalCode(matrix) if not symplectic else codes.QuditCode(matrix)
    matrix = code.matrix if not symplectic else -math.symplectic_conjugate(code.matrix)
    repeat = 2 if symplectic else 1
    block_length = matrix.shape[1] // repeat

    errors: Iterator[npt.NDArray[np.int_]]
    if probability_cutoff:
        assert error_channel is not None
        errors = _iter_errors_above_probability_cutoff(
            code.field,
            block_length,
            repeat,
            max_weight,
            dtype,
            error_channel,
            probability_cutoff,
        )
    else:
        errors = _iter_errors_up_to_weight(
            code.field.order, block_length, repeat, max_weight, dtype
        )

    for error in errors:
        syndrome = (matrix @ error.view(code.field)).view(np.ndarray)
        if syndrome_mask is not None:
            if np.any(syndrome[~syndrome_mask]):
                continue  # a post-selected syndrome bit is nontrivial
            syndrome = syndrome[syndrome_mask]
        yield error, syndrome


def _iter_errors_up_to_weight(
    field_order: int,
    block_length: int,
    repeat: int,
    max_weight: int,
    dtype: npt.DTypeLike,
) -> Iterator[npt.NDArray[np.int_]]:
    """Yield every error with weight at most max_weight, in order of decreasing weight."""
    local_errors = tuple(itertools.product(range(field_order), repeat=repeat))[1:]
    for weight in range(max_weight, -1, -1):
        for error_sites in itertools.combinations(range(block_length), weight):
            error_site_indices = list(error_sites)
            for site_errors in itertools.product(local_errors, repeat=weight):
                error = np.zeros((repeat, block_length), dtype=dtype)
                error[:, error_site_indices] = np.asarray(site_errors, dtype=dtype).T
                yield error.ravel()


def _iter_errors_above_probability_cutoff(
    field: type[galois.FieldArray],
    block_length: int,
    repeat: int,
    max_weight: int,
    dtype: npt.DTypeLike,
    error_channel: npt.NDArray[np.floating],
    probability_cutoff: float,
) -> Iterator[npt.NDArray[np.int_]]:
    """Yield errors above a Bernoulli-probability cutoff without exhaustively generating them.

    Each site (a bit, or a qudit if symplectic) is either inactive or carries a nonzero local error.
    Sites are sorted by the likelihood ratio of their best local error, so the most likely way to
    complete a partial error is to activate the next remaining sites in sorted order.  A depth-first
    search over the active sites of each weight prunes any branch whose most likely completion falls
    below the cutoff, and checks the exact probability of each complete error before yielding it.
    """
    probabilities = error_channel.reshape(repeat, block_length)
    active_probabilities = probabilities / (field.order - 1)
    with np.errstate(divide="ignore"):
        log_active_probabilities = np.log(active_probabilities)
        log_inactive_probabilities = np.log1p(-probabilities)

    # score the possible local errors at each site, most likely first
    local_errors = tuple(itertools.product(range(field.order), repeat=repeat))[1:]
    sites: list[_ScoredErrorSite] = []
    for site_index in range(block_length):
        site_log_active = log_active_probabilities[:, site_index]
        site_log_inactive = log_inactive_probabilities[:, site_index]
        scored_errors: list[_ScoredLocalError] = []
        for local_error in local_errors:
            is_active = np.asarray(local_error, dtype=bool)
            log_probability = float(np.sum(np.where(is_active, site_log_active, site_log_inactive)))
            if np.isfinite(log_probability):
                scored_errors.append(_ScoredLocalError(local_error, log_probability))
        if scored_errors:  # otherwise this site is never active, so we can ignore it
            scored_errors.sort(key=lambda error: error.log_probability, reverse=True)
            sites.append(
                _ScoredErrorSite(site_index, float(np.sum(site_log_inactive)), tuple(scored_errors))
            )

    # A site with a probability-one mechanism has an infinite likelihood ratio and an inactive log
    # probability of -inf, so it sorts first and every bound that leaves it inactive is pruned.
    sites.sort(
        key=lambda site: site.errors[0].log_probability - site.inactive_log_probability,
        reverse=True,
    )

    # prefix/suffix sums give the log probability of activating a window of sorted sites
    best_log_prefix = [0.0]
    for site in sites:
        best_log_prefix.append(best_log_prefix[-1] + site.errors[0].log_probability)
    inactive_log_suffix = [0.0] * (len(sites) + 1)
    for index in range(len(sites) - 1, -1, -1):
        inactive_log_suffix[index] = (
            inactive_log_suffix[index + 1] + sites[index].inactive_log_probability
        )

    def get_best_completion(start: int, count: int) -> float:
        """Log probability of activating sites start, ..., start + count - 1 and no later sites."""
        stop = start + count
        return best_log_prefix[stop] - best_log_prefix[start] + inactive_log_suffix[stop]

    # Log-space bounds are only used to prune, while yielded errors are checked exactly, so pruning
    # only needs a margin that exceeds the rounding error of any bound: a sum of at most
    # 3 * len(sites) terms whose magnitudes are bounded by log_scale.
    log_cutoff = float(np.log(probability_cutoff))
    log_scale = 1 + abs(log_cutoff)
    for site in sites:
        log_scale += max(abs(error.log_probability) for error in site.errors)
        if np.isfinite(site.inactive_log_probability):
            log_scale += abs(site.inactive_log_probability)
    prune_threshold = log_cutoff - 4 * (len(sites) + 1) * np.finfo(float).eps * log_scale

    # enumerate weights from heavy to light, matching the exhaustive path's tie-breaking
    for weight in range(min(len(sites), max_weight), -1, -1):
        # each stack entry: (next sorted site, remaining sites to activate, log probability of the
        # sites decided so far, linked list of the selected local errors)
        stack: list[tuple[int, int, float, _ErrorSelection | None]] = [(0, weight, 0.0, None)]
        while stack:
            start, remaining, log_probability, selection = stack.pop()
            if remaining == 0:
                error = np.zeros((repeat, block_length), dtype=dtype)
                while selection is not None:
                    error[:, selection.site_index] = selection.error
                    selection = selection.parent
                probability = np.prod(
                    np.where(error.astype(bool), active_probabilities, 1 - probabilities)
                )
                if probability >= probability_cutoff:
                    yield error.ravel()
                continue

            # choose the next active site, leaving all sites from start up to it inactive
            children: list[tuple[int, int, float, _ErrorSelection]] = []
            skipped_log_probability = 0.0
            for position in range(start, len(sites) - remaining + 1):
                site = sites[position]
                base_log_probability = log_probability + skipped_log_probability
                best_later = get_best_completion(position + 1, remaining - 1)
                if (
                    base_log_probability + site.errors[0].log_probability + best_later
                    < prune_threshold
                ):
                    break  # later positions have smaller likelihood ratios, so they are pruned too
                for scored_error in site.errors:
                    child_log_probability = base_log_probability + scored_error.log_probability
                    if child_log_probability + best_later < prune_threshold:
                        break  # less likely local errors at this site are pruned too
                    children.append(
                        (
                            position + 1,
                            remaining - 1,
                            child_log_probability,
                            _ErrorSelection(selection, site.site_index, scored_error.error),
                        )
                    )
                skipped_log_probability += site.inactive_log_probability
            stack.extend(reversed(children))
