# SPDX-License-Identifier: Apache-2.0

"""Observable decoder backed by the optional Frontier package.

Frontier performs approximate maximum-likelihood decoding over the logical classes of a binary
detector error model, using dynamic programming that prunes unlikely partial solutions.  It predicts
observable flips rather than an error, so this module provides only observable decoders.  Frontier
is imported only when settings are compiled for a detector error model.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import scipy.sparse
import stim

from ..dems import DetectorErrorModelArrays
from ..protocols import BatchObservableDecoder

# PyPI rejects direct-URL dependencies, so frontier cannot be a qldpc extra.
_INSTALL_COMMAND = (
    "pip install 'frontier @ git+https://github.com/aleverrier/frontier.git"
    "@5d5a60968182eb17cdc12c5ac1949732ead0905b'"
)

_METRIC_MODES = ("logsumexp_float", "frontier_lite")
_COLUMN_ORDERS = ("deadline_reorder", "time_order")


def _get_frontier() -> Any:
    """Import the optional upstream dependency or raise an actionable error."""
    try:
        import frontier
        import frontier.progressive
    except ModuleNotFoundError as error:
        if error.name != "frontier":
            raise
        raise ModuleNotFoundError(
            "The Frontier decoder requires the optional 'frontier' package. "
            f"Install it with `{_INSTALL_COMMAND}`."
        ) from error
    return frontier


@dataclasses.dataclass(frozen=True, slots=True)
class FrontierDecoder:
    """Settings for a Frontier decoder, which compile it for a detector error model.

    Frontier (https://github.com/aleverrier/frontier) scans the error mechanisms of a binary
    detector error model in a fixed order.  After each step, it groups partial solutions by the
    detectors and observables that they flip, and prunes unlikely groups.  It then predicts the most
    likely observable flips among the remaining groups that are consistent with the syndrome.
    Pruning makes this prediction approximate.

    Frontier only predicts observable flips, so these settings are accepted wherever an
    observable-decoder compiler is, such as by decoders.get_observable_decoder and
    decoders.SinterDecoder, but not where an error decoder is required.

    See help(decoders.get_observable_decoder_frontier) for the meaning of each setting.
    """

    K: int = 128
    Delta: float = 8.0
    score_alpha: float = 0.8
    metric_mode: Literal["logsumexp_float", "frontier_lite"] = "logsumexp_float"
    int_metric_scale: int = 1024
    column_order: Literal["deadline_reorder", "time_order"] = "deadline_reorder"
    committee: bool = False
    add_erasure_bit: bool = False

    def __post_init__(self) -> None:
        """Validate the settings, which Frontier would otherwise check only when decoding."""
        if self.K <= 0:
            raise ValueError("K must be positive")
        if not self.Delta >= 0:
            raise ValueError("Delta must be non-negative")
        if not (math.isfinite(self.score_alpha) and self.score_alpha >= 0):
            raise ValueError("score_alpha must be finite and non-negative")
        if self.metric_mode not in _METRIC_MODES:
            raise ValueError(f"metric_mode must be one of {_METRIC_MODES}")
        if self.int_metric_scale <= 0:
            raise ValueError("int_metric_scale must be positive")
        if self.column_order not in _COLUMN_ORDERS:
            raise ValueError(f"column_order must be one of {_COLUMN_ORDERS}")

    @property
    def options(self) -> dict[str, object]:
        """Return a copy of the settings, including defaults."""
        return {field.name: getattr(self, field.name) for field in dataclasses.fields(self)}

    def compile_decoder_for_dem(self, dem: stim.DetectorErrorModel) -> FrontierObservableDecoder:
        """Build a Frontier decoder that predicts the observable flips of a detector error model."""
        frontier = _get_frontier()
        dem_arrays = DetectorErrorModelArrays(dem)

        # Frontier requires at least one detector, so pad detector-free models with a dummy one.
        num_detectors = max(dem_arrays.num_detectors, 1)
        columns, layout = self._get_scan(
            frontier,
            _get_columns(dem_arrays, frontier.progressive.ProgressiveColumn),
            num_detectors,
        )
        backward_columns = backward_layout = None
        if self.committee:
            # Frontier would otherwise rebuild the reverse scan for every syndrome
            reversed_columns = [
                dataclasses.replace(column, index=index)
                for index, column in enumerate(reversed(columns))
            ]
            backward_columns, backward_layout = self._get_scan(
                frontier, reversed_columns, num_detectors
            )
        model = frontier.FrontierModel(
            columns=columns,
            layout=layout,
            num_detectors=num_detectors,
            num_observables=dem_arrays.num_observables,
            backward_columns=backward_columns,
            backward_layout=backward_layout,
        )
        return FrontierObservableDecoder(
            model,
            frontier.decode_frontier_committee if self.committee else frontier.decode_frontier,
            num_detectors=dem_arrays.num_detectors,
            decode_options={
                "K": self.K,
                "Delta": self.Delta,
                "score_alpha": self.score_alpha,
                "metric_mode": self.metric_mode,
                "int_metric_scale": self.int_metric_scale,
            },
            add_erasure_bit=self.add_erasure_bit,
        )

    def _get_scan(
        self, frontier: Any, columns: list[Any], num_detectors: int
    ) -> tuple[tuple[Any, ...], Any]:
        """Order the columns of one scan, and build the Frontier layout for that order."""
        if self.column_order == "deadline_reorder":
            columns, _ = frontier.progressive.optimize_column_order(
                columns, num_detectors=num_detectors
            )
        layout = frontier.progressive.build_frontier_layout(columns, num_detectors=num_detectors)
        return tuple(columns), layout

    def __repr__(self) -> str:
        """Show the helper call that reproduces these settings."""
        options = ", ".join(
            f"{field.name}={getattr(self, field.name)!r}"
            for field in dataclasses.fields(self)
            if getattr(self, field.name) != field.default
        )
        return f"decoders.frontier({options})"


class FrontierObservableDecoder(BatchObservableDecoder):
    """Frontier decoder that predicts the observable flips of one detector error model.

    Build one with decoders.get_observable_decoder_frontier, or by compiling FrontierDecoder
    settings for a detector error model.
    """

    def __init__(
        self,
        model: Any,
        decode_func: Callable[..., Any],
        *,
        num_detectors: int,
        decode_options: Mapping[str, object],
        add_erasure_bit: bool = False,
    ) -> None:
        """Initialize from a Frontier model and decoding function.

        Args:
            model: A frontier.FrontierModel, which may have more detectors than the syndromes to
                decode, in which case the extra detectors are never flipped.
            decode_func: frontier.decode_frontier or frontier.decode_frontier_committee.
            num_detectors: The number of detectors in a syndrome to decode.
            decode_options: Keyword arguments for decode_func.
            add_erasure_bit: Whether to append an erasure flag to every prediction.
        """
        self.model = model
        self.decode_func = decode_func
        self.num_detectors = num_detectors
        self.num_observables = int(model.num_observables)
        self.decode_options = dict(decode_options)
        self.has_erasure_bit = add_erasure_bit

    def decode_observables(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode one syndrome to predicted observable flips, followed by any erasure flag."""
        syndrome = np.asarray(syndrome, dtype=np.uint8)
        if syndrome.shape != (self.num_detectors,):
            raise ValueError(
                f"Expected a syndrome of shape ({self.num_detectors},), got {syndrome.shape}"
            )
        padding = np.zeros(self.model.num_detectors - self.num_detectors, dtype=np.uint8)
        result = self.decode_func(
            self.model, np.concatenate([syndrome, padding]), **self.decode_options
        )
        if result.status == "ok":
            logical_hat, erased = int(result.logical_hat), False
        elif result.status == "no_path":
            logical_hat, erased = 0, True
        else:
            raise RuntimeError(f"Frontier returned an unexpected status: {result.status!r}")
        flips = [(logical_hat >> index) & 1 for index in range(self.num_observables)]
        return np.array(flips + [erased] * self.has_erasure_bit, dtype=int)

    def decode_observables_batch(self, syndromes: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode a batch of syndromes, one per row, to predicted observable flips."""
        syndromes = np.asarray(syndromes, dtype=np.uint8)
        if syndromes.ndim != 2:
            raise ValueError(f"Expected a 2D batch of syndromes, got shape {syndromes.shape}")
        predictions = [self.decode_observables(syndrome) for syndrome in syndromes]
        return np.array(predictions, dtype=int).reshape(
            len(syndromes), self.num_observables + self.has_erasure_bit
        )


def get_observable_decoder_frontier(
    dem: stim.DetectorErrorModel,
    *,
    K: int = 128,
    Delta: float = 8.0,
    score_alpha: float = 0.8,
    metric_mode: Literal["logsumexp_float", "frontier_lite"] = "logsumexp_float",
    int_metric_scale: int = 1024,
    column_order: Literal["deadline_reorder", "time_order"] = "deadline_reorder",
    committee: bool = False,
    add_erasure_bit: bool = False,
) -> FrontierObservableDecoder:
    """Build a Frontier decoder that predicts the observable flips of a detector error model.

    Frontier is not a qLDPC dependency.  If it is missing, this function raises an error that shows
    how to install the version that qLDPC is tested against.  See help(decoders.FrontierDecoder) for
    a description of the algorithm.

    Args:
        dem: The binary detector error model to decode.
        K: The maximum number of groups of partial solutions to keep after each step.
        Delta: The maximum gap between the score of a kept group and the best score.  A score is
            the log-probability of a group, plus score_alpha times an estimate of how likely the
            remaining error mechanisms are to resolve the detectors that it flips.
        score_alpha: The weight of the estimate of future consistency in a score.
        metric_mode: ``"logsumexp_float"`` to add the probabilities of merged partial solutions,
            or ``"frontier_lite"`` to keep only the largest, using integer arithmetic.
            ``"frontier_lite"`` requires the native extension of Frontier.
        int_metric_scale: The scale of the integer log-probabilities used by ``"frontier_lite"``.
        column_order: ``"deadline_reorder"`` to let Frontier reorder error mechanisms so that
            detectors are resolved as early as possible, or ``"time_order"`` to scan them in the
            order of the detector error model.
        committee: If True, also scan the error mechanisms in reverse order (reordered again if
            column_order is ``"deadline_reorder"``), and keep the prediction of the scan that
            retains more probability.
        add_erasure_bit: If True, append an erasure flag to every prediction.  The flag is set if
            no remaining group is consistent with the syndrome, in which case the prediction is no
            observable flips, and is clear otherwise.

    Returns:
        A FrontierObservableDecoder.
    """
    return FrontierDecoder(
        K=K,
        Delta=Delta,
        score_alpha=score_alpha,
        metric_mode=metric_mode,
        int_metric_scale=int_metric_scale,
        column_order=column_order,
        committee=committee,
        add_erasure_bit=add_erasure_bit,
    ).compile_decoder_for_dem(dem)


def _get_columns(dem_arrays: DetectorErrorModelArrays, column_type: type) -> list[Any]:
    """Convert the error mechanisms of a detector error model into Frontier columns.

    Each column is a binary factor whose detector and observable flips are little-endian bit masks.
    """
    detector_matrix = dem_arrays.detector_flip_matrix
    observable_matrix = dem_arrays.observable_flip_matrix
    columns = []
    for index, probability in enumerate(dem_arrays.error_probs):
        detector_mask = _get_column_mask(detector_matrix, index)
        columns.append(
            column_type(
                family="qldpc",
                index=index,
                label=f"error_{index}",
                instruction_offset=index,
                prior_probs=(1 - float(probability), float(probability)),
                detector_response_masks=(0, detector_mask),
                logical_response_masks=(0, _get_column_mask(observable_matrix, index)),
                detector_support_mask=detector_mask,
                original_column_index=index,
            )
        )
    return columns


def _get_column_mask(matrix: scipy.sparse.csc_matrix, column: int) -> int:
    """Encode the support of a column of a binary sparse matrix as a little-endian bit mask."""
    rows = matrix.indices[matrix.indptr[column] : matrix.indptr[column + 1]]
    return sum(1 << int(row) for row in rows)


__all__ = [
    "FrontierDecoder",
    "FrontierObservableDecoder",
    "get_observable_decoder_frontier",
]
