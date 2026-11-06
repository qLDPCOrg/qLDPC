#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Prototype: scan one layer of the Brouwer-Zimmermann distance search with parallel threads.

The exact-distance search of qldpc.codes.distance spends nearly all of its time scanning coefficient
layers in the compiled kernel ``_scan_brouwer_zimmermann``.  This script tests whether that scan
parallelizes, without changing the library:

1. It runs ``get_distance_exact`` on a code until the search reaches a chosen layer, records the
   packed state of that layer, and stops the search.  It reports how long the search took to get
   there, and how much of that time was spent outside the compiled kernel (the serial setup).
2. It splits the supports of the layer, in lexicographic order, into equal ranges of combination
   indices.  To scan a range, it converts the first index to a support, rebuilds the prefix XORs of
   that support, and calls the unchanged kernel, compiled with ``nogil=True``, with a batch size
   equal to the length of the range.
3. With ``--serial``, it scans the layer in one call and range by range, and simulates dynamic
   scheduling of the measured range times on more workers than the machine may have.
4. It scans the ranges with a thread pool for each requested number of threads, sharing the best
   weight between ranges, and reports the speedup.

The number of supports scanned by a layer is fixed when the layer starts, so splitting it changes
neither the work nor the minimum weight found.  Layers grow by a factor of roughly ``r / t``, so a
layer that takes seconds is followed by one that takes minutes.  For example, layer 6 of bb288
takes a few minutes on one thread.

Example, using up to four threads::

    python experiments/parallel_distance/prototype.py bb288 5 --threads 1,2,4 --serial
"""

from __future__ import annotations

import argparse
import itertools
import math
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any
from unittest import mock

import numba
import numpy as np
import numpy.typing as npt
import sympy

import qldpc
from qldpc.codes.distance import _brouwer_zimmermann as bz

X, Y = sympy.symbols("x y")

CODES: dict[str, Callable[[], qldpc.codes.ClassicalCode | qldpc.codes.QuditCode]] = {
    "bb144": lambda: qldpc.codes.BBCode({X: 12, Y: 6}, X**3 + Y + Y**2, Y**3 + X + X**2),
    "bb288": lambda: qldpc.codes.BBCode({X: 12, Y: 12}, X**3 + Y**2 + Y**7, Y**3 + X + X**2),
    "bb90-noncss": lambda: qldpc.codes.QuditCode(
        qldpc.codes.BBCode({X: 15, Y: 3}, X**9 + Y + Y**2, 1 + X**2 + X**7).matrix
    ),
    "random-classical": lambda: qldpc.codes.ClassicalCode.random(200, 100, seed=3),
}

# The kernel of the library, compiled to release the GIL so that threads can run it concurrently.
SCAN_NOGIL = numba.njit(nogil=True)(bz._scan_brouwer_zimmermann.py_func)


@dataclass
class LayerState:
    """The packed search bases of one layer, as ``_BrouwerZimmermannSearch.advance`` scans it."""

    generators: npt.NDArray[np.uint64]
    labels: npt.NDArray[np.uint64]
    ranks: npt.NDArray[np.int64]
    widths: npt.NDArray[np.int64]
    label_widths: npt.NDArray[np.int64]
    best: int
    cutoff: int
    weight: int
    dimension: int

    @property
    def num_supports(self) -> int:
        """Number of supports in this layer."""
        return math.comb(self.dimension, self.weight)


class _Captured(Exception):
    """Raised to stop a search once the state of the target layer is recorded."""


def capture_layer(
    code: qldpc.codes.ClassicalCode | qldpc.codes.QuditCode, weight: int
) -> tuple[LayerState, float, float]:
    """Run the exact-distance search of a code until it reaches a layer, and record that layer.

    Returns the layer, the time the search took to reach it, and the part of that time spent in the
    compiled kernel.
    """
    states: list[LayerState] = []
    scan_seconds = [0.0]
    original_advance = bz._BrouwerZimmermannSearch.advance
    original_scan = bz._scan_brouwer_zimmermann

    def advance(search: bz._BrouwerZimmermannSearch) -> int:
        if not search.finished and search.next_weight == weight:
            num_sets = search.certificate.get_num_sets_to_certify(weight, search.best)
            states.append(
                LayerState(
                    search.generators[:num_sets].copy(),
                    search.labels[:num_sets].copy(),
                    search.ranks[:num_sets].copy(),
                    search.widths[:num_sets].copy(),
                    search.label_widths[:num_sets].copy(),
                    search.best,
                    max(-1, search.cutoff),
                    weight,
                    search.dimension,
                )
            )
            raise _Captured
        return original_advance(search)

    def scan(*args: Any) -> tuple[int, int, int]:
        start = time.perf_counter()
        result = original_scan(*args)
        scan_seconds[0] += time.perf_counter() - start
        return result

    start = time.perf_counter()
    with (
        mock.patch.object(bz._BrouwerZimmermannSearch, "advance", advance),
        mock.patch.object(bz, "_scan_brouwer_zimmermann", scan),
    ):
        try:
            distance = code.get_distance_exact()
        except _Captured:
            return states[0], time.perf_counter() - start, scan_seconds[0]
    raise SystemExit(f"The search finished, with distance {distance}, before layer {weight}")


def unrank_support(rank: int, dimension: int, weight: int) -> npt.NDArray[np.int64]:
    """The support of a given index among the weight-subsets of range(dimension), in lex order."""
    support = []
    row = 0
    for position in range(weight):
        while rank >= (count := math.comb(dimension - row - 1, weight - position - 1)):
            rank -= count
            row += 1
        support.append(row)
        row += 1
    return np.array(support, dtype=np.int64)


def scan_range(state: LayerState, start: int, count: int, best: int) -> int:
    """Scan the supports with indices in [start, start + count), and return the best weight."""
    support = unrank_support(start, state.dimension, state.weight)
    num_sets, _, num_words = state.generators.shape
    prefixes = np.zeros((state.weight + 1, num_sets, num_words), dtype=np.uint64)
    for level in range(state.weight - 1):
        prefixes[level + 1] = prefixes[level] ^ state.generators[:, support[level], :]
    best, depth, visited = SCAN_NOGIL(
        state.generators,
        state.labels,
        state.ranks,
        state.widths,
        state.label_widths,
        support,
        prefixes,
        state.weight - 1,
        best,
        state.cutoff,
        count,
    )
    assert visited == count or depth < 0
    return int(best)


def scan_in_threads(state: LayerState, ranges: list[tuple[int, int]], num_threads: int) -> int:
    """Scan ranges of a layer with a thread pool, sharing the best weight between ranges."""
    shared_best = [state.best]

    def work(start_count: tuple[int, int]) -> int:
        best = scan_range(state, *start_count, shared_best[0])
        shared_best[0] = min(shared_best[0], best)
        return best

    with ThreadPoolExecutor(num_threads) as pool:
        return min(pool.map(work, ranges))


def main() -> None:
    """Measure the speedup of scanning one layer in parallel threads."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("code", choices=sorted(CODES))
    parser.add_argument("layer", type=int, help="coefficient weight of the layer to scan (>= 2)")
    parser.add_argument("--threads", default="1,2", help="comma-separated thread counts to time")
    parser.add_argument("--chunks", type=int, default=256, help="number of ranges in the layer")
    parser.add_argument(
        "--serial",
        action="store_true",
        help="also time a serial scan, and simulate scheduling of the measured range times",
    )
    args = parser.parse_args()
    thread_counts = [int(num) for num in args.threads.split(",")]

    state, elapsed, scan_seconds = capture_layer(CODES[args.code](), args.layer)
    print(
        f"{args.code}: reached layer {args.layer} after {elapsed:.3f}s, of which {scan_seconds:.3f}s"
        f" in the kernel and {elapsed - scan_seconds:.3f}s of serial setup"
    )
    print(
        f"layer {args.layer}: {state.num_supports:,} supports in {len(state.ranks)} search bases"
        f" of dimension {state.dimension}, {state.generators.shape[2]} words per row,"
        f" best weight {state.best} at the start"
    )
    bounds = np.linspace(0, state.num_supports, args.chunks + 1).astype(np.int64)
    ranges = [(int(start), int(stop - start)) for start, stop in itertools.pairwise(bounds)]
    scan_range(state, 0, 1, state.best)  # compile the kernel

    serial_seconds = None
    if args.serial:
        start = time.perf_counter()
        best = scan_range(state, 0, state.num_supports, state.best)
        serial_seconds = time.perf_counter() - start
        print(f"serial scan in one call: {serial_seconds:.3f}s, best weight {best}")

        times = []
        best = state.best
        for range_start, count in ranges:
            start = time.perf_counter()
            best = min(best, scan_range(state, range_start, count, best))
            times.append(time.perf_counter() - start)
        range_times = np.array(times)
        mean = range_times.mean()
        print(
            f"serial scan in {args.chunks} ranges: {range_times.sum():.3f}s, best weight {best},"
            f" range time max/mean {range_times.max() / mean:.2f}, min/mean"
            f" {range_times.min() / mean:.2f}"
        )
        for num_workers in (2, 4, 8, 16, 32, 64):
            # Each worker that becomes free takes the next range.
            loads = np.zeros(num_workers)
            for range_time in range_times:
                loads[loads.argmin()] += range_time
            print(f"  simulated {num_workers} workers: {range_times.sum() / loads.max():.2f}x")

    for num_threads in thread_counts:
        start = time.perf_counter()
        best = scan_in_threads(state, ranges, num_threads)
        seconds = time.perf_counter() - start
        speedup = "" if serial_seconds is None else f", {serial_seconds / seconds:.2f}x"
        print(f"{num_threads} threads: {seconds:.3f}s, best weight {best}{speedup}", flush=True)


if __name__ == "__main__":
    main()
