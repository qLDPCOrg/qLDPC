# SPDX-License-Identifier: Apache-2.0

"""Benchmark Brouwer-Zimmermann exact distance against exhaustive enumeration."""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import qldpc
from qldpc.objects import Pauli


@dataclass(frozen=True)
class BenchmarkCase:
    """One exact-distance benchmark case."""

    name: str
    get_distance: Callable[[qldpc.codes.DistanceMethod], int]


def _median_seconds(call: Callable[[], int], repeats: int) -> tuple[int, float]:
    """Return the common result and median end-to-end runtime of a callable."""
    call()
    results: set[int] = set()
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        results.add(call())
        samples.append(time.perf_counter() - start)
    if len(results) != 1:
        raise RuntimeError(f"Exact-distance calls disagreed: {sorted(results)}")
    return results.pop(), statistics.median(samples)


def _get_cases() -> list[BenchmarkCase]:
    """Build representative classical, CSS, and non-CSS benchmark cases."""
    hamming = qldpc.codes.HammingCode(5)
    steane = qldpc.codes.CSSCode.stack([qldpc.codes.SteaneCode()] * 7)
    five_qubit = qldpc.codes.QuditCode.stack([qldpc.codes.FiveQubitCode()] * 5)

    return [
        BenchmarkCase(
            "Hamming(5)",
            lambda method: qldpc.codes.get_distance_classical(
                hamming.generator,
                method=method,
            ),
        ),
        BenchmarkCase(
            "7 x Steane, X sector",
            lambda method: qldpc.codes.get_distance_quantum(
                steane.get_logical_ops(Pauli.X),
                steane.get_stabilizer_ops(Pauli.X),
                homogeneous=True,
                method=method,
            ),
        ),
        BenchmarkCase(
            "5 x FiveQubit",
            lambda method: qldpc.codes.get_distance_quantum(
                five_qubit.get_logical_ops(),
                five_qubit.get_stabilizer_ops(),
                method=method,
            ),
        ),
    ]


def _run_comparisons(
    cases: list[BenchmarkCase],
    *,
    repeats: int,
    minimum_speedup: float,
) -> None:
    """Run and report BZ-versus-brute-force comparisons."""
    print(f"\nEnd-to-end timings (median of {repeats})")
    print("-" * 88)
    for case in cases:
        distance_bz, seconds_bz = _median_seconds(
            partial(case.get_distance, "brouwer_zimmermann"),
            repeats,
        )
        distance_brute, seconds_brute = _median_seconds(
            partial(case.get_distance, "brute_force"),
            repeats,
        )
        if distance_bz != distance_brute:
            raise RuntimeError(
                f"{case.name}: BZ returned {distance_bz}, brute force returned {distance_brute}"
            )
        speedup = seconds_brute / seconds_bz
        print(
            f"{case.name:28} d={distance_bz:<3} "
            f"BZ={seconds_bz:9.6f}s brute={seconds_brute:9.6f}s speedup={speedup:7.1f}x"
        )
        if speedup < minimum_speedup:
            raise RuntimeError(
                f"{case.name}: {speedup:.2f}x speedup is below the required {minimum_speedup:.2f}x"
            )


def main() -> None:
    """Run exact-distance performance comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--minimum-speedup", type=float, default=2.0)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.minimum_speedup <= 0:
        parser.error("--minimum-speedup must be positive")

    cases = _get_cases()
    _run_comparisons(
        cases,
        repeats=args.repeats,
        minimum_speedup=args.minimum_speedup,
    )

    large_hamming = qldpc.codes.HammingCode(6)
    distance, seconds = _median_seconds(
        lambda: qldpc.codes.get_distance_classical(large_hamming.generator),
        args.repeats,
    )
    print(
        f"\nBZ-only scalability: Hamming(6) d={distance}, "
        f"median={seconds:.6f}s (brute force has 2**57 codewords)"
    )


if __name__ == "__main__":
    main()
