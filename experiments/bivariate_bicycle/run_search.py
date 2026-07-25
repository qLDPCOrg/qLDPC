#!/usr/bin/env python3
"""Script to perform a brute-force search for bivariate bicycle codes

Copyright 2023 The qLDPC Authors and Infleqtion Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import concurrent.futures
import itertools
import os
import sys

import diskcache
from sympy.abc import x, y

import qldpc
import qldpc.cache

MIN_ORDER = 3  # minimum cyclic group order
MIN_RATE = 1 / 36  # ignore codes with lower encoding rates
NUM_TRIALS = 1000  # for code distance calculations

CACHE_DIR = os.path.dirname(__file__)
CACHE_NAME = ".code_cache"


def get_quasi_cyclic_code_params(
    dims: tuple[int, int],
    exponents: tuple[int, int, int, int],
    min_rate: float,
    num_trials: int,
    *,
    silent: bool = False,
) -> tuple[int, int, int | None]:
    """Compute the code parameters of a bivariate bicycle code.

    If the code dimension (number of encoded logical qubits) is below some cutoff, return None.
    """
    # construct the code itself
    ax, ay, bx, by = exponents
    poly_a = 1 + x + x**ax * y**ay
    poly_b = 1 + y + x**bx * y**by
    code = qldpc.codes.BBCode(dims, poly_a, poly_b)

    if code.dimension < code.num_qubits * min_rate:
        return code.num_qubits, code.dimension, None

    if not silent:
        print(" starting", dims, exponents)

    distance = code.get_distance_bound(num_trials=num_trials)
    assert isinstance(distance, int)

    return code.num_qubits, code.dimension, distance


def run_and_save(
    dims: tuple[int, int],
    exponents: tuple[int, int, int, int],
    min_rate: float,
    num_trials: int,
    cache: diskcache.Cache,
    *,
    silent: bool = False,
) -> None:
    """Compute and save bivariate bicycle code parameters."""
    if not silent and exponents[2:] == (1, 0):
        print(dims, exponents)

    key = (dims, exponents, num_trials)

    # check whether we already have data for this code
    if key in cache:
        nn, kk, dd = cache[key]
        # if we know the distance or the encoding rate is too low, there's nothing more to do
        if dd is not None or kk < nn * min_rate:
            return

    # compute and save code parameters
    params = get_quasi_cyclic_code_params(dims, exponents, min_rate, num_trials, silent=silent)
    cache[key] = params

    nn, kk, dd = params
    if not silent and dd is not None:
        merit = kk * dd**2 / nn
        print("", dims, exponents, (nn, kk, dd), f"{merit:.2f}")


def redundant_or_trivial(dims: tuple[int, int], exponents: tuple[int, int, int, int]) -> bool:
    """Is the given bivariate bicycle code redundant?"""
    dim_x, dim_y = dims
    ax, ay, bx, by = exponents
    return (
        (dim_x, ax, ay) < (dim_y, by, bx)  # enforce torus width >= height
        or (ax, bx) > ((1 - ax) % dim_x, -bx % dim_x)  # reflection about x axis
        or (by, ay) > ((1 - by) % dim_y, -ay % dim_y)  # reflection about y axis
        or (ax, ay) in [(0, 0), (1, 0)]  # "trivial" polynomial
        or (bx, by) in [(0, 0), (0, 1)]  # "trivial" polynomial
    )


if __name__ == "__main__":
    dim_x = int(sys.argv[1])

    max_concurrent_jobs = num_cpus - 2 if (num_cpus := os.cpu_count()) else 1
    cache = qldpc.cache.get_disk_cache(CACHE_NAME, cache_dir=CACHE_DIR)

    # run multiple jobs in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_concurrent_jobs) as executor:
        for dim_y in range(MIN_ORDER, dim_x + 1):
            dims = (dim_x, dim_y)
            for exponents in itertools.product(
                range(dim_x), range(dim_y), range(dim_x), range(dim_y)
            ):
                if not redundant_or_trivial(dims, exponents):
                    executor.submit(run_and_save, dims, exponents, MIN_RATE, NUM_TRIALS, cache)
