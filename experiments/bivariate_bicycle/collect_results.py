#!/usr/bin/env python3
"""Script to save bivariate bicycle code search results to data files.

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

import os

import numpy as np
from run_search import CACHE_DIR, CACHE_NAME, NUM_TRIALS

import qldpc.cache

# directory in which to save results
save_dir = os.path.join(os.path.dirname(__file__))

# data file headers
headers = [
    "AUTHOR: Michael A. Perlin, 2024",
    "bivariate bicycle codes of arXiv:2308.07915, with generating polynomials",
    "    A = 1 + x + x**ax * y**ay",
    "    B = 1 + y + x**bx * y**by",
    "here x and y are generators of cyclic groups with orders Rx and Ry",
    "code parameters [[n, k, d]] indicate",
    "    n = number of physical qubits",
    "    k = number of logical qubits",
    "    d = code distance (minimal weight of a nontrivial logical operator)",
    "code distance is estimated by the method of arXiv:2308.07915,"
    + f" minimizing over {NUM_TRIALS} trials",
    "the last column reports r = k d^2 / n",
    "topological 2D codes such as the toric code strictly satisfy r <= 1",
    "we only keep track of codes with r > 1",
    "",
    "Rx, Ry, ax, ay, bx, by, n, k, d, r",
]

# data format
fmt = "%d, %d, %d, %d, %d, %d, %d, %d, %d, %.3f"

##################################################

data: list[tuple[int | float, ...]] = []
last_print_val = -1

# iterate over all entries in the cache
cache = qldpc.cache.get_disk_cache(CACHE_NAME, cache_dir=CACHE_DIR)
for key in cache.iterkeys():
    # identify cyclic group orders and polynomial exponents
    dims, exponents, num_trials = key
    if num_trials != NUM_TRIALS:
        continue

    if (print_val := exponents[0]) != last_print_val:
        last_print_val = print_val
        print(dims, exponents)

    # retrieve code parameters
    nn, kk, dd = cache[key]
    if dd is None:
        continue

    # only report codes that outperform the surface code
    merit = kk * dd**2 / nn
    if merit > 1:
        code_data = (*dims, *exponents, nn, kk, dd, merit)
        data.append(code_data)

##################################################

# save codes to a data file
os.makedirs(save_dir, exist_ok=True)
path = os.path.join(save_dir, "codes.csv")
header = "\n".join(headers)
np.savetxt(path, data, header=header, fmt=fmt)
