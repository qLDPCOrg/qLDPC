#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Script to collect search results into a local database."""

from __future__ import annotations

import ast
import glob
import os

import run_randomized_search as search

import qldpc.cache

# identify where codes are saved
file_dir = os.path.dirname(__file__)
save_dir = os.path.join(file_dir, "codes")

# initialize database
cache = qldpc.cache.get_disk_cache(".code_cache", cache_dir=file_dir)

# loop over all groups and base codes
for group_order, group_index in search.get_small_groups(search.MAX_ORDER):
    group_id = f"SmallGroup-{group_order}-{group_index}"
    for _, base_code in search.get_base_codes():
        paths = f"{save_dir}/qtcode_{group_id}_{base_code}_s*.txt"

        # collect all data for into a list
        data = []
        for path in glob.glob(paths):
            seed = path.split("_")[-1].removesuffix(".txt")[1:]
            with open(path, "r") as file:
                num_trials = ast.literal_eval(file.readline().split(":")[-1])
                param_text = file.readline().split(":")[-1]
                if "0, nan)" in param_text:
                    # this is a trivial code, we can ignore it
                    continue
                params = ast.literal_eval(param_text)
                weight = ast.literal_eval(file.readline().split(":")[-1])

            data.append((*params, num_trials, weight))

        if data:
            # save data to the database
            cache[group_order, group_index, base_code] = data
