#!/usr/bin/env python3
"""Script to perform a randomized search for quantum Tanner codes

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
import hashlib
import os
from collections.abc import Hashable, Iterator

from qldpc import abstract, codes

SAVE_DIR = os.path.join(os.path.dirname(__file__), "codes")
MAX_ORDER = 10  # consider all finite groups up to this order
NUM_SAMPLES = 100  # per choice of group and subcode
NUM_TRIALS = 1000  # for code distance calculations


def get_deterministic_hash(*inputs: Hashable, num_bytes: int = 4) -> int:
    """Get a deterministic hash from the given inputs."""
    input_bytes = repr(inputs).encode("utf-8")
    hash_bytes = hashlib.sha256(input_bytes).digest()
    return int.from_bytes(hash_bytes[:num_bytes], byteorder="big", signed=False)


def get_small_groups(max_order: int = MAX_ORDER) -> Iterator[tuple[int, int]]:
    """Finite groups by order and index."""
    for order in range(1, max_order + 1):
        for index in range(1, abstract.SmallGroup.number(order) + 1):
            yield order, index


def get_cordaro_wagner_code(length: int) -> codes.ClassicalCode:
    """Cordaro Wagner code with a given block length."""
    return codes.ClassicalCode.from_name(f"CordaroWagnerCode({length})")


def get_mittal_code(length: int) -> codes.ClassicalCode:
    """Modified Hammming code of a given block length."""
    name = "MittalCode"
    full_code = codes.HammingCode(3)
    if length == 4:
        code = full_code.shortened([2, 3]).punctured([4])
    elif length == 5:
        code = full_code.shortened([2, 3])
    elif length == 6:
        code = full_code.shortened([3])
    else:
        raise ValueError(f"Unrecognized length for {name}: {length}")
    setattr(code, "_name", name)
    return code


def get_base_codes() -> Iterator[tuple[codes.ClassicalCode, str]]:
    """Iterator over classical codes and their identifiers."""
    for nn in [2, 3]:
        yield codes.HammingCode(nn), f"Hamming-{nn}"
    for nn in [3, 4, 5, 6]:
        yield get_cordaro_wagner_code(nn), f"CordaroWagner-{nn}"
    for nn in [4, 5, 6]:
        yield get_mittal_code(nn), f"Mittal-{nn}"


def run_and_save(
    group_order: int,
    group_index: int,
    base_code: codes.ClassicalCode,
    base_code_id: str,
    sample: int,
    num_samples: int = NUM_SAMPLES,
    num_trials: int = NUM_TRIALS,
    *,
    save_dir: str = SAVE_DIR,
    identify_completion_text: bool = False,
    override_existing_data: bool = False,
    silent: bool = False,
) -> None:
    """Generate a random quantum Tanner code, compute its distance, and save it to a text file.

    The multiprocessing module (which calls this method) is unable to properly handle SymPy
    PermutationGroup objects, so we have to construct groups here from their order and index.
    Passing ClassicalCode objects seems to be fine, though.
    """
    if group_order < base_code.num_bits:
        # No subset of this group can be large enough to have as many elements as there are bits in
        # the base code, so random quantum Tanner codes with the given input data do not exist.
        return None

    group_id = f"SmallGroup-{group_order}-{group_index}"
    seed = get_deterministic_hash(group_order, group_index, base_code.matrix.tobytes(), sample)
    file = f"qtcode_{group_id}_{base_code_id}_s{seed}.txt"
    path = os.path.join(save_dir, file)

    if os.path.isfile(path) and not override_existing_data:
        # we already have the data for this code, so there is nothing to do
        return None

    if not silent:
        job_id = f"{group_id} {base_code_id} {sample}/{num_samples}"
        print(job_id)

    # construct a random code and compute its parameters
    group = abstract.SmallGroup(group_order, group_index)
    code = codes.QTCode.random(group, base_code, seed=seed)
    code_params = code.get_code_params(bound=num_trials)
    weight = code.get_weight()

    # save code and computed parameters
    headers = [
        f"distance trials: {num_trials}",
        f"code parameters: {code_params}",
        f"stabilizer weight: {weight}",
    ]
    code.save(path, *headers)

    if not silent:
        completion_text = " "
        if identify_completion_text:
            completion_text += f"({job_id}) "
        completion_text += "code parameters: " + str(code_params + (weight,))
        print(completion_text)


if __name__ == "__main__":
    max_concurrent_jobs = num_cpus - 2 if (num_cpus := os.cpu_count()) else 1

    # run multiple jobs in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_concurrent_jobs) as executor:
        # iterate over all combinations of group, base code, and sample index
        for group_order, group_index in get_small_groups():
            for base_code, base_code_id in get_base_codes():
                for sample in range(NUM_SAMPLES):
                    # submit this job to the job queue
                    executor.submit(
                        run_and_save,
                        group_order,
                        group_index,
                        base_code,
                        base_code_id,
                        sample,
                        identify_completion_text=max_concurrent_jobs > 1,
                    )

                    if base_code.num_bits == group_order:
                        # There is only one instance of this random quantum Tanner code, namely the
                        # one in which subset_a = subset_b = group, so we only need one sample.
                        break
