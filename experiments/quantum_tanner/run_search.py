#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Randomized search for quantum Tanner codes.

This script samples random quantum Tanner codes (arXiv:2202.13641) built from every finite group of
order at most --max-order, as indexed by the GAP SmallGroups library, and every classical base code
in BASE_CODE_IDS.  It constructs each code with qldpc.codes.QTCode.random, which draws two random
symmetric subsets (closed under inverses) of the group, whose size is the block length of the base
code.  The base code is the local code on one subset, and its dual code is the local code on the
other.  The code lives on the quadripartite Cayley complex of arXiv:2206.07571, which is built on
four copies of the group (G x Z_2 x Z_2) so that the subsets need not satisfy the conditions of the
bipartite complex of arXiv:2202.13641; see qldpc.objects.CayleyComplex.

The subsets are not uniformly random (see qldpc.abstract.Group.random_symmetric_subset).  The
frequencies of code parameters among saved codes therefore reflect this sampler, rather than the
fractions of all symmetric subsets that give these parameters.

For each group and base code, the script draws --num-samples codes with deterministic seeds, or one
code if the base code has as many bits as the group has elements.  It upper-bounds their code
distances by minimizing over --num-trials randomized trials, and saves each code with its parameters
to a text file in the "codes" subdirectory of --results-dir (by default, results/codes in this
directory).  Codes that already have a saved file are skipped unless --overwrite is passed, even if
that file used a different --num-trials, so an interrupted search resumes where it left off.

Afterwards, collect_results.py summarizes the parameters of saved codes in a CSV file in the same
results directory, and verify_codes.py checks that every saved code can be reproduced from its
seed.  Both scripts rely on helpers in this module.

Example:
    python experiments/quantum_tanner/run_search.py --max-order 6 --workers 8
"""

from __future__ import annotations

import argparse
import concurrent.futures
import functools
import hashlib
import math
import os
import pathlib
import re
import traceback
from collections.abc import Callable, Iterable, Iterator

from qldpc import abstract, codes

RESULTS_DIR = pathlib.Path(__file__).parent / "results"  # default directory of all results
CODES_NAME = "codes"  # subdirectory of the results directory that holds saved codes

# default search settings
MAX_ORDER = 10  # consider all finite groups up to this order
NUM_SAMPLES = 100  # per choice of group and base code
NUM_TRIALS = 1000  # for code distance calculations
NUM_WORKERS = max(1, (os.cpu_count() or 1) - 2)  # number of parallel jobs

# identifiers of the classical base codes to search over; see get_base_code
BASE_CODE_IDS = (
    "Hamming-2",
    "Hamming-3",
    "CordaroWagner-3",
    "CordaroWagner-4",
    "CordaroWagner-5",
    "CordaroWagner-6",
    "Mittal-4",
    "Mittal-5",
    "Mittal-6",
)

####################################################################################################
# command-line interface


def main() -> None:
    """Sample random quantum Tanner codes in parallel, saving each code to a file."""
    args = parse_args()
    jobs = (
        functools.partial(
            run_and_save,
            group_order,
            group_index,
            base_code_id,
            sample,
            num_code_samples=num_code_samples,
            num_trials=args.num_trials,
            codes_dir=args.results_dir / CODES_NAME,
            overwrite=args.overwrite,
        )
        for group_order, group_index, base_code_id, sample, num_code_samples in (
            get_search_space(args.max_order, args.num_samples)
        )
    )
    run_jobs(jobs, args.workers)


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--max-order",
        type=int,
        default=MAX_ORDER,
        help="consider all finite groups up to this order (default: %(default)s)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=NUM_SAMPLES,
        help="number of random codes per choice of group and base code, or one code if the base"
        " code has as many bits as the group has elements (default: %(default)s)",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=NUM_TRIALS,
        help="number of randomized trials for upper-bounding code distances (default: %(default)s)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=NUM_WORKERS,
        help="number of parallel jobs (default: %(default)s)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="regenerate and overwrite codes that already have a saved file",
    )
    add_results_dir_argument(parser)
    args = parser.parse_args()
    for name in ["max_order", "num_samples", "num_trials", "workers"]:
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def add_results_dir_argument(parser: argparse.ArgumentParser) -> None:
    """Add the --results-dir option, which is shared by all quantum Tanner code scripts."""
    parser.add_argument(
        "--results-dir",
        type=pathlib.Path,
        default=RESULTS_DIR,
        help="directory of saved codes and their summary"
        " (default: the 'results' directory beside this script)",
    )


####################################################################################################
# code search


def get_search_space(max_order: int, num_samples: int) -> Iterator[tuple[int, int, str, int, int]]:
    """Iterate over the random quantum Tanner codes to sample.

    Yields tuples (group_order, group_index, base_code_id, sample, num_code_samples), where
    num_code_samples is the number of samples for this group and base code.  Skips groups with fewer
    elements than the base code has bits, which define no quantum Tanner code with that base code.
    """
    for group_order, group_index in get_small_groups(max_order):
        for base_code_id in BASE_CODE_IDS:
            # A group with fewer elements than the base code has bits has no subset as large as the
            # block length of the base code, and thereby defines no quantum Tanner code with it.
            num_bits = get_base_code(base_code_id).num_bits
            if group_order < num_bits:
                continue

            # If the base code has as many bits as the group has elements, there is only one random
            # quantum Tanner code, namely the one with subset_a = subset_b = group.
            num_code_samples = 1 if num_bits == group_order else num_samples
            for sample in range(num_code_samples):
                yield group_order, group_index, base_code_id, sample, num_code_samples


def run_and_save(
    group_order: int,
    group_index: int,
    base_code_id: str,
    sample: int,
    *,
    num_code_samples: int,
    num_trials: int,
    codes_dir: pathlib.Path,
    overwrite: bool = False,
) -> None:
    """Generate a random quantum Tanner code, compute its parameters, and save it to a text file.

    The random seed for the code is a deterministic hash of the group, the bytes of the parity check
    matrix of the base code, and the sample index, so rerunning this function with the same inputs
    reproduces the same code.  These bytes depend on the rows and data type of the matrix, so a
    change to how a base code is constructed changes the seeds of its codes.

    SymPy PermutationGroup objects do not survive pickling, which a process pool uses to pass
    arguments to its workers.  This function therefore constructs the group from its order and
    index.

    Args:
        group_order: The order of the group that defines the code.
        group_index: The index of the group among those of the same order in the SmallGroups
            library.
        base_code_id: An identifier of the classical base code that defines local constraints of
            the code; see get_base_code.
        sample: The index of this code among random samples with the same group and base code.
        num_code_samples: The number of samples for this group and base code, used to report
            progress.
        num_trials: The number of randomized trials for upper-bounding the code distance.
        codes_dir: The directory in which to save the code.
        overwrite: If True, regenerate and save a code that already has a saved file.
    """
    # skip codes that already have a saved file, unless overwriting them
    base_code = get_base_code(base_code_id)
    seed = get_deterministic_hash(group_order, group_index, base_code.matrix.tobytes(), sample)
    path = get_code_path(codes_dir, group_order, group_index, base_code_id, seed)
    if path.is_file() and not overwrite:
        return

    group_id = get_group_id(group_order, group_index)
    job_id = f"{group_id} {base_code_id} {sample + 1}/{num_code_samples}"
    print(job_id)

    # construct a random code and compute its parameters
    group = abstract.SmallGroup(group_order, group_index)
    code = codes.QTCode.random(group, base_code, seed=seed)
    code_params = code.get_code_params(bound=num_trials)
    weight = code.get_weight()

    # Save the code to a temporary file first, and then move it into place, so that an interrupted
    # job leaves no partial file that a resumed search would mistake for a finished code.
    temp_path = path.with_name(path.name + ".tmp")
    code.save(str(temp_path), *get_code_header(num_trials, code_params, weight))
    temp_path.replace(path)

    print(f" ({job_id}) (n, k, d) = {code_params}, stabilizer weight = {weight}")


def get_small_groups(max_order: int) -> Iterator[tuple[int, int]]:
    """Iterate over (order, index) for all groups in the SmallGroups library up to a given order."""
    for order in range(1, max_order + 1):
        for index in range(1, abstract.SmallGroup.count_groups(order) + 1):
            yield order, index


####################################################################################################
# base codes


@functools.cache
def get_base_code(base_code_id: str) -> codes.ClassicalCode:
    """Construct a classical base code from an identifier of the form "{family}-{parameter}".

    The family is "Hamming", with the number r of parity checks as its parameter and block length
    2^r - 1, or "CordaroWagner" or "Mittal", with the block length of the code as its parameter.
    The base codes in BASE_CODE_IDS have parameters [n, k, d]:
    - Hamming-2: [3, 1, 3], and Hamming-3: [7, 4, 3],
    - CordaroWagner-3 to CordaroWagner-6: [3, 2, 2], [4, 2, 2], [5, 2, 3], and [6, 2, 4], and
    - Mittal-4 to Mittal-6: [4, 2, 2], [5, 2, 3], and [6, 3, 3].
    """
    family, parameter = base_code_id.split("-")
    if family == "Hamming":
        return codes.HammingCode(int(parameter))
    if family == "CordaroWagner":
        return get_cordaro_wagner_code(int(parameter))
    if family == "Mittal":
        return get_mittal_code(int(parameter))
    raise ValueError(f"Base code not recognized: {base_code_id}")


def get_cordaro_wagner_code(length: int) -> codes.ClassicalCode:
    """Get the Cordaro-Wagner code with a given block length, as constructed by GUAVA.

    Cordaro-Wagner codes are binary codes of dimension 2 with the largest minimum distance possible
    for their block length.
    """
    return codes.ClassicalCode.from_name(f"CordaroWagnerCode({length})")


def get_mittal_code(length: int) -> codes.ClassicalCode:
    """Get a modified [7, 4] Hamming code with a given block length (4, 5, or 6).

    Lengths 6 and 5 are obtained by shortening the Hamming code, and length 4 is obtained by further
    puncturing the length-5 code.
    """
    full_code = codes.HammingCode(3)
    if length == 4:
        return full_code.shortened([2, 3]).punctured([4])
    if length == 5:
        return full_code.shortened([2, 3])
    if length == 6:
        return full_code.shortened([3])
    raise ValueError(f"Unrecognized length for a Mittal code: {length}")


####################################################################################################
# saved code files


def get_code_path(
    codes_dir: pathlib.Path,
    group_order: int,
    group_index: int,
    base_code_id: str,
    seed: int,
) -> pathlib.Path:
    """Get the path of the file that saves a quantum Tanner code; see parse_code_path."""
    group_id = get_group_id(group_order, group_index)
    return codes_dir / f"qtcode_{group_id}_{base_code_id}_s{seed}.txt"


def get_saved_code_paths(codes_dir: pathlib.Path) -> list[pathlib.Path]:
    """Get the sorted paths of all saved quantum Tanner codes in a directory.

    Raises a FileNotFoundError if there are none.
    """
    paths = sorted(codes_dir.glob("qtcode_*.txt"))
    if not paths:
        raise FileNotFoundError(f"No saved quantum Tanner codes found in {codes_dir}")
    return paths


def parse_code_path(path: pathlib.Path) -> tuple[int, int, str, int]:
    """Get the group order, group index, base code ID, and seed of a saved code from its path."""
    match = re.fullmatch(r"qtcode_SmallGroup-(\d+)-(\d+)_([A-Za-z]+-\d+)_s(\d+)\.txt", path.name)
    if match is None:
        raise ValueError(f"Unrecognized quantum Tanner code file name: {path}")
    group_order, group_index, base_code_id, seed = match.groups()
    return int(group_order), int(group_index), base_code_id, int(seed)


def get_group_id(group_order: int, group_index: int) -> str:
    """Get an identifier for a group in the SmallGroups library."""
    return f"SmallGroup-{group_order}-{group_index}"


def get_code_header(
    num_trials: int, code_params: tuple[int, int, int | float], weight: int
) -> list[str]:
    """Get the header lines of a saved quantum Tanner code file; see read_code_header."""
    return [
        f"distance trials: {num_trials}",
        f"code parameters: {code_params}",
        f"stabilizer weight: {weight}",
    ]


def read_code_header(path: pathlib.Path) -> tuple[int, tuple[int, int, int | float], int]:
    """Read the header of a saved quantum Tanner code file; see get_code_header.

    Returns the number of distance trials, the code parameters (n, k, d), and the maximum weight of
    a parity check of the code.  The distance d is nan if the code has no logical qubits.
    """
    with open(path) as file:
        num_trials = int(file.readline().split(":")[-1])
        param_text = file.readline().split(":")[-1]
        weight = int(file.readline().split(":")[-1])

    # parse code parameters written as "(n, k, d)", where d is nan if k = 0
    num_qudits, dimension, distance = (
        float(value) for value in param_text.strip(" ()\n").split(",")
    )
    code_params = (
        int(num_qudits),
        int(dimension),
        distance if math.isnan(distance) else int(distance),
    )
    return num_trials, code_params, weight


####################################################################################################
# helper functions


def get_deterministic_hash(*inputs: object) -> int:
    """Hash inputs by their repr, consistently across Python sessions (unlike the built-in hash).

    Each input needs a repr that is the same in every Python session.  Seeds of saved codes are such
    hashes, so changing this function changes which codes a search samples, and stops a search from
    resuming from codes saved before the change.
    """
    input_bytes = repr(inputs).encode("utf-8")
    hash_bytes = hashlib.sha256(input_bytes).digest()
    return int.from_bytes(hash_bytes[:4], byteorder="big", signed=False)


def run_jobs(jobs: Iterable[Callable[[], None]], num_workers: int) -> None:
    """Run picklable jobs in parallel processes, submitting a few at a time to bound memory use.

    Prints the traceback of every job that fails, and raises a RuntimeError at the end if any did.
    """
    max_pending_jobs = 4 * num_workers
    num_jobs = num_failures = 0
    pending: set[concurrent.futures.Future[None]] = set()
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for job in jobs:
            while len(pending) >= max_pending_jobs:
                done, pending = concurrent.futures.wait(
                    pending, return_when=concurrent.futures.FIRST_COMPLETED
                )
                num_failures += sum(report_failure(future) for future in done)
            pending.add(executor.submit(job))
            num_jobs += 1
        remaining = concurrent.futures.as_completed(pending)
        num_failures += sum(report_failure(future) for future in remaining)
    if num_failures:
        raise RuntimeError(f"{num_failures} of {num_jobs} jobs failed")


def report_failure(future: concurrent.futures.Future[None]) -> bool:
    """Print the traceback of a finished job if it failed, and return whether it failed."""
    if (exception := future.exception()) is not None:
        traceback.print_exception(exception)
        return True
    return False


if __name__ == "__main__":
    main()
