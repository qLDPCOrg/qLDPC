#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Randomized search for quantum Tanner codes.

This script samples random quantum Tanner codes (arXiv:2202.13641) built from finite groups and
pairs of classical local codes.  It constructs each code with qldpc.codes.QTCode.random, which draws
two random symmetric subsets (closed under inverses) of a group, whose sizes are the block lengths
of the two local codes.  The bits of each local code are labeled by the members of one subset.

The search covers the groups of a family chosen by --groups, up to order --max-order: all groups in
the GAP SmallGroups library (the default), or dihedral groups of order 6 or more, which
arXiv:2508.05095 uses for the expansion of their Cayley graphs.  The local codes come from the
classical codes in --base-codes (by default, BASE_CODE_IDS; see saved_codes.get_local_code for
identifiers).  By default, each pair of local codes is a base code and its dual.  With "--pairs
all", the pairs are instead every ordered pair of codes with equal block lengths among the base
codes and their duals, so that local codes need not be dual, as in arXiv:2508.05095.

By default, the codes live on the quadripartite Cayley complex of arXiv:2206.07571.  This complex
is built on four copies of the group (G x Z_2 x Z_2), so that the subsets need not satisfy the Total
No Conjugacy condition, which the bipartite complex of arXiv:2202.13641 requires; see
qldpc.objects.CayleyComplex.  With --bipartite, the codes instead live on the bipartite complex.  A
code on this complex has at most as many qubits as the code with the same subsets on the
quadripartite complex, and often half as many.  The search tries up to MAX_TNC_ATTEMPTS seeds per
sample to find subsets that satisfy the Total No Conjugacy condition, and skips a sample if none of
them does.

The subsets are not uniformly random (see qldpc.abstract.Group.random_symmetric_subset).  The
frequencies of code parameters among saved codes therefore reflect this sampler, rather than the
fractions of all symmetric subsets that give these parameters.  Different seeds can also draw the
same subsets, and thereby the same code.

For each group and pair of local codes, the script draws --num-samples codes with deterministic
seeds, or one code if both local codes have as many bits as the group has elements.  It upper-bounds
their code distances by minimizing over --num-trials randomized trials, and saves each code with its
parameters to a text file in the "codes" subdirectory of --results-dir (by default, results/codes in
this directory).  Codes that already have a saved file are skipped unless --overwrite is passed,
even if that file used a different --num-trials, so an interrupted search resumes where it left off.

Afterwards, compute_distances.py computes exact distances of saved codes where it can,
collect_results.py summarizes the parameters of saved codes in a CSV file in the same results
directory, and verify_codes.py checks that every saved code can be reproduced from its seed.

Examples:
    python experiments/quantum_tanner/run_search.py --max-order 6 --workers 8
    python experiments/quantum_tanner/run_search.py --groups dihedral --max-order 20 --bipartite \
        --pairs all --base-codes Hamming-2 Random-5-2-0 Random-5-2-1 Random-5-2-2
"""

from __future__ import annotations

import argparse
import concurrent.futures
import functools
import os
import pathlib
import traceback
from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING

from qldpc import codes

# Scripts in this directory import each other as top-level modules, which mypy finds only by their
# paths in the repository.
if TYPE_CHECKING:
    from experiments.quantum_tanner import saved_codes
else:
    import saved_codes

# default search settings
MAX_ORDER = 10  # largest order of the groups to consider
NUM_SAMPLES = 100  # per choice of group and local codes
NUM_TRIALS = 1000  # for code distance calculations
NUM_WORKERS = max(1, (os.cpu_count() or 1) - 2)  # number of parallel jobs
MAX_TNC_ATTEMPTS = 100  # seeds to try per sample for a bipartite complex

# identifiers of the classical base codes to search over; see saved_codes.get_local_code
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
    search_space = list(
        get_search_space(
            saved_codes.get_group_ids(args.groups, args.max_order),
            get_local_code_pairs(args.base_codes, args.pairs),
            bipartite=args.bipartite,
            num_samples=args.num_samples,
        )
    )
    print(f"Searching over {len(search_space)} samples")
    jobs = (
        functools.partial(
            run_and_save,
            spec,
            sample,
            num_spec_samples=num_spec_samples,
            num_trials=args.num_trials,
            codes_dir=args.results_dir / saved_codes.CODES_NAME,
            overwrite=args.overwrite,
        )
        for spec, sample, num_spec_samples in search_space
    )
    run_jobs(jobs, args.workers)


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--groups",
        choices=["small", "dihedral"],
        default="small",
        help="family of groups to search over: all groups in the SmallGroups library, or dihedral"
        " groups of order 6 or more (default: %(default)s)",
    )
    parser.add_argument(
        "--max-order",
        type=int,
        default=MAX_ORDER,
        help="largest order of the groups to consider (default: %(default)s)",
    )
    parser.add_argument(
        "--base-codes",
        nargs="+",
        default=list(BASE_CODE_IDS),
        metavar="CODE_ID",
        help="identifiers of classical base codes (default: %(default)s)",
    )
    parser.add_argument(
        "--pairs",
        choices=["dual", "all"],
        default="dual",
        help="pairs of local codes: each base code with its dual, or all ordered pairs of codes"
        " with equal block lengths among the base codes and their duals (default: %(default)s)",
    )
    parser.add_argument(
        "--bipartite",
        action="store_true",
        help="build codes on the bipartite Cayley complex, rather than the quadripartite one",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=NUM_SAMPLES,
        help="number of random codes per choice of group and local codes, or one code if both"
        " local codes have as many bits as the group has elements (default: %(default)s)",
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
    saved_codes.add_results_dir_argument(parser)
    args = parser.parse_args()
    for name in ["max_order", "num_samples", "num_trials", "workers"]:
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for code_id in args.base_codes:
        try:
            saved_codes.get_local_code(code_id)
        except ValueError as error:
            parser.error(str(error))
    return args


####################################################################################################
# code search


def get_search_space(
    group_ids: Iterable[str],
    local_code_pairs: Iterable[tuple[str, str]],
    *,
    bipartite: bool,
    num_samples: int,
) -> Iterator[tuple[saved_codes.CodeSpec, int, int]]:
    """Iterate over the random quantum Tanner codes to sample.

    Yields tuples (spec, sample, num_spec_samples), where num_spec_samples is the number of samples
    with the specification spec.  Skips groups that are too small for a pair of local codes.  Such
    groups have fewer elements than a local code has bits, so they have no subset as large as the
    block length of that code.

    For a bipartite complex, also skips groups whose subsets cannot satisfy the Total No Conjugacy
    condition, which a common member s of both subsets violates with s * e == e * s.  The subsets
    must therefore be disjoint, so the group needs at least as many elements as the two local codes
    have bits together.  Moreover, a symmetric subset with an odd number of members contains a
    member that is its own inverse, which is the identity element in a group of odd order.  A group
    of odd order thereby has no disjoint symmetric subsets of odd sizes.
    """
    local_code_pairs = list(local_code_pairs)
    for group_id in group_ids:
        order = saved_codes.get_group_order(group_id)
        for code_a_id, code_b_id in local_code_pairs:
            # skip groups that are too small for these local codes
            num_bits_a = saved_codes.get_local_code(code_a_id).num_bits
            num_bits_b = saved_codes.get_local_code(code_b_id).num_bits
            min_order = num_bits_a + num_bits_b if bipartite else max(num_bits_a, num_bits_b)
            if order < min_order:
                continue

            # skip bipartite complexes whose subsets would both contain the identity element
            if bipartite and order % 2 == num_bits_a % 2 == num_bits_b % 2 == 1:
                continue

            # If both local codes have as many bits as the group has elements, there is only one
            # random quantum Tanner code, namely the one with subset_a = subset_b = group.
            num_spec_samples = 1 if num_bits_a == num_bits_b == order else num_samples
            spec = saved_codes.CodeSpec(group_id, code_a_id, code_b_id, bipartite)
            for sample in range(num_spec_samples):
                yield spec, sample, num_spec_samples


def get_local_code_pairs(base_code_ids: Iterable[str], pairing: str) -> list[tuple[str, str]]:
    """Get the identifiers of the pairs of local codes to search over.

    If pairing is "dual", pair each base code with its dual.  If pairing is "all", take every
    ordered pair of codes with equal block lengths among the base codes and their duals.
    """
    if pairing == "dual":
        return [(code_id, saved_codes.get_dual_id(code_id)) for code_id in base_code_ids]
    if pairing != "all":
        raise ValueError(f"Pairing of local codes not recognized: {pairing}")
    local_code_ids = list(
        dict.fromkeys(
            code_id
            for base_code_id in base_code_ids
            for code_id in (base_code_id, saved_codes.get_dual_id(base_code_id))
        )
    )
    return [
        (code_a_id, code_b_id)
        for code_a_id in local_code_ids
        for code_b_id in local_code_ids
        if saved_codes.get_local_code(code_a_id).num_bits
        == saved_codes.get_local_code(code_b_id).num_bits
    ]


def run_and_save(
    spec: saved_codes.CodeSpec,
    sample: int,
    *,
    num_spec_samples: int,
    num_trials: int,
    codes_dir: pathlib.Path,
    overwrite: bool,
) -> None:
    """Generate a random quantum Tanner code, compute its parameters, and save it to a text file.

    Skips a code that already has a saved file, unless overwrite is True.  The random seed for the
    code is deterministic (see saved_codes.CodeSpec.get_seed), so rerunning this function with the
    same inputs reproduces the same code.  For a bipartite complex, this function tries up to
    MAX_TNC_ATTEMPTS seeds, and skips the code if none of them satisfies the Total No Conjugacy
    condition.

    SymPy PermutationGroup objects do not survive pickling, which a process pool uses to pass
    arguments to its workers.  The specification of a code therefore identifies its group, which
    this function constructs.

    Args:
        spec: The specification of the code.
        sample: The index of this code among random samples with the same specification.
        num_spec_samples: The number of samples with this specification, used to report
            progress.
        num_trials: The number of randomized trials for upper-bounding the code distance.
        codes_dir: The directory in which to save the code.
        overwrite: If True, regenerate and save a code that already has a saved file.
    """
    # skip codes that already have a saved file, unless overwriting them
    num_attempts = MAX_TNC_ATTEMPTS if spec.bipartite else 1
    seeds = [spec.get_seed(sample, attempt) for attempt in range(num_attempts)]
    if not overwrite and any(spec.get_path(codes_dir, seed).is_file() for seed in seeds):
        return

    complex_text = " bipartite" if spec.bipartite else ""
    job_id = (
        f"{spec.group_id} {spec.code_a_id}+{spec.code_b_id}{complex_text}"
        f" {sample + 1}/{num_spec_samples}"
    )
    print(job_id)

    # construct a random code
    built = build_first_valid_code(spec, seeds)
    if built is None:
        print(f" ({job_id}) no subsets satisfying Total No Conjugacy in {num_attempts} attempts")
        return
    code, seed = built

    # compute the parameters of the code
    code_params = code.get_code_params(bound=num_trials)
    weight = code.get_weight()

    # Save the code to a temporary file first, and then move it into place, so that an interrupted
    # job leaves no partial file that a resumed search would mistake for a finished code.
    path = spec.get_path(codes_dir, seed)
    temp_path = path.with_name(path.name + ".tmp")
    code.save(str(temp_path), *saved_codes.get_code_header(num_trials, code_params, weight))
    temp_path.replace(path)

    print(f" ({job_id}) (n, k, d) = {code_params}, stabilizer weight = {weight}")


def build_first_valid_code(
    spec: saved_codes.CodeSpec, seeds: Iterable[int]
) -> tuple[codes.QTCode, int] | None:
    """Construct a code from the first of the given seeds that defines a valid code.

    Returns the code and its seed, or None if no seed defines a valid code.  Every seed defines a
    valid code on a quadripartite complex.  A seed defines a valid code on a bipartite complex only
    if its subsets satisfy the Total No Conjugacy condition.
    """
    # construct the group and local codes first, so that their errors are not mistaken below for
    # violations of the Total No Conjugacy condition
    saved_codes.get_group(spec.group_id)
    saved_codes.get_local_code(spec.code_a_id)
    saved_codes.get_local_code(spec.code_b_id)

    for seed in seeds:
        try:
            return spec.build(seed), seed
        except ValueError:
            # The group, local codes, and subset sizes are valid, so the only error expected here is
            # a violation of the Total No Conjugacy condition by the subsets of a bipartite complex.
            if not spec.bipartite:
                raise
    return None


####################################################################################################
# helper functions


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
