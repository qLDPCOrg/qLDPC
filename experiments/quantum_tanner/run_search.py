#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Search for quantum Tanner codes, enumerating subsets where cheap and sampling them otherwise.

This script constructs quantum Tanner codes (arXiv:2202.13641) from finite groups and pairs of
classical local codes.  Each code is defined by two symmetric subsets (closed under inverses) of a
group, whose sizes are the block lengths of the two local codes.  The bits of each local code are
labeled by the members of one subset.

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
quadripartite complex, and often half as many.  The search skips some groups and local codes
whose subsets cannot satisfy the Total No Conjugacy condition at all; see get_search_space.

By default, the subsets may contain the identity element of the group.  With --exclude-identity,
they exclude it, as in arXiv:2508.05095, whose subsets generate Cayley graphs.  Such generating
sets usually exclude the identity, which would add a loop at every vertex.

For each group and pair of local codes, the script enumerates the pairs of symmetric subsets of the
required sizes if there are at most --num-samples of them, keeping only pairs that satisfy the Total
No Conjugacy condition on a bipartite complex (see saved_codes.get_subset_pairs).  The enumeration
covers each such pair exactly once, and never constructs more codes than random sampling would.
Different pairs can still give the same code.

Otherwise, the script draws --num-samples random pairs with qldpc.codes.QTCode.random, using
deterministic seeds (see saved_codes.CodeSpec.build).  On a bipartite complex, it tries up to
saved_codes.MAX_TNC_ATTEMPTS seeds per random sample to find subsets that satisfy the Total No
Conjugacy condition, and skips the sample if none of them does.  Random sampling has two
limitations.  First, the subsets are not uniformly random (see
qldpc.abstract.Group.random_symmetric_subset), so the frequencies of code parameters among random
samples reflect this sampler, rather than the fractions of pairs of symmetric subsets that give
these parameters.  Second, different seeds can draw the same subsets, and thereby the same code.

The script upper-bounds the code distances by minimizing over --num-trials randomized trials, and
saves each code with its parameters to a text file in the "codes" subdirectory of --results-dir (by
default, results/codes in this directory).  Codes that already have a saved file are skipped unless
--overwrite is passed, even if that file used a different --num-trials, so an interrupted search
resumes where it left off.

Afterwards, compute_distances.py computes exact distances of saved codes where it can,
collect_results.py summarizes the parameters of saved codes in a CSV file in the same results
directory, and verify_codes.py checks that every saved code can be reproduced from its file name.

Examples:
    python experiments/quantum_tanner/run_search.py --max-order 6 --workers 8
    python experiments/quantum_tanner/run_search.py --groups dihedral --max-order 20 --bipartite \
        --exclude-identity --pairs all --base-codes Hamming-2 Random-5-2-0 Random-5-2-1 Random-5-2-2
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

# Scripts in this directory import each other as top-level modules.  mypy knows these files only as
# experiments.quantum_tanner.*, and would silently treat a bare "import saved_codes" as untyped.
if TYPE_CHECKING:
    from experiments.quantum_tanner import saved_codes
else:
    import saved_codes

# default search settings
MAX_ORDER = 10  # largest order of the groups to consider
NUM_SAMPLES = 100  # random samples per group and local codes, and the most pairs to enumerate
NUM_TRIALS = 1000  # for code distance calculations
NUM_WORKERS = max(1, (os.cpu_count() or 1) - 2)  # number of parallel jobs

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
    """Construct quantum Tanner codes in parallel, saving each code to a file."""
    args = parse_args()
    search_space = list(
        get_search_space(
            saved_codes.get_group_ids(args.groups, args.max_order),
            get_local_code_pairs(args.base_codes, args.pairs),
            bipartite=args.bipartite,
            exclude_identity=args.exclude_identity,
            num_samples=args.num_samples,
        )
    )
    print(f"Searching over {len(search_space)} samples")
    jobs = (
        functools.partial(
            run_and_save,
            spec,
            sample_id,
            progress=progress,
            num_trials=args.num_trials,
            codes_dir=args.results_dir / saved_codes.CODES_NAME,
            overwrite=args.overwrite,
        )
        for spec, sample_id, progress in search_space
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
        "--exclude-identity",
        action="store_true",
        help="exclude the identity element of each group from the subsets that define codes",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=NUM_SAMPLES,
        help="number of random codes per choice of group and local codes, unless there are at most"
        " this many pairs of subsets, which are then enumerated (default: %(default)s)",
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
    exclude_identity: bool,
    num_samples: int,
) -> Iterator[tuple[saved_codes.CodeSpec, str, str]]:
    """Iterate over the quantum Tanner codes to construct.

    Yields tuples (spec, sample_id, progress), where progress labels a sample among the samples of
    its specification spec, such as "3/100".  If spec has at most num_samples pairs of symmetric
    subsets of the required sizes, the samples are the enumerated pairs "e{index}", keeping only the
    pairs that satisfy the Total No Conjugacy condition on a bipartite complex (see
    saved_codes.get_subset_pairs).  Otherwise, the samples are num_samples random samples
    "r{index}".  A specification without pairs of subsets of the required sizes, such as one whose
    group has fewer elements than a local code has bits, has no samples.

    For a bipartite complex, also skips some groups whose subsets cannot satisfy the Total No
    Conjugacy condition.  A member s of both subsets violates this condition, since s * e == e * s
    for the identity element e.  The subsets must therefore be disjoint, so the group needs at least
    as many elements as the two local codes have bits together, plus one if the subsets exclude the
    identity element.  Moreover, a symmetric subset with an odd number of members contains a member
    that is its own inverse, which in a group of odd order is the identity element.  A group of odd
    order thereby has no disjoint symmetric subsets of odd sizes.  Other groups can still have no
    valid subsets, such as dihedral groups of order 2m with odd m for subsets of odd sizes without
    the identity (Lemma 3 of arXiv:2508.05095).  Their enumerations are empty, and their random
    samples are skipped after saved_codes.MAX_TNC_ATTEMPTS attempts each.
    """
    local_code_pairs = list(local_code_pairs)
    for group_id in group_ids:
        order = saved_codes.get_group_order(group_id)
        for code_a_id, code_b_id in local_code_pairs:
            # skip bipartite complexes whose subsets cannot be disjoint
            num_bits_a = saved_codes.get_local_code(code_a_id).num_bits
            num_bits_b = saved_codes.get_local_code(code_b_id).num_bits
            if bipartite and (
                order < num_bits_a + num_bits_b + exclude_identity
                or order % 2 == num_bits_a % 2 == num_bits_b % 2 == 1
            ):
                continue

            # enumerate pairs of subsets if there are few, and sample random pairs otherwise
            spec = saved_codes.CodeSpec(group_id, code_a_id, code_b_id, bipartite, exclude_identity)
            if saved_codes.count_subset_pairs(spec) <= num_samples:
                sample_ids = [
                    f"e{index}" for index in range(len(saved_codes.get_subset_pairs(spec)))
                ]
            else:
                sample_ids = [f"r{index}" for index in range(num_samples)]
            for index, sample_id in enumerate(sample_ids):
                yield spec, sample_id, f"{index + 1}/{len(sample_ids)}"


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
    sample_id: str,
    *,
    progress: str,
    num_trials: int,
    codes_dir: pathlib.Path,
    overwrite: bool,
) -> None:
    """Construct a quantum Tanner code, compute its parameters, and save it to a text file.

    Skips a code that already has a saved file, unless overwrite is True.  The subsets of the code
    are deterministic (see saved_codes.CodeSpec.build), so rerunning this function with the same
    inputs reproduces the same code.  For a random sample on a bipartite complex, this function
    skips the code if no attempt draws subsets that satisfy the Total No Conjugacy condition.

    SymPy PermutationGroup objects do not survive pickling, which a process pool uses to pass
    arguments to its workers.  The specification of a code therefore identifies its group, which
    this function constructs.

    Args:
        spec: The specification of the code.
        sample_id: The sample identifier of the code; see saved_codes.
        progress: A label of this sample among the samples of its specification, such as "3/100".
        num_trials: The number of randomized trials for upper-bounding the code distance.
        codes_dir: The directory in which to save the code.
        overwrite: If True, regenerate and save a code that already has a saved file.
    """
    # skip codes that already have a saved file, unless overwriting them
    path = spec.get_path(codes_dir, sample_id)
    if path.is_file() and not overwrite:
        return

    options_text = (" bipartite" if spec.bipartite else "") + (
        " without identity" if spec.exclude_identity else ""
    )
    job_id = f"{spec.group_id} {spec.code_a_id}+{spec.code_b_id}{options_text} {progress}"
    print(job_id)

    # construct the code
    try:
        code = spec.build(sample_id)
    except saved_codes.NoValidSubsetsError as error:
        print(f" ({job_id}) {error}")
        return

    # compute the parameters of the code
    code_params = code.get_code_params(bound=num_trials)
    weight = code.get_weight()

    # Save the code to a temporary file first, and then move it into place, so that an interrupted
    # job leaves no partial file that a resumed search would mistake for a finished code.
    temp_path = path.with_name(path.name + ".tmp")
    code.save(str(temp_path), *saved_codes.get_code_header(num_trials, code_params, weight))
    temp_path.replace(path)

    print(f" ({job_id}) (n, k, d) = {code_params}, stabilizer weight = {weight}")


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
