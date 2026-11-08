# SPDX-License-Identifier: Apache-2.0

"""Sample random codes, and save their exact distances and numbers of minimum-weight operators.

Run with
``python -m experiments.code_annealing.baseline.run_random N K [MIN_DISTANCE] --num-seeds 100``.
This is the baseline for annealing: single random codes of the same size that
have at least the minimum distance, a general CSS code for ``--code-type css`` and a self-dual code
for ``--code-type swel``.  These are not the starting codes of the campaign, whose searches start
from the lowest-energy code of many random codes, and whose CSS searches start from self-dual codes.
Each seed gives one code, and one line ``seed distance count`` of a file in the logs of the results
directory named like ``random_css_n30_k04_d5_s0-99.txt``, where ``count`` is the number of
minimum-weight X-only plus Z-only logical operators.  The seeds run from ``--first-seed`` for
``--num-seeds`` seeds, and ``plot_random_histogram.py`` reads the files.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import IO, cast

from ..core.atomic import write_atomic
from ..core.cli import add_results_dir_argument, make_parser
from ..core.paths import get_logs_dir
from ..core.proposals import CODE_TYPES, CodeType, random_initial_code

# A file of random codes, whose code type is one of CODE_TYPES; other files are not matched.
_FILE_NAME = re.compile(rf"random_({'|'.join(CODE_TYPES)})_n(\d+)_k(\d+)_d(\d+)_s(\d+)-(\d+)\.txt")


def main(argv: Sequence[str] | None = None) -> None:
    """Save the distance and number of minimum-weight logical operators of each random code."""
    parser = make_parser(__doc__)
    parser.add_argument("num_qubits", type=int, help="number of physical qubits")
    parser.add_argument("num_logicals", type=int, help="number of logical qubits")
    parser.add_argument(
        "min_distance", type=int, nargs="?", default=1, help="least distance of the codes"
    )
    parser.add_argument(
        "--first-seed", type=int, default=0, help="seed of the first code (default: %(default)s)"
    )
    parser.add_argument(
        "--num-seeds", type=int, default=100, help="codes to draw (default: %(default)s)"
    )
    parser.add_argument(
        "--code-type",
        choices=CODE_TYPES,
        default="css",
        help="code type of the random codes (default: %(default)s)",
    )
    add_results_dir_argument(parser)
    args = parser.parse_args(argv)
    if args.first_seed < 0 or args.num_seeds < 1:
        parser.error("--first-seed must be nonnegative and --num-seeds positive")

    seeds = range(args.first_seed, args.first_seed + args.num_seeds)
    output = get_logs_dir(args.results_dir) / _get_file_name(
        args.code_type, args.num_qubits, args.num_logicals, args.min_distance, seeds
    )

    def write(stream: IO[str]) -> None:
        for seed in seeds:
            counter = random_initial_code(
                args.num_qubits,
                args.num_logicals,
                seed=seed,
                self_dual=args.code_type == "swel",
                min_distance=args.min_distance,
            )
            counts = counter.count_logical_ops()
            print(seed, counts.weight, counts.count_x + counts.count_z, file=stream)

    write_atomic(output, write)
    print(f"Wrote {len(seeds)} codes to {output}")


def find_random_code_sizes(logs_dir: str | Path) -> list[tuple[CodeType, int, int]]:
    """The (code type, number of qubits, number of logical qubits) of every set of random codes."""
    sizes: set[tuple[CodeType, int, int]] = set()
    for path in Path(logs_dir).glob("random_*.txt"):
        match = _FILE_NAME.fullmatch(path.name)
        if match is not None:
            code_type = cast(CodeType, match.group(1))
            sizes.add((code_type, int(match.group(2)), int(match.group(3))))
    return sorted(sizes)


def read_random_counts(
    logs_dir: str | Path, code_type: CodeType, num_qubits: int, num_logicals: int
) -> dict[int, list[int]]:
    """The numbers of minimum-weight logical operators of random codes, by code distance.

    The counts come from the files that ``run_random`` saved in the logs directory.  A seed draws
    its codes from one stream, and keeps the first code that reaches the minimum distance of its
    file, so files with different minimum distances can hold the same code of a seed.  Each code
    counts once.

    Args:
        logs_dir: The logs directory of a results directory.
        code_type: The code type of the random codes.
        num_qubits: The number of physical qubits of the random codes.
        num_logicals: The number of logical qubits of the random codes.

    Returns:
        The counts of the random codes of each distance, by distance in increasing order.

    Raises:
        ValueError: If a line of a file is not ``seed distance count``.
    """
    results: dict[tuple[int, int], tuple[int, int]] = {}  # (min distance, seed) -> (d, count)
    for path in sorted(Path(logs_dir).glob("random_*.txt")):
        match = _FILE_NAME.fullmatch(path.name)
        if match is None:
            continue
        file_code_type, file_qubits, file_logicals, min_distance = match.groups()[:4]
        if (file_code_type, int(file_qubits), int(file_logicals)) != (
            code_type,
            num_qubits,
            num_logicals,
        ):
            continue
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            try:
                seed, distance, count = map(int, line.split())
            except ValueError:
                raise ValueError(
                    f"{path.name}, line {number}: expected 'seed distance count'"
                ) from None
            results[int(min_distance), seed] = distance, count

    # The code of a seed in a file of a smaller minimum distance is also the code of a larger one
    # that it reaches, since every earlier code of the stream falls short of the smaller one.
    unique = {
        (min_distance, seed): value
        for (min_distance, seed), value in results.items()
        if not any(
            other_seed == seed and other_min < min_distance <= other_value[0]
            for (other_min, other_seed), other_value in results.items()
        )
    }
    counts_by_distance: defaultdict[int, list[int]] = defaultdict(list)
    for distance, count in unique.values():
        counts_by_distance[distance].append(count)
    return dict(sorted(counts_by_distance.items()))


####################################################################################################
# helper methods


def _get_file_name(
    code_type: CodeType, num_qubits: int, num_logicals: int, min_distance: int, seeds: range
) -> str:
    """The name of the file of random codes, such as ``random_css_n30_k04_d5_s0-99.txt``."""
    return (
        f"random_{code_type}_n{num_qubits}_k{num_logicals:02d}_d{min_distance}"
        f"_s{seeds[0]}-{seeds[-1]}.txt"
    )


if __name__ == "__main__":
    main()
