# SPDX-License-Identifier: Apache-2.0

"""Collect the best codes of a campaign's logs, and the data computed from them.

Run with ``python -m experiments.code_annealing.collect_data [--scrape] [--test]``.  The script runs
these scripts of ``analysis/`` on the results directory, in order:

1. ``collect_annealed_codes`` keeps the best code of each size, from the logs and from earlier
   collections, and recounts each exactly.
2. ``half_supports`` counts the half and majority supports of those codes.
3. With ``--scrape``, ``scrape_qecdb`` saves a dated comparison with qecdb.org, which needs network
   access.
4. ``latex_tables`` writes the LaTeX tables and listings of the codes, compared with the newest
   scrape.  It is skipped if there is no scrape, or if the newest one is unreadable or does not
   match the current codes.

The files go to ``data/`` of the results directory, and the section "What the scripts write" of
``README.md`` lists them.  Each step also runs alone, with more options, as
``python -m experiments.code_annealing.analysis.NAME``.
"""

from __future__ import annotations

from collections.abc import Sequence

from .analysis import collect_annealed_codes, half_supports, latex_tables, scrape_qecdb
from .core.cli import (
    add_results_dir_argument,
    add_test_argument,
    get_results_dir,
    make_parser,
    run_step,
)
from .core.paths import find_code_files


def main(argv: Sequence[str] | None = None) -> None:
    """Collect the best codes of the logs, compute tables from them, and write the LaTeX files."""
    parser = make_parser(__doc__)
    parser.add_argument(
        "--scrape",
        action="store_true",
        help="query qecdb.org and save today's comparison to a dated data file",
    )
    add_results_dir_argument(parser)
    add_test_argument(parser)
    args = parser.parse_args(argv)

    results_dir = get_results_dir(args)
    options = ["--results-dir", str(results_dir)]
    run_step("collect_annealed_codes", collect_annealed_codes.main, options)
    run_step("half_supports", half_supports.main, options)
    if args.scrape:
        run_step("scrape_qecdb", scrape_qecdb.main, options)
    scraped = scrape_qecdb.find_data_file(results_dir)
    if scraped is None:
        print("\nSkipping latex_tables: no earlier scrape, and --scrape was not given")
        return
    try:
        scrape_qecdb.read_best_known(find_code_files(results_dir), scraped)
    except ValueError as error:
        print(f"\nSkipping latex_tables: {error}; use --scrape")
        return
    run_step("latex_tables", latex_tables.main, options)


if __name__ == "__main__":
    main()
