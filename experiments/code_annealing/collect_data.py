# SPDX-License-Identifier: Apache-2.0

"""Turn the logs of a campaign into data: the best codes, and tables computed from them.

Run with ``python -m experiments.code_annealing.collect_data [--scrape] [--test]``.  The script
runs the scripts of ``analysis/``, in order, on the same results directory:

1. ``collect_annealed_codes`` saves the best code of each size that the logs hold, with the codes
   that earlier collections found, to ``data/annealed_codes.txt`` (CSS) and
   ``data/annealed_codes_swel.txt`` (SWEL).  Every code is recounted exactly.
2. ``half_supports`` counts the half-weight subsets of the supports of the minimum-weight logical
   operators of those codes, and saves the counts to ``data/half_supports.txt``.
3. With ``--scrape``, ``scrape_qecdb`` asks qecdb.org about the best known codes next to each code,
   which needs network access, and saves the answers to ``data/scrape_qecdb_YYYY-MM-DD.txt``, named
   by the date of the scrape.
4. ``latex_tables`` writes the LaTeX tables of the codes, with a comparison with qecdb.org from the
   newest scrape, to ``data/table_css.tex`` and ``data/table_swel.tex``.  This step is skipped if
   there is no scrape, or if the codes changed since the newest one.

Pass ``--test`` to collect the data of the smoke-test campaign of ``write_jobs.py --test``.  Every
step can be run alone, with more options, as ``python -m experiments.code_annealing.analysis.NAME``.
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
    """Collect the best codes of the logs, compute tables from them, and print the LaTeX tables."""
    parser = make_parser(__doc__)
    parser.add_argument(
        "--scrape", action="store_true", help="query qecdb.org and rewrite its data file"
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
    elif not scrape_qecdb.is_up_to_date(find_code_files(results_dir), scraped):
        print("\nSkipping latex_tables: the codes changed since the earlier scrape; use --scrape")
    else:
        run_step("latex_tables", latex_tables.main, options)


if __name__ == "__main__":
    main()
