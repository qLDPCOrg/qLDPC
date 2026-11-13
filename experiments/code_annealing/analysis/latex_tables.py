# SPDX-License-Identifier: Apache-2.0

"""Write the LaTeX tables and the listings of the best codes, one of each for each code type.

Run with ``python -m experiments.code_annealing.analysis.latex_tables``.  The tables are those of
the paper whose code search this experiment implements, arXiv:2607.19574.  Each row is a code of
the code files, in their order, with:

- ``n``, ``k``, and ``d``;
- ``delta_QGV`` and ``delta_CSS``, the excess of ``d`` over the stabilizer and CSS
  Gilbert-Varshamov bounds of ``(n, k)`` (see ``gv_bounds.py``), with a dot for zero;
- the numbers of minimum-weight X-only and Z-only logical operators, in bold if neither exceeds
  ``--few-logicals``;
- the mean and maximum weight of the stabilizer generators, after ``reduce_weights``;
- the best known neighbors of the code at qecdb.org, from a scrape of the database,
  ``data/scrape_qecdb_YYYY-MM-DD.txt`` (see ``scrape_qecdb.py``): the newest one, or the one that
  ``--data-file`` names.  If no code of the database is at least as good in all of ``n``, ``k``,
  and ``d``, the code beats its three neighbors: its parameters are highlighted in green, and those
  of the neighbors in red.

Every distance and count is recomputed exactly, and a code whose distance or count differs from the
claims of its code file is an error.  The data file of qecdb.org must describe the current codes:
run ``collect_data --scrape`` to update it.

The tables go to ``data/table_css.tex`` and ``data/table_swel.tex`` of the results directory, and a
document includes one with ``\\input{table_css.tex}``.  Each file defines the commands of its
highlights within a group, so nothing leaks into the document, and a document that defines
``\\hlgreen`` and ``\\hlred`` itself keeps its own definitions.  The document only needs
``\\usepackage[table]{xcolor}`` and ``\\usepackage{makecell}`` in its preamble.

The listings go to ``data/listing_css.txt`` and ``data/listing_swel.txt``, for a document to show
verbatim, for example with ``\\lstinputlisting``.  They are the code files with a header such as
``# CSS (20, 4, 4)`` for each code.  A rerun with the same data writes the same files.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ..core.atomic import write_text_atomic
from ..core.cli import add_results_dir_argument, make_parser
from ..core.code_files import (
    CodeEntry,
    format_generators,
    get_reduced_weights,
    read_code_file,
    recount,
)
from ..core.paths import find_code_files, get_data_dir, get_listing_file, get_table_file
from ..core.proposals import CodeType
from .gv_bounds import get_css_gv_distance, get_qgv_distance
from .scrape_qecdb import BestKnown, find_data_file, read_best_known

_HEADER = (
    r"$n$ & $k$ & $d$ & $\delta_{\mathrm{QGV}}$ & $\delta_{\mathrm{CSS}}$"
    r" & \thead{min-weight\\nontrivial\\logicals} & \thead{mean\\stab.\\weight}"
    r" & \thead{max\\stab.\\weight} & $n_\mathrm{best}^{\mathrm{prior}}$"
    r" & $k_\mathrm{best}^{\mathrm{prior}}$ & $d_\mathrm{best}^{\mathrm{prior}}$"
)
_COLUMNS = "c|c|c||c|c||c|c|c||c|c|c"
# The column of the maximum stabilizer weight in ``_HEADER``, counted from 0, whose numbers are
# right-aligned in the LaTeX source so that their digits line up.
_MAX_WEIGHT_COLUMN = 7

# The preamble and the end of a table file.  A comment ends every line that does not end in a
# control word, so that the file adds no spaces where it is input.
_PREAMBLE = r"""% The {code_type} codes of the table of arXiv:2607.19574, written by
% python -m experiments.code_annealing.analysis.latex_tables from {code_file} and
% {data_file}.  Rerun the script instead of editing this file.
% Needs \usepackage[table]{{xcolor}} and \usepackage{{makecell}} in the preamble.
\begingroup
\providecommand{{\hlred}}[1]{{\cellcolor{{red!25}}#1}}%
\providecommand{{\hlgreen}}[1]{{\cellcolor{{green!25}}#1}}%
% Some classes, such as quantumarticle, leave \col@sep at 0pt inside tabular, so the highlight of a
% cell would not span the \tabcolsep gutters; restore it for this table.
\makeatletter
\col@sep\tabcolsep
\makeatother
"""
_END = "\\endgroup%\n"


def main(argv: Sequence[str] | None = None) -> None:
    """Write the table and the listing of each code type to their files."""
    parser = make_parser(__doc__)
    parser.add_argument(
        "--few-logicals",
        type=int,
        default=5,
        help="bold the counts of minimum-weight logical operators that are at most this"
        " (default: %(default)s)",
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        help="the scrape of qecdb.org to compare with (default: the newest in the data directory)",
    )
    add_results_dir_argument(parser)
    args = parser.parse_args(argv)

    # Read the codes and their neighbors at qecdb.org, and write a table and listing per code type.
    code_files = find_code_files(args.results_dir)
    data_file = args.data_file or find_data_file(args.results_dir)
    if data_file is None:
        raise SystemExit(
            f"No scrape of qecdb.org in {get_data_dir(args.results_dir)}: run collect_data --scrape"
        )
    try:
        best_known = read_best_known(code_files, data_file)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    for code_type, neighbors in best_known.items():
        code_file = code_files[code_type]
        entries = read_code_file(code_file)
        table = _format_table(entries, neighbors, args.few_logicals)
        preamble = _PREAMBLE.format(
            code_type=code_type.upper(), code_file=code_file.name, data_file=data_file.name
        )
        path = get_table_file(args.results_dir, code_type)
        write_text_atomic(path, preamble + table + "\n" + _END)
        print(f"Wrote {path}")
        path = get_listing_file(args.results_dir, code_type)
        write_text_atomic(path, _format_listing(entries, code_type))
        print(f"Wrote {path}")


####################################################################################################
# helper methods


def _format_table(
    entries: Sequence[CodeEntry], neighbors: Sequence[BestKnown], few_logicals: int
) -> str:
    """The ``tabular`` environment of the codes of one code file, with aligned columns."""
    rows = [
        _get_cells(entry, best_known, few_logicals)
        for entry, best_known in zip(entries, neighbors, strict=True)
    ]
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    lines = [
        " & ".join(
            cell.rjust(width) if column == _MAX_WEIGHT_COLUMN else cell.ljust(width)
            for column, (cell, width) in enumerate(zip(row, widths, strict=True))
        )
        for row in rows
    ]
    body = [rf"{line} \\ \hline" for line in lines[:-1]] + [lines[-1].rstrip()]
    return "\n".join(
        [
            rf"\begin{{tabular}}{{{_COLUMNS}}}",
            rf"  {_HEADER} \\ \hline\hline",
            *(f"  {line}" for line in body),
            r"\end{tabular}%",
        ]
    )


def _format_listing(entries: Sequence[CodeEntry], code_type: CodeType) -> str:
    """The blocks of the codes of a code file, each with a header such as ``# CSS (20, 4, 4)``."""
    return "\n".join(
        f"# {code_type.upper()} {entry.params}\n"
        + format_generators(entry.code.matrix_x, entry.code.matrix_z)
        for entry in entries
    )


def _get_cells(entry: CodeEntry, best_known: BestKnown, few_logicals: int) -> list[str]:
    """The cells of the row of a code, unpadded except for the decimal point of the mean weight."""
    nn, kk, dd = entry.params
    counts = recount(entry)
    logicals = f"{counts.count_x} / {counts.count_z}"
    if max(counts.count_x, counts.count_z) <= few_logicals:
        logicals = rf"\textbf{{{logicals}}}"
    weights = get_reduced_weights(entry.code)
    best = [
        "--" if value is None else str(value)
        for value in (best_known.n_best, best_known.k_best, best_known.d_best)
    ]
    undominated = best_known.undominated
    return [
        *(_highlight("hlgreen", str(value), undominated) for value in (nn, kk, dd)),
        _format_excess(dd - get_qgv_distance(nn, kk)),
        _format_excess(dd - get_css_gv_distance(nn, kk)),
        logicals,
        _format_mean(sum(weights) / len(weights)),
        str(max(weights)),
        *(_highlight("hlred", text, undominated) for text in best),
    ]


def _highlight(command: str, text: str, undominated: bool) -> str:
    """A cell wrapped in a highlighting command if the annealed code beats the database."""
    return rf"\{command}{{{text}}}" if undominated else text


def _format_excess(excess: int) -> str:
    """The excess of a distance over a bound: a dot for zero, and math mode for a negative one."""
    if excess == 0:
        return r"$\cdot$"
    return f"${excess}$" if excess < 0 else str(excess)


def _format_mean(mean: float) -> str:
    """A mean weight with two decimals, or none if it is a whole number, aligned on its point."""
    rounded = round(mean, 2)
    return f"{int(rounded):2d}   " if rounded == int(rounded) else f"{rounded:5.2f}"


if __name__ == "__main__":
    main()
