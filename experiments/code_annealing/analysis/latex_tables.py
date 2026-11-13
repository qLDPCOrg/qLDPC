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
  ``data/scrape_qecdb_YYYY-MM-DD.txt`` (see ``scrape_qecdb.py``): the newest one, or those that
  ``--data-file`` names.  If no code of the database is at least as good in all of ``n``, ``k``,
  and ``d``, the code beats its three neighbors: its parameters are highlighted in green, and those
  of the neighbors in red.

One scrape must list exactly the current codes, in order: run ``collect_data --scrape`` to update
it.  ``--data-file`` can instead name several scrapes, oldest first, to keep the comparisons of an
older scrape, such as the one before a paper, for the codes that it lists.  Each row then takes its
neighbors from the first scrape that has a line for its exact ``(n, k, d)``, since its neighbors
depend on all three, and a row from a later scrape gets a dagger on its ``n``.  A row whose code
beats its scrape, but not the newest scrape, keeps the comparison with its scrape and gets a double
dagger; if the newest scrape does not list the code, the row gets none.  A scrape names no
code types, so each of several scrapes must have one block for each code type, as the scrape of a
results directory with codes of every code type has.  It is an error if no scrape lists a code.
The header comment of a table names the scrapes that its rows use.

Every distance and count is recomputed exactly, and a code whose distance or count differs from the
claims of its code file is an error.

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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..core.atomic import write_text_atomic
from ..core.cli import add_results_dir_argument, make_parser
from ..core.code_files import (
    CodeEntry,
    CodeParams,
    format_generators,
    get_reduced_weights,
    read_code_file,
    recount,
)
from ..core.paths import find_code_files, get_data_dir, get_listing_file, get_table_file
from ..core.proposals import CodeType
from .gv_bounds import get_css_gv_distance, get_qgv_distance
from .scrape_qecdb import BestKnown, find_data_file, read_best_known, read_data_file

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
% {data_files}.  Rerun the script instead of editing this file.
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
        nargs="+",
        help="the scrapes of qecdb.org to compare with, oldest first; each code uses the first that"
        " has a line for its exact (n, k, d); a code from a later scrape gets a dagger, and a code"
        " that beats its scrape but not the newest a double dagger (default: the newest in the"
        " data directory)",
    )
    add_results_dir_argument(parser)
    args = parser.parse_args(argv)

    # Read the codes and their neighbors at qecdb.org, and write a table and listing per code type.
    code_files = find_code_files(args.results_dir)
    data_files = args.data_file or [find_data_file(args.results_dir)]
    if data_files[0] is None:
        raise SystemExit(
            f"No scrape of qecdb.org in {get_data_dir(args.results_dir)}: run collect_data --scrape"
        )
    try:
        if len(data_files) == 1:
            # One scrape describes exactly the current codes, in order, as it always has.
            best_known = read_best_known(code_files, data_files[0])
            scrapes = [
                {
                    code_type: {row.params: row for row in rows}
                    for code_type, rows in best_known.items()
                }
            ]
        else:
            scrapes = [read_data_file(data_file) for data_file in data_files]
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    for code_type, code_file in code_files.items():
        entries = read_code_file(code_file)
        comparisons = [
            _find_neighbors(entry, [scrape[code_type] for scrape in scrapes]) for entry in entries
        ]
        table = _format_table(entries, comparisons, args.few_logicals)
        preamble = _PREAMBLE.format(
            code_type=code_type.upper(),
            code_file=code_file.name,
            data_files=_describe_data_files(data_files, comparisons),
        )
        path = get_table_file(args.results_dir, code_type)
        write_text_atomic(path, preamble + table + "\n" + _END)
        print(f"Wrote {path}")
        path = get_listing_file(args.results_dir, code_type)
        write_text_atomic(path, _format_listing(entries, code_type))
        print(f"Wrote {path}")


####################################################################################################
# helper methods


@dataclass(frozen=True)
class _Comparison:
    """The neighbors of a code in the scrape that its row uses, and the markers of the row."""

    best_known: BestKnown
    scrape_index: int
    """The index of the scrape, among those given, oldest first."""
    dominated_since: bool
    """Whether the newest scrape lists a code at least as good, although this scrape does not."""


def _find_neighbors(
    entry: CodeEntry, scrapes: Sequence[Mapping[CodeParams, BestKnown]]
) -> _Comparison:
    """The neighbors of a code in the first scrape that lists it, compared with the newest scrape.

    Raises:
        SystemExit: If no scrape lists the code.
    """
    newest = scrapes[-1].get(entry.params)
    for index, scrape in enumerate(scrapes):
        if entry.params in scrape:
            best_known = scrape[entry.params]
            dominated_since = (
                best_known.undominated and newest is not None and not newest.undominated
            )
            return _Comparison(best_known, index, dominated_since)
    raise SystemExit(
        f"No scrape of qecdb.org lists the code {entry.params} of {entry.path.name}: run"
        " collect_data --scrape"
    )


def _describe_data_files(data_files: Sequence[Path], comparisons: Sequence[_Comparison]) -> str:
    """The scrapes that the rows of a table use, and what its markers mean, for its header comment.

    The rows without a dagger use the first scrape, and the rows with one use later scrapes.
    """
    used = {comparison.scrape_index for comparison in comparisons}
    later = ", ".join(
        path.name for index, path in enumerate(data_files) if index > 0 and index in used
    )
    if 0 not in used:
        text = f"{later}, which all rows use and which mark them $^\\dagger$"
    elif not later:
        text = data_files[0].name
    else:
        text = f"{data_files[0].name}, or for the rows marked $^\\dagger$,\n% {later}"
    if any(comparison.dominated_since for comparison in comparisons):
        text += (
            ".  The codes of the rows marked $^\\ddagger$ beat their scrape, but not\n%"
            f" {data_files[-1].name}"
        )
    return text


def _format_table(
    entries: Sequence[CodeEntry], comparisons: Sequence[_Comparison], few_logicals: int
) -> str:
    """The ``tabular`` environment of the codes of one code file, with aligned columns.

    A row whose neighbors come from a scrape other than the first gets a dagger on its ``n``, and a
    row whose code beats its scrape but not the newest scrape gets a double dagger.
    """
    rows = [
        _get_cells(entry, comparison, few_logicals)
        for entry, comparison in zip(entries, comparisons, strict=True)
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


def _get_cells(entry: CodeEntry, comparison: _Comparison, few_logicals: int) -> list[str]:
    """The cells of the row of a code, unpadded except for the decimal point of the mean weight."""
    nn, kk, dd = entry.params
    counts = recount(entry)
    logicals = f"{counts.count_x} / {counts.count_z}"
    if max(counts.count_x, counts.count_z) <= few_logicals:
        logicals = rf"\textbf{{{logicals}}}"
    weights = get_reduced_weights(entry.code)
    best_known = comparison.best_known
    best = [
        "--" if value is None else str(value)
        for value in (best_known.n_best, best_known.k_best, best_known.d_best)
    ]
    undominated = best_known.undominated
    params = [_highlight("hlgreen", str(value), undominated) for value in (nn, kk, dd)]
    if comparison.scrape_index > 0:
        params[0] += r"$^\dagger$"
    if comparison.dominated_since:
        params[0] += r"$^\ddagger$"
    return [
        *params,
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
