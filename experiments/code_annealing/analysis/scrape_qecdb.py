# SPDX-License-Identifier: Apache-2.0

"""Find the best known codes of the qecdb.org database next to each annealed code.

Run with ``python -m experiments.code_annealing.analysis.scrape_qecdb``, which needs network access.

For every annealed code ``[[n, k, d]]`` of the two code files, this finds the best known
neighbors in the database: the smallest ``n`` with at least the same ``k`` and ``d``, the largest
``k`` with at most the same ``n`` and at least the same ``d``, and the largest ``d`` with at most
the same ``n`` and at least the same ``k``.  CSS codes are compared with the CSS codes of the
database, and SWEL codes with its self-dual CSS codes.  The script prints, and saves to the data
file ``data/scrape_qecdb_YYYY-MM-DD.txt`` of the results directory, one line per annealed code,
``(n, k, d) n_best k_best d_best id_n id_k id_d``, in one block per code file, with ``None`` for a
neighbor that the database does not have.  Each ID is the ID at qecdb.org, as in
``https://qecdb.org/codes/ID``, of a code that has the parameter of its neighbor: the first code
that the database lists, since several codes may share the best value.  The distance of an
annealed code is the one that its code file claims, and that of a database code is the one that
the database lists.

The database grows, and may come to list the annealed codes themselves, so a data file records the
database on the date of its scrape (in UTC), which its name carries.  ``latex_tables.py`` reads the
newest data file, or the one that its ``--data-file`` names, and earlier ones are kept, so that a
table can be rebuilt from the snapshot behind it.  The snapshot behind the paper's table is in
``paper_data/``.

Scraping uses only the standard library, makes three paced requests to the database for each
annealed code, tries a request again if it fails in transit or the server is busy or failing, and
writes the data file only after every request has succeeded.
"""

from __future__ import annotations

import datetime
import http.client
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.atomic import write_text_atomic
from ..core.cli import add_results_dir_argument, make_parser
from ..core.code_files import CodeParams, read_code_file, split_blocks
from ..core.paths import find_code_files, get_data_dir
from ..core.proposals import CODE_TYPES, CodeType

_DATA_FILE_PATTERN = "scrape_qecdb_????-??-??.txt"
_QECDB_API_URL = "https://qecdb.org/api/codes"

# Seconds to wait for the database to answer.
_REQUEST_TIMEOUT = 30

# Seconds to wait before every request, to be polite to the database.
_REQUEST_DELAY = 0.25

# Attempts of a request that fails in transit, which can be transient.
_REQUEST_ATTEMPTS = 3

# Seconds to wait before another attempt of a request.
_RETRY_DELAY = 5

_DATA_LINE_PATTERN = re.compile(
    r"\((\d+), (\d+), (\d+)\) (\d+|None) (\d+|None) (\d+|None) (\w+) (\w+) (\w+)"
)
# The ID of a code at qecdb.org, which never reads "None".
_ID_PATTERN = re.compile(r"\w+")


def main(argv: Sequence[str] | None = None) -> None:
    """Ask the database about every annealed code, and save the data file of today."""
    parser = make_parser(__doc__)
    add_results_dir_argument(parser)
    args = parser.parse_args(argv)

    today = datetime.datetime.now(datetime.UTC).date()
    data_file = get_data_dir(args.results_dir) / f"scrape_qecdb_{today.isoformat()}.txt"
    try:
        _scrape_data_file(find_code_files(args.results_dir), data_file)
    except RuntimeError as error:
        raise SystemExit(f"Scraping failed, and nothing was written: {error}") from error


def find_data_file(results_dir: Path) -> Path | None:
    """The newest data file of the results directory, by the date in its name, or None."""
    data_files = sorted(get_data_dir(results_dir).glob(_DATA_FILE_PATTERN))
    return data_files[-1] if data_files else None


@dataclass(frozen=True)
class BestKnown:
    """The parameters of an annealed code and of its best known neighbors in the database."""

    params: CodeParams
    n_best: int | None
    """The smallest ``n`` with at least the same ``k`` and ``d``."""
    k_best: int | None
    """The largest ``k`` with at most the same ``n`` and at least the same ``d``."""
    d_best: int | None
    """The largest ``d`` with at most the same ``n`` and at least the same ``k``."""
    ids: tuple[str | None, str | None, str | None]
    """The IDs at qecdb.org of codes that have ``n_best``, ``k_best``, and ``d_best``."""

    def __str__(self) -> str:
        """The line of the data file."""
        ids = " ".join(str(code_id) for code_id in self.ids)
        return f"{self.params} {self.n_best} {self.k_best} {self.d_best} {ids}"

    @property
    def undominated(self) -> bool:
        """Whether no code of the database is at least as good in all of ``n``, ``k``, and ``d``.

        Each neighbor decides this alone.  For example, the annealed code has a smaller ``n`` than
        ``n_best`` (or there is no ``n_best``) exactly when no code of the database has at most its
        ``n``, at least its ``k``, and at least its ``d``.  The same holds for ``k_best`` and
        ``d_best``, so the code beats its three neighbors together or none of them.
        """
        return self.n_best is None or self.params[0] < self.n_best

    @staticmethod
    def parse(line: str) -> BestKnown:
        """Read a line of the data file."""
        match = _DATA_LINE_PATTERN.fullmatch(line.strip())
        if match is None:
            raise ValueError(
                f"Not a line of the form '(n, k, d) n_best k_best d_best id_n id_k id_d': {line!r}"
            )
        nn, kk, dd = (int(text) for text in match.groups()[:3])
        n_best, k_best, d_best = (
            None if text == "None" else int(text) for text in match.groups()[3:6]
        )
        id_n, id_k, id_d = (None if text == "None" else text for text in match.groups()[6:])
        if any(
            (value is None) != (code_id is None)
            for value, code_id in [(n_best, id_n), (k_best, id_k), (d_best, id_d)]
        ):
            raise ValueError(
                f"A neighbor has a value without an ID, or an ID without a value: {line!r}"
            )
        return BestKnown((nn, kk, dd), n_best, k_best, d_best, (id_n, id_k, id_d))


def is_up_to_date(code_files: Mapping[CodeType, Path], data_file: Path) -> bool:
    """Does a data file describe exactly the codes of the code files, in the same order?"""
    return [
        [result.params for result in block] for block in _read_blocks(data_file)
    ] == _get_expected_params(code_files)


def read_best_known(
    code_files: Mapping[CodeType, Path], data_file: Path
) -> dict[CodeType, list[BestKnown]]:
    """The neighbors of every code of the code files, by code type, in the order of the files.

    Raises:
        ValueError: If the data file does not describe exactly the codes of the code files.
    """
    if not is_up_to_date(code_files, data_file):
        raise ValueError(
            f"{data_file} is out of date, because the codes changed: run collect_data --scrape"
        )
    return dict(zip(_sorted_code_types(code_files), _read_blocks(data_file), strict=True))


def query_best_known(params: CodeParams, code_type: CodeType) -> BestKnown:
    """The best known neighbors of a code ``(n, k, d)`` of a code type in the database.

    The ``css`` code type holds all of the CSS codes of the database, and the ``swel`` code type
    those that are self-dual.  Each neighbor is one request.

    Args:
        params: The parameters ``(n, k, d)`` of the code.
        code_type: The code type, which selects the codes of the database to compare with.

    Returns:
        The parameters of the code and of its best known neighbors.

    Raises:
        RuntimeError: If the database cannot be reached or answers in an unexpected format.
    """
    nn, kk, dd = params
    (n_best, id_n), (k_best, id_k), (d_best, id_d) = (
        _query_extreme("n", "asc", code_type, kMin=kk, dMin=dd),
        _query_extreme("k", "desc", code_type, nMax=nn, dMin=dd),
        _query_extreme("d", "desc", code_type, nMax=nn, kMin=kk),
    )
    return BestKnown(params, n_best, k_best, d_best, (id_n, id_k, id_d))


####################################################################################################
# helper methods


def _sorted_code_types(code_files: Mapping[CodeType, Path]) -> list[CodeType]:
    """The code types of the code files, in the order of the blocks of the data file.

    Callers pass the files in this order; sorting keeps the public functions of this module correct
    for a mapping in any order.
    """
    return sorted(code_files, key=CODE_TYPES.index)


def _get_expected_params(code_files: Mapping[CodeType, Path]) -> list[list[CodeParams]]:
    """The parameters of the codes of the code files, in the blocks of the data file."""
    return [
        [entry.params for entry in read_code_file(code_files[code_type])]
        for code_type in _sorted_code_types(code_files)
    ]


def _read_blocks(data_file: Path) -> list[list[BestKnown]]:
    """The blocks of a data file, which are separated by blank lines.

    Raises:
        ValueError: If a line is not a line of the data file, naming the file and the line.
    """
    blocks: list[list[BestKnown]] = []
    for lines in split_blocks(data_file.read_text()):
        blocks.append([])
        for number, line in lines:
            try:
                blocks[-1].append(BestKnown.parse(line))
            except ValueError as error:
                raise ValueError(f"{data_file}, line {number}: {error}") from error
    return blocks


def _scrape_data_file(code_files: Mapping[CodeType, Path], data_file: Path) -> None:
    """Ask the database about every annealed code, print the neighbors, and save them to a file.

    The code files are read first, and the file is written after the last request, so a failure
    keeps the previous file.
    """
    entries = {code_type: read_code_file(code_file) for code_type, code_file in code_files.items()}
    blocks: list[list[BestKnown]] = []
    for code_type in _sorted_code_types(code_files):
        if blocks:
            print()
        block = []
        for entry in entries[code_type]:
            block.append(query_best_known(entry.params, code_type))
            print(block[-1], flush=True)
        blocks.append(block)
    write_text_atomic(
        data_file, "\n".join("".join(f"{result}\n" for result in block) for block in blocks)
    )


def _query_extreme(
    field: str, direction: str, code_type: CodeType, **bounds: int
) -> tuple[int, str] | tuple[None, None]:
    """The first value of a parameter of the codes within bounds, sorted by that parameter.

    Args:
        field: The parameter, ``n``, ``k``, or ``d``, to sort by and return.
        direction: The sort order, ``asc`` or ``desc``.
        code_type: The code type, which selects all CSS codes or only the self-dual ones.
        **bounds: Query parameters of the database, such as ``nMax`` or ``kMin``.

    Returns:
        The parameter and the ID of the first code, or ``(None, None)`` if no code matches.

    Raises:
        RuntimeError: If the database cannot be reached, answers in an unexpected format, or does
            not list the parameter or the ID of the first code that it returns.
    """
    query: dict[str, str | int] = {"cssOnly": 1, "sort": field, "dir": direction, **bounds}
    if code_type == "swel":
        query["selfdual"] = 1
    time.sleep(_REQUEST_DELAY)
    data = _fetch_json(
        f"{_QECDB_API_URL}?{urllib.parse.urlencode({**query, 'page': 0, 'pageSize': 1})}"
    )
    # Only an empty list of codes means that none matches; any other shape is a malformed answer.
    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise RuntimeError(f"qecdb.org answered in an unexpected format: {data!r}")
    if not items:
        return None, None
    code_id = items[0].get("_id")
    if not isinstance(code_id, str) or not _ID_PATTERN.fullmatch(code_id) or code_id == "None":
        raise RuntimeError(
            f"Could not read the ID of the first code that qecdb.org lists: {code_id!r}"
        )
    value = items[0].get(field)
    # an integer, or the decimal text of one; anything else, even a float, is a malformed answer
    if isinstance(value, int) and not isinstance(value, bool):
        return value, code_id
    if isinstance(value, str) and value.strip().isdecimal() and value.strip().isascii():
        return int(value), code_id
    raise RuntimeError(
        f"Could not read the {field} of the first code that qecdb.org lists: {value!r}"
    )


def _fetch_json(url: str) -> Any:
    """Retrieve JSON data from a URL, in up to ``_REQUEST_ATTEMPTS`` attempts.

    A request is tried again after a failure in transit, or an answer that the server is busy (HTTP
    429) or failing (HTTP 5xx), but not after another HTTP error, which a new attempt would repeat.
    """
    for attempt in range(1, _REQUEST_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(url, timeout=_REQUEST_TIMEOUT) as response:
                body = response.read()
            break
        except (OSError, http.client.HTTPException) as exception:
            retry = not isinstance(exception, urllib.error.HTTPError) or (
                exception.code == 429 or exception.code >= 500
            )
            if not retry or attempt == _REQUEST_ATTEMPTS:
                raise RuntimeError(f"Cannot access {url}: {exception}") from exception
            time.sleep(_RETRY_DELAY)
    try:
        return json.loads(body)  # which decodes the bytes, and raises a ValueError if it cannot
    except ValueError as exception:
        raise RuntimeError(
            f"qecdb.org answered {url} with something other than JSON"
        ) from exception


if __name__ == "__main__":
    main()
