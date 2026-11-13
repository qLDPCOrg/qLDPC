# SPDX-License-Identifier: Apache-2.0

"""A table compares each code with the oldest scrape that lists it, and marks what changed since."""

from __future__ import annotations

from pathlib import Path

from ..analysis import latex_tables
from ..core.code_files import format_code, write_code_file
from ..core.paths import get_code_files, get_table_file

# The Steane code, with 7 + 7 minimum-weight logicals, and the [[4, 2, 2]] code, with 6 + 6.
_CHECKS_STEANE = [[1, 0, 0, 1, 0, 1, 1], [0, 1, 0, 1, 1, 0, 1], [0, 0, 1, 0, 1, 1, 1]]
_CHECKS_422 = [[1, 1, 1, 1]]


def test_each_row_uses_the_oldest_scrape_that_lists_its_exact_parameters(tmp_path: Path) -> None:
    write_code_file(
        get_code_files(tmp_path)["css"],
        [
            format_code(_CHECKS_STEANE, _CHECKS_STEANE, distance=3, count=14),
            format_code(_CHECKS_422, _CHECKS_422, distance=2, count=12),
        ],
    )
    # The old scrape lists the Steane code, and a code of the size of [[4, 2, 2]] with another
    # distance, whose neighbors do not apply.  The new scrape lists both codes, and a code that is
    # at least as good as the Steane code.  Each scrape has a block for each code type, CSS first.
    old, new = tmp_path / "scrape_qecdb_2000-01-01.txt", tmp_path / "scrape_qecdb_2000-02-02.txt"
    old.write_text("(7, 1, 3) None None None\n(4, 2, 1) 2 2 1\n\n(7, 1, 3) 7 1 3\n")
    new.write_text("(7, 1, 3) 7 1 3\n(4, 2, 2) 4 2 2\n\n(7, 1, 3) 7 1 3\n")
    latex_tables.main(["--results-dir", str(tmp_path), "--data-file", str(old), str(new)])
    table = get_table_file(tmp_path, "css").read_text()
    cells = _get_cells(table)
    # The Steane code beats the old scrape's empty neighbors, but not the new scrape; [[4, 2, 2]]
    # uses the new scrape.
    assert cells[0][:3] == [r"\hlgreen{7}$^\ddagger$", r"\hlgreen{1}", r"\hlgreen{3}"]
    assert cells[0][8:] == [r"\hlred{--}", r"\hlred{--}", r"\hlred{--}"]
    assert cells[1][:3] == [r"4$^\dagger$", "2", "2"]
    assert cells[1][8:] == ["4", "2", "2"]
    assert old.name in table and new.name in table

    # A newer scrape that lists no code as good as the Steane code gives it no double dagger.
    newer = tmp_path / "scrape_qecdb_2000-03-03.txt"
    newer.write_text("(7, 1, 3) None None None\n(4, 2, 2) 4 2 2\n\n(7, 1, 3) 7 1 3\n")
    latex_tables.main(["--results-dir", str(tmp_path), "--data-file", str(old), str(newer)])
    assert _get_cells(get_table_file(tmp_path, "css").read_text())[0][0] == r"\hlgreen{7}"


def _get_cells(table: str) -> list[list[str]]:
    """The cells of the rows of codes of a table."""
    rows = [line.split(r"\\")[0] for line in table.splitlines() if line.count("&") == 10][1:]
    return [[cell.strip() for cell in row.split("&")] for row in rows]
