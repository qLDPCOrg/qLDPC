# SPDX-License-Identifier: Apache-2.0

"""Check the headline claims of the paper against its codes, without changing any file.

Run with ``python -m experiments.code_annealing.check_claims``, which takes seconds.  It reads the
codes, the dated qecdb.org snapshot, and the automorphisms of the paper from ``paper_data/``, and
prints what it observes:

1. Every code has the distance and number of minimum-weight logical operators that its file claims,
   recounted exactly.
2. How many codes of each code type meet or exceed the stabilizer and the CSS Gilbert-Varshamov
   bounds (the columns delta_QGV and delta_CSS of the paper's table).
3. How many codes of each code type no code that qecdb.org listed before the paper matches or
   betters in all of ``n``, ``k``, and ``d``, by the parameters that the database listed, which
   are a trusted input rather than recomputed.
4. The order of the group of logical Clifford gates, modulo logical Pauli gates, that the
   SWAP-transversal automorphisms of the [[20, 6, 4]] SWEL code generate; the paper reports 2304.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import stim

from qldpc.circuits import get_logical_tableau

from .analysis.gv_bounds import get_css_gv_distance, get_qgv_distance
from .analysis.scrape_qecdb import read_best_known
from .core.cli import make_parser
from .core.code_files import read_code_file, recount, split_blocks
from .core.paths import CODE_FILE_NAMES, PAPER_DATA_DIR

_AUTOMORPHISMS_FILE = PAPER_DATA_DIR / "automorphisms_swel_n20_k06.txt"


def main(argv: Sequence[str] | None = None) -> None:
    """Print what the paper's codes show about each headline claim."""
    make_parser(__doc__).parse_args(argv)
    code_files = {code_type: PAPER_DATA_DIR / name for code_type, name in CODE_FILE_NAMES.items()}
    entries = {code_type: read_code_file(path) for code_type, path in code_files.items()}

    print("1. Exact recount of every code against its file")
    for code_type, codes in entries.items():
        for entry in codes:
            recount(entry)
        print(f"   {code_type}: {len(codes)} codes match their claimed distance and count")

    print("2. Codes with distance at least the Gilbert-Varshamov bounds")
    for code_type, codes in entries.items():
        qgv = sum(entry.distance >= get_qgv_distance(*entry.params[:2]) for entry in codes)
        css = sum(entry.distance >= get_css_gv_distance(*entry.params[:2]) for entry in codes)
        print(f"   {code_type}: stabilizer {qgv}/{len(codes)}, CSS {css}/{len(codes)}")

    print("3. Codes that no code at qecdb.org before the paper matches or betters in n, k, and d")
    snapshot = max(PAPER_DATA_DIR.glob("scrape_qecdb_*.txt"))
    for code_type, neighbors in read_best_known(code_files, snapshot).items():
        wins = sum(any(best_known.wins) for best_known in neighbors)
        print(f"   {code_type}: {wins}/{len(neighbors)} (snapshot {snapshot.name})")

    print(
        "4. Logical Clifford gates of the SWAP-transversal automorphisms of the [[20, 6, 4]] code"
    )
    code = next(entry for entry in entries["swel"] if entry.params[:2] == (20, 6)).code.to_swel()
    order = _get_group_order(
        [_get_symplectic_matrix(get_logical_tableau(code, gate)) for gate in _read_gates()]
    )
    print(f"   {order} gates modulo logical Pauli gates (the paper reports 2304)")


####################################################################################################
# helper methods


def _read_gates() -> list[stim.Circuit]:
    """The Stim circuits of the automorphism file, which blank lines separate.

    A block of comments alone, such as the description at the top of the file, is no circuit.
    """
    circuits = [
        stim.Circuit("\n".join(line for _, line in block))
        for block in split_blocks(_AUTOMORPHISMS_FILE.read_text())
    ]
    return [circuit for circuit in circuits if len(circuit)]


def _get_symplectic_matrix(tableau: stim.Tableau) -> np.ndarray:
    """The binary symplectic matrix of a Clifford tableau, which ignores its Pauli signs."""
    x2x, x2z, z2x, z2z, _, _ = tableau.to_numpy()
    return np.block([[x2x, x2z], [z2x, z2z]]).astype(np.uint8)


def _get_group_order(generators: Sequence[np.ndarray]) -> int:
    """The order of the group of binary matrices that the generators generate."""
    identity = np.eye(len(generators[0]), dtype=np.uint8)
    seen = {identity.tobytes()}
    frontier = [identity]
    while frontier:
        products = [(element @ generator) % 2 for element in frontier for generator in generators]
        frontier = []
        for product in products:
            if product.tobytes() not in seen:
                seen.add(product.tobytes())
                frontier.append(product)
    return len(seen)


if __name__ == "__main__":
    main()
