# SPDX-License-Identifier: Apache-2.0

"""Print the routes that ``overlapping_minimum_work`` gives the BZ searches of a coset workload.

A ``coset`` workload asks each BZ search of ``CosetLeaderSearch`` for the minimum weights of its
cosets up to a maximum weight, which it answers with the route of that weight (see
``qldpc.codes.distance._brouwer_zimmermann.coset_routes``).  For each threshold given, the script
builds the search of the workload's state under that value of ``overlapping_minimum_work``, chooses
the route of the maximum weight of each of its BZ searches (the X and Z parts of a CSS state, or the
Pauli errors), and prints:

- the predicted examinations with the disjoint family at each weight from 2 to the maximum weight;
- the target weight at which the search builds its other families, the smallest weight from 2
  whose predicted examinations reach the threshold, or None if the route keeps the disjoint family
  without building others;
- the number of sets of the family of the route, its summed work at the maximum weight, whether it
  scans by tails, and a fingerprint of its rows and pivots, which is the same exactly when two
  thresholds give the same family.

Two thresholds whose routes have the same fingerprints and tails make the same decisions, so they
take the same time, apart from building the families.  The script answers no query.

Example:

    python experiments/distance_tuning/coset_families.py coset/pp232/css/10/10/3000 \\
        --thresholds 10000 100000 1000000
"""

from __future__ import annotations

import argparse
import hashlib
from typing import TYPE_CHECKING

from qldpc.codes.distance import CosetLeaderSearch, _tuning
from qldpc.codes.distance._brouwer_zimmermann import coset
from qldpc.codes.distance._brouwer_zimmermann.coset_routes import CosetRoutes

# Scripts in this directory import each other as top-level modules.  mypy knows these files only as
# experiments.distance_tuning.*, and would silently treat a bare "import workloads" as untyped.
if TYPE_CHECKING:
    from experiments.distance_tuning import workloads
else:
    import workloads


def main() -> None:
    """Print the routes of the workload under each threshold."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("workload", help="a coset workload, such as coset/pp232/css/10/10/3000")
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=int,
        default=[10**3, 10**4, 3 * 10**4, 10**5, 3 * 10**5, 10**6, 10**7],
        help="values of overlapping_minimum_work (default: %(default)s)",
    )
    args = parser.parse_args()
    parts = args.workload.split("/")
    if parts[0] != "coset" or len(parts) != 6:
        raise ValueError("Give a coset workload, such as coset/pp232/css/10/10/3000")
    _, code_name, weighing, _, max_weight_text, _ = parts
    if weighing not in ("css", "physical"):
        raise ValueError(f"Unknown weighing {weighing!r}; choose 'css' or 'physical'")
    if int(max_weight_text) < 2:
        raise ValueError("Routes build other families only for maximum weights of two or more")
    stabilizers = workloads.get_zero_state_stabilizers(code_name)
    # No error weighs more than the number of qubits.
    max_weight = min(int(max_weight_text), len(stabilizers))
    labels = "XZ" if weighing == "css" else "P"
    for threshold in args.thresholds:
        with _tuning.use_tuning(overlapping_minimum_work=threshold):
            search = CosetLeaderSearch(stabilizers, css_weight=weighing == "css")
            for label, bz_search in zip(labels, search._brouwer_zimmermann_searches, strict=True):
                print_route(label, threshold, bz_search._routes, max_weight)
        coset._get_css_searches.cache_clear()
        coset._get_pauli_search.cache_clear()


def print_route(label: str, threshold: int, routes: CosetRoutes, max_weight: int) -> None:
    """Choose the route of the maximum weight of one search, and print it."""
    disjoint = (routes._disjoint, routes.disjoint_tables)
    examinations = {
        weight: routes._get_examinations(*disjoint, weight) for weight in range(2, max_weight + 1)
    }
    tables, tails = routes.get_route(max_weight)
    candidates = [disjoint, *(routes._candidates or [])]
    instance = next(family for family, family_tables in candidates if family_tables is tables)
    # The routes build their other families once, at the target weight, only past the gate.
    target = None
    if routes._candidates is not None:
        target = next(weight for weight, count in examinations.items() if count >= threshold)
    by_tails = tails is not None
    work = sum(routes._get_works(instance, tables, max_weight))
    fingerprint = hashlib.sha256(instance.pivots.tobytes() + instance.generators.tobytes())
    print(
        f"{label} threshold {threshold}: target weight {target}, {len(instance.ranks)} sets,"
        f" work {work:.3g}, by tails {by_tails}, family {fingerprint.hexdigest()[:12]}"
    )
    print("    examinations:", ", ".join(f"{w}: {e:.3g}" for w, e in examinations.items()))


if __name__ == "__main__":
    main()
