# SPDX-License-Identifier: Apache-2.0

"""Report the hit rates of the caches of exact searches over a sequence of workloads.

The script warms up as ``trial.py`` does for each kind of workload, empties the caches, runs
workloads (see ``workloads.py``) one after another in this process, ``--cycles`` times over, and
then prints ``cache_info()`` of each cache whose size ``qldpc.codes.distance._tuning`` fixes:
``BINOMIAL_CACHE_SIZE``, ``CSS_SEARCH_CACHE_SIZE``, and ``PAULI_SEARCH_CACHE_SIZE``.  A cache whose
misses stop falling as its size grows is large enough, so run a workload that cycles through more
inputs than a cache holds to see the cost of misses: each cycle after the first should then hit, if
the cache is large enough.  The time of a workload in the first cycle less that in the second is
then the setup of the searches that it missed.

Example:

    python experiments/distance_tuning/cache_stats.py --cycles 2 \\
        coset/gross/css/6/6/100 coset/pp110/css/6/6/100 hbound/pp232/2/100
"""

from __future__ import annotations

import argparse
import time
from typing import TYPE_CHECKING, Any

from qldpc.codes.distance._brouwer_zimmermann import coset
from qldpc.codes.distance._information_sets import certificates

# Scripts in this directory import each other as top-level modules.  mypy knows these files only as
# experiments.distance_tuning.*, and would silently treat a bare "import workloads" as untyped.
if TYPE_CHECKING:
    from experiments.distance_tuning import trial, workloads
else:
    import trial
    import workloads


def main() -> None:
    """Run the workloads, and print the statistics of the caches."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("workloads", nargs="+", help="workload names")
    parser.add_argument(
        "--cycles", type=int, default=2, help="passes over the workloads (default: %(default)s)"
    )
    args = parser.parse_args()
    caches: dict[str, Any] = {
        "BINOMIAL_CACHE_SIZE": certificates.get_capped_binomials,
        "CSS_SEARCH_CACHE_SIZE": coset.get_css_searches,
        "PAULI_SEARCH_CACHE_SIZE": coset.get_pauli_search,
    }
    for kind in sorted({name.split("/")[0] for name in args.workloads}):
        trial.warm_up(kind)
    for cache in caches.values():
        cache.cache_clear()
    for cycle in range(args.cycles):
        for name in args.workloads:
            start = time.perf_counter()
            workloads.run_workload(name)
            print(f"cycle {cycle} {name}: {time.perf_counter() - start:.3f} s", flush=True)
    for name, cache in caches.items():
        print(f"{name}: {cache.cache_info()}")


if __name__ == "__main__":
    main()
