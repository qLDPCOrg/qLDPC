# Experiments

This directory holds research scripts that use `qldpc`.
They are not part of the installed `qldpc` package, and the test suite does not cover them.

| Directory | What it does | Entry points |
| --- | --- | --- |
| [`bivariate_bicycle/`](bivariate_bicycle/) | Brute-force search for bivariate bicycle codes with toric layouts | `run_search.py`, then `collect_results.py` |
| [`quantum_tanner/`](quantum_tanner/) | Randomized search for quantum Tanner codes | `run_search.py`, then `collect_results.py` and `verify_codes.py` |
| [`connected_support_distance/`](connected_support_distance/) | Timings of the connected-support and Brouwer–Zimmermann exact distance searches on pair-partition and other codes | `benchmark.py` |
| [`lattice_surgery/`](lattice_surgery/) | Logical error rates of lattice-surgery gadgets versus quantum memory (experimental; see the [warning](#lattice-surgery)) | `collect_ler.py` |

Each script's module docstring summarizes its method and outputs; run it with `--help` to see its options.
The quantum Tanner `run_search.py` and `verify_codes.py` require [GAP](https://www.gap-system.org) for their groups (the SmallGroups library) and Cordaro–Wagner base codes (the GUAVA package), whereas `collect_results.py` only reads saved files.
Both code searches bound distances with the default backend of `get_distance_bound`, which uses `sqetch` if installed, then GAP's QDistRnd package if available, and a decoder-based method otherwise.
See the [GAP integration instructions](../docs/source/installation.rst#gap-integration) to set up GAP, for example with `pip install 'qldpc[gap]'`.
Commands below are written relative to the repository root.

## Results

Every experiment that saves results stores them in a `results/` directory beside its scripts, which Git ignores.
Its scripts accept `--results-dir` to use another directory instead, such as a scratch directory for a quick trial run; pass the same `--results-dir` to all scripts of an experiment.
Each results directory holds a CSV summary, together with the resumable intermediate data from which that summary is built:

| Experiment | Intermediate data | CSV summary |
| --- | --- | --- |
| Bivariate bicycle | `results/cache/`: disk cache of code parameters | `results/codes.csv`: codes with `k d^2 / n > 1` |
| Quantum Tanner | `results/codes/`: one text file per saved code | `results/codes.csv`: parameters of every saved code with `k > 0` |
| Lattice surgery | `results/{preset}_{run_id}_progress.csv`: Sinter progress | `results/{preset}.csv`: sample counts, from which logical error rates follow |

Earlier versions of these scripts saved results beside the scripts instead, where Git no longer ignores them.
To keep using such results, move them into the new layout:

```bash
mkdir -p experiments/bivariate_bicycle/results experiments/quantum_tanner/results
mv experiments/bivariate_bicycle/.code_cache experiments/bivariate_bicycle/results/cache
mv experiments/bivariate_bicycle/codes.csv experiments/bivariate_bicycle/results/codes.csv
mv experiments/quantum_tanner/codes experiments/quantum_tanner/results/codes
```

Then rerun `quantum_tanner/collect_results.py` to write `results/codes.csv`, which supersedes the old `quantum_tanner/.code_cache` directory, and delete that directory.

## Bivariate bicycle codes

[`bivariate_bicycle/`](bivariate_bicycle/) performs a brute-force search for bivariate bicycle codes of [arXiv:2308.07915](https://arxiv.org/abs/2308.07915) with toric layouts.
In a toric layout, qubits and checks sit on a periodic two-dimensional lattice, and four of the six data qubits addressed by each check are its neighbors on that lattice, as in the toric code; the remaining two connections may be nonlocal.
A bivariate bicycle code has a manifestly toric layout if its generating polynomials have the form `A = 1 + x + ...` and `B = 1 + y + ...`, where `x` and `y` generate cyclic groups whose orders are the width and height of the torus.
The search therefore covers codes with generating polynomials `A = 1 + x + x^ax y^ay` and `B = 1 + y + x^bx y^by`.
Every bivariate bicycle code with a toric layout and three terms per polynomial is equivalent to a code of this form; see `BBCode.get_equivalent_toric_layout_code_data` and the [bivariate bicycle notebook](../examples/bivariate_bicycle_codes.ipynb).

1. `run_search.py DIM_X` searches over every torus with width `DIM_X` and height between `--min-order` and `DIM_X`, skipping codes that are equivalent by symmetry or whose generating polynomials reduce to monomials.
   It saves code parameters `(n, k, d)` to the disk cache `results/cache/`, and resumes from that cache if interrupted.
   To save time, codes with an encoding rate `k / n` below `--min-rate` are saved as `(n, k, None)`, without computing their distance.
2. `collect_results.py` exports every cached code with `r = k d^2 / n > 1` to `results/codes.csv`, with one row `Rx, Ry, ax, ay, bx, by, n, k, d, r` per code, where `Rx` and `Ry` are the orders of `x` and `y`.

```bash
python experiments/bivariate_bicycle/run_search.py 12 --workers 8
python experiments/bivariate_bicycle/collect_results.py
```

Code distances are upper bounds, each minimized over `--num-trials` randomized trials.
Before symmetry reduction, each torus height contributes roughly `DIM_X^4` candidate codes, so the cost of a search grows rapidly with `DIM_X`.
`collect_results.py` only exports results computed with its own `--num-trials`, so pass the same value to both scripts.
To try the pipeline cheaply first, write a small search to a new private scratch directory:

```bash
RESULTS=$(mktemp -d)
python experiments/bivariate_bicycle/run_search.py 6 --num-trials 10 --results-dir "$RESULTS"
python experiments/bivariate_bicycle/collect_results.py --num-trials 10 --results-dir "$RESULTS"
```

## Quantum Tanner codes

[`quantum_tanner/`](quantum_tanner/) performs a randomized search over small quantum Tanner codes, which [arXiv:2202.13641](https://arxiv.org/abs/2202.13641) introduced.
The search builds codes from small finite groups and small classical base codes.
It constructs each code with `codes.QTCode.random`, which draws two random symmetric subsets of a group, that is, subsets closed under inverses.
Both subsets have the block length of the base code as their size.
The base code is the local code on one subset, and its dual code is the local code on the other.
The codes live on the quadripartite Cayley complex of [arXiv:2206.07571](https://arxiv.org/abs/2206.07571), the default of `codes.QTCode.random`.
This complex is built on four copies of the group, so the subsets need not satisfy the conditions of the bipartite complex of arXiv:2202.13641.
See the docstrings of `codes.QTCode` and `objects.CayleyComplex` for the construction.

`abstract.Group.random_symmetric_subset` does not draw subsets uniformly.
The frequencies of code parameters in the results therefore reflect this sampler, rather than the fractions of all symmetric subsets that give these parameters.
Small groups have few symmetric subsets of a given size, so different seeds often draw the same subsets, and thereby the same code.
With default settings, the search saves 10,913 codes, of which 4,515 are distinct.

1. `run_search.py` samples codes for every group of order at most `--max-order` and every base code in `BASE_CODE_IDS`, and saves each code to a text file in `results/codes/`.
   Groups with fewer elements than a base code has bits are skipped for that base code, because they cannot define a quantum Tanner code with it.
   File names identify the group, base code, and random seed of each code.
   Codes that already have a file are skipped, even if that file used a different `--num-trials`, unless `--overwrite` is passed.
   Each code is written to a temporary `*.tmp` file and then renamed, so an interrupted search leaves no partial code file.
   Leftover `*.tmp` files are ignored by the other scripts, and overwritten when the search resumes.
   Run one search per results directory at a time.
2. `collect_results.py` summarizes the parameters of saved codes in `results/codes.csv`, with one row `group_order, group_index, base_code, seed, n, k, d, num_trials, stabilizer_weight` per saved code that has `k > 0`.
   Here `group_order` and `group_index` identify a group in the SmallGroups library, and `stabilizer_weight` is the maximum weight of a parity check (stabilizer generator) of the code.
   Codes saved under different seeds can be identical, so the same code can have several rows.
3. `verify_codes.py` checks that every saved code can be loaded, and that regenerating it from its seed reproduces it.

```bash
python experiments/quantum_tanner/run_search.py --workers 8
python experiments/quantum_tanner/collect_results.py
python experiments/quantum_tanner/verify_codes.py
```

To try the pipeline cheaply first, write a small search to a new private scratch directory:

```bash
RESULTS=$(mktemp -d)
python experiments/quantum_tanner/run_search.py --max-order 6 --num-samples 2 --num-trials 10 --results-dir "$RESULTS"
python experiments/quantum_tanner/collect_results.py --results-dir "$RESULTS"
python experiments/quantum_tanner/verify_codes.py --results-dir "$RESULTS"
```

With default settings, the search covers the 18 groups of order at most 10.
It samples up to 100 random codes per group and base code, or only one if the base code has as many bits as the group has elements.
Each code requires a randomized distance estimate with `--num-trials` trials, so a full search can take many CPU-hours.
Distances `d` are upper bounds.
These bounds are randomized without a seed, so they can differ between runs for the same code.
`verify_codes.py` does not check them.
Saved code files record `d` as `nan` for codes with no logical qubits, and `collect_results.py` omits such codes.
Verification requires `codes.QTCode.random` to sample the same code for the same seed, so codes saved with a different version of `qldpc` may fail verification.

## Connected-support distance

[`connected_support_distance/benchmark.py`](connected_support_distance/benchmark.py) times the exact distance searches of `qldpc` on pair-partition codes of Okada and Kasai and of Lee et al. (see [the connected-support guide](../src/qldpc/codes/distance/connected_support.md)), and on a few other codes.
It prints its results and saves nothing, so it takes no `--results-dir`.
For example:

```bash
python experiments/connected_support_distance/benchmark.py 110 232 276
python experiments/connected_support_distance/benchmark.py 232 --methods brouwer_zimmermann
python experiments/connected_support_distance/benchmark.py gross surface:11 surface:15
python experiments/connected_support_distance/benchmark.py 110 232 --hadamards
```

The option `--hadamards` applies Hadamards to the second half of the qubits of each code, and times the resulting code, which is not CSS.

Numba compiles the searches on first use, so run the script twice for the times of a warm cache.
Some searches run for hours or more, so run it under a deadline.

## Lattice surgery

> [!WARNING]
> Lattice-surgery support in `qldpc` (`qldpc.experimental.surgery`) is experimental and unstable, and its implementation is still pending careful expert review.
> Its API and behavior may change without notice or deprecation.
> Validate results independently, and pin the `qldpc` version if you depend on them.

[`lattice_surgery/`](lattice_surgery/) samples logical error rates of the experimental lattice-surgery constructions in `qldpc.experimental.surgery`, and of quantum memory baselines, with [Sinter](https://pypi.org/project/sinter/).
`collect_ler.py --preset PRESET` runs one of the presets `bb72`, `bb18`, or `steane-joint`, and writes aggregate sample counts, from which its logical error rates follow, to `results/{preset}.csv`.
Collection is resumable, and some presets take hours to days at publication scale.
See the [lattice-surgery README](lattice_surgery/README.md) for example commands, provenance tracking, and costs.
