# Experiments

This directory holds research scripts that use `qldpc`.
They are not part of the installed `qldpc` package, and the test suite does not cover them.

| Directory | What it does | Entry points |
| --- | --- | --- |
| [`bivariate_bicycle/`](bivariate_bicycle/) | Brute-force search for bivariate bicycle codes with toric layouts | `run_search.py`, then `collect_results.py` |
| [`quantum_tanner/`](quantum_tanner/) | Randomized search for quantum Tanner codes | `run_search.py`, then `collect_results.py` and `verify_codes.py` |
| [`lattice_surgery/`](lattice_surgery/) | Logical error rates of lattice-surgery gadgets versus quantum memory (experimental; see the [warning](#lattice-surgery)) | `collect_ler.py` |

Each script's module docstring summarizes its method and outputs; run it with `--help` to see its options.
The quantum Tanner `run_search.py` and `verify_codes.py` require [GAP](https://www.gap-system.org) for their groups (the SmallGroups library) and Cordaro–Wagner base codes (the GUAVA package), whereas `collect_results.py` only reads saved files.
Both code searches bound distances with the default backend of `get_distance_bound`, which uses `sqetch` if installed, then GAP's QDistRnd package if available, and a decoder-based method otherwise.
See the [GAP integration](../README.md#gap-integration) section of the main README to set up GAP, for example with `pip install 'qldpc[gap]'`.
Commands below are written relative to the repository root.

## Results

Every experiment stores its results in a `results/` directory beside its scripts, which Git ignores.
Every script accepts `--results-dir` to use another directory instead, such as a scratch directory for a quick trial run; pass the same `--results-dir` to all scripts of an experiment.
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
To try the pipeline cheaply first, write a small search to a scratch directory:

```bash
python experiments/bivariate_bicycle/run_search.py 6 --num-trials 10 --results-dir /tmp/bbcodes
python experiments/bivariate_bicycle/collect_results.py --num-trials 10 --results-dir /tmp/bbcodes
```

## Quantum Tanner codes

[`quantum_tanner/`](quantum_tanner/) performs a randomized search over quantum Tanner codes built from small finite groups and small classical base codes.

1. `run_search.py` samples codes for every group of order at most `--max-order` and every base code in `BASE_CODE_IDS`, and saves each code to a text file in `results/codes/`.
   Groups with fewer elements than a base code has bits are skipped for that base code, because they cannot define a quantum Tanner code with it.
   File names identify the group, base code, and random seed of each code.
   Codes that already have a file are skipped, even if that file used a different `--num-trials`, unless `--overwrite` is passed.
2. `collect_results.py` summarizes the parameters of saved codes in `results/codes.csv`, with one row `group_order, group_index, base_code, seed, n, k, d, num_trials, stabilizer_weight` per saved code that has `k > 0`.
   Here `group_order` and `group_index` identify a group in the SmallGroups library, and `stabilizer_weight` is the maximum weight of a stabilizer of the code.
3. `verify_codes.py` checks that regenerating every saved code from its seed reproduces that code.

```bash
python experiments/quantum_tanner/run_search.py --workers 8
python experiments/quantum_tanner/collect_results.py
python experiments/quantum_tanner/verify_codes.py
```

To try the pipeline cheaply first, write a small search to a scratch directory:

```bash
python experiments/quantum_tanner/run_search.py --max-order 6 --num-samples 2 --num-trials 10 --results-dir /tmp/qtcodes
python experiments/quantum_tanner/collect_results.py --results-dir /tmp/qtcodes
python experiments/quantum_tanner/verify_codes.py --results-dir /tmp/qtcodes
```

With default settings, the search covers the 18 groups of order at most 10, with up to 100 random codes per group and base code (only one if the base code has as many bits as the group has elements), which amounts to roughly 11,000 saved codes.
Each code requires a randomized distance estimate with `--num-trials` trials, so a full search can take many CPU-hours.
Distances `d` are upper bounds.
Saved code files record `d` as `nan` for codes with no logical qubits, and `collect_results.py` omits such codes.
Verification requires `codes.QTCode.random` to sample the same code for the same seed, so codes saved with a different version of `qldpc` may fail verification.

## Lattice surgery

> [!WARNING]
> Lattice-surgery support in `qldpc` (`qldpc.experimental.surgery`) is experimental and unstable, and its implementation is still pending careful expert review.
> Its API and behavior may change without notice or deprecation.
> Validate results independently, and pin the `qldpc` version if you depend on them.

[`lattice_surgery/`](lattice_surgery/) samples logical error rates of the experimental lattice-surgery constructions in `qldpc.experimental.surgery`, and of quantum memory baselines, with [Sinter](https://pypi.org/project/sinter/).
`collect_ler.py --preset PRESET` runs one of the presets `bb72`, `bb18`, or `steane-joint`, and writes aggregate sample counts, from which its logical error rates follow, to `results/{preset}.csv`.
Collection is resumable, and some presets take hours to days at publication scale.
See the [lattice-surgery README](lattice_surgery/README.md) for example commands, provenance tracking, and costs.
