# Quantum codes from classical annealing

This experiment implements and reproduces the code search of the paper [Quantum codes from classical annealing](https://arxiv.org/abs/2607.19574) by Michael A. Perlin, Matthew Steinberg, and Ben Criger.
It also rebuilds the paper's table and figures from the paper's codes.

The experiment searches for quantum error-correcting codes with high encoding rates and large distances by adaptive simulated annealing.
It searches two classes of codes: CSS codes, and self-dual CSS codes with equivalent logicals (SWEL codes), which admit transversal logical Hadamard and phase gates.
Each search fixes the numbers `n` of physical and `k` of logical qubits, and walks over local Clifford conjugations of a code, lowering an energy.
Every proposed code is scored exactly, by its distance and its number of minimum-weight X-only and Z-only logical operators, with no sampling.
The energy computed from that score is a leading-order proxy for the rate of undetected logical errors of the code.

## Where to start

| To | Read |
| --- | --- |
| Set up and run the scripts | [Installation](#installation), then [ENVIRONMENT.md](ENVIRONMENT.md) for details |
| Reproduce the paper's table and figures, or check its headline claims | [Reproducing the paper](#reproducing-the-paper), then [REPRODUCTION.md](REPRODUCTION.md) for every figure and table |
| Know what the paper's data files are | [paper_data/README.md](paper_data/README.md) |
| Rerun the code searches, at any scale | [The searches, at any scale](#the-searches-at-any-scale) |
| Find the code of each part of the paper's method | [How the experiment implements the paper](#how-the-experiment-implements-the-paper) |
| Know what each file and directory holds | [Layout](#layout) |
| Cite the paper | [CITATION.md](CITATION.md) |

## Installation

Clone the qLDPC repository, install its `qldpc` package with matplotlib, and run every command from the root of the repository:

```bash
git clone https://github.com/qLDPCOrg/qLDPC.git
cd qLDPC
python -m pip install . matplotlib
```

[ENVIRONMENT.md](ENVIRONMENT.md) is the source of truth for the environment: the tested platform, developer setup, threads and caches, and running on a server.

## Reproducing the paper

### The table and figures, from the paper's codes

[`paper_data/`](paper_data/) keeps the codes of the paper, and the parameters of the best codes that a scrape of [qecdb.org](https://qecdb.org) found on 2026-05-21, before the paper.
No script writes that directory.
One command checks every code of the paper, and rebuilds the table and figures of the paper from them:

```bash
python -m experiments.code_annealing.reproduce_paper
```

It recounts the distance and the minimum-weight logical operators of every code exactly.
It writes the table of the paper to `experiments/code_annealing/results/paper/data/table_css.tex` and `table_swel.tex`.
It simulates the codes to save the figures of the paper to `experiments/code_annealing/results/paper/figures/`, with a figure `error_rates.pdf` of both code types together.
The table takes moments, whereas the figures take a while: they simulate every code with many sampled errors, and these simulations take most of the time.
Add `--skip-figures` for the table alone, or a small `--num-samples` and `--max-qubits` for a quick look at the figures.

[REPRODUCTION.md](REPRODUCTION.md) maps every figure and table of the paper to its command, input, and output, and records whether it reproduces.
`python -m experiments.code_annealing.check_claims` checks the paper's headline claims against its codes in seconds, without changing any file.
[paper_data/README.md](paper_data/README.md) describes the data, and shows how to load the codes into qLDPC.

### The searches, at any scale

The paper's code search runs a campaign of many independent searches, one for each code type (CSS or SWEL), code size, and seed.
The searches are random, so a rerun finds its own codes, which need not match those of the paper one for one.
The full campaign takes many CPU hours, but every part of it can run on its own, at a smaller scale, on a laptop.

The pipeline has three steps, and each runs as a module with `python -m experiments.code_annealing.NAME`:

| Step | Script | What it does |
| --- | --- | --- |
| 1 | `write_jobs` | Writes the jobs of a campaign, one search each, for [GNU parallel](https://www.gnu.org/software/parallel/) |
| | `progress_report` | Shows how many searches of each size have finished, and the best code of each so far |
| 2 | `collect_data` | Keeps the best code of each size, computes tables from those codes, and writes tables in the format of the paper's, with a comparison with the newest scrape of qecdb.org (`--scrape` takes a new one) |
| 3 | `make_figures` | Simulates the best codes, and plots the figures of the paper and `error_rates.pdf` |

First check the pipeline at smoke-test scale, with `--test`, which keeps its files in `results/test/`.
Here `collect_data --test` skips the LaTeX tables, which need a scrape of qecdb.org; add `--scrape`, which needs network access, to write them:

```bash
python -m experiments.code_annealing.write_jobs --test
parallel -j 4 < experiments/code_annealing/results/test/logs/jobs.txt
python -m experiments.code_annealing.progress_report --test
python -m experiments.code_annealing.collect_data --test
python -m experiments.code_annealing.make_figures --test
```

Then choose a part of the campaign that suits your computer.
The options `--qubits`, `--code-types`, and `--max-seeds` of `write_jobs` select code sizes, code types, and seeds, and `--epochs` sets the number of epochs of every search.
For example, the smallest code sizes of the paper, with a few short searches each:

```bash
python -m experiments.code_annealing.write_jobs --qubits 20 22 24 --max-seeds 4 --epochs 100
parallel -j 4 < experiments/code_annealing/results/logs/jobs.txt
python -m experiments.code_annealing.progress_report   # at any time
python -m experiments.code_annealing.collect_data --scrape
python -m experiments.code_annealing.make_figures
```

`make_figures` takes a while too, because it simulates every best code; pass `--workers N` to simulate the codes in N parallel processes, or a small `--num-samples` or `--max-qubits` for a quick look.
`collect_data --scrape` compares the codes with those of qecdb.org today, which needs network access.
It saves the comparison in a file named by the date of the scrape, and writes the tables in the format of the paper's.
Without `--scrape`, it writes the table from the newest earlier scrape, if that scrape covered the same codes.
qecdb.org may list codes that were published after the paper, including those of the paper itself, so this comparison can differ from that of the paper.
Without GNU parallel, `bash experiments/code_annealing/results/logs/jobs.txt` runs the jobs one at a time.

After an interruption, run `write_jobs` again with the same options, and then `parallel`: the new jobs file resumes the unfinished searches and skips the finished ones.
A resumed search continues exactly as an uninterrupted one would, in the same numerical environment, so a campaign can run in sessions of any length.
The jobs are ordered by seed, so a campaign that is stopped early has about as many finished searches for every code size.
To extend a campaign, run `write_jobs` with more code sizes or seeds, and the finished searches are skipped.
A search locks its files, so two processes never run the same search, but the lock works only between the processes of one machine: run all the jobs of a results directory on one machine.

## How the experiment implements the paper

Each part of the method of the paper is implemented by one part of the experiment:

| In the paper | In the experiment |
| --- | --- |
| The CSS and SWEL searches, and the gates that they propose | `core/proposals.py` |
| The energy function | `energy()` in `core/annealing.py` |
| Adaptive annealing | `core/annealing.py` |
| Warm starts from the best of many random codes | `--init-samples` of `core/run_annealing.py` |
| The campaign: code sizes, epochs, seeds, and starting codes | `CAMPAIGN`, `NUM_SEEDS`, and `INIT_SAMPLES` in `write_jobs.py` |
| Reducing stabilizer weights | `reduce_weights()` in `core/code_files.py` |
| The Gilbert-Varshamov bounds | `analysis/gv_bounds.py` |
| The table of best codes | `analysis/latex_tables.py` |
| The comparison with qecdb.org | `analysis/scrape_qecdb.py` |
| The logical error rates of a lookup decoder with discards | `figures/common.py` |
| The estimates of logical error rates from counts of minimum-weight logical operators | `analysis/half_supports.py` and `figures/plot_error_estimates.py` |

Every script explains itself in its module docstring, and with `--help`.
To run one search by hand, for example a short one that prints one JSON line per epoch and writes nothing:

```bash
python -m experiments.code_annealing.core.run_annealing 7 1 \
  --epochs 2 --steps-per-epoch 32 --seed 7 --dry-run
```

The first two arguments are the numbers of physical and logical qubits, and `--code-type swel` searches SWEL codes.
The docstring of `core/run_annealing.py` explains how searches are saved, resumed, and refined.

## More details

### What the scripts write

Everything goes under `experiments/code_annealing/results/`, which Git ignores; every script that reads or writes results accepts `--results-dir DIR` to use another directory.
`check_claims` reads only `paper_data/`.
A search is named `{code_type}_n{n}_k{k}_s{seed}`, with `k` padded to two digits, the seed to three, and `_{tag}` appended if `write_jobs --tag` is given.

| Directory | Contents |
| --- | --- |
| `logs/` | `jobs.txt`, and for each search its checkpoint `{name}.json`, its records `{name}.lowest-energy.json` and `{name}.greatest-distance.json` of its two best codes, its `{name}.log`, and an empty `{name}.lock` file; also the files `random_*.txt` of the random baseline |
| `data/` | The best codes `annealed_codes.txt` (CSS) and `annealed_codes_swel.txt` (SWEL), `half_supports.txt`, a dated `scrape_qecdb_YYYY-MM-DD.txt` for each scrape, the LaTeX tables `table_css.tex` and `table_swel.tex`, the listings `listing_css.txt` and `listing_swel.txt` of the codes for the paper's appendix, and `saved_codes.csv` if requested |
| `figures/` | The PDF figures |
| `cache/` | Simulations that the figure scripts reuse, by code and settings but not by qLDPC version, so delete it after updating qLDPC |
| `test/` | The same layout, for runs with `--test` |
| `paper/` | The same layout, for `reproduce_paper` |

Keep a results directory private to you, because `jobs.txt` holds shell commands and `cache/` holds pickled simulations.

### Optional scripts

- `baseline/run_random.py` samples random codes of one size as a baseline, and `make_figures` plots a histogram for every size that was sampled.
- `analysis/print_annealed_codes.py` lists every code that the searches saved, not only the best ones, with their exact counts, and can write them to `data/saved_codes.csv`.

### Layout

```text
experiments/code_annealing/
├── README.md             this guide
├── REPRODUCTION.md       the figures and tables of the paper, and how each reproduces
├── ENVIRONMENT.md        setting up and running
├── CITATION.md           how to cite the paper
├── reproduce_paper.py    the table and figures of the paper, from its codes
├── check_claims.py       check the headline claims of the paper against its codes
├── write_jobs.py         step 1: write the jobs of a campaign
├── progress_report.py    check the progress of a campaign
├── collect_data.py       step 2: collect the best codes and compute tables
├── make_figures.py       step 3: plot figures
├── paper_data/           the codes of the paper, qecdb.org before the paper, and automorphisms
├── core/                 the search, its command line, and the files and paths that all steps share
├── analysis/             what collect_data.py runs, and other tools for saved runs
├── figures/              what make_figures.py runs
├── baseline/             the random-code baseline
├── tests/                offline tests of the search and of the numbers derived from codes
└── results/              everything that the scripts write (Git-ignored)
```

### Terms

| Term | Meaning |
| --- | --- |
| count, multiplicity | The number of minimum-weight X-only and Z-only logical operators of a code, which the score and the energy use, and the tables list |
| code type | The kind of search: `css` searches CSS codes with CNOT gates by default, or larger random CSS gates with `--gate-size`, and `swel` searches SWEL codes with the elementary SWEL-preserving gate |
| search, run | One simulated-annealing chain: a code type, a size `(n, k)`, and a seed |
| job | The shell command that performs one search |
| epoch | A fixed number of proposed codes at one temperature, which is refit each epoch to a target acceptance of uphill proposals that falls geometrically |
| checkpoint | The file that holds a whole search, so that it can resume exactly |
| record | The file that keeps one of the two best codes of a search as soon as it is found: the code of the lowest energy, and the code of the greatest distance |
| best code of a size | The code of the greatest distance that any search of a code type and size found, and then with the fewest minimum-weight logical operators |
| half supports, majority supports | The distinct sets of `(d + 1) // 2` (half) and `d // 2 + 1` (majority) qubits that lie in the supports of minimum-weight logical operators, which are the same sets for odd `d`; the counts of majority supports refine the estimates of logical error rates |

## Tests

The tests cover what could go wrong unnoticed.
That is the search modules of `core/`, and the numbers that the scripts derive from codes beyond the exact counts.
Those numbers are the supports of minimum-weight logical operators (`analysis/half_supports.py` and `analysis/min_weight_supports.py`), and the low-weight generators of code files (`reduce_weights`).
The tests compare with small codes whose answers are derived by hand, or with brute force, rather than repeating the code that they test.
The other scripts and helpers have no tests: they are plain formulas or file handling, and when one of them breaks, its output shows it.
The tests are offline and local-only, and do not run in CI.
They need the development install of [ENVIRONMENT.md](ENVIRONMENT.md), which provides pytest and its plugins:

```bash
python -m pip install -e '.[dev]' matplotlib
python -m pytest experiments/code_annealing
```
