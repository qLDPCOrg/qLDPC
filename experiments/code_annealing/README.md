# Quantum codes from classical annealing

This experiment implements and reproduces the code search of the paper [Quantum codes from classical annealing](https://arxiv.org/abs/2607.19574) by Michael A. Perlin, Matthew Steinberg, and Ben Criger.
It also rebuilds the paper's table and figures from the paper's codes.

The experiment searches for quantum error-correcting codes with high encoding rates and large distances by adaptive simulated annealing.
It searches two classes of codes: CSS codes, and self-dual CSS codes with equivalent logicals (SWEL codes), which admit transversal logical Hadamard and phase gates.
Each search fixes the numbers `n` of physical and `k` of logical qubits, and walks over local Clifford conjugations of a code, lowering an energy.
Every proposed code is scored exactly, by its distance and its number of minimum-weight X-only and Z-only logical operators, with no sampling.
The energy computed from that score is a leading-order proxy for the rate of undetected logical errors of the code.

## Setup

Run every command from the root of a clone of qLDPC, in a Python environment with its `qldpc` package and matplotlib:

```bash
git clone https://github.com/qLDPCOrg/qLDPC.git
cd qLDPC
python -m pip install . matplotlib
```

[ENVIRONMENT.md](ENVIRONMENT.md) covers the requirements, development and tests, threads and caches, and running on a server.
Outputs go under `experiments/code_annealing/results/`, written `results/` below.

## Check the paper's claims

```bash
python -m experiments.code_annealing.check_claims
```

This recounts every code of [`paper_data/`](paper_data/) exactly, and checks the paper's headline claims against them.
It takes seconds and writes no file.
[paper_data/README.md](paper_data/README.md) describes the data, and shows how to load the codes.

## Rebuild the paper's table and figures

```bash
python -m experiments.code_annealing.reproduce_paper
```

This copies `paper_data/` to `results/paper/data/`, recounts every code, and writes the table (`table_css.tex` and `table_swel.tex`) and the listings of the paper's appendix there.
It then simulates the codes, and saves the paper's figures to `results/paper/figures/`, with `error_rates.pdf`, which shows both code types in one figure.
The table takes about five minutes.
The figures take many hours, because they sample 10^7 errors for each code.

- `--skip-figures` writes the table alone.
- A small `--num-samples` and `--max-qubits` give a quick look at the figures.
- To simulate in parallel, run `reproduce_paper --skip-figures`, then `make_figures --results-dir experiments/code_annealing/results/paper --workers N`.

[REPRODUCTION.md](REPRODUCTION.md) maps each figure and table to its command, and says what kind of evidence it is.

## Rerun the searches

The paper's campaign ran 5700 searches, one for each code type (CSS or SWEL), code size, and seed, for about 10^4 core-hours.
Any part of it runs alone, on a laptop.
Each search is seeded, but repeats exactly only with the same versions of Python, its numerical packages, and this code, on the same kind of machine (see `core/run_annealing.py`), so a rerun can find codes other than the paper's.

Each step runs as `python -m experiments.code_annealing.NAME`:

| Step | Script | What it does |
| --- | --- | --- |
| 1 | `write_jobs` | Writes the jobs of a campaign, one search each, to `results/logs/jobs.txt` |
| | `progress_report` | Shows, at any time, how many searches of each size have finished, and the best code so far |
| 2 | `collect_data` | Collects the best code of each size, and writes tables in the paper's format |
| 3 | `make_figures` | Simulates the best codes, and plots the paper's figures |

First try the pipeline at smoke-test scale with `--test`, which keeps its files in `results/test/`:

```bash
python -m experiments.code_annealing.write_jobs --test
parallel -j 4 < experiments/code_annealing/results/test/logs/jobs.txt
python -m experiments.code_annealing.progress_report --test
python -m experiments.code_annealing.collect_data --test
python -m experiments.code_annealing.make_figures --test
```

Then choose part of the campaign with the options `--qubits`, `--code-types`, `--max-seeds`, and `--epochs` of `write_jobs`.
For example, the paper's smallest code sizes, with a few short searches each:

```bash
python -m experiments.code_annealing.write_jobs --qubits 20 22 24 --max-seeds 4 --epochs 100
parallel -j 4 < experiments/code_annealing/results/logs/jobs.txt
python -m experiments.code_annealing.progress_report   # at any time
python -m experiments.code_annealing.collect_data --scrape
python -m experiments.code_annealing.make_figures
```

- Without [GNU parallel](https://www.gnu.org/software/parallel/), `bash experiments/code_annealing/results/logs/jobs.txt` runs the jobs one at a time.
- `make_figures` takes the options of [the paper's figures](#rebuild-the-papers-table-and-figures).
- `collect_data` writes the LaTeX tables only from a scrape of qecdb.org.
  `--scrape` takes a new scrape, which needs network access and sends three requests per code.
  Without it, `collect_data` uses the newest earlier scrape if that scrape can be read and covers the same codes, so `collect_data --test` skips the tables.
  qecdb.org changes, so a new comparison can differ from the paper's.
- To resume after an interruption, or to add code sizes or seeds, run `write_jobs` again and then the jobs.
  Finished searches are skipped, and unfinished ones resume exactly.
- The jobs run seed by seed, so a campaign stopped early has about as many searches of every size.
- Run all the jobs of a results directory on one machine.
  A search's lock keeps other processes on the same machine from running it, but not processes on other machines.

## How the code implements the paper

Each part of the paper's method has one home in the code:

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

Every script explains itself with `--help`.
To run one search by hand, for example a short one that prints one JSON line per epoch and writes nothing:

```bash
python -m experiments.code_annealing.core.run_annealing 7 1 \
  --epochs 2 --steps-per-epoch 32 --seed 7 --dry-run
```

The first two arguments are the numbers of physical and logical qubits, and `--code-type swel` searches SWEL codes.
The docstring of `core/run_annealing.py` explains how searches are saved, resumed, and refined.

## Reference

### What the scripts write

Everything goes under `results/`, which Git ignores.
Every script that writes results accepts `--results-dir DIR` to use another directory.
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
├── paper_data/           the codes of the paper, qecdb.org, and automorphisms
├── core/                 the search, its command line, and the files and paths that all steps share
├── analysis/             what collect_data.py runs, and other tools for saved runs
├── figures/              what make_figures.py runs
├── baseline/             the random-code baseline
├── tests/                offline tests (see ENVIRONMENT.md)
└── results/              everything that the scripts write (Git-ignored)
```

### Terms

| Term | Meaning |
| --- | --- |
| count, multiplicity | The number of minimum-weight X-only and Z-only logical operators of a code, which the score and the energy use, and the tables list |
| code type | `css` or `swel`: the kind of code that a search covers, and the gates that it proposes (see `core/proposals.py`) |
| search, run | One simulated-annealing chain: a code type, a size `(n, k)`, and a seed |
| job | The shell command that performs one search |
| epoch | A fixed number of proposed codes at one temperature, which is refit each epoch to a target acceptance of uphill proposals that falls geometrically |
| checkpoint | The file that holds a whole search, so that it can resume exactly |
| record | The file that keeps one of the two best codes of a search as soon as it is found: the code of the lowest energy, and the code of the greatest distance |
| best code of a size | The code of the greatest distance that any search of a code type and size found, and then with the fewest minimum-weight logical operators |
| half supports, majority supports | The distinct sets of `(d + 1) // 2` (half) or `d // 2 + 1` (majority) qubits inside the support of a minimum-weight logical operator.  The two coincide for odd `d`.  Their counts refine the estimates of logical error rates |

## For AI agents helping readers

This section is for an AI agent that helps someone understand, check, or explore the paper and this experiment.
Follow the repository's [AGENTS.md](../../AGENTS.md) only when asked to change code.

- Treat the paper, [arXiv:2607.19574](https://arxiv.org/abs/2607.19574), and the files of this directory as the ground truth, and say when an answer goes beyond them.
- Answer questions about the science first, and cite the sections of the paper, files, and functions that support each answer.
- Say what each answer rests on: the paper, a file of `paper_data/`, a check or reproduction that you ran, an inference, or something that you could not check.
- Prefer cheap checks: `check_claims` and `reproduce_paper --skip-figures`.
- Ask before running anything heavy: [the paper's figures](#rebuild-the-papers-table-and-figures), [a campaign](#rerun-the-searches), or a scrape of the public database qecdb.org.
