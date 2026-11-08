# Environment

This file is the source of truth for setting up and running the experiment.

## Summary

- Tested platforms: macOS on Apple silicon (arm64), Python 3.13, October 2026; and Linux on x86-64, Python 3.11, November 2026.
- Operating system: Unix, such as Linux or macOS, since the searches lock their files with `fcntl`.
- Toolchain: Python 3.11 or later, with the `qldpc` package of the same repository, and matplotlib for the figures.
- Environment required: yes, a Python environment with `qldpc` installed from the clone.
- External requirements: none for reproducing the paper's table and figures.
  Scraping qecdb.org needs network access, and running many searches at once uses [GNU parallel](https://www.gnu.org/software/parallel/), which is optional.

## Dependency files

- `pyproject.toml` at the root of the qLDPC repository lists the dependencies of `qldpc`, which include NumPy, SciPy, numba, galois, and Stim.
- matplotlib, which only the figures need, is installed alongside.

## Setup from the root of the repository

```bash
git clone https://github.com/qLDPCOrg/qLDPC.git
cd qLDPC
python -m pip install . matplotlib
```

Install `qldpc` from the same clone, so that the library matches the experiment.
Developers of the experiment instead install the clone in editable mode with its development tools:

```bash
python -m pip install -e '.[dev]' matplotlib
```

## Running

Run every script from the root of the repository, as a module of the Python environment in which `qldpc` is installed:

```bash
python -m experiments.code_annealing.NAME [options]
```

Each search uses one core: set `NUMBA_NUM_THREADS=1` to keep numba from starting more threads, as the jobs of `write_jobs` do.
qLDPC caches its compiled kernels in the user's cache directory.
Set `NUMBA_CACHE_DIR` to a writable directory if that is read-only, or to one that all jobs share.

## Running on a server

Copy the repository to the server, and install it as above, without matplotlib if you plot elsewhere.

```bash
python -m experiments.code_annealing.write_jobs
nohup parallel -j "$(nproc)" --joblog experiments/code_annealing/results/logs/joblog.txt \
  < experiments/code_annealing/results/logs/jobs.txt > /dev/null 2>&1 &
```

Here `--joblog` is an option of `parallel` that records the exit status of every job.
Copy `results/logs/` back to a local copy of the repository to collect the data and plot the figures there.
