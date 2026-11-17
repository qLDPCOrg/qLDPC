# Environment

## Requirements

- Python 3.11 or later, on a Unix system such as Linux or macOS, because the searches lock their files with `fcntl`.
- The `qldpc` package of the same clone, which brings NumPy, SciPy, numba, galois, and Stim.
- matplotlib, for the figures.
- Optional: network access to scrape qecdb.org, and [GNU parallel](https://www.gnu.org/software/parallel/) to run many searches at once.

Tested on macOS on Apple silicon with Python 3.13 (October 2026), and on Linux on x86-64 with Python 3.11 (November 2026).
[README.md](README.md#setup) gives the installation.

## Development and tests

To change the experiment, install the clone in editable mode with its development tools, and run the offline tests, which CI does not run:

```bash
python -m pip install -e '.[dev]' matplotlib
python -m pytest experiments/code_annealing
```

The tests cover what could go wrong unnoticed: the search in `core/`, the supports of minimum-weight logical operators, and the minimum-weight generators of `reduce_weights`.
They compare with small codes solved by hand or by brute force.
The other scripts have no tests, because their failures show in their output.

## Threads and caches

Each search uses one core.
The jobs of `write_jobs` set `NUMBA_NUM_THREADS=1` and `QLDPC_NUM_THREADS=1`; set them too when you run a search by hand.
qLDPC caches its compiled kernels in the user's cache directory.
Set `NUMBA_CACHE_DIR` to a writable directory if that one is read-only, or to share one cache among jobs.

## Running on a server

Clone and install as in the README; matplotlib is needed only where you plot.

```bash
python -m experiments.code_annealing.write_jobs
nohup parallel -j "$(nproc)" --joblog experiments/code_annealing/results/logs/joblog.txt \
  < experiments/code_annealing/results/logs/jobs.txt > /dev/null 2>&1 &
```

`--joblog` records the exit status of every job.
Copy `experiments/code_annealing/results/logs/` back to a local clone to collect the data and plot the figures there.
