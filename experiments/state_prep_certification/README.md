# State preparation certification

This directory holds experiments on the exact certification of fault-tolerant state preparation circuits, and on the exact distance searches that it relies on.
Each experiment is a subdirectory that plugs into shared code at this level:

| Experiment | What it measures | Inputs |
| --- | --- | --- |
| [`certification/`](certification/) | Exact certification of bounded fault propagation in published state preparation circuits: feasibility, listed faults against all accepted fault combinations, smallest counterexamples, and the cost of the two weight oracles | MQT QECC's shipped and synthesized circuits, the hand-translated circuits of Goto and of Chen et al. (referenced in certification's README), and weakened variants |
| [`distances/`](distances/) | Exact code distances by qLDPC's Brouwer-Zimmermann search ([guide](../../src/qldpc/codes/distance/brouwer_zimmermann.md)) with everything on, against the same search with four of its extensions turned off | Codes drawn by a fixed rule with fixed seeds, and named anchors |
| [`ranking/`](ranking/) | Exploratory; certification and distances do not use it. Whether decoder-free scores of certified preparation circuits order them as stratified direct simulation of a fixed consumer does, and at what cost | The certified t = 1 circuits of the certification experiment, with its code files |

The experiments run only qLDPC and its Python dependencies, so that they can run on a Linux node without practical network access.
The shared workflow builds everything that the node needs into one offline bundle on a workstation, runs each experiment on the node with GNU parallel, and brings one archive back for analysis on the workstation.
Times measured on a workstation are never results: the analyses of certification and distances refuse them unless given `--allow-local`, and then mark every output.

## Shared code

- [`run_node.py`](run_node.py) runs one experiment for its label: the acceptance tests, the task lists with their ceiling of CPU-hours, the two passes, and the archive.
- [`node.sh`](node.sh) is the node's entry point: a preflight check of the node, an offline install from the bundle, and then `run_node.py`.
- [`bundle.py`](bundle.py) builds the offline bundle on the workstation.
- [`pins.json`](pins.json) pins the qLDPC commit, CPython, the Linux platform of the wheels, and GNU parallel.
- [`common/`](common/) holds the task runner (`runner.py` and `run_task.py`), the result store (`store.py`), the task lists (`tasks.py`), the label and manifest (`manifest.py`), the plug-in interface (`plugin.py`), `local_runs.py`, which runs tasks locally for the acceptance tests, and `acceptance.py`, which runs each experiment's acceptance tests.
  `common/selftest.py` tests them on a stand-in experiment (`selftest_plugin.py`), locally and in node mode.

### Labels and results

A run's label is the experiment's name and a hash of `pins.json`, the experiment's frozen input files, and the commit of this tree, as [`common/manifest.py`](common/manifest.py) describes.
The frozen files include the experiment's configuration and the data that fix its inputs.
From a bundle, the label also covers the bundle's resolved requirements.
Any change to these inputs gives a new label, and so a new directory `results/<experiment>/<label>/`, which Git ignores.
`common/manifest.py` gives the suffixes that the directory's name can carry.
That directory holds the manifest (`manifest.json`), one line per start (`starts.jsonl`, written as the start begins, before its acceptance tests), the record of the acceptance tests (`acceptance.json`), the task lists, GNU parallel's job logs, `runs.jsonl` with one row per task (see [Runs](#runs)), the tail of each run's output in `logs/`, the records of GNU parallel and of the live runs in `active/`, the lock of the start that owns the label (`lock`), and the archive.
Each attempt of the acceptance tests writes its runs to a fresh directory, `results/acceptance/<label directory>/<UTC>-<random>/`, which `acceptance.json` records; failed attempts are kept.
The manifest records the machine, the installed package versions, and the environment variables that can change a run, such as `PYTHONPATH` and `NUMBA_*`.
A node start refuses a few of them, such as `PYTHONPATH`, unless `--allow-env NAME` accepts one, as [`run_node.py`](run_node.py) says.
The analyses of certification and distances name in every output each such variable that a start accepted, and each recorded variable whose value differs among the label's starts.

### Runs

Each run of a task is a fresh process pinned to one core, with the thread variables of numerical libraries set to 1; the runner that watches it runs on the reserved cores (see below).
The runner times one call of the task's code, the measured call, whose wall time is T.
A run is a `timeout` if T reaches the task's limit, even if a memory kill followed.
Otherwise it is `ok` if the measured call returns and its result reaches the runner, a `memout` if its memory exceeds the task's memory cap before its result arrives, and an `error` otherwise.
For example, a run is an `error` if process start, imports, and untimed preparation exceed the task's start-up cap.
The runner polls the run's summed memory every second.
Since the polls can miss a short peak, it also checks the peak that the operating system records for the run's process and the processes that it waited for, which the process sends with its result or the runner reads when it exits: an `ok` run whose peak until its result exceeded the cap becomes a `memout`.
These are peaks of single processes, so a short peak of the run's summed memory between polls can still go unseen.
If the experiment sends its measured result before untimed post-processing, running out of the time that the task allows for it (`post_s`), or a peak above the cap after the result, does not change an `ok` status, but marks the row's `post_status`.
A run is discarded if numba reports a compilation during its measured call, and the task runs once more in a fresh process that loads the compiled code from numba's disk cache.
The retry is kept, and its row is the task's one row.
If the retry compiles too, its row has `compiled_in_timing` set.
So T includes loading compiled code and the other first-call work of a fresh process, but no compilation that numba reports, unless the row has `compiled_in_timing` set.
Such a row is kept so that it is visible, but its timing enters no analysis; each experiment's `analyze.py` says how it treats such rows.
[`common/runner.py`](common/runner.py) gives these rules in full, and [`common/tasks.py`](common/tasks.py) gives the limits that each task can set and their defaults.

Two physical cores and 16 GiB of memory are reserved for the runners, GNU parallel, and the system.
Each job slot has its own free physical core and the experiment's memory per slot (`JOB_MEM_GIB`).
There are as many slots as the free cores and memory allow, and at most N if `node.sh` or `run_node.py` is given `--jobs N`.
Pass 1 holds every first run in an order shuffled with a fixed seed.
Pass 2 holds the repeats that each experiment's repeat policy asks for after pass 1, such as two more runs of a fast run.

The ceiling of CPU-hours that an experiment needs approved allows for the runner's retries, as [`common/tasks.py`](common/tasks.py) explains.

## Workflow

Each experiment's data, such as its codes or circuits, are not committed: the experiment reads them from its data directory, which Git ignores.
Each experiment's README lists the files that its data directory must hold, says which of them the experiment's scripts make, and which are inputs that it takes as given.
On the workstation, put those inputs in place, make the rest with the experiment's scripts, and check them as its README says, such as with `circuits.py --check`.
Then build the bundle, which carries the data, and note its checksum:

```bash
python experiments/state_prep_certification/bundle.py EXPERIMENT --out /path/to/bundles
```

Name in a bundle every experiment whose directory holds a frozen file of an experiment that it names; `bundle.py` refuses a bundle that lacks a named experiment's frozen file, and says which experiment to add.

The bundle needs a pinned qLDPC commit in `pins.json`; `--unfrozen` builds a trial bundle without one, which `run_node.py` refuses on the node unless `--allow-unfrozen` is given.

On the node, check the bundle, unpack it, and run the experiment from its `qLDPC/` directory, with the CPU-hours H approved for it:

```bash
sha256sum -c spc-bundle-<id>.tar.sha256 && tar -xf spc-bundle-<id>.tar
cd spc-bundle-<id>/qLDPC
bash experiments/state_prep_certification/node.sh EXPERIMENT --approved-cpu-hours H
```

`node.sh --check` runs only the preflight, and needs only `node.sh` and `pins.json`, so it can run on the node before any bundle exists.
`node.sh` keeps the results, each bundle's installs, and each bundle's preflight record in `results/` beside it, or in the directory that `FTSP_RESULTS` names, which bundles may share.
Before the experiment first runs for its label, `run_node.py`, which `node.sh` calls, runs the self-tests of the shared code in node mode and the experiment's acceptance tests, and stops if any fails.
Repeating the command resumes the run, on any host with the same CPU model, but not while the label's `active/` directory lists runs of another host.
One start, or `--cleanup`, owns a label at a time, and a start refuses while GNU parallel or a run of an earlier start is alive.
If a pass ends with tasks that have no final row, as after damage to `runs.jsonl`, `run_node.py` stops and lists them in `missing-pass<N>.txt` of the label directory, which `--status` counts.
To rerun them under all the checks of a start, delete their lines from the pass's job log, `joblog-pass<N>.tsv`, and repeat the command: GNU parallel's `--resume-failed` reruns a task that the job log does not list, and the tasks that have a final row end at once.
After a hard kill of `node.sh` or GNU parallel, run `node.sh EXPERIMENT --cleanup` before resuming, which kills GNU parallel and the runs that the label's `active/` directory lists, if their processes are still the recorded ones; `--status` prints the progress.
Both act on one label directory, which both print: give them the run's `--subset` if it was not `main`, and the same `FTSP_RESULTS`.
The archive that the run prints at its end is the one file to copy back.

For development on a workstation, `--local` runs the tasks one at a time, without GNU parallel or pinning, and `--subset` selects tasks, as in the commands below.
A local run has its own label directory, with the suffix `-local`, and the analyses of certification and distances refuse its rows unless given `--allow-local`, with which every output says how many rows of local runs it includes.
After a hard kill of a local run, run `run_node.py EXPERIMENT --local --cleanup` with the run's `--subset`, before any change to the tree, since the uncommitted changes of a dirty tree name its label directory; check that the label directory that it prints is the run's.

```bash
python experiments/state_prep_certification/run_node.py EXPERIMENT --local --subset smoke --approved-cpu-hours H
python experiments/state_prep_certification/common/selftest.py
```
