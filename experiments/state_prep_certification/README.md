# State preparation certification

**Prototype.** Everything in `state_prep_certification/` is a prototype, subject to change: its code, interfaces, inputs, outputs, and documentation.

This directory holds experiments on the exact certification of fault-tolerant state preparation circuits, and on the exact distance searches that certification relies on.
Each experiment is a subdirectory that plugs into shared code at this level:

| Experiment | What it measures | Inputs |
| --- | --- | --- |
| [`certification/`](certification/) | Exact certification of bounded fault propagation in state preparation circuits by qLDPC's certifier: which circuits certify, how many faults the certifier lists compared with all accepted fault combinations, smallest counterexamples, and the cost of the certifier's two [weight oracles](certification/README.md#terms) (Brouwer-Zimmermann and connected-cluster) | MQT QECC's shipped and synthesized circuits, Chen et al.'s hand-translated circuit and Goto's circuit as a qLDPC example notebook writes it (both cited in certification's README), and weakened variants |
| [`distances/`](distances/) | Exact code distances by qLDPC's Brouwer-Zimmermann search ([guide](../../src/qldpc/codes/distance/brouwer_zimmermann.md)) with everything on, at its defaults, against the same search with four of its extensions turned off | Codes drawn by a fixed rule with fixed seeds, and anchors: named codes, most with a reference distance |
| [`ranking/`](ranking/) | Exploratory; certification and distances do not use it. Whether decoder-free scores of certified preparation circuits order them as stratified direct simulation of a fixed consumer does, and at what cost | The certified t = 1 circuits of the certification experiment, with its code files |

The experiments run only qLDPC, its Python dependencies with the extras that `pins.json` names (`qldpc_extras`), and the few extra wheels that an experiment's plug-in names (`EXTRA_WHEELS`, such as matplotlib), so that they can run on a Linux node without practical network access.
The shared workflow builds everything that the node needs into one offline bundle on a workstation, runs each experiment on the node with GNU parallel, and brings one archive back for analysis on the workstation.
Times measured on a workstation are never results (see [Local runs](#local-runs)).

## Terms

- The **label** of an experiment names its inputs and code (see [Labels and results](#labels-and-results)), and the **label directory** holds its results.
- A **task** is one measurement that an experiment defines: one **arm**, a method being measured, on one input, with its limits and a unique `key`; [`common/tasks.py`](common/tasks.py) lists a task's fields.
- A **run** is one execution of a task in a fresh process, which the task runner watches, and whose **measured call** it times (see [Runs](#runs)).
  Each task gets one **row** in the label directory's `runs.jsonl`.
- A **retry** is the runner's second run of a task whose first run compiled code during its measured call; it replaces the first run, and gives the task's row.
- **Pass 1** runs each task of an experiment's task list, and **pass 2** the repeats that the experiment's repeat policy asks for after pass 1; the **task lists** are the files that name each pass's tasks ([`common/tasks.py`](common/tasks.py)).
- A **repeat** is a task of pass 2 that measures the same arm on the same input as a task of pass 1, with its own key, `repeat` number, and row.
- A **start** is one invocation of `run_node.py` for a label: on the node through `node.sh`, or on a workstation with `--local`.
- The **ceiling** bounds the CPU-hours of the runs of a label directory's pass-1 tasks, with their retries and repeats; a start's `--approved-cpu-hours` must cover it (see [Scheduling](#scheduling)).
- In **node mode**, the node runs tasks with GNU parallel, each in a **job slot** with a CPU of its own; **local mode** (`--local`) is for development on a workstation (see [Local runs](#local-runs)).

## Shared code

- [`run_node.py`](run_node.py) runs one experiment for its label: the acceptance tests, the task lists with their ceiling, the two passes, and the archive.
- [`node.sh`](node.sh) is the node's entry point: a preflight check of the node, an offline install from the bundle, and then `run_node.py`.
- [`bundle.py`](bundle.py) builds the offline bundle on the workstation.
- [`pins.json`](pins.json) pins the qLDPC commit, CPython, the Linux platform of the wheels, GNU parallel, and qLDPC's extras that the bundle installs (`qldpc_extras`).
  It also holds an `interfaces` entry, which its `interfaces_note` explains, and other notes, which enter the label like the rest of the file.
- [`common/`](common/) holds:
  - the task runner, `runner.py` and `run_task.py`;
  - the result store and archive, `store.py`;
  - the task lists and the ceiling, `tasks.py`;
  - the label and manifest, `manifest.py`;
  - the plug-in interface, `plugin.py`;
  - `local_runs.py`, which runs tasks locally for the acceptance tests, and `acceptance.py`, which runs each experiment's acceptance tests;
  - `selftest.py`, which tests the shared code on a stand-in experiment (`selftest_plugin.py`), locally and in node mode.

### Labels and results

A label is the experiment's name and a hash of three things: `pins.json`; the experiment's **frozen files** (`FROZEN_FILES` of its plug-in, [`common/plugin.py`](common/plugin.py)), such as its configuration and the data that fix its inputs; and HEAD of the qLDPC checkout.
From a bundle, the hash also covers the bundle's resolved requirements; [`common/manifest.py`](common/manifest.py) gives the details.
So any change to these, even a commit that changes only documentation, gives a new label directory `results/<experiment>/<label>/`, which Git ignores.
In such paths, `results/` stands for the results directory: by default `results/` beside `node.sh`, or else the directory that `FTSP_RESULTS` names for `node.sh` or `--results-dir` for `run_node.py`.
`common/manifest.py` gives the suffixes that the label directory's name can carry.
The label directory holds, among others:

- `manifest.json`: the machine, the installed package versions, Python's optimization level, and the environment variables that can change a run, such as `PYTHONPATH` and `NUMBA_*`;
- `starts.jsonl`: the host, CPU model, and environment of every start;
- `acceptance.json`: the record of the acceptance tests;
- the task lists and GNU parallel's job logs;
- `runs.jsonl`: one row per task (see [Runs](#runs)), with the tail of each run's output in `logs/`;
- `active/`: the records of GNU parallel and of the runs that are, or were, in progress;
- `lock`: the lock of the start that owns the label directory;
- `archive/`: the archive.

Each experiment adds its own files, such as certification's copy of `circuits.json` and its `oracle_calls/`, and distances' copy of `sample.json`, its `codes/`, and `draws.jsonl`.
A node start refuses to run while some of the recorded environment variables are set, such as `PYTHONPATH`, unless `--allow-env NAME` accepts one (`REFUSED_ENVIRONMENT` in [`run_node.py`](run_node.py)).
The analyses of certification and distances mark their outputs where a start accepted such a variable or where it, or a Python that it started, ran with -O, and where the recorded variables differ among the label's starts ([`common/store.py`](common/store.py)).

### Runs

On the node, each run of a task is a fresh process pinned to one logical CPU of its own physical core, with the thread variables of numerical libraries set to 1; the runner that watches it runs on the reserved cores (see [Scheduling](#scheduling)).
The runner times one call of the task's code, the measured call, whose wall time is T, and enforces the task's limits: a start-up cap, a time limit, a memory cap, and an allowance for untimed work after the measured call (post-processing).
A run is a `timeout` if T reaches the time limit; otherwise it is a `memout` if its memory exceeded the cap before its result arrived, `ok` if its result arrived, and an `error` otherwise, as when start-up exceeds its cap.
A run whose measured call compiled code, as numba reports it, gets a retry (see [Terms](#terms)), so T includes no compilation that numba reports, unless the retry compiled too.
Then the row has `compiled_in_timing` set, and no analysis uses its T.
[`common/runner.py`](common/runner.py) gives these rules in full, such as how the runner measures memory, when a `timeout` or `memout` row keeps a result, and when a late kill or a late peak of memory marks only the row's `post_status`; [`common/tasks.py`](common/tasks.py) gives the limits that each task can set and their defaults.
Each experiment's `analyze.py` says how it combines the rows of a task and its repeats, how it uses the results that `timeout` and `memout` rows keep, and how it treats a row with `compiled_in_timing` set: [certification's](certification/analyze.py) and [distances'](distances/analyze.py).

### Scheduling

`RESERVED_CORES` physical cores and `RESERVED_MEM_GIB` GiB of memory, constants of [`common/tasks.py`](common/tasks.py), are reserved for the runners, GNU parallel, and the system.
Each job slot gets its own free physical core and the experiment's memory per slot, `JOB_MEM_GIB` of its plug-in ([`common/plugin.py`](common/plugin.py)), as `tasks.choose_cpus` describes.
There are as many slots as the free cores and memory allow, and at most N if `node.sh` or `run_node.py` is given `--jobs N`.
Pass 1 runs its tasks in an order shuffled with a fixed seed, and pass 2 runs the repeats, such as two more measurements of a task whose first run was fast.

The ceiling, as [`common/tasks.py`](common/tasks.py) derives it, leaves out some time, which that module lists, such as reruns after a hard kill, so an allocation of exactly the ceiling can run out.
A start prints the number of tasks of pass 1 and the ceiling, and stops with exit status 1 unless `--approved-cpu-hours` covers the ceiling.
On a workstation, `run_node.py EXPERIMENT --local --subset S --dry-run` prints them without running the acceptance tests.

## Workflow

### On the workstation

Install qLDPC from a checkout, as qLDPC's [installation guide](../../docs/source/installation.rst) describes.
The workstation also needs Git, `uv`, and `uvx` for `bundle.py`, and GNU parallel (`parallel` on the `PATH`) for the self-tests.
Each experiment's data, such as its codes or circuits, are not committed: the experiment reads them from its data directory, which Git ignores.
Each experiment's README lists the files that its data directory must hold, says which of them the experiment's scripts make, and which are inputs that it takes as given.
Put those inputs in place, make the rest with the experiment's scripts, and check them as its README says, such as with `circuits.py --check`.
Then build the bundle, which carries the data, and note its checksum:

```bash
python experiments/state_prep_certification/bundle.py EXPERIMENT --out /path/to/bundles
```

If an experiment's frozen files lie in another experiment's directory, pass that experiment to `bundle.py` too.
`bundle.py` refuses a bundle that lacks a named experiment's frozen file, and says which experiment to add, or which file to make.

The bundle needs a pinned qLDPC commit in `pins.json`; `--unfrozen` builds a trial bundle without one, which `run_node.py` refuses on the node unless `--allow-unfrozen` is given.

### On the node

Check the bundle, unpack it, and run the experiment from its `qLDPC/` directory, with the CPU-hours H approved for it:

```bash
sha256sum -c spc-bundle-<id>.tar.sha256 && tar -xf spc-bundle-<id>.tar
cd spc-bundle-<id>/qLDPC
bash experiments/state_prep_certification/node.sh EXPERIMENT --approved-cpu-hours H
```

Here `<id>` is the bundle's id, a hash of the pins, HEAD of the qLDPC checkout, the names of the experiments, the platforms, and the resolved requirements (see `bundle.py`'s docstring); it is not a label.
The experiments' data directories do not enter it, so two bundles that differ only in their data get the same name, and the later tar replaces the earlier one in `--out`.

`node.sh --check` runs only the preflight, and needs only `node.sh` and `pins.json`, so it can run on the node before any bundle exists.
`node.sh` keeps the results, each bundle's installs, and each bundle's preflight record in the results directory (see [Labels and results](#labels-and-results)), which bundles may share.
Before the experiment first runs for its label directory, `run_node.py`, which `node.sh` calls, runs the self-tests of the shared code in node mode and the experiment's acceptance tests, and stops if any fails.
Repeating the command resumes the run; a task with a row, whatever its status, never runs again.
If a pass leaves tasks without a row, the start stops and says how to rerun them.
After a hard kill of `node.sh` or GNU parallel, run `node.sh EXPERIMENT --cleanup` before resuming; `node.sh EXPERIMENT --status` prints the progress.
Use the same `FTSP_RESULTS` for `--status` and `--cleanup` as for the run.
If the run used a subset other than `main`, pass its `--subset` too.
Check that the label directory that they print is the run's.
[`run_node.py`](run_node.py)'s docstring, which its `--help` prints, gives the rules for resuming on another host, the refusals of a start, and what `--cleanup` and `--status` do.

### Analysis

The archive that the start prints at its end, with the `.sha256` file beside it, is what to copy back.
A start writes the archive only once both passes leave every task with a row; for a run that cannot get there, copy back the label directory itself, which `analyze.py --label-dir` accepts.
Its analysis prints a note when tasks have no row.
Certification reports each such task as `missing`; distances reports an arm as `missing` if its first run has no row, and leaves absent repeats out of its timing summary, as `distances/analyze.py`'s docstring describes.
On the workstation, analyze the archive with the experiment's `analyze.py --archive PATH --out DIR`.
The analyses of certification and distances check the archive against the `.sha256` file and extract it into a temporary directory, which they remove when they end, so with `--archive` they write their outputs only to `--out`, which they then require ([`common/store.py`](common/store.py)'s `open_archive`).
The experiment's README gives the analysis's other options and its outputs.

### Local runs

For development on a workstation, `--local` runs the tasks one at a time, without GNU parallel, pinning, or the acceptance tests, unless `--subset acceptance` asks for these tests.
`--subset` selects tasks, such as `smoke`, a small subset that each experiment defines, as in the commands below.
A local run has its own label directory, with the suffix `-local`.
Its rows were not timed as on the node, so the analyses of certification and distances refuse them unless given `--allow-local`, and then mark every output with the number of local rows ([`common/store.py`](common/store.py)).
After a hard kill of a local run, run `run_node.py EXPERIMENT --local --cleanup` with the run's `--subset` before any change to the tree.
The label directory's name depends on the tree: with uncommitted changes to tracked files, it carries `-dirty-<hash>` (see `common/manifest.py`), so after a change, `--cleanup` looks in another directory.
[`run_node.py`](run_node.py)'s `cleanup` says what a local `--cleanup` kills and what it can leave behind.
If the tree has changed since, each record in the run's `active/` directory names the run's process id (`pid`) and its runner's (`runner_pid`), with their start times, which you can check with `ps` and kill by hand.

```bash
python experiments/state_prep_certification/run_node.py EXPERIMENT --local --subset smoke --approved-cpu-hours H
python experiments/state_prep_certification/common/selftest.py
```
