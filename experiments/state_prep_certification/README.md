# State preparation certification

This directory holds experiments on the exact certification of fault-tolerant state preparation circuits, and on the exact distance searches that it relies on.
Each experiment is a subdirectory that plugs into shared code at this level:

| Experiment | What it measures | Inputs |
| --- | --- | --- |
| [`distances/`](distances/) | Exact code distances with qLDPC's extensions of the Brouwer-Zimmermann search on, against the same search configured with only the techniques that some public exact tool shares with it | Codes drawn by a fixed rule with fixed seeds, and named anchors |

The experiments run only qLDPC and its Python dependencies, so that they can run on a Linux node without practical network access.
The shared workflow builds everything that the node needs into one offline bundle on a workstation, runs each experiment on the node with GNU parallel, and brings one archive back for analysis on the workstation.
Times measured on a workstation are never used as results.

## Shared code

- [`run_node.py`](run_node.py) runs one experiment for its label: the acceptance tests, the task lists with their ceiling of CPU-hours, the two passes, and the archive.
- [`node.sh`](node.sh) is the node's entry point: a preflight check of the node, an offline install from the bundle, and then `run_node.py`.
- [`bundle.py`](bundle.py) builds the offline bundle on the workstation.
- [`pins.json`](pins.json) pins the qLDPC commit, CPython, the Linux platform of the wheels, and GNU parallel.
- [`common/`](common/) holds the task runner (`runner.py` and `run_task.py`), the result store (`store.py`), the task lists (`tasks.py`), the label and manifest (`manifest.py`), and the plug-in interface (`plugin.py`).
  `common/selftest.py` tests them locally.

### Labels and results

A run's label is the experiment's name and the first 12 hexadecimal digits of the SHA-256 of `pins.json`, the experiment's frozen input files, and the commit of this tree.
Any change to them gives a new label, and so a new directory `results/<experiment>/<label>/`, which Git ignores.
From a bundle, the label also covers the bundle's resolved requirements, so that it fixes the versions of the third-party packages too; the manifest records the installed versions.
That directory holds the manifest of the machine and software, the task lists, GNU parallel's job logs, `runs.jsonl` with one row per run, the tail of each run's output in `logs/`, and the archive.

### Runs

Each task runs in a fresh process pinned to one core, with the thread variables of numerical libraries set to 1.
Only the measured process and its descendants are pinned to that core; the runner that watches them, and polls their memory every second, runs on the cores left free.
The runner times only the measured call, from the moment that the run reports that timing has started.
A run is `ok` if the measured call returns within the task's limit, and a `timeout` if its measured time reaches the limit, even if it returned later.
Process start, imports, and warm-up are capped at 120 s, beyond which the run is an `error`.
The runner kills a run above its memory cap as a `memout`, and kills every process of a stopped run at once, then waits up to 30 s for them to exit.
Work after the measured call, such as describing a counterexample, has its own allowance if the experiment sends its measured result first; running out of it then marks only the row's `post_status`.

There is one job slot per physical core, with two cores and 16 GiB of memory left free and as many slots as the memory caps of the runs allow.
Pass 1 holds every first run in an order shuffled with a fixed seed.
Pass 2 holds the repeats that each experiment's repeat policy asks for after pass 1, such as two more runs of a fast run.

## Workflow

On the workstation, with the experiment's inputs frozen and committed, build the bundle and note its checksum:

```bash
python experiments/state_prep_certification/bundle.py EXPERIMENT --out /path/to/bundles
```

The bundle needs a pinned qLDPC commit in `pins.json`; `--unfrozen` builds a trial bundle without one, which `run_node.py` refuses on the node unless `--allow-unfrozen` is given.

On the node, check the bundle, unpack it, and run the experiment from its `qLDPC/` directory, with the CPU-hours H approved for it:

```bash
sha256sum -c spc-bundle-<id>.tar.sha256 && tar -xf spc-bundle-<id>.tar
cd spc-bundle-<id>/qLDPC
bash experiments/state_prep_certification/node.sh EXPERIMENT --approved-cpu-hours H
```

`node.sh --check` runs only the preflight, and needs only `node.sh` and `pins.json`, so it can run on the node before any bundle exists.
Before the experiment, `node.sh` runs the self-tests of the shared code in node mode and the experiment's acceptance tests, and stops if any fails.
Repeating the command resumes the run.
After a hard kill of `node.sh` or GNU parallel, run `node.sh EXPERIMENT --cleanup` before resuming; `--status` prints the progress.
The archive that the run prints at its end is the one file to copy back.

For development on a workstation, `--local` runs the tasks one at a time, without GNU parallel or pinning, and `--subset` selects tasks, as in:

```bash
python experiments/state_prep_certification/run_node.py EXPERIMENT --local --subset smoke --approved-cpu-hours H
python experiments/state_prep_certification/common/selftest.py
```
