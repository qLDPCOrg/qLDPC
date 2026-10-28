# SPDX-License-Identifier: Apache-2.0

"""Write the annealing jobs of a campaign to ``results/logs/jobs.txt``, one shell command per line.

Run with ``python -m experiments.code_annealing.write_jobs``, then run the jobs, from any
directory, with GNU parallel::

    parallel -j 16 < experiments/code_annealing/results/logs/jobs.txt

Give ``parallel`` the option ``--joblog FILE`` to record the exit status of every job.  Each job
runs one chain with one core, and writes its checkpoint, its two code records and a log to the logs
of the results directory (see ``run_annealing.py``).  Interrupted runs lose at most one epoch: run
``write_jobs`` again, and its new ``jobs.txt`` resumes the unfinished runs and skips the finished
ones, in the same order, so that rerunning it after any interruption completes the campaign.  Pass
``--tag`` to start another campaign beside the files of earlier runs.  Progress is shown by
``progress_report.py``, and the codes that the runs found are collected by
``analysis/collect_annealed_codes.py``.

Pass ``--test`` to write a smoke-test campaign instead: the 20- and 22-qubit rows, 2 seeds, 2 epochs
of 100 proposals, starting from the lowest-energy code of 10 random codes, with the files in the
``test/`` subdirectory of the results directory.  Give the same flag to the other high-level
scripts to follow that campaign.

The campaign, ``CAMPAIGN`` in this script, lists each code size with its number of epochs and its
final target acceptance of uphill proposals.  Every code size runs both code types, CSS (starting
from a self-dual code) and SWEL, with the seeds of ``CampaignRow.num_seeds``, and every search
starts from the lowest-energy code of ``INIT_SAMPLES`` random codes, drawn with its own seed.
Running all of it takes a very long time: choose jobs with ``--qubits``, ``--max-seeds`` and
``--code-types``.
"""

from __future__ import annotations

import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .core.atomic import write_text_atomic
from .core.cli import (
    add_results_dir_argument,
    add_test_argument,
    get_results_dir,
    make_parser,
)
from .core.paths import EXPERIMENT_DIR, get_logs_dir, get_run_name
from .core.proposals import CODE_TYPES, CodeType
from .core.run_annealing import build_parser as build_command_parser
from .core.saved_runs import load_run_progress

_REPO_ROOT = EXPERIMENT_DIR.parents[1]
STEPS_PER_EPOCH = 10_000
NUM_SEEDS = 100
INIT_SAMPLES = 1000
UPHILL_ACCEPTANCE_START = 0.01


_RunStatus = Literal["new", "unfinished", "finished"]


@dataclass(frozen=True)
class _SmokeScale:
    """The part of the campaign that ``--test`` runs: enough to exercise every step."""

    qubits: tuple[int, ...]
    max_seeds: int
    epochs: int
    steps_per_epoch: int
    init_samples: int


SMOKE_SCALE = _SmokeScale(
    qubits=(20, 22), max_seeds=2, epochs=2, steps_per_epoch=100, init_samples=10
)


@dataclass(frozen=True)
class CampaignRow:
    """A code size to search, with its epochs and final target acceptance of uphill proposals.

    The number of seeds of each code type depends on the size (see ``num_seeds``).
    """

    num_qubits: int
    num_logicals: int
    epochs: int
    uphill_acceptance_stop: float

    @property
    def num_seeds(self) -> int:
        """The number of seeds of each code type, which count up from 0.

        There are ``NUM_SEEDS`` below 40 qubits, half as many below 50 qubits, and a quarter as many
        from 50 qubits.
        """
        if self.num_qubits < 40:
            return NUM_SEEDS
        return NUM_SEEDS // 2 if self.num_qubits < 50 else NUM_SEEDS // 4


# The code sizes of the campaign: qubits, logical qubits, epochs, final uphill acceptance.
CAMPAIGN = (
    CampaignRow(20, 4, 1000, 1e-3),
    CampaignRow(20, 6, 1000, 1e-3),
    CampaignRow(22, 4, 1000, 1e-3),
    CampaignRow(22, 6, 1000, 1e-3),
    CampaignRow(24, 4, 1000, 1e-4),
    CampaignRow(24, 6, 1000, 1e-4),
    CampaignRow(25, 5, 1000, 1e-4),
    CampaignRow(26, 4, 1000, 1e-4),
    CampaignRow(26, 6, 1000, 1e-4),
    CampaignRow(30, 4, 1000, 1e-4),
    CampaignRow(30, 6, 1000, 1e-4),
    CampaignRow(32, 6, 1000, 1e-4),
    CampaignRow(32, 8, 1000, 1e-4),
    CampaignRow(36, 6, 2000, 1e-4),
    CampaignRow(36, 8, 2000, 1e-4),
    CampaignRow(38, 8, 2000, 1e-4),
    CampaignRow(38, 10, 2000, 1e-4),
    CampaignRow(38, 12, 2000, 1e-4),
    CampaignRow(40, 10, 2000, 1e-4),
    CampaignRow(40, 12, 2000, 1e-4),
    CampaignRow(42, 8, 2000, 1e-4),
    CampaignRow(42, 10, 2000, 1e-4),
    CampaignRow(42, 12, 2000, 1e-4),
    CampaignRow(44, 8, 2000, 1e-4),
    CampaignRow(48, 6, 3000, 1e-4),
    CampaignRow(48, 8, 3000, 1e-4),
    CampaignRow(50, 6, 3000, 1e-4),
    CampaignRow(50, 8, 3000, 1e-4),
    CampaignRow(50, 10, 3000, 1e-4),
    CampaignRow(52, 10, 1500, 1e-4),
    CampaignRow(52, 12, 1500, 1e-4),
    CampaignRow(56, 12, 1500, 1e-4),
    CampaignRow(56, 14, 1500, 1e-4),
)


def main(argv: Sequence[str] | None = None) -> None:
    """Write the jobs file of the campaign, largest codes first."""
    parser = make_parser(__doc__)
    parser.add_argument(
        "--code-types",
        nargs="+",
        choices=CODE_TYPES,
        default=CODE_TYPES,
        help="code types to run",
    )
    parser.add_argument("--qubits", type=int, nargs="+", help="only these numbers of qubits")
    parser.add_argument("--max-seeds", type=int, help="only the first seeds of each code size")
    parser.add_argument("--epochs", type=int, help="override the epochs of every code size")
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        help=f"proposals of an epoch (default: {STEPS_PER_EPOCH}, or {SMOKE_SCALE.steps_per_epoch}"
        " with --test)",
    )
    parser.add_argument(
        "--init-samples",
        type=int,
        help=f"random codes to choose the starting code of a search from (default: {INIT_SAMPLES},"
        f" or {SMOKE_SCALE.init_samples} with --test)",
    )
    parser.add_argument(
        "--tag", default="", help="suffix of the search names and files, to tell campaigns apart"
    )
    parser.add_argument("--python", default="python", help="the Python command of the jobs")
    add_results_dir_argument(parser)
    add_test_argument(parser)
    args = parser.parse_args(argv)
    for name in ("max_seeds", "epochs", "steps_per_epoch", "init_samples"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    if args.test:
        qubits = args.qubits or SMOKE_SCALE.qubits
        max_seeds = args.max_seeds or SMOKE_SCALE.max_seeds
        epochs = args.epochs or SMOKE_SCALE.epochs
    else:
        qubits, max_seeds, epochs = args.qubits, args.max_seeds, args.epochs
    settings = _JobSettings(
        epochs=epochs,
        steps_per_epoch=args.steps_per_epoch
        or (SMOKE_SCALE.steps_per_epoch if args.test else STEPS_PER_EPOCH),
        init_samples=args.init_samples or (SMOKE_SCALE.init_samples if args.test else INIT_SAMPLES),
        python=args.python,
    )

    rows = [row for row in CAMPAIGN if qubits is None or row.num_qubits in qubits]
    if not rows:
        parser.error(f"the campaign has no code size with {qubits} qubits")
    # the jobs change into the repository root, so their paths must not be relative
    logs_dir = get_logs_dir(get_results_dir(args)).resolve()
    command_parser = build_command_parser()
    jobs = []
    skipped = 0
    # Long jobs first, so that the last jobs of a parallel run are short.
    for row in sorted(rows, key=lambda row: -row.num_qubits):
        for code_type in args.code_types:
            for seed in range(row.num_seeds)[:max_seeds]:
                name = get_run_name(code_type, row.num_qubits, row.num_logicals, seed, args.tag)
                checkpoint = logs_dir / f"{name}.json"
                status = _get_run_status(checkpoint)
                if status == "finished":
                    skipped += 1
                    continue
                if status == "unfinished":
                    arguments = ["--resume", str(checkpoint)]
                else:
                    arguments = _get_new_run_arguments(settings, row, code_type, seed, checkpoint)
                command_parser.parse_args(arguments)  # bad arguments fail here, not in a job
                log = logs_dir / f"{name}.log"
                jobs.append(
                    _get_job(settings.python, arguments, log, append=status == "unfinished")
                )

    output = logs_dir / "jobs.txt"
    # writing the jobs file creates the logs directory, which the shell opens before the jobs run
    write_text_atomic(output, "".join(f"{job}\n" for job in jobs))
    print(f"Wrote {len(jobs)} jobs to {output}, and skipped {skipped} finished runs")


@dataclass(frozen=True)
class _JobSettings:
    """What the jobs share beyond the code size: the scale of a search and the Python command."""

    epochs: int | None
    """Epochs of every search, or None for those of its code size."""

    steps_per_epoch: int
    init_samples: int
    python: str


####################################################################################################
# helper methods


def _get_new_run_arguments(
    settings: _JobSettings, row: CampaignRow, code_type: CodeType, seed: int, checkpoint: Path
) -> list[str]:
    """The arguments of ``run_annealing`` for a new chain."""
    epochs = row.epochs if settings.epochs is None else settings.epochs
    arguments = [
        str(row.num_qubits),
        str(row.num_logicals),
        "--code-type",
        code_type,
        "--init-samples",
        str(settings.init_samples),
        "--seed",
        str(seed),
        "--epochs",
        str(epochs),
        "--steps-per-epoch",
        str(settings.steps_per_epoch),
        "--uphill-acceptance-start",
        str(UPHILL_ACCEPTANCE_START),
        "--uphill-acceptance-stop",
        str(row.uphill_acceptance_stop),
        "--output",
        str(checkpoint),
    ]
    return arguments + ["--self-dual-start"] if code_type == "css" else arguments


def _get_job(python: str, arguments: list[str], log: Path, *, append: bool) -> str:
    """A shell command that runs ``run_annealing`` from the repository root, logging to a file.

    The job imports ``qldpc`` from the repository that it runs in, so a copy of the repository
    works without being installed.

    Args:
        python: The Python command that runs the job.
        arguments: The arguments of ``run_annealing``.
        log: The file that receives the output of the job.
        append: Whether to append to the log, as a resumed run does, rather than overwrite it.

    Returns:
        The shell command, with every argument quoted.
    """
    quoted = " ".join(shlex.quote(argument) for argument in arguments)
    root = shlex.quote(str(_REPO_ROOT))
    path = shlex.quote(f"{_REPO_ROOT}/src:{_REPO_ROOT}")
    return (
        f"cd {root} && PYTHONPATH={path} NUMBA_NUM_THREADS=1 {shlex.quote(python)}"
        f" -m experiments.code_annealing.core.run_annealing {quoted}"
        f" {'>>' if append else '>'} {shlex.quote(str(log))} 2>&1"
    )


def _get_run_status(checkpoint: Path) -> _RunStatus:
    """Whether a run is ``new``, ``unfinished`` or ``finished``, from its checkpoint file."""
    if not checkpoint.exists():
        return "new"
    progress = load_run_progress(checkpoint)
    if progress is None:
        raise SystemExit(f"{checkpoint} is not a chain checkpoint")
    return "finished" if progress.finished else "unfinished"


if __name__ == "__main__":
    main()
