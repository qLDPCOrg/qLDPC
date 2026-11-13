# SPDX-License-Identifier: Apache-2.0

"""Search binary CSS or SWEL codes by annealing, scoring each proposal exactly.

Run with ``python -m experiments.code_annealing.core.run_annealing N K [options]`` to search codes
with N physical and K logical qubits, or with ``--help`` for the options.  Exact enumeration is
exponential, so use small ``--epochs`` and ``--steps-per-epoch`` for smoke tests, and ``--dry-run``,
which writes nothing.

Every search but a dry run compiles the Python set-up of its exact counts before its pilot, or
before its next epoch when it resumes.  This changes no result, and pays off over a long search.  It
would slow down the counts of the ``--init-samples`` codes slightly, so they come first.
``--skip-compile`` skips it for a short search.

Starting code.  A new search starts from the first random code that it draws, or from the
lowest-energy code of ``--init-samples`` random codes, which are scored exactly.  The energy of a
random code varies a lot, and so does the first temperature, which a pilot fits to one epoch's worth
of proposals from the starting code, so the lowest-energy code of many makes the first epochs less
erratic.

Files.  A run saves its chain checkpoint, such as ``css_n20_k04_s003.json`` for the code type css,
20 qubits, 4 logical qubits, and seed 3, after every completed epoch, and records of its
lowest-energy (``.lowest-energy.json``) and greatest-distance (``.greatest-distance.json``) codes as
soon as either improves, so they keep discoveries of an unfinished epoch.  They go to
``results/logs/``, or to the logs of ``--results-dir``.  ``--tag`` adds a suffix to the names, and
``--output`` names the chain explicitly, relative to the current directory, like ``--resume`` and
``--refine``.  A new search refuses to overwrite an existing file.  While a run is active it holds a
lock on a ``.lock`` file beside its checkpoint, so that a second process cannot run the same chain.
Redirect the progress that it prints to a ``.log`` file of the same name, as the jobs of
``write_jobs.py`` do.  Standard output gets one JSON line of scores per epoch, which a seeded run
repeats exactly.  Standard error gets the wall-clock times (UTC) at which the epochs of the run
start or resume and at which each epoch finishes, and the counting method of each epoch with
``--method auto``.

Resuming and refining.  ``--resume CHECKPOINT`` restores the saved policy, codes, scores, random
generator, temperature, and epoch count exactly, so an interrupted and resumed run reproduces an
uninterrupted one.  It rejects new code or policy options, and writes to the files of its
checkpoint, so it takes no ``--tag`` or ``--output``.  An interruption loses only the unfinished
epoch, never a record, and a replayed epoch never replaces a better record with a worse one.  The
first checkpoint is saved after the pilot, so a run interrupted before then, while it scores its
``--init-samples`` codes or its pilot, has no checkpoint and starts over.  ``--stop-after-epochs N``
stops after N more complete epochs.  Exact restart needs the same numerical runtime, operating
system, and architecture, and the same version of this code.  Otherwise, or to try new settings,
``--refine PATH`` starts a new chain from the lowest-energy code of a chain checkpoint or from the
code of a record, either of which may come from another runtime or schema version.

Scoring.  Every proposal is scored by its distance and its numbers of minimum-weight X-only and
Z-only logical operators.  ``--method`` is ``auto`` (the default), or one of the exact methods of
``CSSLogicalCounter.count_logical_ops``: ``brouwer_zimmermann``, ``macwilliams``, or
``brute_force``.  Before the pilot and at the start of each epoch, ``auto`` races
Brouwer--Zimmermann against MacWilliams and keeps the faster, or uses Brouwer--Zimmermann alone for
codes of large check rank (see ``annealing.py``).
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime
import itertools
import json
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

from qldpc.codes.distance import CSSLogicalCounter, compile_search_setup

from .annealing import (
    METHODS,
    SearchConfig,
    SearchState,
    energy,
    get_lowest_energy_code,
    run_epochs,
    start_search,
)
from .atomic import hold_lock
from .checkpoints import (
    RECORD_KINDS,
    get_record_path,
    load_checkpoint,
    load_code,
    read_saved_file,
    save_checkpoint,
    save_record,
)
from .cli import add_results_dir_argument, make_parser
from .paths import get_logs_dir, get_run_name
from .proposals import (
    CODE_TYPES,
    CodeType,
    random_initial_codes,
)

####################################################################################################
# command-line interface


def main(argv: Sequence[str] | None = None) -> None:
    """Run or resume an exact CSS/SWEL search, printing one JSON line of scores per epoch."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.stop_after_epochs is not None and args.stop_after_epochs < 0:
        parser.error("--stop-after-epochs must be nonnegative")
    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.resume is not None and (args.tag or args.output is not None):
        parser.error(
            "--resume continues the files of its checkpoint, so it takes no --tag or --output"
        )

    # Never overwrite files owned by another chain; resuming owns its own checkpoint and records.
    # Check before a slow initialization if the name is known, and again, once this process holds
    # the lock, before writing.
    output = args.output or args.resume or _get_output_from_arguments(args)
    new_files = args.resume is None and not args.dry_run
    if output is not None and new_files:
        _check_output_is_free(parser, output)

    # Only a chain checkpoint can be resumed, so that a lock file is never left beside another file.
    if args.resume is not None:
        if not args.resume.is_file():
            parser.error(f"{args.resume} is not a file")
        with _report_invalid_input(parser):
            if read_saved_file(args.resume, any_schema=True)[0] != "chain":
                raise ValueError(f"{args.resume} is a record of a code, not a chain; use --refine")

    # Lock the chain before reading it, so that no other process changes it while this one reads and
    # initializes it.
    with contextlib.ExitStack() as locks:
        if output is not None and not args.dry_run:
            _lock_chain(parser, locks, output)
        compile_setup = not (args.dry_run or args.skip_compile)
        if args.resume:
            with _report_invalid_input(parser):
                state = _resume_search(parser, args)
            if compile_setup:
                compile_search_setup()
        else:
            with _report_invalid_input(parser):
                initial, config = _prepare_new_search(parser, args)
            if compile_setup:
                compile_search_setup()
            # The pilot scores a whole epoch of proposals, so an error there keeps its traceback.
            state = start_search(initial, config, seed=_get_seed(args))
        if output is None:
            counter = state.current.counter
            output = _name_output(
                args, state.config.code_type, counter.num_qubits, counter.dimension
            )
            if not args.dry_run:
                _lock_chain(parser, locks, output)
        if new_files:
            _check_output_is_free(parser, output)
        _run_search(
            state,
            output,
            dry_run=args.dry_run,
            stop_after_epochs=args.stop_after_epochs,
        )


def build_parser() -> argparse.ArgumentParser:
    """The parser of the command line, which ``write_jobs`` uses to check the commands it writes."""
    defaults = SearchConfig()
    parser = make_parser(__doc__)
    parser.add_argument("num_qubits", type=int, nargs="?", help="number of physical qubits")
    parser.add_argument("num_logicals", type=int, nargs="?", help="number of logical qubits")

    # Search policy, saved in the checkpoint.  Names match SearchConfig fields, which document them.
    policy = parser.add_argument_group("search policy (see SearchConfig)")
    policy.add_argument("--code-type", choices=CODE_TYPES, help=f"(default: {defaults.code_type})")
    policy.add_argument(
        "--gate-size",
        type=int,
        choices=(2, 3, 4),
        help=f"qubits per proposed gate (default: {defaults.gate_size}, 4 for swel)",
    )
    policy.add_argument("--epochs", type=int, help=f"(default: {defaults.epochs})")
    policy.add_argument(
        "--steps-per-epoch", type=int, help=f"(default: {defaults.steps_per_epoch})"
    )
    policy.add_argument(
        "--uphill-acceptance-start",
        type=float,
        help=f"target acceptance of uphill proposals in the first epoch"
        f" (default: {defaults.uphill_acceptance_start})",
    )
    policy.add_argument(
        "--uphill-acceptance-stop",
        type=float,
        help=f"target acceptance of uphill proposals in the last epoch"
        f" (default: {defaults.uphill_acceptance_stop})",
    )
    policy.add_argument(
        "--temperature-memory",
        type=float,
        help="weight of the previous log-temperature in the next one"
        f" (default: {defaults.temperature_memory})",
    )
    policy.add_argument(
        "--physical-error-rate",
        type=float,
        help=f"the physical error rate p of the energy (default: {defaults.physical_error_rate})",
    )
    policy.add_argument(
        "--method",
        choices=METHODS,
        help=f"how proposals are scored, which never changes a score (default: {defaults.method})",
    )

    start = parser.add_argument_group("starting code of a new search")
    start.add_argument(
        "--seed", type=int, help="seed of the search and of its random codes (default: 0)"
    )
    start.add_argument(
        "--self-dual-start",
        action="store_true",
        help="draw self-dual random codes, also for a css search",
    )
    start.add_argument(
        "--init-samples",
        type=int,
        help="start from the lowest-energy code of this many random codes,"
        " instead of the first one",
    )
    start.add_argument(
        "--refine",
        type=Path,
        help="start from the lowest-energy code of this chain checkpoint,"
        " or the code of this record",
    )

    add_results_dir_argument(parser)
    files = parser.add_argument_group("files and execution")
    files.add_argument("--resume", type=Path, help="continue the chain of this checkpoint exactly")
    files.add_argument(
        "--output", type=Path, help="explicit checkpoint path; overrides --results-dir"
    )
    files.add_argument(
        "--tag",
        default="",
        help="suffix of the default file names, to tell batches apart",
    )
    files.add_argument("--dry-run", action="store_true", help="run, but write no file")
    files.add_argument(
        "--skip-compile",
        action="store_true",
        help="do not compile the Python set-up of the exact counts,"
        " which only pays off in a long search",
    )
    files.add_argument(
        "--stop-after-epochs",
        type=int,
        help="stop after this many additional complete epochs, without changing saved policy",
    )
    return parser


####################################################################################################
# helper methods


def _get_seed(args: argparse.Namespace) -> int:
    return 0 if args.seed is None else args.seed


def _get_output_from_arguments(args: argparse.Namespace) -> Path | None:
    """The checkpoint of a new search, if the command line gives its code size, or None.

    A refined search takes its size from its code, so its checkpoint is named once it has one.
    """
    if args.num_qubits is None or args.num_logicals is None or args.refine is not None:
        return None
    code_type = args.code_type or SearchConfig().code_type
    return _name_output(args, code_type, args.num_qubits, args.num_logicals)


def _name_output(
    args: argparse.Namespace, code_type: CodeType, num_qubits: int, num_logicals: int
) -> Path:
    """The checkpoint of a run in the logs, named by its code type, size, and seed."""
    name = get_run_name(code_type, num_qubits, num_logicals, _get_seed(args), args.tag)
    return get_logs_dir(args.results_dir) / f"{name}.json"


def _check_output_is_free(parser: argparse.ArgumentParser, output: Path) -> None:
    """Report a usage error if a new chain would overwrite the checkpoint or records of another."""
    if output.exists() or any(get_record_path(output, kind).exists() for kind in RECORD_KINDS):
        parser.error(
            f"{output} or its records already exist: use --resume to continue that run,"
            " or --tag or --output to start another"
        )


def _lock_chain(parser: argparse.ArgumentParser, locks: contextlib.ExitStack, output: Path) -> None:
    """Hold the lock of a chain until ``locks`` closes, or report that a process already has it."""
    try:
        locks.enter_context(hold_lock(output.with_suffix(".lock")))
    except BlockingIOError:
        parser.error(f"Another process is running {output}")


@contextlib.contextmanager
def _report_invalid_input(parser: argparse.ArgumentParser) -> Iterator[None]:
    """Report a rejected option, size, or file as a usage error instead of a traceback."""
    try:
        yield
    except (OSError, ValueError) as error:
        parser.error(str(error))


def _resume_search(parser: argparse.ArgumentParser, args: argparse.Namespace) -> SearchState:
    """Restore a saved chain, whose policy and initial code cannot be overridden."""
    new_search_options = (
        *(field.name for field in dataclasses.fields(SearchConfig)),
        "refine",
        "num_qubits",
        "num_logicals",
        "seed",
        "init_samples",
    )
    if args.self_dual_start or any(getattr(args, name) is not None for name in new_search_options):
        parser.error("--resume restores its saved policy; do not supply new code/policy options")
    return load_checkpoint(args.resume)


def _prepare_new_search(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> tuple[CSSLogicalCounter, SearchConfig]:
    """The starting code and the policy of a new search, checked against each other.

    The starting code is a refined saved code, or a random initial code.
    """
    options = {
        field.name: getattr(args, field.name)
        for field in dataclasses.fields(SearchConfig)
        if getattr(args, field.name) is not None
    }
    if args.code_type == "swel" and args.gate_size is None:
        options["gate_size"] = 4
    config = SearchConfig(**options)
    if args.refine is not None:
        initial = _load_refinement_code(parser, args)
    else:
        if args.num_qubits is None or args.num_logicals is None:
            parser.error("New searches require num_qubits and num_logicals")
        initial = _get_random_code(parser, args, config)
    config.validate_code(initial)
    return initial, config


def _get_random_code(
    parser: argparse.ArgumentParser, args: argparse.Namespace, config: SearchConfig
) -> CSSLogicalCounter:
    """A random initial code: the first one drawn, or the lowest-energy one of several."""
    self_dual = config.code_type == "swel" or args.self_dual_start
    counters = random_initial_codes(
        args.num_qubits, args.num_logicals, self_dual=self_dual, seed=_get_seed(args)
    )
    if args.init_samples is None:
        return next(counters)
    if args.init_samples < 1:
        parser.error("--init-samples must be positive")
    initial = get_lowest_energy_code(itertools.islice(counters, args.init_samples), config)
    print(
        f"Started from the lowest-energy code of {args.init_samples} random codes:"
        f" distance {initial.score.distance}, energy {energy(initial.score, config):.4f}",
        file=sys.stderr,
    )
    return initial.counter


def _load_refinement_code(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> CSSLogicalCounter:
    """Import the saved code to refine, checking it against any explicit size."""
    if args.self_dual_start or args.init_samples is not None:
        parser.error("Refinement does not use random initialization options")
    initial = load_code(args.refine)
    if (args.num_qubits is not None and args.num_qubits != initial.num_qubits) or (
        args.num_logicals is not None and args.num_logicals != initial.dimension
    ):
        parser.error("Requested n/k differ from the refinement code")
    return initial


def _run_search(
    state: SearchState,
    output: Path,
    *,
    dry_run: bool,
    stop_after_epochs: int | None,
) -> None:
    """Run the remaining epochs, saving the chain and its two records as they improve."""

    def save_and_report_epoch(state: SearchState) -> None:
        if not dry_run:
            save_checkpoint(state, output)
        print(
            json.dumps(
                {
                    "epoch": state.completed_epochs,
                    "current_score": dataclasses.asdict(state.current.score),
                    "lowest_energy_score": dataclasses.asdict(state.lowest_energy.score),
                    "greatest_distance_score": dataclasses.asdict(state.greatest_distance.score),
                    **dataclasses.asdict(state.history[-1]),
                }
            ),
            flush=True,
        )
        # Wall-clock times, and the method that ``auto`` chooses, which depends on timing, vary from
        # run to run, so they go to stderr, which keeps stdout deterministic.
        method = f", scored with {state.epoch_method}" if state.config.method == "auto" else ""
        print(f"epoch {state.completed_epochs}: finished at {_get_time()}{method}", file=sys.stderr)

    def preserve_lowest_energy(state: SearchState) -> None:
        save_record(state, output, "lowest-energy")

    def preserve_greatest_distance(state: SearchState) -> None:
        save_record(state, output, "greatest-distance")

    print(
        f"epochs started at {_get_time()}, after {state.completed_epochs} completed epochs",
        file=sys.stderr,
    )
    # Save the initial chain and its pilot optima, so that even an interrupted first epoch is kept.
    if not dry_run:
        if not output.exists():
            save_checkpoint(state, output)
        preserve_lowest_energy(state)
        preserve_greatest_distance(state)
    run_epochs(
        state,
        stop_after_epochs=stop_after_epochs,
        on_epoch=save_and_report_epoch,
        on_lowest_energy=None if dry_run else preserve_lowest_energy,
        on_greatest_distance=None if dry_run else preserve_greatest_distance,
    )


def _get_time() -> str:
    """The current wall-clock time in UTC, to the millisecond."""
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="milliseconds")


if __name__ == "__main__":
    main()
