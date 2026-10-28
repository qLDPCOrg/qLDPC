# SPDX-License-Identifier: Apache-2.0

"""Search binary CSS/SWEL codes with exact physical logical-weight scores.

Run with ``python -m experiments.code_annealing.core.run_annealing N K [options]`` to search codes
with N physical and K logical qubits, or with ``--help`` for the options.  Exact enumeration is
exponential, so use small ``--epochs`` and ``--steps-per-epoch`` for smoke tests, and ``--dry-run``,
which writes nothing.  Numba compiles its kernels on first use, which is not a steady-state cost.
Every search but a dry run also compiles the set-up code of the counts when it starts (see
:func:`qldpc.codes.distance.compile_search_setup`), which pays off over the many counts of a search
and changes no result; ``--skip-compile`` skips it for a short search.  Compiling stays on for the
rest of the process.

Starting code.  A new search starts from the first random code that it draws, which has at least
``--min-distance``, or from the lowest-energy code of ``--init-samples`` random codes, which are
scored exactly.  The energy of a random code varies a lot, and so does the temperature that the
pilot epoch fits to the neighbors of the starting code, so the lowest-energy code of many makes the
first epochs less erratic.  ``--init-samples`` cannot be combined with ``--min-distance`` or
``--max-attempts``.

Files.  A run saves its chain checkpoint, such as ``css_n20_k04_s003.json`` for the code type css,
20 qubits, 4 logical qubits and seed 3, after every completed epoch, and records of its
lowest-energy (``.lowest-energy.json``) and greatest-distance (``.greatest-distance.json``) codes as
soon as either improves, so they keep discoveries of an unfinished epoch.  They go to
``results/logs/``, or to the logs of ``--results-dir``.  ``--tag`` adds a suffix to the names, and
``--output`` names the chain explicitly, relative to the current directory, like ``--resume`` and
``--refine``.  A new search refuses to overwrite an existing file.  While a run is active it holds a
lock on a ``.lock`` file beside its checkpoint, so that a second process cannot run the same chain.
Redirect the progress that it prints to a ``.log`` file of the same name, as the jobs of
``write_jobs.py`` do.

Resuming and refining.  ``--resume CHECKPOINT`` restores the saved policy, codes, scores, random
generator, temperature and epoch count exactly, so an interrupted and resumed run reproduces an
uninterrupted one, and it rejects new code or policy options.  An interruption loses only the
unfinished epoch, never a record, and a replayed epoch never replaces a better record with a worse
one.  ``--stop-after-epochs N`` stops after N more complete epochs.  Exact restart needs the same
numerical runtime, operating system and architecture, and the same version of this code.  Otherwise,
or to try new settings, ``--refine RECORD`` starts a new chain from the lowest-energy code of a
chain, or the code of either record.

Scoring.  ``--method`` is ``auto`` (the default), ``brouwer_zimmermann``, ``macwilliams`` or
``brute_force``, the methods of ``CSSLogicalCounter.count_logical_ops`` and all exact, and
``--objective`` is ``homogeneous`` (X-only and Z-only operators) or ``pauli`` (all Pauli
representatives; the MacWilliams method needs ``--code-type swel`` for it).  See ``annealing.py``.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import itertools
import json
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

from qldpc.codes import CSSLogicalCounter
from qldpc.codes.distance import compile_search_setup

from .annealing import (
    METHODS,
    OBJECTIVES,
    SearchConfig,
    SearchState,
    energy,
    get_library_method,
    get_lowest_energy_code,
    run_epochs,
    score_code,
    start_search,
)
from .atomic import hold_lock
from .checkpoints import (
    RECORD_KINDS,
    get_record_path,
    load_checkpoint,
    load_code,
    save_checkpoint,
    save_record,
)
from .cli import add_results_dir_argument, make_parser
from .paths import get_logs_dir, get_run_name
from .proposals import (
    CODE_TYPES,
    DEFAULT_MAX_ATTEMPTS,
    CodeType,
    random_initial_code,
    random_initial_codes,
)

####################################################################################################
# command-line interface


def main(argv: Sequence[str] | None = None) -> None:
    """Run or resume an exact CSS/SWEL search, writing explicit per-epoch observables."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.stop_after_epochs is not None and args.stop_after_epochs < 0:
        parser.error("--stop-after-epochs must be nonnegative")
    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.resume is not None and args.tag:
        parser.error("--resume continues the files of its checkpoint, so it takes no --tag")

    # Never overwrite files owned by another chain; resuming owns its own checkpoint and records.
    # Check before a slow initialization if the name is known, and again before claiming it.
    output = args.output or args.resume or _guess_output(args)
    if output is not None and not args.dry_run:
        _check_output_is_free(parser, args, output)

    # Lock the chain before reading it, so that no other process changes it while this one waits.
    with contextlib.ExitStack() as locks:
        if output is not None and not args.dry_run:
            _lock_chain(parser, locks, output)
        if args.resume:
            with _report_invalid_input(parser):
                state = _resume_search(parser, args)
        else:
            with _report_invalid_input(parser):
                initial, config = _prepare_new_search(parser, args)
            # The pilot scores a whole epoch of proposals, so an error there keeps its traceback.
            state = start_search(initial, config, seed=_get_seed(args))
        if output is None:
            output = _get_output(args, state.config.code_type, state.current.counter)
            if not args.dry_run:
                _lock_chain(parser, locks, output)
        if not args.dry_run:
            _check_output_is_free(parser, args, output)
        _run_search(
            state,
            output,
            dry_run=args.dry_run,
            compile_setup=not (args.dry_run or args.skip_compile),
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
        help="weight of the previous temperature in the next one"
        f" (default: {defaults.temperature_memory})",
    )
    policy.add_argument(
        "--physical-error-rate",
        type=float,
        help=f"the physical error rate p of the energy (default: {defaults.physical_error_rate})",
    )
    policy.add_argument("--objective", choices=OBJECTIVES, help=f"(default: {defaults.objective})")
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
        "--min-distance",
        type=int,
        help="draw random codes until one has this distance (default: 1)",
    )
    start.add_argument(
        "--max-attempts",
        type=int,
        help=f"random codes to try for --min-distance (default: {DEFAULT_MAX_ATTEMPTS})",
    )
    start.add_argument(
        "--walk-steps",
        type=int,
        help="SWEL gates of the random walk of a self-dual code (default: 30 per qubit)",
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
        help="do not compile the set-up code of the counts, which only pays off in a long search",
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


def _guess_output(args: argparse.Namespace) -> Path | None:
    """The checkpoint of a new search, if the command line gives its code size.

    A refined search takes its size from its code, so its checkpoint is named once it has one.
    """
    if args.num_qubits is None or args.num_logicals is None or args.refine is not None:
        return None
    return _name_output(args, args.code_type or "css", args.num_qubits, args.num_logicals)


def _get_output(args: argparse.Namespace, code_type: CodeType, counter: CSSLogicalCounter) -> Path:
    return _name_output(args, code_type, counter.num_qubits, counter.dimension)


def _name_output(
    args: argparse.Namespace, code_type: CodeType, num_qubits: int, num_logicals: int
) -> Path:
    """The checkpoint of a run in the logs, named by its code type, size and seed."""
    name = get_run_name(code_type, num_qubits, num_logicals, _get_seed(args), args.tag)
    return get_logs_dir(args.results_dir) / f"{name}.json"


def _check_output_is_free(
    parser: argparse.ArgumentParser, args: argparse.Namespace, output: Path
) -> None:
    existing = output.exists() or any(
        get_record_path(output, kind).exists() for kind in RECORD_KINDS
    )
    own_resume = args.resume is not None and output.resolve() == args.resume.resolve()
    if existing and not own_resume:
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
    """Report a rejected option, size or file as a usage error instead of a traceback."""
    try:
        yield
    except (OSError, ValueError, RuntimeError) as error:
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
        "min_distance",
        "max_attempts",
        "walk_steps",
    )
    if args.self_dual_start or any(
        getattr(args, name, None) is not None for name in new_search_options
    ):
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
        if getattr(args, field.name, None) is not None
    }
    if args.code_type == "swel" and args.gate_size is None:
        options["gate_size"] = 4
    config = SearchConfig(**options)
    if args.refine is not None:
        initial = _load_refinement_code(parser, args, config)
    else:
        if args.num_qubits is None or args.num_logicals is None:
            parser.error("New searches require num_qubits and num_logicals")
        initial = _get_random_code(parser, args, config)
    config.validate_code(initial)
    return initial, config


def _get_random_code(
    parser: argparse.ArgumentParser, args: argparse.Namespace, config: SearchConfig
) -> CSSLogicalCounter:
    """A random initial code: the first of the minimum distance, or the lowest-energy of several."""
    options = {
        "seed": _get_seed(args),
        "self_dual": config.code_type == "swel" or args.self_dual_start,
        "walk_steps": args.walk_steps,
    }
    filters = {
        name: value
        for name, value in (
            ("min_distance", args.min_distance),
            ("max_attempts", args.max_attempts),
        )
        if value is not None
    }
    if args.init_samples is None:
        return random_initial_code(
            args.num_qubits,
            args.num_logicals,
            method=get_library_method(config),
            **filters,
            **options,
        )
    if args.init_samples < 1:
        parser.error("--init-samples must be positive")
    if filters:
        parser.error(
            "--init-samples does not filter codes: do not use --min-distance or --max-attempts"
        )
    counters = random_initial_codes(args.num_qubits, args.num_logicals, **options)
    initial = get_lowest_energy_code(itertools.islice(counters, args.init_samples), config)
    score = score_code(initial, config)
    print(
        f"Started from the lowest-energy code of {args.init_samples} random codes:"
        f" distance {score.distance}, energy {energy(score, config):.4f}",
        file=sys.stderr,
    )
    return initial


def _load_refinement_code(
    parser: argparse.ArgumentParser, args: argparse.Namespace, config: SearchConfig
) -> CSSLogicalCounter:
    """Import the saved code to refine, checking it against any explicit size or distance."""
    if (
        args.self_dual_start
        or args.walk_steps is not None
        or args.max_attempts is not None
        or args.init_samples is not None
    ):
        parser.error("Refinement does not use random initialization options")
    initial = load_code(args.refine)
    if (args.num_qubits is not None and args.num_qubits != initial.num_qubits) or (
        args.num_logicals is not None and args.num_logicals != initial.dimension
    ):
        parser.error("Requested n/k differ from the refinement code")
    if args.min_distance is not None:
        minimum = initial.count_logical_ops(method=get_library_method(config)).weight
        if not 1 <= args.min_distance <= minimum:
            parser.error("Refinement code does not satisfy --min-distance")
    return initial


def _run_search(
    state: SearchState,
    output: Path,
    *,
    dry_run: bool,
    compile_setup: bool,
    stop_after_epochs: int | None,
) -> None:
    """Run the remaining epochs, saving the chain and its two records as they improve.

    A search that writes files is long enough to repay compiling the set-up code of its counts,
    which ``compile_setup`` does before the first epoch.
    """

    def save_and_report_epoch(state: SearchState) -> None:
        if not dry_run:
            save_checkpoint(state, output)
        print(
            json.dumps(
                {
                    "epoch": state.completed_epochs,
                    "objective": state.config.objective,
                    "current_score": dataclasses.asdict(state.current.score),
                    "lowest_energy_score": dataclasses.asdict(state.lowest_energy.score),
                    "greatest_distance_score": dataclasses.asdict(state.greatest_distance.score),
                    **dataclasses.asdict(state.history[-1]),
                }
            ),
            flush=True,
        )
        # The choice depends on timing, so it goes to stderr, which keeps the results deterministic.
        if state.config.method == "auto":
            print(
                f"epoch {state.completed_epochs}: scored with {state.epoch_method}", file=sys.stderr
            )

    def preserve_lowest_energy(state: SearchState) -> None:
        save_record(state, output, "lowest-energy")

    def preserve_greatest_distance(state: SearchState) -> None:
        save_record(state, output, "greatest-distance")

    # Save the initial chain and its pilot optima, so that even an interrupted first epoch is kept.
    if not dry_run:
        if not output.exists():
            save_checkpoint(state, output)
        preserve_lowest_energy(state)
        preserve_greatest_distance(state)
    if compile_setup:
        compile_search_setup()
    run_epochs(
        state,
        stop_after_epochs=stop_after_epochs,
        on_epoch=save_and_report_epoch,
        on_lowest_energy=None if dry_run else preserve_lowest_energy,
        on_greatest_distance=None if dry_run else preserve_greatest_distance,
    )


if __name__ == "__main__":
    main()
