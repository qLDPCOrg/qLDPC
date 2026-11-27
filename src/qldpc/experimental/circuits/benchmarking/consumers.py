# SPDX-License-Identifier: Apache-2.0

"""Consumers of prepared code blocks, and calibrations of the figure of merit for them.

A ``Consumer`` simulates a noisy circuit that uses a code block, and ``Consumer.calibrate``
estimates a ``ConsumerCalibration`` for it.  A ``ConsumerCalibration`` combines X-type and Z-type
excess-failure tables with residual-weight histograms.  The tables are either estimated or supplied
by the caller.  ``get_steane_swap_consumer`` builds a consumer from a short run of steps of
swap-type Steane quantum error correction (QEC).  With it, two calls score a noisy state-preparation
circuit::

    calibration = get_steane_swap_consumer(code, noise_model).calibrate()
    phi, standard_error = calibration.estimate_excess_failure_probability(circuit)

See the :guide:`consumer-calibration guide
<experimental/circuits/benchmarking/consumer_calibration.md>` for the model, the scoring formula,
its approximations, and validation guidance.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
import scipy.linalg
import scipy.sparse
import stim

from qldpc import circuits, codes, decoders
from qldpc import math as qldpc_math
from qldpc._binary import pack_bits, row_reduce_binary
from qldpc.circuits.certification.fault_propagation import _get_unmeasured_qubits
from qldpc.decoders._resolution import PLACEHOLDER_ERROR_RATE, reject_prebuilt_decoder
from qldpc.objects import Pauli, PauliXZ, PauliXZLike

from .residual_weights import (
    _CHUNK_SIZE,
    ResidualWeightHistogram,
    _check_integer,
    _get_combinations,
    _get_fault_members,
    _get_fault_weight_probabilities,
    _get_proposal_probabilities,
    _get_selection,
    _ResidualWeightSetup,
    _sample_bernoulli_faults,
    _validate_indices,
    estimate_residual_weight_histogram,
)

# the floating-point tolerance of checks that a value is a probability, or equal to another value
_TOLERANCE = 1e-12

# By default, a calibration stops at the smallest consumer fault weight beyond which the remaining
# probability is at most this fraction of the probability of a reference fault weight (see
# Consumer.calibrate).
_FAULT_WEIGHT_TOLERANCE = 1e-3

# The window of a Steane-swap consumer starts this many noisy steps before the step by which the
# block under test enters, and ends this many steps after it (see
# ``consumer_calibration.md#sec-window``).
_NUM_STEPS_BEFORE_ENTRY = 2
_NUM_STEPS_AFTER_ENTRY = 2
# The default decoder of a Steane-swap consumer refuses a lookup table, or a list of the errors of
# one weight, with more than this many entries.
_MAX_TABLE_ENTRIES = 2**24
# The name of the default decoder of a Steane-swap consumer, in its description and its error
# messages.
_DEFAULT_DECODER_NAME = "the default lookup-table decoder"

####################################################################################################
# consumers


@dataclasses.dataclass(frozen=True, eq=False)
class Consumer:
    """A noisy circuit that uses a code block, with a decoder that predicts its observables.

    A shot of the consumer runs the noiseless ``block_preparation``, which prepares the block in its
    ideal stabilizer state, and then the noisy ``circuit``.  The decoder predicts the flips of every
    observable of the shot from its detectors.  The consumer fails if it mispredicts any of them.
    The observables need not be logical operators.  As Stim requires, each is deterministic without
    noise.  It is a parity of measurement outcomes, and Stim's ``OBSERVABLE_INCLUDE`` also allows it
    to include Pauli operators on qubits, as the observables of ``get_steane_swap_consumer`` do.
    The consumer does not post-select.  Every operation must be a Clifford operation or Pauli noise,
    so that a Pauli error on the block flips a fixed set of detectors and observables.  See
    :guide:`Consumers and profiles
    <experimental/circuits/benchmarking/consumer_calibration.md#sec-consumers>` for what a consumer
    models.

    Only the first three attributes can be passed by position.

    Attributes:
        block_preparation: A noiseless Stim circuit that prepares the block in its ideal state,
            starting from Stim's all-zero state.
        circuit: The noisy Stim circuit that uses the block after ``block_preparation``.
        decoder: An observable decoder or a decoder specification, such as
            ``decoders.mwpm(decompose_errors=True)``.  An observable decoder decodes the detectors
            of ``block_preparation + circuit``, in batches if it is a
            ``decoders.BatchObservableDecoder``.  A decoder specification builds one from the
            detector error model of these circuits.  This model keeps the detector coordinates of
            the circuits.  It also has the decompositions of errors into graphlike components that
            Stim suggests, if Stim can find them, which matching decoders can use.  The decoder
            must predict one flip per observable, so decoders that signal erasure fail when they
            are used.
        block_qubits: The qubits of the block, in order.  Default: the qubits that
            ``block_preparation`` acts on and never measures.
        correctable_weight: Optionally, the number ``t`` of errors that the consumer is designed to
            correct.  It sets the default largest residual weight ``t + 1`` of ``calibrate``.
            Default: None.
        description: A short description of the consumer, which its calibrations keep.  Default:
            an empty string.

    Raises:
        TypeError: If ``decoder`` is neither an observable decoder nor a decoder specification,
            ``block_qubits`` is not a sequence of integers, or ``correctable_weight`` is not an
            integer or None.
        ValueError: If ``block_preparation`` has noise or the consumer has no observables.  Also if
            ``block_qubits`` is empty or has repeated or out-of-range entries, or
            ``correctable_weight`` is negative.
    """

    block_preparation: stim.Circuit
    circuit: stim.Circuit
    decoder: decoders.ObservableDecoder | decoders.DecoderSpec[Any]
    block_qubits: Sequence[int] | None = dataclasses.field(default=None, kw_only=True)
    correctable_weight: int | None = dataclasses.field(default=None, kw_only=True)
    description: str = dataclasses.field(default="", kw_only=True)

    def __post_init__(self) -> None:
        """Check the consumer, and store its block qubits as a tuple."""
        # a decoder class passes the protocol check, since it has the decoding methods
        if isinstance(self.decoder, type) or not isinstance(
            self.decoder, (decoders.DecoderSpec, decoders.ObservableDecoder)
        ):
            raise TypeError(
                "The decoder of a consumer must be an observable decoder or a decoder"
                f" specification, not {self.decoder!r}"
            )
        if self.correctable_weight is not None:
            correctable_weight = _check_integer(self.correctable_weight, "correctable_weight", 0)
            object.__setattr__(self, "correctable_weight", correctable_weight)
        if self.block_preparation != self.block_preparation.without_noise():
            raise ValueError("The block preparation of a consumer must be noiseless")
        qubits = _validate_indices(
            _get_unmeasured_qubits(self.block_preparation)
            if self.block_qubits is None
            else self.block_qubits,
            self.block_preparation.num_qubits,
            "block_qubits",
        )
        object.__setattr__(self, "block_qubits", qubits)
        if not (self.block_preparation + self.circuit).num_observables:
            raise ValueError("A consumer needs at least one observable")

    def calibrate(
        self,
        max_residual_weight: int | None = None,
        num_samples: int = 10_000,
        *,
        max_fault_weight: int | None = None,
        paulis: Iterable[PauliXZLike] = (Pauli.X, Pauli.Z),
        seed: int | None = None,
    ) -> ConsumerCalibration:
        """Estimate the excesses ``Δ_s(m, g) = f_s(m, g) - f_s(0, g)`` of the consumer.

        ``f_s(m, g)`` is the probability that the consumer fails when its block carries a type-``s``
        error of residual weight ``m`` and exactly ``g`` error mechanisms of its detector error
        model occur.  Each excess is the mean paired change in failure, with and without a random
        block error, over the samples whose block errors have residual weight ``m``.  The baseline
        ``f(0, g)`` is the fraction of all samples of fault weight ``g`` that fail without their
        block errors.  The point estimates are sample means.  The standard errors are posterior
        standard deviations under a Jeffreys prior, so they are positive for every sampled cell
        and baseline.  A cell ``(m, g)`` that no sample reaches counts as certain failure, with the
        excess ``1 - f(0, g)``, a standard error of zero, and a sample count of zero.  See
        :guide:`How it is sampled
        <experimental/circuits/benchmarking/consumer_calibration.md#sec-sampling>` for the sampled
        errors, :guide:`Standard errors of a simulated calibration
        <experimental/circuits/benchmarking/consumer_calibration.md#sec-standard-errors>` for the
        posteriors, and :guide:`Approximations and uncertainty
        <experimental/circuits/benchmarking/consumer_calibration.md#sec-approximations>` for the
        approximations of the calibration.

        Args:
            max_residual_weight: The largest residual weight to calibrate.  Heavier residual weights
                count as certain failure.  Default: ``correctable_weight + 1``.
            num_samples: The number of samples for each Pauli type, fault weight, and physical
                error weight.  Default: 10_000.

        Keyword args:
            max_fault_weight: The largest fault weight ``G`` of the consumer to calibrate.  Heavier
                fault weights count as certain failure.  Default: let ``N`` be the number of error
                mechanisms of the consumer and ``b = min(max_residual_weight, N)``.  Choose the
                smallest ``G >= b`` for which ``Pr[g > G]`` is at most a thousandth of
                ``Pr[g = b]``, searching no further than ``min(max_residual_weight + 40, N)``.
            paulis: The Pauli types to calibrate, ``Pauli.X`` and ``Pauli.Z`` (or ``"X"`` and
                ``"Z"``).  Default: both.
            seed: A seed for the random number generator.  Default: None, for fresh randomness.

        Returns:
            A ``ConsumerCalibration`` with standard errors and sample counts.  It also holds the
            probabilities of the error mechanisms of the consumer, the signed stabilizers of the
            ideal state of its block, and the description of the consumer.  Its cells are all
            estimated, in the sense of ``ConsumerCalibration``.

        Raises:
            TypeError: If an integer argument is not an integer.
            ValueError: If ``max_residual_weight`` is None and the consumer has no
                ``correctable_weight``, an integer argument is out of range, or ``paulis`` is empty
                or has repeated or invalid Pauli types.  Also if the block preparation does not
                leave the block in a definite CSS stabilizer state, an error mechanism has
                probability one, or the decoder predicts observable flips of the wrong shape.
        """
        if max_residual_weight is None:
            if self.correctable_weight is None:
                raise ValueError(
                    "Pass max_residual_weight, or give the consumer a correctable_weight"
                )
            max_residual_weight = self.correctable_weight + 1
        max_residual_weight = _check_integer(max_residual_weight, "max_residual_weight", 0)
        num_samples = _check_integer(num_samples, "num_samples", 1)
        pauli_types = [Pauli.coerce_xz(pauli) for pauli in paulis]
        if not pauli_types or len(set(pauli_types)) != len(pauli_types):
            raise ValueError(f"paulis must have distinct Pauli types, at least one, not {paulis}")
        rng = np.random.default_rng(seed)
        simulation = _ConsumerSimulation(self, rng)
        probabilities = simulation.probabilities
        if np.any(probabilities >= 1):
            raise ValueError("The consumer has an error mechanism with probability one")
        if max_fault_weight is not None:
            max_fault_weight = min(
                _check_integer(max_fault_weight, "max_fault_weight", 0), len(probabilities)
            )
        distribution, _ = _get_fault_weight_probabilities(
            probabilities, max_fault_weight, max_residual_weight, _FAULT_WEIGHT_TOLERANCE
        )
        statistics = {
            pauli: _sample_failure_changes(
                simulation, pauli, max_residual_weight, len(distribution) - 1, num_samples, rng
            )
            for pauli in pauli_types
        }
        # pool the baseline samples across all requested Pauli types
        failures = np.sum([stats[3] for stats in statistics.values()], axis=0)
        shots = np.sum([stats[4] for stats in statistics.values()], axis=0)
        baseline = failures / shots
        excesses: dict[PauliXZLike, npt.NDArray[np.float64]] = {}
        errors: dict[PauliXZLike, npt.NDArray[np.float64]] = {}
        for pauli, (increases, decreases, counts, _, _) in statistics.items():
            reached = counts > 0
            means = np.divide(
                increases - decreases, counts, out=np.zeros(counts.shape), where=reached
            )
            # cells that no sample reaches count as certain failure
            excesses[pauli] = np.where(reached, means, 1 - baseline)
            deviations = _get_jeffreys_deviations(increases, decreases, counts)
            errors[pauli] = np.where(reached, deviations, 0)
        return ConsumerCalibration(
            excesses,
            baseline,
            probabilities,
            excess_standard_errors=errors,
            baseline_standard_errors=_get_jeffreys_deviations(failures, None, shots),
            excess_num_samples={pauli: stats[2] for pauli, stats in statistics.items()},
            state_stabilizers=simulation.stabilizer_strings,
            description=self.description,
        )

    def estimate_excess_failure_probability(
        self,
        circuit: stim.Circuit,
        num_samples: int = 100_000,
        *,
        pauli: PauliXZLike | None = None,
        output_qubits: Sequence[int] | None = None,
        add_flags: bool = False,
        post_select: bool | Collection[int] | None = None,
        seed: int | None = None,
    ) -> tuple[float, float]:
        """Directly estimate the change in failure that a preparation circuit causes.

        Each accepted shot of the preparation circuit leaves a Pauli error on its output qubits.
        The block qubits take the place of the output qubits, in order.  Each error is paired with
        one independently noisy shot of the consumer, which is decoded with and without the error.
        The estimate is the mean paired change in failure per accepted shot.  It validates a
        calibration at the cost of a simulation for every circuit (see :guide:`How to validate
        <experimental/circuits/benchmarking/consumer_calibration.md#sec-validation>`).

        The point estimate is the sample mean.  The standard error is the posterior standard
        deviation of the excess per accepted shot under a Jeffreys prior, which rejected shots do
        not affect (see :guide:`Standard errors of a simulated calibration
        <experimental/circuits/benchmarking/consumer_calibration.md#sec-standard-errors>`).

        Args:
            circuit: A noisy Stim Clifford circuit starting in Stim's all-zero state, whose
                noiseless version prepares the ideal state of the block on its output qubits.
            num_samples: The number of shots of the preparation circuit to simulate, including those
                that post-selection discards.  Default: 100_000.

        Keyword args:
            pauli: ``Pauli.X`` or ``Pauli.Z`` (or ``"X"`` or ``"Z"``) to keep only that part of each
                error, or None (the default) to keep the whole error.
            output_qubits: The output qubits of the circuit, in the order of the qubits of the
                block, as for ``estimate_residual_weight_histogram``.
            add_flags: Whether to add flag detectors, as for ``estimate_residual_weight_histogram``.
                Default: False.
            post_select: The detectors to post-select on, as for
                ``estimate_residual_weight_histogram``.
            seed: A seed for the random number generators.  Default: None, for fresh randomness.

        Returns:
            The estimated excess failure probability per accepted shot, and its standard error.

        Raises:
            TypeError: If an argument has an invalid type.
            ValueError: If an argument is invalid, or the noiseless circuit does not prepare the
                ideal state of the block on its output qubits.  Also if post-selection discards
                every shot, or the decoder predicts observable flips of the wrong shape.
        """
        num_samples = _check_integer(num_samples, "num_samples", 1)
        pauli = None if pauli is None else Pauli.coerce_xz(pauli)
        full_circuit, qubits, selected = _get_selection(
            circuit, output_qubits, add_flags, post_select
        )
        rng = np.random.default_rng(seed)
        simulation = _ConsumerSimulation(self, rng)
        _check_same_state(full_circuit.without_noise(), qubits, simulation.stabilizer_strings)

        # the paired changes in failure are -1, 0, or 1, so their counts determine the estimate
        num_accepted = num_increases = num_decreases = 0
        for num_shots in _get_chunks(num_samples):
            simulator = stim.FlipSimulator(
                batch_size=num_shots,
                num_qubits=full_circuit.num_qubits,
                disable_stabilizer_randomization=True,
                seed=_get_seed(rng),
            )
            simulator.do(full_circuit)
            xs, zs, _, detector_flips, _ = simulator.to_numpy(
                output_xs=True, output_zs=True, output_detector_flips=True, transpose=True
            )
            accepted = ~np.any(detector_flips[:, list(selected)], axis=1)
            errors = np.concatenate(
                [
                    xs[accepted][:, qubits] & (pauli is not Pauli.Z),
                    zs[accepted][:, qubits] & (pauli is not Pauli.X),
                ],
                axis=1,
            ).astype(np.uint8)
            detectors, observables = simulation.sampler.sample(
                len(errors), separate_observables=True
            )
            detectors, observables = detectors.astype(np.uint8), observables.astype(np.uint8)
            baseline = simulation.get_failures(detectors, observables).astype(int)
            changes = simulation.get_failures(detectors, observables, errors) - baseline
            num_accepted += len(changes)
            num_increases += int(np.count_nonzero(changes > 0))
            num_decreases += int(np.count_nonzero(changes < 0))
        if not num_accepted:
            raise ValueError("Post-selection discarded every shot")
        deviation = _get_jeffreys_deviations(num_increases, num_decreases, num_accepted)
        return (num_increases - num_decreases) / num_accepted, float(deviation)


def _get_jeffreys_deviations(
    increases: npt.ArrayLike, decreases: npt.ArrayLike | None, counts: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Posterior standard deviations of expected changes, under a Jeffreys prior.

    Each of ``n`` samples changes a quantity by ``1``, ``0``, or ``-1``, with counts ``n₊``,
    ``n₀``, and ``n₋``.  If ``decreases`` is None, the changes are only ``1`` or ``0``, so that the
    expected change is a probability.  The Jeffreys prior gives the probabilities of the possible
    changes a Dirichlet posterior, with parameters ``a₊ = n₊ + 1/2``, ``a₀ = n₀ + 1/2``, and, if
    there are decreases, ``a₋ = n₋ + 1/2``.  Let ``A`` be the sum of these parameters, and let
    ``a₋ = 0`` without decreases.  The expected change ``p₊ - p₋`` then has the posterior variance
    ``((a₊ + a₋) A - (a₊ - a₋)²) / (A² (A + 1))``, with or without decreases, as
    ``consumer_calibration.md#sec-standard-errors`` shows.

    Returns:
        The standard deviations, in the shape of the counts.
    """
    plus = np.asarray(increases, dtype=float) + 0.5
    minus = np.zeros(plus.shape) if decreases is None else np.asarray(decreases, dtype=float) + 0.5
    total = np.asarray(counts, dtype=float) + (1 if decreases is None else 1.5)
    return np.sqrt(((plus + minus) * total - (plus - minus) ** 2) / (total**2 * (total + 1)))


####################################################################################################
# calibrations


@dataclasses.dataclass(frozen=True, eq=False)
class ConsumerCalibration:
    """Calibration tables for the excess failure probabilities of a consumer.

    Let ``f_s(m, g)`` be the probability that the consumer fails when its block carries a type-``s``
    error of residual weight ``m`` and exactly ``g`` of its own independent error mechanisms occur.
    For each calibrated Pauli type ``s``, a calibration stores the excesses
    ``Δ_s(m, g) = f_s(m, g) - f(0, g)``.  It also stores the baseline ``f(0, g)``, which all Pauli
    types share, and the probabilities of the error mechanisms, which determine the distribution of
    ``g``.

    A cell ``(m, g)`` of a table is exact if its excess and the baseline ``f(0, g)`` both have zero
    standard error.  The failure probability ``f(0, g) + Δ_s(m, g)`` of an exact cell must lie in
    [0, 1].  See :guide:`What is estimated
    <experimental/circuits/benchmarking/consumer_calibration.md#sec-estimand>` and :guide:`The
    figure of merit
    <experimental/circuits/benchmarking/consumer_calibration.md#sec-figure-of-merit>` for the
    scoring model, :guide:`Approximations and uncertainty
    <experimental/circuits/benchmarking/consumer_calibration.md#sec-approximations>` for its
    approximations, :guide:`Supplying a consumer's response
    <experimental/circuits/benchmarking/consumer_calibration.md#sec-supplied-response>` for costs
    measured at a fixed noise level, and :guide:`How to validate
    <experimental/circuits/benchmarking/consumer_calibration.md#sec-validation>` for validation
    guidance.

    ``Consumer.calibrate`` estimates a calibration by simulation.  Only the first three attributes
    can be passed by position.

    Attributes:
        excess_failure_probabilities: The tables of ``Δ_s(m, g)`` in [-1, 1], keyed by ``Pauli.X``,
            ``Pauli.Z``, ``"X"``, or ``"Z"``.  Each row is a residual weight ``m = 1, 2, ...``, and
            each column is a fault weight ``g = 0, ..., G``.  A table may have no rows.  Tables
            are stored as arrays, X first.
        baseline_failure_probabilities: The failure probabilities ``f(0, g)`` in [0, 1], one for
            each fault weight ``g = 0, ..., G``.
        mechanism_probabilities: The probability of each error mechanism of the consumer, in
            [0, 1].  Default: none, so that ``Pr[g = 0] = 1``.
        excess_standard_errors: Optional standard errors of the excesses, in the layout of
            ``excess_failure_probabilities``.  Default: None, for exact excesses.
        baseline_standard_errors: Optional standard errors of the baseline, one per fault weight.
            Default: None, for an exact baseline.
        excess_num_samples: Optional sample counts of the excesses, in the layout of
            ``excess_failure_probabilities``.  A count of zero marks a cell that no sample reached.
            Such a cell counts as certain failure, so its excess must be ``1 - f(0, g)``.  Its
            standard error, if given, must be zero, because its uncertainty is that of ``f(0, g)``.
            Default: None.
        state_stabilizers: Optionally, signed stabilizer generators of the ideal state of the
            block, as ``stim.PauliString`` objects or strings such as ``"+XX_"``, one per qubit of
            the block, in order.  Preparation circuits are checked against them before they are
            scored.  Default: None, for no check.
        description: A short description of the consumer.  Default: an empty string.

    Raises:
        ValueError: If there are no excess tables, a key is repeated or not a Pauli type, or the
            baseline, tables, standard errors, or sample counts have inconsistent shapes or invalid
            values.  Also if an exact cell has a failure probability outside [0, 1], a cell with a
            sample count of zero has an excess other than ``1 - f(0, g)`` or a nonzero standard
            error, a mechanism probability is not in [0, 1], or the state stabilizers are not
            independent commuting generators of a state.
    """

    excess_failure_probabilities: Mapping[PauliXZLike, npt.ArrayLike]
    baseline_failure_probabilities: npt.ArrayLike
    mechanism_probabilities: npt.ArrayLike = ()
    excess_standard_errors: Mapping[PauliXZLike, npt.ArrayLike] | None = dataclasses.field(
        default=None, kw_only=True
    )
    baseline_standard_errors: npt.ArrayLike | None = dataclasses.field(default=None, kw_only=True)
    excess_num_samples: Mapping[PauliXZLike, npt.ArrayLike] | None = dataclasses.field(
        default=None, kw_only=True
    )
    state_stabilizers: Sequence[str | stim.PauliString] | None = dataclasses.field(
        default=None, kw_only=True
    )
    description: str = dataclasses.field(default="", kw_only=True)

    def __post_init__(self) -> None:
        """Store the tables as arrays keyed by Pauli type, X first, and check them."""
        baseline = np.array(self.baseline_failure_probabilities, dtype=float).reshape(-1)
        if not len(baseline) or not np.all((baseline >= 0) & (baseline <= 1)):
            raise ValueError(
                "baseline_failure_probabilities must have one probability in [0, 1] per fault"
                " weight, at least one"
            )
        object.__setattr__(self, "baseline_failure_probabilities", baseline)
        num_columns = len(baseline)
        tables = _get_tables(
            self.excess_failure_probabilities, "excess_failure_probabilities", num_columns
        )
        if not all(np.all(np.abs(table) <= 1) for table in tables.values()):
            raise ValueError("Excess failure probabilities must lie in [-1, 1]")
        object.__setattr__(self, "excess_failure_probabilities", tables)
        shapes = {pauli: table.shape for pauli, table in tables.items()}
        values: dict[str, dict[PauliXZ, npt.NDArray[Any]]] = {}
        for name, integer in (("excess_standard_errors", False), ("excess_num_samples", True)):
            if getattr(self, name) is not None:
                values[name] = _get_tables(getattr(self, name), name, num_columns, integer)
                if {pauli: value.shape for pauli, value in values[name].items()} != shapes or any(
                    np.any(value < 0) for value in values[name].values()
                ):
                    raise ValueError(f"{name} must be nonnegative, in the layout of the tables")
                object.__setattr__(self, name, values[name])
        for pauli, counts in values.get("excess_num_samples", {}).items():
            unsampled = counts == 0
            errors = values.get("excess_standard_errors", {}).get(pauli, np.zeros(counts.shape))
            if np.any(
                unsampled & ~np.isclose(tables[pauli], 1 - baseline, rtol=0, atol=_TOLERANCE)
            ) or np.any(errors[unsampled]):
                raise ValueError(
                    "Cells with a sample count of zero count as certain failure, so their excess"
                    " must be one minus the baseline, with a standard error of zero"
                )
        baseline_errors = np.zeros(num_columns)
        if self.baseline_standard_errors is not None:
            baseline_errors = np.array(self.baseline_standard_errors, dtype=float).reshape(-1)
            if baseline_errors.shape != baseline.shape or not np.all(
                (baseline_errors >= 0) & np.isfinite(baseline_errors)
            ):
                raise ValueError(
                    "baseline_standard_errors must have one nonnegative number per fault weight"
                )
            object.__setattr__(self, "baseline_standard_errors", baseline_errors)
        # exact cells must be probabilities
        # statistical noise may move estimated cells outside [0, 1]
        for pauli, table in tables.items():
            excess_errors = values.get("excess_standard_errors", {}).get(pauli, 0)
            exact = (baseline_errors == 0) & (excess_errors == 0)
            failures = baseline + table
            if np.any(exact & ((failures < -_TOLERANCE) | (failures > 1 + _TOLERANCE))):
                raise ValueError(
                    "Exact cells, whose excess and baseline have no standard errors, must have"
                    " failure probabilities f(0, g) + Δ_s(m, g) in [0, 1]"
                )
        probabilities = np.array(self.mechanism_probabilities, dtype=float).reshape(-1)
        if not np.all((probabilities >= 0) & (probabilities <= 1)):
            raise ValueError("Mechanism probabilities must lie in [0, 1]")
        object.__setattr__(self, "mechanism_probabilities", probabilities)
        if self.state_stabilizers is not None:
            stabilizers = tuple(stim.PauliString(op) for op in self.state_stabilizers)
            try:
                stim.Tableau.from_stabilizers(list(stabilizers))
            except ValueError as error:
                raise ValueError(
                    "state_stabilizers must be independent commuting generators of a state, one"
                    f" per qubit: {error}"
                ) from error
            object.__setattr__(self, "state_stabilizers", stabilizers)

    @property
    def fault_weight_probabilities(self) -> npt.NDArray[np.float64]:
        """The probabilities ``Pr[g]`` that exactly ``g`` error mechanisms of the consumer occur.

        There is one entry for each calibrated fault weight ``g = 0, ..., G``.  These are fault
        weights of the consumer, unlike the attribute of the same name of a
        ``ResidualWeightHistogram``, which counts the faults of a preparation circuit.
        """
        return self._get_distribution()[0]

    @property
    def truncated_probability(self) -> float:
        """The probability ``Pr[g > G]`` of the fault weights that count as certain failure."""
        return self._get_distribution()[1]

    def rescaled(self, scale: float) -> ConsumerCalibration:
        """The calibration with every error probability of the consumer multiplied by ``scale``.

        This recomputes ``Pr[g]`` and reuses the tables of excesses and the baseline, which is an
        approximation for nearby low error rates (see :guide:`Approximations and uncertainty
        <experimental/circuits/benchmarking/consumer_calibration.md#sec-approximations>`).

        Args:
            scale: A nonnegative factor for every error probability of the consumer.

        Returns:
            A calibration with scaled mechanism probabilities, and a description that says so.

        Raises:
            ValueError: If the calibration has no error mechanisms, or if a scaled error probability
                is not in [0, 1].
        """
        probabilities = np.asarray(self.mechanism_probabilities)
        if not len(probabilities):
            raise ValueError("A calibration without error mechanisms cannot be rescaled")
        description = f"{self.description} (consumer error probabilities scaled by {scale:g})"
        return dataclasses.replace(
            self, mechanism_probabilities=scale * probabilities, description=description.strip()
        )

    def get_failure_probabilities(self, pauli: PauliXZLike) -> npt.NDArray[np.float64]:
        """The failure probabilities ``c_s(m)`` of the consumer for the residual weights of a type.

        Here ``c(0) = c_s(0) = Σ_{g ≤ G} Pr[g] f(0, g) + Pr[g > G]`` is the failure probability
        without a block error, and ``c_s(m) = c(0) + Σ_{g ≤ G} Pr[g] Δ_s(m, g)`` for the rows ``m``
        of the table of type ``s``, as :guide:`What is estimated
        <experimental/circuits/benchmarking/consumer_calibration.md#sec-estimand>` defines them.  As
        for a ``ConsumerProfile``, every heavier residual weight counts as certain failure.  ``Φ``
        adds, over the Pauli types, the average of ``c_s(m) - c(0)`` over the histogram of that
        type.  The values are not clipped, so sampling noise in estimated cells, or rounding, can
        put them slightly outside [0, 1].

        Args:
            pauli: The Pauli type ``s``, as ``Pauli.X``, ``Pauli.Z``, ``"X"``, or ``"Z"``.

        Returns:
            The array ``c(0), c_s(1), ..., c_s(R)``, for ``R`` the number of rows of the table.

        Raises:
            ValueError: If ``pauli`` is not a calibrated Pauli type.
        """
        pauli = Pauli.coerce_xz(pauli)
        if pauli not in self._tables:
            raise ValueError(
                f"The calibration has no table of Pauli type {pauli}, only of"
                f" {[str(calibrated) for calibrated in self._tables]}"
            )
        _, no_block_error, excesses = self._get_excesses()
        return no_block_error + np.concatenate([[0.0], excesses[pauli]])

    def get_excess_failure_probability(
        self, *histograms: ResidualWeightHistogram
    ) -> tuple[float, float]:
        """The calibrated figure of merit ``Φ`` of a preparation circuit, from its histograms.

        For each Pauli type, this is the expectation of the excess ``c_s(m) - c(0)`` over the
        residual weight ``m`` of the histogram of that type, and ``Φ`` is their sum, as
        :guide:`The figure of merit
        <experimental/circuits/benchmarking/consumer_calibration.md#sec-figure-of-merit>` defines.
        Residual weights beyond the table of a type count as certain failure, with excess
        ``1 - c(0)``.  The histograms must describe the same preparation circuit, output qubits,
        and post-selection.  ``estimate_excess_failure_probability`` builds such histograms and
        checks the state of the circuit.

        Args:
            histograms: One residual-weight histogram of the preparation circuit for each calibrated
                Pauli type, from ``estimate_residual_weight_histogram`` with its ``pauli`` option,
                resolving the residual weights of the table of that type.  Their sampling errors
                are treated as independent, so give each histogram its own seed, or None.

        Returns:
            ``Φ`` and its standard error, propagated to first order as :guide:`Approximations and
            uncertainty
            <experimental/circuits/benchmarking/consumer_calibration.md#sec-approximations>`
            describes.  The standard error combines in quadrature the sampling errors of the
            histograms, the standard errors of the cells of the tables, and the error of the
            baseline.  Every certain failure, beyond a table or in a cell with a sample count of
            zero, depends on the baseline, so the error of each ``f(0, g)`` adds linearly across the
            residual weights and Pauli types that use it, before the quadrature sum.  The histograms
            of distinct Pauli types, the cells, and the baseline estimates for distinct ``g`` are
            treated as independent.  The calculation uses marginal standard errors only.
            ``get_standard_error_parts`` splits this standard error into its histogram and
            calibration parts.

        Raises:
            ValueError: If there is not exactly one histogram for each calibrated Pauli type, or if
                a histogram does not resolve the residual weights of its table.
        """
        value, histogram_error, terms = self._get_error_terms(histograms)
        return value, math.hypot(histogram_error, float(np.linalg.norm(terms)))

    def get_standard_error_parts(self, *histograms: ResidualWeightHistogram) -> tuple[float, float]:
        """The histogram and calibration parts of the standard error of ``Φ``.

        The histogram part propagates the sampling errors of the histograms, and the calibration
        part the standard errors of the cells and the baseline, as
        ``get_excess_failure_probability`` does.  To first order, more samples in the histograms
        reduce only the first part, and more samples in the calibration only the second.  Neither
        part includes the bias of cells with a sample count of zero, which more calibration
        samples remove (see :guide:`Approximations and uncertainty
        <experimental/circuits/benchmarking/consumer_calibration.md#sec-approximations>`).

        Args:
            histograms: The histograms of a preparation circuit, as for
                ``get_excess_failure_probability``.

        Returns:
            The histogram part and the calibration part.  The standard error that
            ``get_excess_failure_probability`` returns is their quadrature sum.

        Raises:
            ValueError: As for ``get_excess_failure_probability``.
        """
        _, histogram_error, terms = self._get_error_terms(histograms)
        return histogram_error, float(np.linalg.norm(terms))

    def get_excess_failure_probability_difference(
        self, first: Sequence[ResidualWeightHistogram], second: Sequence[ResidualWeightHistogram]
    ) -> tuple[float, float]:
        """The difference ``Φ(a) - Φ(b)`` of the figures of merit of preparation circuits a and b.

        Both circuits are scored with this calibration.  An error of a cell or a baseline moves both
        scores, so it largely cancels from their difference when the circuits have similar
        residual-weight distributions.  The standard error of the difference keeps this
        cancellation, so it can then be much smaller than the standard errors of the two scores
        added in quadrature.  It excludes two biases, each of which can move the difference either
        way.  The first is that of residual weights beyond a table and of unsampled cells, which
        count as certain failure.  The second is that of the truncation of consumer fault weights
        above ``G``.  :guide:`Approximations and uncertainty
        <experimental/circuits/benchmarking/consumer_calibration.md#sec-approximations>` explains
        this cancellation and these biases.

        Args:
            first: The histograms of circuit a, as for ``get_excess_failure_probability``.
            second: The histograms of circuit b, as for ``get_excess_failure_probability``.  Every
                histogram of either circuit needs its own seed, or None.

        Returns:
            ``Φ(a) - Φ(b)`` and its standard error, propagated to first order.  The standard error
            combines in quadrature the histogram parts of both circuits, which are treated as
            independent, and the calibration part of the difference.  That part propagates the
            standard error of each cell and baseline through the difference of the two circuits'
            derivatives with respect to it.

        Raises:
            ValueError: As for ``get_excess_failure_probability``, or if one histogram object
                describes both circuits.
        """
        if any(histogram is other for histogram in first for other in second):
            raise ValueError(
                "The two circuits need independently sampled histograms, not a shared histogram"
            )
        value, histogram_error, terms = self._get_error_terms(first)
        other_value, other_histogram_error, other_terms = self._get_error_terms(second)
        calibration_error = float(np.linalg.norm(terms - other_terms))
        return value - other_value, math.hypot(
            histogram_error, other_histogram_error, calibration_error
        )

    def estimate_excess_failure_probability(
        self,
        circuit: stim.Circuit,
        num_samples: int = 1_000_000,
        *,
        output_qubits: Sequence[int] | None = None,
        add_flags: bool = False,
        post_select: bool | Collection[int] | None = None,
        seed: int | None = None,
    ) -> tuple[float, float]:
        """Estimate the calibrated figure of merit ``Φ`` of a preparation circuit.

        If the calibration has ``state_stabilizers``, this first checks that the noiseless circuit
        prepares that state on its output qubits.  It then estimates one residual-weight histogram
        of the circuit for each calibrated Pauli type, from independent random numbers.
        ``get_excess_failure_probability`` combines these histograms, as :guide:`The figure of merit
        <experimental/circuits/benchmarking/consumer_calibration.md#sec-figure-of-merit>`
        describes.  The probability that a shot is accepted is the ``acceptance_probability`` of a
        histogram from ``estimate_residual_weight_histogram`` with the same circuit and options.

        Args:
            circuit: A noisy Stim Clifford circuit starting in Stim's all-zero state, which
                prepares the state of the block on its output qubits.
            num_samples: The number of faults to sample for each histogram, as for
                ``estimate_residual_weight_histogram``.  Default: 1_000_000.

        Keyword args:
            output_qubits: The output qubits of the circuit, in the order of the qubits of the
                block, as for ``estimate_residual_weight_histogram``.
            add_flags: Whether to add flag detectors, as for ``estimate_residual_weight_histogram``.
                Default: False.
            post_select: The detectors to post-select on, as for
                ``estimate_residual_weight_histogram``.
            seed: A seed for the random number generators.  Default: None, for fresh randomness.

        Returns:
            ``Φ`` and its standard error, as for ``get_excess_failure_probability``.

        Raises:
            TypeError: If an argument has an invalid type.
            ValueError: If the calibration has ``state_stabilizers`` and the noiseless circuit does
                not prepare that state on its output qubits, or as for
                ``estimate_residual_weight_histogram``.
        """
        if self.state_stabilizers is not None:
            _, qubits, _ = _get_selection(circuit, output_qubits, False, False)
            _check_same_state(circuit.without_noise(), qubits, list(self.state_stabilizers))
        child_seeds = np.random.SeedSequence(seed).spawn(len(self._tables))
        histograms = [
            estimate_residual_weight_histogram(
                circuit,
                len(table),
                num_samples,
                output_qubits=output_qubits,
                add_flags=add_flags,
                post_select=post_select,
                pauli=pauli,
                seed=int(child_seed.generate_state(1, dtype=np.uint64)[0]),
            )
            for (pauli, table), child_seed in zip(self._tables.items(), child_seeds)
        ]
        return self.get_excess_failure_probability(*histograms)

    def to_dict(self) -> dict[str, Any]:
        """Return the fields of the calibration as JSON-compatible values.

        Returns:
            A dictionary keyed by field name, which ``from_dict`` loads.  Tables become
            dictionaries keyed by ``"X"`` or ``"Z"``, arrays become lists, and stabilizers become
            strings.
        """
        data: dict[str, Any] = {}
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, Mapping):
                value = {str(pauli): np.asarray(table).tolist() for pauli, table in value.items()}
            elif isinstance(value, np.ndarray):
                value = value.tolist()
            elif isinstance(value, tuple):
                value = [str(op) for op in value]
            data[field.name] = value
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ConsumerCalibration:
        """Build a calibration from a mapping returned by ``to_dict``.

        Args:
            data: The serialized fields of a calibration.

        Returns:
            The validated calibration.

        Raises:
            TypeError: If required fields are missing or unexpected fields are present.
            ValueError: As for ``ConsumerCalibration``.
        """
        return cls(**data)

    @property
    def _tables(self) -> dict[PauliXZ, npt.NDArray[np.float64]]:
        """The validated tables of excesses, keyed by Pauli type."""
        return cast(dict[PauliXZ, npt.NDArray[np.float64]], self.excess_failure_probabilities)

    def _get_distribution(self) -> tuple[npt.NDArray[np.float64], float]:
        """The fault-weight distribution ``Pr[g]`` for ``g <= G``, and ``Pr[g > G]``."""
        num_columns = len(np.asarray(self.baseline_failure_probabilities))
        probabilities = np.asarray(self.mechanism_probabilities, dtype=float)
        return _get_fault_weight_probabilities(probabilities, num_columns - 1)

    def _get_excesses(
        self,
    ) -> tuple[npt.NDArray[np.float64], float, dict[PauliXZ, npt.NDArray[np.float64]]]:
        """``Pr[g]`` for ``g <= G``, ``c(0)``, and the excesses ``c_s(m) - c(0)`` of each table."""
        distribution, truncated_probability = self._get_distribution()
        no_block_error = (
            float(np.asarray(self.baseline_failure_probabilities) @ distribution)
            + truncated_probability
        )
        excesses = {pauli: table @ distribution for pauli, table in self._tables.items()}
        return distribution, no_block_error, excesses

    def _get_error_terms(
        self, histograms: Sequence[ResidualWeightHistogram]
    ) -> tuple[float, float, npt.NDArray[np.float64]]:
        """Score a preparation circuit, and linearize the errors of its score.

        The linearization is the first-order propagation of errors of
        ``consumer_calibration.md#sec-approximations``.

        Args:
            histograms: The histograms of the circuit, as for ``get_excess_failure_probability``.

        Returns:
            ``Φ``, the histogram part of its standard error, and the error terms of the calibration.
            The error term of a cell ``Δ_s(m, g)`` or a baseline ``f(0, g)`` is the derivative of
            ``Φ`` with respect to it, times its standard error.  A cell with a sample count of zero
            has no error of its own, and moves with its baseline in the derivative.  The terms of
            the cells of each table, in the order of the tables and row by row, precede those of
            the baselines.

        Raises:
            ValueError: As for ``get_excess_failure_probability``.
        """
        # check that the histograms match the tables
        tables = self._tables
        types = [histogram.pauli for histogram in histograms]
        if len(types) != len(tables) or set(types) != set(tables):
            raise ValueError(
                "Expected one residual-weight histogram for each calibrated Pauli type"
                f" {[str(pauli) for pauli in tables]}, not for {[str(t) for t in types]}"
            )

        # score each Pauli type, and collect the derivatives of the score
        distribution, no_block_error, excesses = self._get_excesses()
        value = histogram_variance = 0.0
        cell_terms = {pauli: np.zeros(table.shape) for pauli, table in tables.items()}
        # the derivative of the score with respect to each f(0, g) is -Pr[g] times this weight
        baseline_weights = np.zeros(len(distribution))
        for histogram in histograms:
            pauli = cast(PauliXZ, histogram.pauli)
            table = tables[pauli]
            num_rows = len(table)
            if num_rows > histogram.max_residual_weight:
                raise ValueError(
                    f"The calibration resolves residual weights of Pauli type {pauli} up to"
                    f" {num_rows}, but the histogram only up to {histogram.max_residual_weight}"
                )
            # outcomes: discarded, residual weights 0, ..., max_residual_weight, and overflow
            num_heavier = histogram.max_residual_weight + 1 - num_rows
            outcome_values = np.concatenate(
                [[0.0, 0.0], excesses[pauli], np.full(num_heavier, 1 - no_block_error)]
            )
            mean, error = histogram._get_conditional_mean(outcome_values)
            value += mean
            histogram_variance += error**2
            probabilities = histogram.get_residual_weight_distribution()
            row_probabilities = probabilities[1 : num_rows + 1]
            baseline_weights += probabilities[num_rows + 1 :].sum()
            if self.excess_num_samples is not None:
                unsampled = np.asarray(self.excess_num_samples[pauli]) == 0
                baseline_weights += row_probabilities @ unsampled
            if self.excess_standard_errors is not None:
                errors = np.asarray(self.excess_standard_errors[pauli])
                cell_terms[pauli] = row_probabilities[:, None] * distribution * errors

        # the error terms of the cells, then those of the baselines
        terms = [cell_terms[pauli].ravel() for pauli in tables]
        baseline_errors = np.zeros(len(distribution))
        if self.baseline_standard_errors is not None:
            baseline_errors = np.asarray(self.baseline_standard_errors)
        terms.append(-baseline_weights * distribution * baseline_errors)
        return value, math.sqrt(histogram_variance), np.concatenate(terms)


def _get_tables(
    tables: Mapping[PauliXZLike, npt.ArrayLike], name: str, num_columns: int, integer: bool = False
) -> dict[PauliXZ, npt.NDArray[Any]]:
    """Convert tables keyed by Pauli type into two-dimensional arrays keyed by PauliXZ, X first.

    Every table has ``num_columns`` columns, and an empty table has no rows.

    Raises:
        ValueError: If there are no tables, or a key is repeated or not a Pauli type.  Also if a
            table is not a finite two-dimensional array with ``num_columns`` columns, or has
            noninteger values when ``integer`` is True.
    """
    converted: dict[PauliXZ, npt.NDArray[Any]] = {}
    for key, table in tables.items():
        pauli = Pauli.coerce_xz(key)
        array: npt.NDArray[Any] = np.array(table, dtype=float)
        if not array.size:
            array = array.reshape(0, num_columns)
        if (
            pauli in converted
            or array.ndim != 2
            or array.shape[1] != num_columns
            or not np.all(np.isfinite(array))
            or (integer and not np.all(array == np.round(array)))
        ):
            raise ValueError(
                f"{name} must have one finite two-dimensional table per Pauli type, with one column"
                " per fault weight" + (" and integer values" if integer else "")
            )
        converted[pauli] = array.astype(int) if integer else array
    if not converted:
        raise ValueError(f"{name} must have at least one table")
    return {pauli: converted[pauli] for pauli in (Pauli.X, Pauli.Z) if pauli in converted}


####################################################################################################
# a window of swap-type Steane QEC


def get_steane_swap_consumer(
    code: codes.CSSCode,
    noise_model: circuits.NoiseModel,
    *,
    basis: PauliXZLike = Pauli.Z,
    decoder: decoders.DecoderSpec[Any] | None = None,
    decoding: Literal["per_step", "window"] = "per_step",
    correctable_weight: int | None = None,
) -> Consumer:
    """A consumer that runs steps of Steane error correction around the entry of a fresh code block.

    Swap-type Steane quantum error correction (QEC), as in Figure 4d of `arXiv:2603.04584
    <https://arxiv.org/abs/2603.04584>`_, teleports the logical qubits of a CSS code from block to
    block.  The current block holds the logical qubits, and each step teleports them onto a fresh
    block, which becomes the current block of the next step.  A Z-correction step applies a
    transversal CNOT from the current block to a fresh ``|0>`` block, and measures the current block
    in the X basis.  An X-correction step applies a transversal CNOT from a fresh ``|+>`` block to
    the current block, and measures the current block in the Z basis.  The checks of the type that
    the step measures define detectors.  The steps do not apply the logical corrections of
    teleportation; instead, each observable includes the logical outcomes that the steps measure.
    :guide:`The built-in consumer
    <experimental/circuits/benchmarking/consumer_calibration.md#sec-built-in-consumer>` describes
    this protocol and the window below.

    The consumer runs a window of five noisy steps of alternating kind, starting from an ideal data
    block in logical ``|0>``.  The block under test is the fresh block of the third step.  For
    ``basis="Z"``, the default, it is a ``|0>`` block that enters by a Z-correction step.  For
    ``basis="X"``, it is a ``|+>`` block that enters by an X-correction step.  Every block starts in
    its ideal state, prepared without noise by ``circuits.get_encoding_circuit``, and a calibration
    adds errors only to the block under test.  The noise model applies to the CNOTs and measurements
    of each step, including any idle noise that it specifies for the two blocks of the step.  It
    sees each step on its own, with the current block on qubits 0, ..., n - 1 and the fresh block on
    qubits n, ..., 2 n - 1, for a code on n qubits.  Finally, the checks of the last block, which
    holds the logical qubits after the last step, are measured without noise.  See :guide:`The
    window <experimental/circuits/benchmarking/consumer_calibration.md#sec-window>` for why the
    window has this form, and :guide:`What a calibration measures
    <experimental/circuits/benchmarking/consumer_calibration.md#sec-calibration-measures>` for the
    approximations of its calibrations.

    The detectors of each step come from a maximal independent subset of the code's own checks of
    the type that the step measures, in their order in the code.  The final readout uses such
    subsets of both types.  The detectors are those of the steps, step by step, followed by those of
    the X-type and then the Z-type checks of the last block.  The observables compare the logical X
    operators, and then the logical Z operators, of the data block with those of the last block.

    By default, the consumer decodes the syndrome of each step, and the final syndrome, on its own.
    This per-step decoding is the reference for scoring preparation circuits, since only it has the
    guarantees that ``decoding`` describes, and decoding the whole window together is a research
    option.  The default decoder looks up the logical flips of a lightest error with each syndrome
    in a table.  It refuses to build a table whose number of syndromes times the number of logical
    qubits exceeds ``2**24``.  It also refuses to enumerate the errors of one weight if their number
    times their weight exceeds ``2**24``.  At these limits, building a table needs up to about
    700 MiB.

    Args:
        code: A CSS stabilizer code on qubits, with at least one logical qubit.
        noise_model: The noise model of the steps.

    Keyword args:
        basis: The block under test: ``Pauli.Z`` (or ``"Z"``, the default) for a fresh ``|0>``
            block, or ``Pauli.X`` (or ``"X"``) for a fresh ``|+>`` block.
        decoder: A decoder specification, or None (the default) for the default decoder described
            above.  For ``decoding="per_step"``, the consumer builds a decoder from the
            specification for each type of check, and decodes every syndrome of that type with it.
            It builds the decoder from a detector error model of one code block, with one error
            mechanism on each qubit, all of one fixed placeholder probability rather than a rate of
            the noise model.  The model's detectors are the independent checks of the type, and its
            observables are the logical operators of the same type.  A matching specification, such
            as ``decoders.mwpm()``, therefore needs each qubit in at most two independent checks of
            each type, as in surface and toric codes.  Other specifications, such as
            ``decoders.bp_osd(max_iter=50, bp_method="minimum_sum")``, can suit a code beyond the
            limits of the default decoder, and the single-error check (see Raises) tells whether one
            corrects every single-qubit error.  For ``decoding="window"``, the consumer builds one
            decoder from the detector error model of the whole window, as for ``Consumer``.  For
            example, ``decoders.mwpm(decompose_errors=True)`` suits a code whose checks give a
            matching graph, like a surface code.
        decoding: ``"per_step"`` (the default) to decode each step, and the final syndrome, on its
            own, or ``"window"`` to decode the whole window together with a decoder built from
            ``decoder``.  Per-step decoding has two guarantees if the prediction of each per-step
            decoder depends on its syndrome alone, which fails for randomized or stateful decoders
            such as ``decoders.relay_bp()``.  First, the errors of the block under test that two
            steps decode cancel without faults of the consumer.  Second, if the per-step decoders
            also correct every single-qubit error, further noisy steps before or after the window
            change the excess failure probability of each residual weight, and so the score of a
            preparation circuit, only at second order in the consumer's error probabilities.  The
            consumer checks that the per-step decoders correct every single-qubit error when the
            correctable weight is at least 1 (see Raises), and no decoder does for a code of
            distance less than 3.  Whole-window decoding has neither guarantee, and its results can
            depend on the window's length.  See :guide:`The window
            <experimental/circuits/benchmarking/consumer_calibration.md#sec-window>` and
            :guide:`Decoders
            <experimental/circuits/benchmarking/consumer_calibration.md#sec-decoders>`.
        correctable_weight: The largest weight ``t`` of errors that the code corrects.  It sets
            the default largest residual weight ``t + 1`` of ``calibrate``.  Default:
            ``(d - 1) // 2`` for the distance ``d`` of the code.  If the code does not know its
            distance, it is computed, in a time that grows exponentially with the size of the code.

    Returns:
        A ``Consumer`` whose block is the block under test, with the correctable weight and a short
        description that names the state of the block under test, the code, the decoder, the
        decoding mode, and the type of the noise model.

    Raises:
        TypeError: If ``decoder`` is neither None, a decoder specification, nor a prebuilt
            decoder, or ``correctable_weight`` is neither an integer nor None.
        ValueError: If the code is not a CSS stabilizer code on qubits with at least one logical
            qubit, ``basis`` is not a Pauli type, ``decoder`` is a prebuilt decoder, ``decoding`` is
            neither ``"per_step"`` nor ``"window"``, or ``correctable_weight`` is negative.  Also if
            ``decoding`` is ``"window"`` and ``decoder`` is None, or ``decoder`` is None and the
            default decoder would exceed the limits described above.  Also if ``decoding`` is
            ``"per_step"``, the correctable weight is at least 1, and the per-step decoder of a type
            of check mispredicts the logical flips of no error, or of a single-qubit error of the
            Pauli type that these checks detect, since the window's guarantees need every such error
            corrected.  During this check, also if a per-step decoder predicts observable flips of
            the wrong shape, as decoders that signal erasure do.
    """
    # check the arguments
    basis = Pauli.coerce_xz(basis)
    if (
        not isinstance(code, codes.CSSCode)
        or code.field.order != 2
        or code.is_subsystem_code
        or not code.dimension
    ):
        raise ValueError(
            "A Steane-swap consumer needs a CSS stabilizer code on qubits with at least one logical"
            " qubit"
        )
    reject_prebuilt_decoder(decoder, "get_steane_swap_consumer builds the circuit to decode")
    if decoder is not None and not isinstance(decoder, decoders.DecoderSpec):
        raise TypeError(
            "get_steane_swap_consumer needs a decoder specification, such as"
            f" decoders.mwpm(decompose_errors=True), or None, not {decoder!r}"
        )
    if decoding not in ("per_step", "window"):
        raise ValueError(f"decoding must be 'per_step' or 'window', not {decoding!r}")
    if decoding == "window" and decoder is None:
        raise ValueError(
            "decoding='window' needs a decoder specification, since the default decoder decodes"
            " each step on its own"
        )

    # build the window
    block_preparation, circuit, block_qubits, measured_bases = _get_steane_swap_window(
        code, noise_model, basis, _NUM_STEPS_BEFORE_ENTRY, _NUM_STEPS_AFTER_ENTRY
    )

    # choose the decoder, and then the correctable weight; a table that is too large then fails
    # before the distance, which can take exponential time, is computed
    decoder_name = _DEFAULT_DECODER_NAME if decoder is None else f"decoders.{decoder._helper_name}"
    consumer_decoder: decoders.ObservableDecoder | decoders.DecoderSpec[Any]
    if decoding == "window":
        assert decoder is not None  # checked above
        consumer_decoder = decoder
        decoding_name = "over the whole window"
    else:
        consumer_decoder = _SteaneSwapDecoder(code, measured_bases, decoder)
        decoding_name = "per step"
    if correctable_weight is None:
        correctable_weight = (int(code.get_distance()) - 1) // 2
    correctable_weight = _check_integer(correctable_weight, "correctable_weight", 0)

    # the guarantees of the window need a per-step decoder that corrects every single-qubit error
    # (see ``consumer_calibration.md#sec-window``)
    if isinstance(consumer_decoder, _SteaneSwapDecoder) and correctable_weight >= 1:
        for check_type in (Pauli.X, Pauli.Z):
            if consumer_decoder.corrects_single_errors(check_type):
                continue
            error_type = check_type.swap_xz()
            if not consumer_decoder.can_correct_single_errors(check_type):
                raise ValueError(
                    f"No decoder corrects every {error_type} error of weight at most 1 of the code,"
                    f" since some {error_type} error of weight 1 or 2 is a nontrivial logical"
                    " operator.  With a correctable weight of at least 1, the window's guarantees"
                    " need a per-step decoder that corrects every such error, so pass"
                    " correctable_weight=0"
                )
            raise ValueError(
                f"{consumer_decoder.name} mispredicts the logical flips of some {error_type} error"
                " of weight at most 1.  With a correctable weight of at least 1, the window's"
                " guarantees need a per-step decoder that corrects every such error.  Try another"
                " decoder specification, which this check tests in turn, such as BP-OSD with more"
                " iterations, minimum-sum BP, or a higher OSD order, as in decoders.bp_osd("
                'max_iter=50, bp_method="minimum_sum"), or pass correctable_weight=0'
            )

    # build the consumer, with a short description
    state = "|0>" if basis is Pauli.Z else "|+>"
    num_steps = len(measured_bases)
    return Consumer(
        block_preparation=block_preparation,
        circuit=circuit,
        decoder=consumer_decoder,
        block_qubits=block_qubits,
        correctable_weight=correctable_weight,
        description=(
            f"{num_steps} swap-type Steane QEC steps around the entry of a fresh {state} block, for"
            f" the [[{len(code)},{code.dimension}]] code {code.name}, with {decoder_name}"
            f" {decoding_name} and {type(noise_model).__name__}"
        ),
    )


class _SteaneSwapDecoder:
    """Observable decoder of the Steane-swap consumer, which decodes each syndrome on its own.

    The detectors and observables are in the order that ``get_steane_swap_consumer`` documents.
    A syndrome of Z-type checks flags X-type errors, which flip the logical Z operators, and a
    syndrome of X-type checks flags Z-type errors, which flip the logical X operators.  The decoder
    keeps one table or decoder for each type of check.  Without a decoder specification, it looks
    up the logical flips of a lightest error with each syndrome in a table.  With one, it decodes
    the syndromes of each type of check with a decoder built from the specification and a detector
    error model of one code block.  That model has one error mechanism on each qubit, each of
    probability ``PLACEHOLDER_ERROR_RATE``, the independent checks of the type as detectors, and
    the logical operators of the type as observables.  ``consumer_calibration.md#sec-decoders``
    describes these decoders.

    Attributes:
        name: The name of the decoder in error messages.
    """

    def __init__(
        self,
        code: codes.CSSCode,
        measured_bases: Sequence[PauliXZ],
        decoder: decoders.DecoderSpec[Any] | None = None,
    ) -> None:
        """Build the decoders of each type of check of a CSS code, for steps with given bases."""
        self.name = (
            _DEFAULT_DECODER_NAME.capitalize()
            if decoder is None
            else f"The per-step decoder that {decoder!r} builds"
        )
        self._dimension = code.dimension
        # the detectors split into one syndrome for each step, and two for the final readout
        self._syndrome_types = [*measured_bases, Pauli.X, Pauli.Z]
        self._checks = {
            check_type: _get_independent_checks(code, check_type)
            for check_type in (Pauli.X, Pauli.Z)
        }
        self._syndrome_sizes = [
            len(self._checks[check_type]) for check_type in self._syndrome_types
        ]
        # a syndrome of type-s checks flags errors that flip the logical operators of type s
        self._logicals = {
            check_type: code.get_logical_ops(check_type).view(np.ndarray).astype(int)
            for check_type in (Pauli.X, Pauli.Z)
        }
        self._tables: dict[PauliXZ, npt.NDArray[np.uint8]] = {}
        self._decoders: dict[PauliXZ, decoders.ObservableDecoder] = {}
        for check_type in (Pauli.X, Pauli.Z):
            checks, logicals = self._checks[check_type], self._logicals[check_type]
            if decoder is None:
                self._tables[check_type] = _get_min_weight_flips(checks, logicals)
            else:
                model = decoders.DetectorErrorModelArrays.from_arrays(
                    checks, logicals, PLACEHOLDER_ERROR_RATE
                )
                self._decoders[check_type] = decoder.build_observable_decoder(model.to_dem())

    def decode_observables(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode the detectors of one shot, and return the predicted observable flips."""
        return self.decode_observables_batch(np.asarray(syndrome)[None, :])[0]

    def decode_observables_batch(self, syndromes: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode the detectors of a batch of shots, and return predicted observable flips."""
        syndromes = np.asarray(syndromes, dtype=np.uint8)
        flips = {
            check_type: np.zeros((len(syndromes), self._dimension), dtype=np.uint8)
            for check_type in (Pauli.X, Pauli.Z)
        }
        # split the detectors into the syndrome of each step and the two of the final readout, and
        # add the predicted flips of each to the observables of its type
        parts = np.split(syndromes, np.cumsum(self._syndrome_sizes)[:-1], axis=1)
        for check_type, part in zip(self._syndrome_types, parts):
            flips[check_type] ^= self._decode_syndromes(check_type, part)
        return np.concatenate([flips[Pauli.X], flips[Pauli.Z]], axis=1).astype(int)

    def corrects_single_errors(self, check_type: PauliXZ) -> bool:
        """Whether the decoder of a type of check corrects every error of weight at most one.

        It corrects an error if it predicts the error's logical flips from its syndrome.  The errors
        are no error and the single-qubit errors of the Pauli type that the checks of the type
        detect.  For a decoder whose predictions depend on the syndrome alone, checking one code
        block checks every step, as ``consumer_calibration.md#sec-decoders`` shows.
        """
        checks, logicals = self._checks[check_type], self._logicals[check_type]
        # no error, then an error on each qubit
        syndromes = np.hstack([np.zeros((len(checks), 1), dtype=int), checks]).T
        expected = np.hstack([np.zeros((len(logicals), 1), dtype=int), logicals]).T
        return np.array_equal(self._decode_syndromes(check_type, syndromes), expected)

    def can_correct_single_errors(self, check_type: PauliXZ) -> bool:
        """Whether any decoder of a type of check passes ``corrects_single_errors``.

        It is False exactly when an error of weight 1 or 2 of the Pauli type that the checks detect
        has no syndrome but flips a logical operator: when a single-qubit error with no syndrome
        flips a logical operator, or two single-qubit errors have the same syndrome but different
        flips.
        """
        checks, logicals = self._checks[check_type], self._logicals[check_type]
        # the syndrome and logical flips of no error and of each single-qubit error, one per row
        signatures = np.hstack(
            [np.zeros((len(checks) + len(logicals), 1), dtype=int), np.vstack([checks, logicals])]
        ).T
        # some syndrome has two signatures exactly when two of these errors share a syndrome but
        # not their flips
        num_syndromes = len(np.unique(signatures[:, : len(checks)], axis=0))
        return len(np.unique(signatures, axis=0)) == num_syndromes

    def _decode_syndromes(
        self, check_type: PauliXZ, syndromes: npt.NDArray[np.int_]
    ) -> npt.NDArray[np.uint8]:
        """The logical flips that the decoder predicts for syndromes of the checks of one type.

        Each row of ``syndromes`` is a syndrome of the checks of type ``check_type``, and each row
        of the result holds the predicted flips of the logical operators of that type.
        """
        if check_type in self._tables:
            return self._tables[check_type][_get_syndrome_indices(syndromes)]
        predictions = _predict_observables(
            self._decoders[check_type], syndromes, self._dimension, self.name
        )
        return predictions.astype(np.uint8)


def _get_steane_swap_window(
    code: codes.CSSCode,
    noise_model: circuits.NoiseModel,
    basis: PauliXZ,
    num_before: int,
    num_after: int,
) -> tuple[stim.Circuit, stim.Circuit, list[int], list[PauliXZ]]:
    """The window of ``get_steane_swap_consumer``, with given numbers of steps around the entry.

    ``basis`` is Z for a fresh ``|0>`` block under test, and X for a fresh ``|+>`` block.  The
    window has ``num_before`` noisy steps before the step by which the block under test enters, and
    ``num_after`` noisy steps after it.  The block under test therefore enters by the step with
    index ``num_before``, counting from zero.  ``consumer_calibration.md#sec-built-in-consumer``
    describes the window.

    Returns:
        A preparation of the block under test, the rest of the window, the qubits of the block
        under test, and the basis that each step measures.
    """
    num_qubits = len(code)
    num_steps = num_before + 1 + num_after
    # the data block, then the fresh blocks in the order in which they enter
    blocks = [
        list(range(index * num_qubits, (index + 1) * num_qubits)) for index in range(num_steps + 1)
    ]
    # each step measures the basis that its fresh block is not prepared in, and the kinds alternate
    measured_bases = [
        basis if (step - num_before) % 2 else basis.swap_xz() for step in range(num_steps)
    ]

    # prepare the data block and every fresh block except the block under test, which the consumer
    # prepares separately, and include the logical operators of the data block in the observables
    circuit = _get_ideal_preparation(code, blocks[0], Pauli.Z)
    for step, measured_basis in enumerate(measured_bases):
        if step != num_before:
            circuit += _get_ideal_preparation(code, blocks[step + 1], measured_basis.swap_xz())
    circuit += circuits.get_observables(code, blocks[0])

    # run the noisy steps
    for current, fresh, measured_basis in zip(blocks, blocks[1:], measured_bases):
        circuit += _get_teleportation_step(code, noise_model, current, fresh, measured_basis)

    # measure the X-type and then the Z-type checks of the last block without noise, as rows of
    # [X|Z] form, and include its logical operators in the observables
    checks = scipy.linalg.block_diag(
        *(_get_independent_checks(code, pauli) for pauli in (Pauli.X, Pauli.Z))
    )
    circuit += circuits.get_pauli_product_measurements(checks, blocks[-1])
    for index in range(-len(checks), 0):
        circuit.append("DETECTOR", [stim.target_rec(index)])
    circuit += circuits.get_observables(code, blocks[-1])

    # prepare the block under test separately, since a calibration adds its errors
    tested_qubits = blocks[num_before + 1]
    preparation = _get_ideal_preparation(code, tested_qubits, basis)
    return preparation, circuit, tested_qubits, measured_bases


def _get_teleportation_step(
    code: codes.CSSCode,
    noise_model: circuits.NoiseModel,
    current_qubits: Sequence[int],
    fresh_qubits: Sequence[int],
    measured_basis: PauliXZ,
) -> stim.Circuit:
    """A noisy step that teleports the logical qubits of the current block to a fresh block.

    The step applies a transversal CNOT and then measures the current block in ``measured_basis``.
    If ``measured_basis`` is X, the CNOT runs from the current block to the fresh ``|0>`` block.
    Otherwise, it runs from the fresh ``|+>`` block to the current block.  The independent checks of
    type ``measured_basis`` (see ``_get_independent_checks``) define detectors.  The step adds the
    measured values of the logical operators of that type to the observables of the same logical
    operators, which tracks its logical correction in software.  The noise model sees only the two
    blocks of the step.
    """
    # build the step on the local qubits 0, ..., n - 1 (current) and n, ..., 2 n - 1 (fresh)
    num_qubits = len(code)
    local_current, local_fresh = range(num_qubits), range(num_qubits, 2 * num_qubits)
    controls, targets = (
        (local_current, local_fresh) if measured_basis is Pauli.X else (local_fresh, local_current)
    )
    step = stim.Circuit()
    step.append("CX", [qubit for pair in zip(controls, targets) for qubit in pair])
    step.append("TICK")
    step.append("MX" if measured_basis is Pauli.X else "M", local_current)
    circuit = circuits.with_remapped_qubits(
        noise_model.noisy_circuit(step), [*current_qubits, *fresh_qubits]
    )

    # the last measurements are those of the current block
    records = [stim.target_rec(qubit - num_qubits) for qubit in range(num_qubits)]
    for row in _get_independent_checks(code, measured_basis):
        circuit.append("DETECTOR", [records[qubit] for qubit in np.flatnonzero(row)])
    # the observables of the logical X operators come first, then those of the logical Z operators
    offset = 0 if measured_basis is Pauli.X else code.dimension
    circuit += circuits.get_observables(
        code,
        basis=measured_basis,
        on_measurements=records,
        observable_indices=range(offset, offset + code.dimension),
    )
    return circuit


def _get_ideal_preparation(
    code: codes.CSSCode, qubits: Sequence[int], basis: PauliXZ
) -> stim.Circuit:
    """A circuit without noise that prepares every logical qubit of a code in ``|0>`` or ``|+>``.

    The state is ``|0>`` if ``basis`` is Z, and ``|+>`` if it is X.
    """
    circuit = stim.Circuit()
    if basis is Pauli.X:
        # the encoding circuit maps qubit j to logical qubit j, for j < k
        circuit.append("H", range(code.dimension))
    circuit += circuits.get_encoding_circuit(code)
    return circuits.with_remapped_qubits(circuit, qubits)


# The default decoder uses this table rather than decoders.lookup.  decoders.lookup needs a maximum
# weight up front, predicts no correction for syndromes that this weight does not cover, and is much
# slower here.  This table enumerates errors by weight, vectorized, until it covers every syndrome,
# and decodes a batch of shots by indexing.  Its enumeration resembles that of _get_syndrome_table
# in residual_weights.py, which keeps the least weight of every syndrome up to a fixed weight
# instead.  The two share _get_combinations, and a helper for the rest would save only a few XORs.
def _get_min_weight_flips(
    checks: npt.NDArray[np.int_], logicals: npt.NDArray[np.int_]
) -> npt.NDArray[np.uint8]:
    """The logical flips of a lightest error with each syndrome of independent checks.

    Row ``s`` holds ``logicals @ e % 2`` for a lightest error ``e`` with the syndrome
    ``checks @ e % 2`` equal to ``s``, with bit ``j`` of ``s`` for check ``j``.  Errors are
    enumerated in order of increasing weight until every syndrome has one.

    Raises:
        ValueError: If the table, or the errors of a weight that it needs, would have more than
            ``_MAX_TABLE_ENTRIES`` entries.
    """
    num_checks, num_qubits = checks.shape
    num_logicals = len(logicals)
    if 2**num_checks * num_logicals > _MAX_TABLE_ENTRIES:
        raise ValueError(
            f"{_DEFAULT_DECODER_NAME.capitalize()} of a Steane-swap consumer needs a table of"
            f" 2**{num_checks} syndromes times {num_logicals} logical operators, which is too"
            " large; pass another decoder"
        )

    # the syndrome of an error on each single qubit, as a table index, and its logical flips
    qubit_syndromes = _get_syndrome_indices(checks.T)
    qubit_flips = logicals.T.astype(np.uint8)
    flips = np.zeros((2**num_checks, num_logicals), dtype=np.uint8)
    filled = np.zeros(2**num_checks, dtype=bool)
    filled[0] = True

    # independent checks give every syndrome an error of weight at most num_checks
    for weight in range(1, num_checks + 1):
        if filled.all():
            break
        if math.comb(num_qubits, weight) * weight > _MAX_TABLE_ENTRIES:
            raise ValueError(
                f"{_DEFAULT_DECODER_NAME.capitalize()} of a Steane-swap consumer needs a table of"
                f" errors of weight {weight} on {num_qubits} qubits, which is too large; pass"
                " another decoder"
            )
        # find the first error with each syndrome, and keep those with new syndromes
        errors = _get_combinations(num_qubits, weight)
        syndromes = np.zeros(len(errors), dtype=np.uint64)
        for column in errors.T:
            syndromes ^= qubit_syndromes[column]
        syndromes, first_errors = np.unique(syndromes, return_index=True)
        new = ~filled[syndromes]
        # the rows are distinct, because np.unique returned their syndromes, so the indexed XOR
        # below updates each row once
        rows = syndromes[new]
        for column in errors[first_errors[new]].T:
            flips[rows] ^= qubit_flips[column]
        filled[rows] = True
    return flips


def _get_syndrome_indices(syndromes: npt.NDArray[np.integer]) -> npt.NDArray[np.uint64]:
    """The rows of the tables of ``_get_min_weight_flips`` for syndromes given as rows of bits.

    Bit ``j`` of an index is bit ``j`` of its syndrome.  One word holds every index, because the
    limits of the tables keep the number of checks below 64.
    """
    words = pack_bits(syndromes.astype(np.uint8))
    return words[:, 0] if words.shape[1] else np.zeros(len(syndromes), dtype=np.uint64)


def _get_independent_checks(code: codes.CSSCode, pauli: PauliXZ) -> npt.NDArray[np.int_]:
    """A maximal independent subset of the checks of one Pauli type of a CSS code, in order.

    The subset has the weights and locality of the code's checks, which decoders built from the
    detector error model of a window, such as matching decoders, rely on.
    """
    matrix = code.matrix_x if pauli is Pauli.X else code.matrix_z
    checks = matrix.view(np.ndarray).astype(np.uint8)
    # the pivots of the transposed checks, in increasing order, are the checks that are independent
    # of the checks before them
    _, _, pivots = row_reduce_binary(checks.T)
    return checks[pivots].astype(int)


####################################################################################################
# simulation of consumers


class _ConsumerSimulation:
    """Samples of a consumer with an ideal block, and the effects of Pauli errors on the block.

    Attributes:
        block_qubits: The qubits of the block.
        stabilizer_strings: Stabilizer generators of the ideal block, with their signs.
        stabilizers: The same stabilizer generators in ``[X|Z]`` form, one per row.
        probabilities: The probability of each error mechanism of the consumer.
        sampler: A detector sampler of the consumer with an ideal block.
        decoder: The observable decoder of the consumer.
    """

    def __init__(self, consumer: Consumer, rng: np.random.Generator) -> None:
        """Build the sampler, the error mechanisms, the decoder, and the effects of block errors."""
        assert consumer.block_qubits is not None
        self.block_qubits = consumer.block_qubits
        num_qubits = len(self.block_qubits)
        self.stabilizer_strings = circuits.get_state_stabilizers(
            consumer.block_preparation, self.block_qubits
        )
        if len(self.stabilizer_strings) != num_qubits:
            raise ValueError(
                "The block preparation of a consumer must leave its block in a definite pure"
                " stabilizer state"
            )
        self.stabilizers = np.array(
            [qldpc_math.string_to_op(op) for op in self.stabilizer_strings], dtype=np.uint8
        )

        full_circuit = consumer.block_preparation + consumer.circuit
        self._num_detectors = full_circuit.num_detectors
        self._num_observables = full_circuit.num_observables
        self.sampler = full_circuit.compile_detector_sampler(seed=_get_seed(rng))
        # without simplification, each error instruction of the model is one error mechanism (see
        # ``consumer_calibration.md#sec-faults``)
        dem_arrays = decoders.DetectorErrorModelArrays(full_circuit, simplify=False)
        mechanisms = np.flatnonzero(dem_arrays.error_probs > 0)
        self.probabilities = np.asarray(dem_arrays.error_probs, dtype=float)[mechanisms]
        # row j holds the detector flips and then the observable flips of error mechanism j
        self._signatures = (
            scipy.sparse.vstack(
                [dem_arrays.detector_flip_matrix, dem_arrays.observable_flip_matrix]
            )
            .T.tocsr()[mechanisms]
            .astype(int)
        )
        self.decoder: decoders.ObservableDecoder = (
            consumer.decoder.build_observable_decoder(_get_decoding_model(full_circuit))
            if isinstance(consumer.decoder, decoders.DecoderSpec)
            else consumer.decoder
        )

        # shot j of 2 n shots has an X error (j < n) or a Z error (j >= n) on block qubit j % n
        simulator = stim.FlipSimulator(
            batch_size=2 * num_qubits,
            num_qubits=full_circuit.num_qubits,
            disable_stabilizer_randomization=True,
        )
        simulator.do(consumer.block_preparation)
        for index, name in enumerate("XZ"):
            mask = np.zeros((full_circuit.num_qubits, 2 * num_qubits), dtype=bool)
            mask[self.block_qubits, index * num_qubits + np.arange(num_qubits)] = True
            simulator.broadcast_pauli_errors(pauli=name, mask=mask)
        simulator.do(consumer.circuit.without_noise())
        _, _, _, detector_flips, observable_flips = simulator.to_numpy(
            output_detector_flips=True, output_observable_flips=True, transpose=True
        )
        self._responses = np.concatenate([detector_flips, observable_flips], axis=1).astype(int)

    def get_flips(
        self, members: npt.NDArray[np.intp]
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
        """The detector and observable flips of sets of error mechanisms, one set per row."""
        num_sets, fault_weight = members.shape
        indicators = scipy.sparse.csr_matrix(
            (
                np.ones(members.size, dtype=int),
                members.ravel(),
                np.arange(num_sets + 1) * fault_weight,
            ),
            shape=(num_sets, self._signatures.shape[0]),
        )
        flips = ((indicators @ self._signatures).toarray() % 2).astype(np.uint8)
        return flips[:, : self._num_detectors], flips[:, self._num_detectors :]

    def get_failures(
        self,
        detectors: npt.NDArray[np.uint8],
        observables: npt.NDArray[np.uint8],
        errors: npt.NDArray[np.uint8] | None = None,
    ) -> npt.NDArray[np.bool_]:
        """Decode shots of the consumer, and report those with a wrong prediction.

        Args:
            detectors: The detector flips of each shot with an ideal block, one shot per row.
            observables: The observable flips of each shot with an ideal block, one shot per row.
            errors: Optional Pauli errors on the block, one per shot, in ``[X|Z]`` form, whose
                effects are added to the shots.

        Returns:
            Whether the decoder predicts a wrong flip of an observable in each shot.
        """
        if errors is not None:
            flips = (errors.astype(int) @ self._responses % 2).astype(np.uint8)
            detectors = detectors ^ flips[:, : self._num_detectors]
            observables = observables ^ flips[:, self._num_detectors :]
        predictions = _predict_observables(
            self.decoder,
            detectors.astype(int),
            self._num_observables,
            "The decoder of the consumer",
        )
        return np.any(predictions != observables, axis=1)


def _sample_failure_changes(
    simulation: _ConsumerSimulation,
    pauli: PauliXZ,
    max_residual_weight: int,
    max_fault_weight: int,
    num_samples: int,
    rng: np.random.Generator,
) -> tuple[
    npt.NDArray[np.int_],
    npt.NDArray[np.int_],
    npt.NDArray[np.int_],
    npt.NDArray[np.int_],
    npt.NDArray[np.int_],
]:
    """Sample paired changes in the failure of a consumer for one Pauli type.

    See ``Consumer.calibrate`` and ``consumer_calibration.md#sec-sampling``.  Each sample is decoded
    with and without its block error, so its paired change in failure is -1, 0, or 1.

    Returns:
        Five arrays.  The first three have one row for each residual weight
        ``m = 1, ..., max_residual_weight`` and one column for each fault weight ``g``.  They count
        the samples whose block errors have residual weight ``m`` and whose paired changes are 1,
        those whose paired changes are -1, and all of these samples.  The last two have one entry
        for each fault weight.  They count the samples that fail without their block errors, and
        all samples.
    """
    num_qubits = len(simulation.block_qubits)
    setup = _ResidualWeightSetup(simulation.stabilizers, True, pauli, max_residual_weight)
    offset = 0 if pauli is Pauli.X else num_qubits
    shape = (max_residual_weight, max_fault_weight + 1)
    increases, decreases, counts = np.zeros((3, *shape), dtype=int)
    failures, shots = np.zeros((2, max_fault_weight + 1), dtype=int)
    for fault_weight in range(max_fault_weight + 1):
        for error_weight in range(1, min(max_residual_weight + 1, num_qubits) + 1):
            for num_shots in _get_chunks(num_samples):
                members = _sample_fault_sets(simulation.probabilities, fault_weight, num_shots, rng)
                detectors, observables = simulation.get_flips(members)
                supports = np.argsort(rng.random((num_shots, num_qubits)), axis=1)[:, :error_weight]
                errors = np.zeros((num_shots, 2 * num_qubits), dtype=np.uint8)
                np.put_along_axis(errors, supports + offset, 1, axis=1)
                residual_weights = setup.get_error_weights(errors)
                baseline = simulation.get_failures(detectors, observables).astype(int)
                changes = simulation.get_failures(detectors, observables, errors) - baseline
                failures[fault_weight] += baseline.sum()
                shots[fault_weight] += num_shots
                kept = (residual_weights >= 1) & (residual_weights <= max_residual_weight)
                rows = residual_weights[kept] - 1
                np.add.at(increases[:, fault_weight], rows, changes[kept] > 0)
                np.add.at(decreases[:, fault_weight], rows, changes[kept] < 0)
                np.add.at(counts[:, fault_weight], rows, 1)
    return increases, decreases, counts, failures, shots


def _sample_fault_sets(
    probabilities: npt.NDArray[np.float64],
    fault_weight: int,
    num_sets: int,
    rng: np.random.Generator,
) -> npt.NDArray[np.intp]:
    """Sample sets of error mechanisms given that exactly ``fault_weight`` of them occur.

    This function draws independent Bernoulli samples of the mechanisms, with their odds
    ``p / (1 - p)`` scaled by a common factor so that the expected fault weight is
    ``fault_weight``.  Their shots with exactly ``fault_weight`` mechanisms are exact samples, as
    ``consumer_calibration.md#sec-sampling`` shows.

    Returns:
        One row of ``fault_weight`` mechanism indices per set.
    """
    if fault_weight == 0:
        return np.zeros((num_sets, 0), dtype=np.intp)
    proposal = _get_proposal_probabilities(probabilities / (1 - probabilities), fault_weight)
    sets = []
    while num_sets > 0:
        fault_weights, grouped_mechanisms = _sample_bernoulli_faults(proposal, _CHUNK_SIZE, rng)
        members = _get_fault_members(fault_weights, grouped_mechanisms, fault_weight)[:num_sets]
        sets.append(members)
        num_sets -= len(members)
    return np.concatenate(sets)


def _get_decoding_model(circuit: stim.Circuit) -> stim.DetectorErrorModel:
    """The detector error model of a circuit for building its decoder, with its metadata.

    The model keeps the detector coordinates of the circuit, and has the decompositions of its
    errors into graphlike components that Stim suggests, such as matching decoders need.  Errors
    that Stim cannot decompose stay whole.
    """
    return circuit.detector_error_model(
        decompose_errors=True, approximate_disjoint_errors=True, ignore_decomposition_failures=True
    )


def _predict_observables(
    decoder: decoders.ObservableDecoder,
    syndromes: npt.NDArray[np.int_],
    num_observables: int,
    name: str,
) -> npt.NDArray[np.int_]:
    """The observable flips that a decoder predicts for a batch of syndromes, as 0 or 1.

    The decoder decodes the batch at once if it can, and one syndrome at a time otherwise.  A
    decoder of one observable may predict one flip per syndrome.

    Raises:
        ValueError: If the predictions do not have one flip per syndrome and observable.  ``name``
            names the decoder in the message.
    """
    if not len(syndromes):
        return np.zeros((0, num_observables), dtype=int)
    if isinstance(decoder, decoders.BatchObservableDecoder):
        predictions = np.asarray(decoder.decode_observables_batch(syndromes))
    else:
        predictions = np.array([decoder.decode_observables(row) for row in syndromes])
    expected_shape = (len(syndromes), num_observables)
    if predictions.ndim == 1 and num_observables == 1:
        predictions = predictions[:, None]
    if predictions.shape != expected_shape:
        raise ValueError(
            f"{name} must predict observable flips of shape {expected_shape} (syndromes,"
            f" observables), not {predictions.shape}"
        )
    return predictions.astype(int) % 2


def _check_same_state(
    circuit: stim.Circuit, qubits: Sequence[int], stabilizers: Sequence[stim.PauliString]
) -> None:
    """Check that a noiseless circuit leaves its qubits in the state with the given stabilizers.

    Raises:
        ValueError: If the numbers of qubits differ, or if a stabilizer, with its sign, does not
            stabilize the output of the circuit.
    """
    if len(qubits) != len(stabilizers):
        raise ValueError(
            f"The circuit has {len(qubits)} output qubits, but the block has {len(stabilizers)}"
        )
    output_stabilizers = circuits.get_state_stabilizers(circuit, qubits)
    if len(output_stabilizers) == len(qubits):
        simulator = stim.TableauSimulator()
        simulator.do_tableau(stim.Tableau.from_stabilizers(output_stabilizers), range(len(qubits)))
        if all(simulator.peek_observable_expectation(op) == 1 for op in stabilizers):
            return
    raise ValueError("The noiseless circuit must prepare the ideal state of the block")


def _get_chunks(num_shots: int) -> list[int]:
    """Split a number of shots into chunks of at most ``_CHUNK_SIZE``."""
    return [min(_CHUNK_SIZE, num_shots - start) for start in range(0, num_shots, _CHUNK_SIZE)]


def _get_seed(rng: np.random.Generator) -> int:
    """A seed for another random number generator."""
    return int(rng.integers(2**63))
