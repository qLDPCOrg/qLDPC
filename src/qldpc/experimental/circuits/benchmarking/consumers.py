# SPDX-License-Identifier: Apache-2.0

"""Consumers of prepared code blocks, and calibrations of the figure of merit for them.

A ``Consumer`` simulates a noisy circuit that uses a code block, and ``Consumer.calibrate``
estimates a ``ConsumerCalibration`` for it.  A ``ConsumerCalibration`` combines X-type and Z-type
excess-failure tables with residual-weight histograms.  The tables are either estimated or supplied
by the caller.  See the `consumer-calibration guide
<https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/experimental/circuits/benchmarking/consumer_calibration.md>`_
for the model, the scoring formula, its approximations, and validation guidance.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import scipy.sparse
import stim

from qldpc import circuits, decoders
from qldpc import math as qldpc_math
from qldpc.circuits.certification.fault_propagation import _get_unmeasured_qubits
from qldpc.objects import Pauli, PauliXZ, PauliXZLike

from .residual_weights import (
    _CHUNK_SIZE,
    ResidualWeightHistogram,
    _check_integer,
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

####################################################################################################
# consumers


@dataclasses.dataclass(frozen=True, eq=False)
class Consumer:
    """A noisy circuit that uses a code block, with a decoder that predicts its observables.

    A shot of the consumer runs the noiseless ``block_preparation``, which prepares the block in its
    ideal stabilizer state, and then the noisy ``circuit``.  The decoder predicts the flips of every
    observable of the shot from its detectors.  The consumer fails if it mispredicts any of them.
    The observables need not be logical operators.  As Stim requires, each is a parity of
    measurements whose value is deterministic without noise.  The consumer does not post-select.
    Every operation must be a Clifford operation or Pauli noise, so that a Pauli error on the block
    flips a fixed set of detectors and observables.  See ``consumer_calibration.md`` for how
    consumers are used.

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
        if not isinstance(self.decoder, (decoders.DecoderSpec, decoders.ObservableDecoder)):
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
        ``consumer_calibration.md`` for the sampled errors, the posteriors, and the approximations
        of the calibration.

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
        calibration at the cost of a simulation for every circuit (see ``consumer_calibration.md``).

        The point estimate is the sample mean.  The standard error is the posterior standard
        deviation of the excess per accepted shot under a Jeffreys prior, which rejected shots do
        not affect (see ``consumer_calibration.md``).

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
    ``a₋ = 0`` without decreases.  The Dirichlet variances ``a (A - a) / (A² (A + 1))`` and the
    covariance ``-a₊ a₋ / (A² (A + 1))`` give the expected change ``p₊ - p₋`` the variance
    ``((a₊ + a₋) A - (a₊ - a₋)²) / (A² (A + 1))``.  Without decreases, this is the variance of the
    posterior ``Beta(n₊ + 1/2, n₀ + 1/2)``.

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
    [0, 1].  See ``consumer_calibration.md`` for the scoring model, its approximations, costs
    measured at a fixed noise level, and validation guidance.

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
    def truncated_probability(self) -> float:
        """The probability ``Pr[g > G]`` of the fault weights that count as certain failure."""
        return self._get_distribution()[1]

    def rescaled(self, scale: float) -> ConsumerCalibration:
        """The calibration with every error probability of the consumer multiplied by ``scale``.

        This recomputes ``Pr[g]`` and reuses the tables of excesses and the baseline, which is an
        approximation for nearby low error rates (see ``consumer_calibration.md``).

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

    def get_excess_failure_probability(
        self, *histograms: ResidualWeightHistogram
    ) -> tuple[float, float]:
        """The calibrated figure of merit ``Φ`` of a preparation circuit, from its histograms.

        For each Pauli type, this is the expectation of the excess ``c_s(m) - c(0)`` over the
        residual weight ``m`` of the histogram of that type, and ``Φ`` is their sum.  Residual
        weights beyond the table of a type count as certain failure, with excess ``1 - c(0)``.  The
        histograms must describe the same preparation circuit, output qubits, and post-selection.
        ``estimate_excess_failure_probability`` builds such histograms and checks the state of the
        circuit.

        Args:
            histograms: One residual-weight histogram of the preparation circuit for each calibrated
                Pauli type, from ``estimate_residual_weight_histogram`` with its ``pauli`` option,
                resolving the residual weights of the table of that type.

        Returns:
            ``Φ`` and its standard error, propagated to first order.  The standard error combines
            the sampling errors of the histograms, the standard errors of the cells of the tables,
            and the error of the baseline in quadrature.  Every certain failure, beyond a table or
            in a cell with a sample count of zero, depends on the baseline.  The contributions of
            the error of each ``f(0, g)`` from all residual weights and Pauli types that use it add
            linearly before the quadrature sum.  The histograms of distinct Pauli types, the cells,
            and the baseline estimates for distinct ``g`` are treated as independent.  The
            calculation uses marginal standard errors only.

        Raises:
            ValueError: If there is not exactly one histogram for each calibrated Pauli type, or if
                a histogram does not resolve the residual weights of its table.
        """
        tables = self._tables
        types = [histogram.pauli for histogram in histograms]
        if len(types) != len(tables) or set(types) != set(tables):
            raise ValueError(
                "Expected one residual-weight histogram for each calibrated Pauli type"
                f" {[str(pauli) for pauli in tables]}, not for {[str(t) for t in types]}"
            )
        distribution, truncated_probability = self._get_distribution()
        no_block_error = (
            float(np.asarray(self.baseline_failure_probabilities) @ distribution)
            + truncated_probability
        )
        value = variance = 0.0
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
                [[0.0, 0.0], table @ distribution, np.full(num_heavier, 1 - no_block_error)]
            )
            mean, error = histogram._get_conditional_mean(outcome_values)
            value += mean
            variance += error**2
            probabilities = histogram.get_residual_weight_distribution()
            row_probabilities = probabilities[1 : num_rows + 1]
            baseline_weights += probabilities[num_rows + 1 :].sum()
            if self.excess_num_samples is not None:
                unsampled = np.asarray(self.excess_num_samples[pauli]) == 0
                baseline_weights += row_probabilities @ unsampled
            if self.excess_standard_errors is not None:
                errors = np.asarray(self.excess_standard_errors[pauli]) * distribution
                variance += float(np.sum((row_probabilities[:, None] * errors) ** 2))
        if self.baseline_standard_errors is not None:
            errors = np.asarray(self.baseline_standard_errors) * distribution
            variance += float(np.sum((baseline_weights * errors) ** 2))
        return value, math.sqrt(variance)

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
        ``get_excess_failure_probability`` combines these histograms.  The probability that a shot
        is accepted is the ``acceptance_probability`` of a histogram from
        ``estimate_residual_weight_histogram`` with the same circuit and options.

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
        # without simplification, each error instruction of the model is one error mechanism
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
        if not len(detectors):
            return np.zeros(0, dtype=bool)
        if errors is not None:
            flips = (errors.astype(int) @ self._responses % 2).astype(np.uint8)
            detectors = detectors ^ flips[:, : self._num_detectors]
            observables = observables ^ flips[:, self._num_detectors :]
        syndromes = detectors.astype(int)
        if isinstance(self.decoder, decoders.BatchObservableDecoder):
            predictions = np.asarray(self.decoder.decode_observables_batch(syndromes))
        else:
            predictions = np.array([self.decoder.decode_observables(row) for row in syndromes])
        expected_shape = (len(detectors), self._num_observables)
        if predictions.ndim == 1 and self._num_observables == 1:
            predictions = predictions[:, None]
        if predictions.shape != expected_shape:
            raise ValueError(
                f"The decoder of the consumer must predict observable flips of shape"
                f" {expected_shape} (shots, observables), not {predictions.shape}"
            )
        predictions = predictions.astype(int) % 2
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

    See ``Consumer.calibrate``.  Each sample is decoded with and without its block error, so its
    paired change in failure is -1, 0, or 1.

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

    Given that exactly ``g`` independent error mechanisms occur, each set of ``g`` mechanisms has a
    probability proportional to the product of their odds ``p / (1 - p)``.  Scaling all odds by a
    common factor keeps this conditional distribution.  This function draws independent Bernoulli
    samples with scaled odds whose expected fault weight is ``g``.  Their shots with exactly ``g``
    mechanisms are exact samples.

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
