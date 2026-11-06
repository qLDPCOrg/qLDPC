# SPDX-License-Identifier: Apache-2.0

"""Consumer-calibrated figures of merit for state-preparation circuits.

``ConsumerCalibration`` combines caller-supplied X-type and Z-type excess-failure tables with
residual-weight histograms.  See the `consumer-calibration guide
<https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/experimental/circuits/benchmarking/consumer_calibration.md>`_
for the model, the scoring formula, its approximations, and validation guidance.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Collection, Mapping, Sequence
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import stim

from qldpc import circuits
from qldpc.objects import Pauli, PauliXZ, PauliXZLike

from .residual_weights import (
    ResidualWeightHistogram,
    _get_fault_weight_probabilities,
    _get_selection,
    estimate_residual_weight_histogram,
)

# the floating-point tolerance of checks that a value is a probability, or equal to another value
_TOLERANCE = 1e-12

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

    Only the first three attributes can be passed by position.

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
