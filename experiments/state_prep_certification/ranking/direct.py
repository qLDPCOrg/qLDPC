# SPDX-License-Identifier: Apache-2.0

"""The stratified direct reference of Phi_true, and its exact leading order.

Phi_true is the excess failure probability that an accepted preparation causes in the consumer:
with preparation faults F and consumer faults T, the paired change in failure is
``delta(F, T) = 1_A(F) (chi(E_F, T) - chi(0, T))``, where ``1_A`` is acceptance by post-selection,
``E_F`` the output error of F, and ``chi`` the failure of the consumer's decoder.  Then

    Phi_true = N / alpha,   N = sum_{f >= 1, g >= 0} pi_f eta_g D(f, g),   alpha = sum_f pi_f a_f,

with pi_f and eta_g the exact laws of the fault weights f of the preparation and g of the consumer,
D(f, g) the mean of delta given both weights, and a_f the acceptance probability given f.

The preparation's error mechanisms are those of qLDPC's residual-weight histograms (flags and
output syndromes from ``_get_primitive_fault_signatures``), and each fault enters the consumer as
the product of the destabilizers of its output syndrome, which equals its output error up to a
stabilizer of the prepared state.  The consumer's mechanisms, decoder, and responses to block
errors are those of qLDPC's ``Consumer``.  The strata are f <= f_max and g <= g_max, with f_max the
histogram's default largest fault weight and g_max the calibration's.  A stratum (f, g) is listed
exactly if it has at most ``max_exact_faults`` pairs of faults, and otherwise sampled.  A sampled
stratum draws F from a proposal for weight f (``sample_faults``), with importance weight
P_f(F) / q_f(F), and T exactly from its conditional law, as ``Consumer.calibrate`` draws it.

The estimate is the ratio of the stratified sums.  Preparation faults heavier than f_max, of total
probability u, count as accepted with no change, and consumer faults heavier than g_max, of total
probability v, are left out.  The first-order (delta-method) standard error ``sigma`` treats the
strata as independent, except that a sampled stratum (f, 0) also estimates a_f from the same
draws.  Since every change lies in [-1, 1], the truncations move the estimate's target by at most
the deterministic allowance ``tau = u / alpha_exact + v``, where ``alpha_exact`` is the acceptance
mass of the exactly listed fault weights of the preparation, f = 0 included.  Neither ``sigma`` nor
``tau`` is a confidence bound, and the ratio estimate is not unbiased.
"""

from __future__ import annotations

import dataclasses
import fractions
import itertools
import math
import time
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import numpy.typing as npt
import stim

from qldpc import circuits
from qldpc import math as qldpc_math
from qldpc.circuits.certification.fault_propagation import (
    _get_primitive_fault_signatures,
    _is_css_state,
)
from qldpc.experimental.circuits.benchmarking import Consumer, consumers, residual_weights
from qldpc.objects import Pauli, PauliXZ

# The variance of a sampled stratum with fewer than this many nonzero changes is floored, and the
# stratum is inadequate if its effective sample size is below this fraction of its draws.
MIN_NONZERO = 10
MIN_ESS_FRACTION = 0.1

####################################################################################################
# error mechanisms


@dataclasses.dataclass(frozen=True)
class Preparation:
    """The error mechanisms of a noisy preparation circuit, as qLDPC's histograms list them.

    Attributes:
        probabilities: The probability of each error mechanism.
        flag_words: Row j holds the post-selected detectors that mechanism j fires, packed.
        errors: Row j is an output error of mechanism j in ``[X|Z]`` form, up to a stabilizer.
        stabilizers: The stabilizer generators of the prepared state, one row per generator.
        css_weight: Whether residual weights are CSS weights.
    """

    probabilities: npt.NDArray[np.float64]
    flag_words: npt.NDArray[np.uint64]
    errors: npt.NDArray[np.uint8]
    stabilizers: npt.NDArray[np.uint8]
    css_weight: bool

    def __len__(self) -> int:
        """The number of error mechanisms."""
        return len(self.probabilities)

    @property
    def flagged(self) -> npt.NDArray[np.bool_]:
        """Whether each mechanism fires a post-selected detector."""
        return np.any(self.flag_words, axis=1)

    def accepted(self, faults: npt.NDArray[np.intp]) -> npt.NDArray[np.bool_]:
        """Whether post-selection accepts each fault, given as a row of mechanism indices."""
        return ~np.any(np.bitwise_xor.reduce(self.flag_words[faults], axis=1), axis=1)

    def fault_errors(self, faults: npt.NDArray[np.intp]) -> npt.NDArray[np.uint8]:
        """The output error of each fault, up to a stabilizer, one row per fault."""
        if faults.shape[1] == 0:
            return np.zeros((len(faults), self.errors.shape[1]), dtype=np.uint8)
        return np.bitwise_xor.reduce(self.errors[faults], axis=1)

    def residual_weights(
        self, errors: npt.NDArray[np.uint8], pauli: PauliXZ | None, max_residual_weight: int
    ) -> npt.NDArray[np.intp]:
        """Residual weights of output errors, capped at ``max_residual_weight + 1``.

        ``pauli`` selects the X-type or Z-type residual weight, as for the histograms.
        """
        setup = residual_weights._ResidualWeightSetup(
            self.stabilizers, self.css_weight, pauli, max_residual_weight
        )
        return setup.get_error_weights(errors)


def get_preparation(
    circuit: stim.Circuit, options: dict[str, Any], max_residual_weight: int
) -> Preparation:
    """The error mechanisms of a noisy preparation circuit with the histogram options of a circuit.

    ``options`` are ``output_qubits``, ``add_flags``, ``post_select``, and ``css_weight``, as for
    ``estimate_residual_weight_histogram``.  The mechanisms, in the order of the DEM's error
    instructions, are those of the histograms of the same circuit and options.
    """
    full, qubits, selected = residual_weights._get_selection(
        circuit, options["output_qubits"], options["add_flags"], options["post_select"]
    )
    strings = circuits.get_state_stabilizers(full, qubits)
    matrix = np.array([qldpc_math.string_to_op(op) for op in strings], dtype=np.uint8)
    css_weight = _is_css_state(matrix) if options["css_weight"] is None else options["css_weight"]
    setup = residual_weights._ResidualWeightSetup(matrix, css_weight, None, max_residual_weight)
    _, dem, flags, syndromes = _get_primitive_fault_signatures(
        full, setup.stabilizers, qubits, selected
    )
    probabilities = [
        instruction.args_copy()[0] for instruction in dem if instruction.type == "error"
    ]
    kept = [index for index, probability in enumerate(probabilities) if probability > 0]

    # the destabilizers of the generators give an error with each syndrome
    tableau = stim.Tableau.from_stabilizers(setup.stabilizers)
    destabilizers = np.array(
        [qldpc_math.string_to_op(tableau.x_output(k)) for k in range(len(qubits))], dtype=np.uint8
    )
    bits = np.array(
        [[(syndromes[index] >> k) & 1 for k in range(len(qubits))] for index in kept], dtype=int
    ).reshape(len(kept), len(qubits))
    num_words = (len(selected) + 63) // 64
    return Preparation(
        probabilities=np.array([probabilities[index] for index in kept], dtype=float),
        flag_words=np.array(
            [residual_weights._to_words(flags[index], num_words) for index in kept],
            dtype=np.uint64,
        ).reshape(len(kept), num_words),
        errors=(bits @ destabilizers % 2).astype(np.uint8),
        stabilizers=setup.matrix,
        css_weight=css_weight,
    )


def fault_weight_law(
    probabilities: npt.NDArray[np.float64], base: int, tolerance: float
) -> tuple[npt.NDArray[np.float64], float]:
    """The exact law of a fault weight, truncated by qLDPC's default rule, and its tail.

    With ``base = max_residual_weight + 1`` and qLDPC's histogram tolerance, this is the
    histogram's; with ``base = max_residual_weight`` and the calibration's tolerance, it is the
    calibration's.
    """
    return residual_weights._get_fault_weight_probabilities(probabilities, None, base, tolerance)


####################################################################################################
# paired changes in failure


class Pairing:
    """Paired changes in the failure of a consumer, for faults of a preparation.

    Decoding is deterministic, so each distinct combination of a consumer fault's flips and a
    block error is decoded once per call.
    """

    def __init__(self, preparation: Preparation, simulation: consumers._ConsumerSimulation) -> None:
        """Pair the mechanisms of a preparation with a simulation of the consumer."""
        self.preparation = preparation
        self.simulation = simulation

    def changes(
        self, errors: npt.NDArray[np.uint8], consumer_faults: npt.NDArray[np.intp]
    ) -> npt.NDArray[np.int_]:
        """``chi(E, T) - chi(0, T)`` for block errors E and consumer faults T, one pair per row."""
        detectors, observables = self.simulation.get_flips(consumer_faults)
        rows = np.concatenate([detectors, observables, errors], axis=1)
        unique, inverse = np.unique(rows, axis=0, return_inverse=True)
        num_detectors = detectors.shape[1]
        num_flips = num_detectors + observables.shape[1]
        flips = unique[:, :num_detectors], unique[:, num_detectors:num_flips]
        baseline = self.simulation.get_failures(*flips).astype(int)
        changed = self.simulation.get_failures(*flips, unique[:, num_flips:]).astype(int)
        return (changed - baseline)[inverse.reshape(-1)]


def build_pairing(
    circuit: stim.Circuit,
    options: dict[str, Any],
    consumer: Consumer,
    max_residual_weight: int,
    rng: np.random.Generator,
) -> Pairing:
    """The pairing of a noisy preparation circuit with a consumer."""
    preparation = get_preparation(circuit, options, max_residual_weight)
    return Pairing(preparation, consumers._ConsumerSimulation(consumer, rng))


####################################################################################################
# the stratified reference


@dataclasses.dataclass
class Stratum:
    """The statistics of one joint stratum (f, g).

    For an exactly listed stratum, ``mean`` is D(f, g), ``acceptance`` is a_f (g = 0 only), the
    variances are zero, and ``plus`` and ``minus`` count the pairs (F, T) with changes +1 and -1.
    For a sampled stratum, they are the estimates D-hat and a-hat with their variances and the
    covariance of the two, ``plus`` and ``minus`` count draws, and ``nonzero`` and ``ess`` are the
    draws with a nonzero change and the Kish effective sample size of the importance weights.
    ``draws``, ``accepted_draws``, and ``max_importance_weight`` are the number of draws, the number
    accepted, and the largest importance weight, all zero or None for an exact stratum; ``cpu_s`` is
    the CPU time of the stratum.
    """

    f: int
    g: int
    exact: bool
    mean: float
    variance: float = 0.0
    acceptance: float | None = None
    acceptance_variance: float = 0.0
    covariance: float = 0.0
    plus: int = 0
    minus: int = 0
    draws: int = 0
    nonzero: int = 0
    accepted_draws: int = 0
    ess: float | None = None
    max_importance_weight: float | None = None
    cpu_s: float = 0.0

    @property
    def inadequate(self) -> bool:
        """Whether a sampled stratum's effective sample size is below its threshold."""
        return not self.exact and (self.ess or 0.0) < MIN_ESS_FRACTION * self.draws


def stratified_reference(
    pairing: Pairing,
    max_residual_weight: int,
    calibration_tolerance: float,
    max_exact_faults: int,
    num_samples: int,
    flagged_odds_factor: float,
    rng: np.random.Generator,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Estimate Phi_true by stratified direct simulation, as the module docstring describes.

    Sampled strata draw ``num_samples`` faults each, from the proposal of ``sample_faults`` with
    the given factor.  Each stratum is reported to ``progress`` when it is done, so that a run
    killed at its limit keeps them, after the laws of both fault weights.

    Returns:
        The estimate ``phi`` with its standard error ``sigma`` and truncation allowance ``tau``,
        the laws and tails of both fault weights, every stratum, and the CPU seconds of the exactly
        listed and the sampled strata.
    """
    preparation = pairing.preparation
    pi, u = fault_weight_law(
        preparation.probabilities,
        max_residual_weight + 1,
        residual_weights._TRUNCATION_TOLERANCE,
    )
    eta, v = fault_weight_law(
        pairing.simulation.probabilities, max_residual_weight, calibration_tolerance
    )
    num_prep, num_consumer = len(preparation), len(pairing.simulation.probabilities)
    laws = {
        "pi": pi.tolist(),
        "u": u,
        "eta": eta.tolist(),
        "v": v,
        "num_prep_mechanisms": num_prep,
        "num_consumer_mechanisms": num_consumer,
    }
    if progress is not None:
        progress(laws)
    strata = []
    for f, g in itertools.product(range(1, len(pi)), range(len(eta))):
        start = time.process_time()
        if math.comb(num_prep, f) * math.comb(num_consumer, g) <= max_exact_faults:
            stratum = _list_stratum(pairing, f, g)
        else:
            stratum = _sample_stratum(pairing, f, g, num_samples, flagged_odds_factor, rng)
        stratum.cpu_s = time.process_time() - start
        strata.append(stratum)
        if progress is not None:
            progress({"stratum": dataclasses.asdict(stratum)})
    return combine_strata(strata, pi, u, eta, v) | laws


def combine_strata(
    strata: list[Stratum],
    pi: npt.NDArray[np.float64],
    u: float,
    eta: npt.NDArray[np.float64],
    v: float,
) -> dict[str, Any]:
    """The ratio estimate of Phi_true from its strata, with sigma and tau (see the module)."""
    numerator = numerator_variance = covariance = 0.0
    # stratum f = 0 is accepted with no change
    acceptance = alpha_exact = float(pi[0])
    acceptance_variance = 0.0
    for stratum in strata:
        weight = pi[stratum.f] * eta[stratum.g]
        numerator += weight * stratum.mean
        numerator_variance += weight**2 * stratum.variance
        if stratum.g == 0:
            assert stratum.acceptance is not None
            acceptance += pi[stratum.f] * stratum.acceptance
            acceptance_variance += pi[stratum.f] ** 2 * stratum.acceptance_variance
            covariance += weight * pi[stratum.f] * stratum.covariance
            if stratum.exact:
                alpha_exact += pi[stratum.f] * stratum.acceptance
    acceptance += u
    phi = numerator / acceptance
    variance = (
        numerator_variance - 2 * phi * covariance + phi**2 * acceptance_variance
    ) / acceptance**2
    return {
        "phi": phi,
        "sigma": math.sqrt(max(variance, 0.0)),
        "tau": u / alpha_exact + v,
        "alpha": acceptance,
        "alpha_exact": alpha_exact,
        "u": u,
        "v": v,
        "inadequate": [[s.f, s.g] for s in strata if s.inadequate],
        "strata": [dataclasses.asdict(stratum) for stratum in strata],
        "cpu_exact_s": sum(s.cpu_s for s in strata if s.exact),
        "cpu_sampled_s": sum(s.cpu_s for s in strata if not s.exact),
    }


def sample_faults(
    preparation: Preparation,
    f: int,
    num_samples: int,
    flagged_odds_factor: float,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.float64]]:
    """Faults of weight f from a proposal, with importance weights P_f(F) / q_f(F).

    The proposal multiplies the odds of flagged mechanisms by ``flagged_odds_factor``, as qLDPC's
    histograms do with their ``_FLAGGED_ODDS_FACTOR``.  Given f, a fault has a probability
    proportional to the product of its odds under both the noise and the proposal, so both
    conditional laws are normalized exactly by elementary symmetric polynomials of the odds, and
    the faults are drawn exactly from the proposal's conditional law.
    """
    odds = preparation.probabilities / (1 - preparation.probabilities)
    flagged = preparation.flagged
    proposal = odds * np.where(flagged, flagged_odds_factor, 1.0)
    faults = _sample_conditional_bernoulli(proposal, f, num_samples, rng)
    log_ratio = _log_elementary_symmetric(proposal, f) - _log_elementary_symmetric(odds, f)
    weights = np.exp(log_ratio - math.log(flagged_odds_factor) * flagged[faults].sum(axis=1))
    return faults, weights


def _list_stratum(pairing: Pairing, f: int, g: int) -> Stratum:
    """The exact stratum (f, g), by listing every pair of faults."""
    preparation, simulation = pairing.preparation, pairing.simulation
    faults = residual_weights._get_combinations(len(preparation), f)
    prep_weights = _conditional_probabilities(preparation.probabilities, faults)
    accepted = preparation.accepted(faults)
    acceptance = float(prep_weights @ accepted) if g == 0 else None

    # accepted faults with the same output error have the same changes
    errors, inverse = np.unique(
        preparation.fault_errors(faults[accepted]), axis=0, return_inverse=True
    )
    inverse = inverse.reshape(-1)
    error_weights = np.bincount(inverse, weights=prep_weights[accepted], minlength=len(errors))
    multiplicities = np.bincount(inverse, minlength=len(errors))
    consumer_faults = residual_weights._get_combinations(len(simulation.probabilities), g)
    consumer_weights = _conditional_probabilities(simulation.probabilities, consumer_faults)

    mean = 0.0
    plus = minus = 0
    for rows, columns in _pair_chunks(len(errors), len(consumer_faults)):
        changes = pairing.changes(errors[rows], consumer_faults[columns])
        mean += float(np.sum(error_weights[rows] * consumer_weights[columns] * changes))
        plus += int(multiplicities[rows] @ (changes > 0))
        minus += int(multiplicities[rows] @ (changes < 0))
    return Stratum(f, g, exact=True, mean=mean, acceptance=acceptance, plus=plus, minus=minus)


def _sample_stratum(
    pairing: Pairing,
    f: int,
    g: int,
    num_samples: int,
    flagged_odds_factor: float,
    rng: np.random.Generator,
) -> Stratum:
    """A sampled stratum (f, g), with exactly normalized importance weights."""
    preparation, simulation = pairing.preparation, pairing.simulation
    faults, weights = sample_faults(preparation, f, num_samples, flagged_odds_factor, rng)
    consumer_faults = consumers._sample_fault_sets(simulation.probabilities, g, num_samples, rng)
    accepted = preparation.accepted(faults)
    changes = np.zeros(num_samples, dtype=int)
    changes[accepted] = pairing.changes(
        preparation.fault_errors(faults[accepted]), consumer_faults[accepted]
    )
    values = weights * changes
    mean, variance = _mean_and_variance(values)
    nonzero = int(np.count_nonzero(changes))
    if nonzero < MIN_NONZERO:
        variance = max(variance, (float(weights.max()) / num_samples) ** 2)
    stratum = Stratum(
        f,
        g,
        exact=False,
        mean=mean,
        variance=variance,
        plus=int(np.count_nonzero(changes > 0)),
        minus=int(np.count_nonzero(changes < 0)),
        draws=num_samples,
        nonzero=nonzero,
        accepted_draws=int(np.count_nonzero(accepted)),
        ess=float(weights.sum() ** 2 / np.sum(weights**2)),
        max_importance_weight=float(weights.max()),
    )
    if g == 0:
        acceptances = weights * accepted
        stratum.acceptance, stratum.acceptance_variance = _mean_and_variance(acceptances)
        stratum.covariance = float(np.cov(values, acceptances)[0, 1]) / num_samples
    return stratum


def _sample_conditional_bernoulli(
    odds: npt.NDArray[np.float64], size: int, num_samples: int, rng: np.random.Generator
) -> npt.NDArray[np.intp]:
    """Exact draws of sets of ``size`` mechanisms with probabilities proportional to their odds.

    Mechanism j joins a draw that still needs r members with probability
    ``o_j e_{r-1}(o_{j+1}, ...) / e_r(o_j, ...)``, with e_r the elementary symmetric polynomials.
    """
    scaled = odds / odds.max()
    num_mechanisms = len(odds)
    # suffix[j, r] = e_r(scaled[j:])
    suffix = np.zeros((num_mechanisms + 1, size + 1))
    suffix[num_mechanisms, 0] = 1.0
    for j in range(num_mechanisms - 1, -1, -1):
        suffix[j] = suffix[j + 1]
        suffix[j, 1:] += scaled[j] * suffix[j + 1, :-1]
    remaining = np.full(num_samples, size)
    chosen = np.zeros((num_samples, num_mechanisms), dtype=bool)
    for j in range(num_mechanisms):
        needed = remaining > 0
        numerator = scaled[j] * suffix[j + 1, np.maximum(remaining - 1, 0)]
        probability = np.divide(
            numerator, suffix[j, remaining], out=np.zeros(num_samples), where=needed
        )
        take = needed & (rng.random(num_samples) < probability)
        chosen[:, j] = take
        remaining -= take
    return np.nonzero(chosen)[1].reshape(num_samples, size)


def _log_elementary_symmetric(odds: npt.NDArray[np.float64], size: int) -> float:
    """The logarithm of the elementary symmetric polynomial of degree ``size`` of the odds."""
    scale = odds.max()
    values = np.zeros(size + 1)
    values[0] = 1.0
    for odd in odds / scale:
        values[1:] = values[1:] + odd * values[:-1]
    return math.log(values[size]) + size * math.log(scale)


def _conditional_probabilities(
    probabilities: npt.NDArray[np.float64], faults: npt.NDArray[np.intp]
) -> npt.NDArray[np.float64]:
    """The probability of each fault of a fixed weight, given that weight, one fault per row."""
    if faults.shape[1] == 0:
        return np.ones(len(faults))
    log_odds = np.log(probabilities / (1 - probabilities))
    log_weights = log_odds[faults].sum(axis=1)
    weights = np.exp(log_weights - log_weights.max())
    return weights / weights.sum()


def _pair_chunks(
    num_rows: int, num_columns: int
) -> Iterator[tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]]:
    """The index pairs of all (row, column) combinations, in chunks of bounded size."""
    chunk = max(1, residual_weights._CHUNK_SIZE * 10 // max(num_columns, 1))
    for start in range(0, num_rows, chunk):
        rows = np.arange(start, min(start + chunk, num_rows))
        yield np.repeat(rows, num_columns), np.tile(np.arange(num_columns), len(rows))


def _mean_and_variance(values: npt.NDArray[np.float64]) -> tuple[float, float]:
    """The sample mean of values and the variance of that mean."""
    mean = float(values.mean())
    return mean, float(values.var(ddof=1)) / len(values)


####################################################################################################
# exact leading order


def first_order_coefficients(
    probabilities: npt.NDArray[np.float64], probe: npt.NDArray[np.float64], probe_rate: float
) -> list[fractions.Fraction]:
    """The exact coefficients c_m of p in the probabilities of mechanisms, as fractions.

    ``probe`` holds the same mechanisms' probabilities at the small rate ``probe_rate``, where
    ``probability / rate`` equals c_m to about ``probe_rate``.  The noise model's coefficients are
    fractions with small denominators, which ``limit_denominator`` recovers exactly.
    """
    if len(probe) != len(probabilities):
        raise ValueError("The mechanisms differ between the two rates")
    coefficients = []
    for value in probe / probe_rate:
        coefficient = fractions.Fraction(float(value)).limit_denominator(10**4)
        if abs(float(coefficient) - value) > 1e-6 * value:
            raise ValueError(f"No small fraction is close to the coefficient {value}")
        coefficients.append(coefficient)
    return coefficients


def leading_order(
    pairing: Pairing,
    prep_coefficients: list[fractions.Fraction],
    consumer_coefficients: list[fractions.Fraction],
    max_residual_weight: int,
) -> dict[str, Any]:
    """The coefficients of the first nonzero power of p in Phi_true and Phi_cal, exactly.

    The decoder is that of ``pairing``, frozen.  The coefficient of p in Phi_true is the sum of
    c_F delta(F, {}) over single preparation faults F.  If every such change is zero, the
    coefficient of p^2 is the sum of c_F c_G delta({F, G}, {}) over pairs of preparation faults and
    of c_F c_T delta(F, T) over a preparation fault and a consumer fault.  Phi_cal uses the exact
    cells of the calibration's ensemble (``exact_cells``) in place of delta, through the X residual
    weight of each accepted fault.  If a coefficient of p is zero only by cancellation, the
    coefficient of p^2 would need the second-order terms of the mechanisms, and is not computed.
    """
    preparation = pairing.preparation
    singles = np.arange(len(preparation))[:, None]
    no_fault = np.zeros((1, 0), dtype=np.intp)
    consumer_singles = np.arange(len(consumer_coefficients))[:, None]

    def terms(
        faults: npt.NDArray[np.intp], consumer_faults: npt.NDArray[np.intp]
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], list[int], list[fractions.Fraction]]:
        """delta and the calibrated excess of every pair of a fault and a consumer fault."""
        cells = exact_cells(pairing, max_residual_weight, consumer_faults)
        accepted = preparation.accepted(faults)
        errors = preparation.fault_errors(faults)
        weights = preparation.residual_weights(errors, Pauli.X, max_residual_weight)
        rows = np.repeat(np.arange(len(faults)), len(consumer_faults))
        columns = np.tile(np.arange(len(consumer_faults)), len(faults))
        changes = pairing.changes(errors[rows], consumer_faults[columns])
        delta = [int(change) if accepted[row] else 0 for row, change in zip(rows, changes)]
        excess = [
            cells[(int(weights[row]), column)] if accepted[row] else fractions.Fraction(0)
            for row, column in zip(rows, columns)
        ]
        return rows, columns, delta, excess

    _, _, delta_1, excess_1 = terms(singles, no_fault)
    result: dict[str, Any] = {}
    for name, values in (("true", [fractions.Fraction(d) for d in delta_1]), ("cal", excess_1)):
        first = sum(
            (c * value for c, value in zip(prep_coefficients, values)), fractions.Fraction()
        )
        if first:
            result[name] = {"order": 1, "coefficient": first}
        elif any(values):
            result[name] = {"order": None, "coefficient": None, "note": "cancellation at order 1"}
        else:
            result[name] = {"order": 2, "coefficient": fractions.Fraction()}

    # the coefficient of p^2, where every term of the coefficient of p vanishes
    if any(entry["order"] == 2 for entry in result.values()):
        doubles = residual_weights._get_combinations(len(preparation), 2)
        rows_2, _, delta_2, excess_2 = terms(doubles, no_fault)
        rows_ft, columns_ft, delta_ft, excess_ft = terms(singles, consumer_singles)
        for name, pair_values, mixed_values in (
            (
                "true",
                list(map(fractions.Fraction, delta_2)),
                list(map(fractions.Fraction, delta_ft)),
            ),
            ("cal", excess_2, excess_ft),
        ):
            if result[name]["order"] != 2:
                continue
            total = fractions.Fraction()
            for row, value in zip(rows_2, pair_values):
                first, second = doubles[row]
                total += prep_coefficients[first] * prep_coefficients[second] * value
            for row, column, value in zip(rows_ft, columns_ft, mixed_values):
                total += prep_coefficients[row] * consumer_coefficients[column] * value
            result[name]["coefficient"] = total
    for entry in result.values():
        coefficient = entry["coefficient"]
        entry["value"] = None if coefficient is None else float(coefficient)
        entry["coefficient"] = None if coefficient is None else str(coefficient)
    return result


def exact_cells(
    pairing: Pairing, max_residual_weight: int, consumer_faults: npt.NDArray[np.intp]
) -> dict[tuple[int, int], fractions.Fraction]:
    """Exact excesses of the calibration's X-type ensemble, for fixed consumer faults.

    Let M be ``max_residual_weight`` and n the number of block qubits.  The ensemble of residual
    weight m draws X errors of each physical weight w = 1, ..., min(M + 1, n) in equal numbers,
    uniformly among the supports of size w, and keeps those of X residual weight m, as
    ``Consumer.calibrate`` samples it with ``paulis=("X",)``.  Key (m, j) holds the mean of
    ``chi(E, T) - chi(0, T)`` over this ensemble, for T row j of ``consumer_faults``.  Key (0, j)
    holds zero, and key (M + 1, j) the excess of certain failure, ``1 - chi(0, T)``, as
    ``ConsumerCalibration`` counts residual weights above M.
    """
    num_qubits = pairing.preparation.stabilizers.shape[1] // 2
    supports = [
        support
        for weight in range(1, min(max_residual_weight + 1, num_qubits) + 1)
        for support in itertools.combinations(range(num_qubits), weight)
    ]
    errors = np.zeros((len(supports), 2 * num_qubits), dtype=np.uint8)
    for row, support in enumerate(supports):
        errors[row, list(support)] = 1
    weights = pairing.preparation.residual_weights(errors, Pauli.X, max_residual_weight)
    masses = [fractions.Fraction(1, math.comb(num_qubits, len(support))) for support in supports]

    cells: dict[tuple[int, int], fractions.Fraction] = {}
    for column, fault in enumerate(consumer_faults):
        changes = pairing.changes(errors, np.repeat(fault[None, :], len(supports), axis=0))
        for m in range(1, max_residual_weight + 1):
            members = [row for row in range(len(supports)) if weights[row] == m]
            total = sum((masses[row] for row in members), fractions.Fraction())
            changed = sum(
                (masses[row] * int(changes[row]) for row in members), fractions.Fraction()
            )
            cells[(m, column)] = changed / total if total else fractions.Fraction()
        detectors, observables = pairing.simulation.get_flips(fault[None, :])
        failure = int(pairing.simulation.get_failures(detectors, observables)[0])
        cells[(0, column)] = fractions.Fraction()
        cells[(max_residual_weight + 1, column)] = fractions.Fraction(1 - failure)
    return cells
