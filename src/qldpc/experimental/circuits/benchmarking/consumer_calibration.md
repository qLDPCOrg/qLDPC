# Consumer-calibrated benchmarks of state preparation

## Purpose and scope

A noisy state-preparation circuit leaves a Pauli error on the code block that it prepares.
How much that error matters depends on the circuit that uses the block next, which we call its **consumer**, such as a round of fault-tolerant error correction.
A logical-observable readout of the prepared state detects only the errors that anticommute with the chosen observable.
A [bounded-fault-propagation certificate](../../../circuits/certification/fault_propagation.md) determines whether small faults can leave disproportionately heavy residual errors.
Consumer calibration instead measures how preparation errors change the failure probability of a specified consumer.

The calibrated figure of merit is the signed change in a supplied consumer failure probability that the errors of a preparation circuit cause, relative to an ideal block.
It is conditioned on the preparation passing post-selection.
The caller supplies the consumer's response as calibration tables or profiles.
Use the figure of merit to compare preparation circuits for one fixed consumer.
It is not a logical error rate, a fault-tolerance certificate, or a bound on the failure of a real computation.

## Definitions

### Error mechanisms, faults, and acceptance

Stim describes the noise of a Clifford circuit by a detector error model (DEM): a list of error instructions, each with a probability and the detectors and logical observables that it flips.
These instructions are the circuit's **error mechanisms**, which are treated as independent events.
Stim approximates some noise channels by independent error mechanisms.
For example, it represents the mutually exclusive components of a `DEPOLARIZE1` instruction as independent mechanisms.
This approximation differs from the channel at second order in its probabilities.
The results called exact below are exact for the independent error mechanisms.

The **fault** of a shot is the set of error mechanisms that occur in it.
Its **fault weight** is the number of those mechanisms.
A shot of a preparation circuit is **accepted** if its fault fires no post-selected detector.
Every distribution of a preparation circuit below is conditioned on acceptance.
The **acceptance probability**, the probability that a shot is accepted, is therefore a separate result.
Report it with the figure of merit.

### Residual weights

A Pauli error on the prepared block leaves the same state as the error times any stabilizer of the ideal state.
The **residual weight** of the error is the smallest Pauli weight among these equivalent errors.
The Pauli weight is either the CSS weight `max(|x|, |z|)` of an error with X support `x` and Z support `z`, or the number of qubits that the error acts on.
For a CSS state, the **residual weight of Pauli type `s`**, for `s = X` or `Z`, is the smallest weight of the type-`s` part of the error times any type-`s` stabilizer.
The two types describe the X and Z parts of an error separately, by their marginal distributions.

`estimate_residual_weight_histogram(circuit, ..., pauli=s)` estimates the distribution `p_s(m) = Pr[M_s = m | accepted]` of the type-`s` residual weight `M_s` of the error that the circuit leaves on its output qubits, as a `ResidualWeightHistogram`.
Its docstring describes the exact enumeration and importance sampling by fault weight that it uses.

### Consumers and profiles

The caller defines the failure event of the consumer and supplies its probabilities.
A **profile** `c_s(m)` is the supplied failure probability for a type-`s` block error of residual weight `m`.
It averages over the consumer's own noise and over the calibration ensemble of errors with that residual weight.
The **excess** `c_s(m) - c_s(0)` is the signed cost of residual weight `m`.
It is normally positive, but an error can occasionally turn a failure of the consumer into a success.
A measured or supplied profile therefore need not increase with `m`.

## The stratified calibration

Let `g` be the fault weight of the consumer, the number of its own error mechanisms that occur.
Let `G` be the largest fault weight to calibrate.
Write

```text
f_s(m, g) = Pr[consumer fails | type-s block error of residual weight m, consumer fault weight g]
```

for the failure probability of the consumer in one stratum `g`.
The figure of merit depends on differences of these probabilities.
For each Pauli type `s`, a `ConsumerCalibration` therefore holds the table of **excesses**

```text
Δ_s(m, g) = f_s(m, g) - f_s(0, g)
```

Each table has a row for each residual weight `m = 1, 2, ...` and a column for each fault weight `g = 0, ..., G`.
The largest tabulated residual weight can differ between Pauli types.
The calibration also holds the **baseline** `f(0, g) = f_s(0, g)`, the failure probability without a block error, which all Pauli types share.
It also holds the probability of each error mechanism of the consumer.
These probabilities determine the Poisson-binomial distribution `Pr[g]` exactly, so that

```text
c_s(m) - c_s(0) = Σ_{g ≤ G} Pr[g] Δ_s(m, g),
c(0) = c_s(0) = Σ_{g ≤ G} Pr[g] f(0, g) + Pr[g > G].
```

Consumer fault weights above `G` count as certain failure both with and without a block error.
They add `Pr[g > G]` to every `c_s(m)` and cancel from the excesses.
A large `truncated_probability` `Pr[g > G]` indicates that `G` is too small.
Residual weights beyond the table of their Pauli type count as certain failure, with excess `1 - c(0)`.
A cell `(m, g)` that no sample of the consumer reached also counts as certain failure, with excess `1 - f(0, g)`.
A sample count of zero in `excess_num_samples` marks such a cell.
Stratifying by `g` computes `Pr[g]` exactly, so only the conditional quantities `Δ_s(m, g)` and `f(0, g)` need estimates.

A cell `(m, g)` is **exact** if its excess and the baseline `f(0, g)` both have zero standard error, and **estimated** otherwise.
The failure probability `f(0, g) + Δ_s(m, g)` of an exact cell must lie in [0, 1].
`ConsumerCalibration` rejects exact cells outside this interval.
Estimated cells may stray outside [0, 1] by their statistical error, so `ConsumerCalibration` accepts them.

## The figure of merit

For a preparation circuit, the calibrated figure of merit averages the excess cost of each Pauli type over the marginal residual-weight distribution of that type.
It then adds the contributions of the Pauli types:

```text
Φ = Σ_s Σ_m p_s(m) (c_s(m) - c_s(0)).
```

`ConsumerCalibration.estimate_excess_failure_probability(circuit)` returns `Φ` and its standard error.
If the calibration has `state_stabilizers`, it first checks that the noiseless circuit prepares that state on its output qubits.
It then estimates one histogram per calibrated Pauli type, from independent random numbers.
`get_excess_failure_probability` combines these histograms.
A Pauli type without a table contributes zero.
Omit the table of a Pauli type for a consumer that ignores errors of that type.
`ResidualWeightHistogram.acceptance_probability` gives the acceptance probability.
Report `Φ`, its standard error, and the acceptance probability of the circuit together.

## Approximations and uncertainty

`Φ` estimates the change in the failure probability of the calibrated consumer, per accepted preparation, relative to an ideal block, under the following approximations:

1. **Independent error mechanisms.**
   The consumer's and the preparation's noise channels are replaced by independent DEM mechanisms, as described [above](#error-mechanisms-faults-and-acceptance).
2. **Separate Pauli types.**
   The X and Z contributions are added, which neglects their interaction when both parts of an error occur in one shot, such as the two parts of a Y error.
3. **The calibration ensemble.**
   Each excess `c_s(m) - c(0)` averages over the ensemble of errors of residual weight `m` with which the consumer was calibrated, not over those of the preparation circuit.
4. **Truncation.**
   Consumer fault weights above `G` count as certain failure, which cancels from the excesses.
   Preparation faults above the histogram's largest fault weight count as overflow, and residual weights above the largest residual weight tabulated for their Pauli type and unsampled cells count as certain failure.
   For exact cells, which are valid probabilities, this can only overstate `Φ`; for estimated cells, this holds only up to their statistical error.
5. **Rescaling.**
   `ConsumerCalibration.rescaled(scale)` multiplies every error probability of the consumer by `scale`, recomputes `Pr[g]`, and reuses the excesses and the baseline.
   For fixed `g`, the distribution of the sets of mechanisms that occur depends only on ratios of their odds.
   These ratios change little when all probabilities are small.
   DEM probabilities are also only approximately proportional to the rates of a noise model.
   Rescaling therefore suits nearby low error rates.
   A calibration made at the target noise avoids this approximation.

The standard errors cover the sampling of the calibration and the histograms, propagated to first order.
They exclude the approximations above.
The reported standard error of `Φ` combines three contributions in quadrature:

- the sampling errors of the histograms, which are independent between Pauli types;
- the standard errors of the cells of the excess tables, treated as independent;
- the error of the baseline.

Every certain failure depends on the baseline, whether it lies beyond a table or in a cell with a sample count of zero.
An unsampled cell has the excess `1 - f(0, g)`, so its uncertainty is that of `f(0, g)`.
The error contributions of each baseline `f(0, g)` therefore add linearly across the cells, residual weights, and Pauli types that use it, before the quadrature sum.
The baseline estimates for distinct `g` are treated as independent.
A calibration supplies only marginal standard errors, so the standard error neglects correlations between cells, and between cells and the baseline.
Cells that share samples are correlated, such as excesses of different residual weights estimated from the same shots of the consumer.
For such cells, the reported standard error only approximates the uncertainty.
Importance-sampled histogram errors can be optimistic when few samples reach the outcomes that matter.
Check the histogram's `num_samples` when an error looks too small.

## Supplying a consumer's response

- **A Poisson profile.**
  `ConsumerProfile.poisson(expected_fault_weight, t)` models a consumer that fails exactly when `m + g > t` for a Poisson-distributed `g`.
  `estimate_downstream_fault_weight` estimates the expected fault weight from an operation-count preset.
  Score the profile with `ResidualWeightHistogram.get_excess_failure_probability` on a histogram computed with `pauli=None`.
  Its hard-threshold response and Poisson approximation make it an approximate score rather than a calibrated one or a bound.
- **Stratified tables.**
  Construct a `ConsumerCalibration` from excesses `Δ_s(m, g)`, the baseline `f(0, g)`, and the probabilities of the consumer's error mechanisms.
  Standard errors, sample counts, and `state_stabilizers` are optional.
  Use `rescaled` to adapt the calibration to a nearby error rate.
- **Costs at a fixed noise level.**
  For failure probabilities `c_s(m)` measured at a fixed noise level, such as on hardware, use one column, `g = 0`.
  Pass the excesses `c_s(m) - c(0)` as the tables, `[c(0)]` as the baseline, and no mechanism probabilities, so that `Pr[g = 0] = 1`.
- **Serialization.**
  `calibration.to_dict()` returns JSON-compatible values, which `ConsumerCalibration.from_dict` loads.

## How to validate

- **Fault tolerance of the consumer.**
  For a consumer that tolerates `t` faults in total, the calibrated excesses `Δ_s(m, g)` with `m + g ≤ t` should vanish.
- **Truncation and coverage.**
  `truncated_probability` should be small compared with the excesses of interest.
  `excess_num_samples` should be nonzero for every reachable cell that a histogram gives weight.
- **Exact small-instance comparison.**
  Compare calibrated estimates and histograms with exact calculations on a small controlled instance.
