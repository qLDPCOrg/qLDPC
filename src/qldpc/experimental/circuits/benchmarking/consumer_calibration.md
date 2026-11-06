# Consumer-calibrated benchmarks of state preparation

## Purpose and scope

A noisy state-preparation circuit leaves a Pauli error on the code block that it prepares.
How much that error matters depends on the circuit that uses the block next, which we call its **consumer**, such as a round of fault-tolerant error correction.
A logical-observable readout of the prepared state detects only the errors that anticommute with the chosen observable.
A [bounded-fault-propagation certificate](../../../circuits/certification/fault_propagation.md) determines whether small faults can leave disproportionately heavy residual errors.
Consumer calibration instead measures how preparation errors change the failure probability of a specified consumer.

The calibrated figure of merit is the signed change in the failure probability of a consumer that the errors of a preparation circuit cause, relative to an ideal block.
It is conditioned on the preparation passing post-selection.
The consumer's response comes either from simulating a `Consumer` or from calibration tables or profiles that the caller supplies.
One calibration of a consumer then scores many preparation circuits.
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

A `Consumer` is a simulated consumer.
It runs a noiseless circuit that prepares the ideal block, and then a noisy Clifford circuit that uses the block.
A decoder predicts the flips of the circuit's observables from its detectors.
These observables need not be logical operators.
As Stim requires, each is a parity of measurements whose value is deterministic without noise.
The consumer fails in a shot if the decoder mispredicts any of these observables.
For other consumers, the caller defines the failure event and supplies its probabilities.
A **profile** `c_s(m)` is the failure probability for a type-`s` block error of residual weight `m`.
It averages over the consumer's own noise and over the calibration ensemble of errors with that residual weight.
The **excess** `c_s(m) - c_s(0)` is the signed cost of residual weight `m`.
It is normally positive, but an error can occasionally turn a failure of the consumer into a success.
A measured or supplied profile therefore need not increase with `m`.

## The stratified calibration

### What is estimated

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
Rare fault weights therefore need no naive sampling.
The conditional quantities can still need many samples when failure within a stratum is rare.

A cell `(m, g)` is **exact** if its excess and the baseline `f(0, g)` both have zero standard error, and **estimated** otherwise.
The failure probability `f(0, g) + Δ_s(m, g)` of an exact cell must lie in [0, 1].
`ConsumerCalibration` rejects exact cells outside this interval.
Estimated cells may stray outside [0, 1] by their statistical error, so `ConsumerCalibration` accepts them.
`Consumer.calibrate` gives every sampled cell and baseline a positive standard error, so its cells are all estimated.

### How it is sampled

`Consumer.calibrate` estimates a calibration by simulation.
Its sampling uses the odds `p / (1 - p)` of each error mechanism of the consumer, so it rejects a consumer with a mechanism of probability one.
For each requested Pauli type `s`, calibrated fault weight `g`, and physical error weight `w = 1, ..., min(max_residual_weight + 1, n)`, where `n` is the number of block qubits, it draws `num_samples` samples.
Each sample has two parts:

1. A set of exactly `g` error mechanisms of the consumer, drawn from their conditional distribution given that exactly `g` occur.
   In this distribution, a set has probability proportional to the product of the odds `p / (1 - p)` of its mechanisms.
   Scaling all odds by a common factor keeps this distribution.
   The shots with exactly `g` mechanisms of independent Bernoulli samples with suitably scaled odds are therefore exact samples.
2. A uniformly random type-`s` error of weight `w` on the block, whose residual weight `m` is at most `w`.

Each sample is decoded with and without its block error.
Its paired change in failure, with the block error minus without it, is `-1`, `0`, or `1`.
The excess `Δ_s(m, g)` is the mean paired change over the samples whose block errors have residual weight `m`.
The two decodings of a sample share the consumer's faults.
Pairing reduces the variance of an excess when the failures with and without the block error are positively correlated, as when the consumer's own faults cause most failures.
It does not guarantee a smaller standard error than independent estimates.
The baseline `f(0, g)` is the fraction of all samples of fault weight `g`, across all requested Pauli types and sampled physical error weights, that fail without their block errors.
Each sample contributes to one excess cell at most, and to the baseline.

A cell `(m, g)` that no sample reaches counts as certain failure.
It gets the excess `1 - f(0, g)`, a standard error of zero, and a sample count of zero in `excess_num_samples`, so its uncertainty is that of `f(0, g)`.
This choice is conservative.
Some residual weights have no errors, such as the Z-type residual weight 2 of the Steane code's logical `|0>` state.
Every histogram gives such a residual weight probability zero, so its unsampled cells do not change `Φ`.
For a reachable residual weight, a zero count means that the calibration needs more samples, or a smaller `max_residual_weight` or `G`.

`Δ_s(m, g)` averages over the errors of residual weight `m` that this procedure draws.
The procedure gives each sampled physical weight the same number of trials and samples uniformly within that weight.
It then pools the trials whose errors reduce to residual weight `m`.
This ensemble generally differs from a uniform distribution over the equivalence classes of residual weight `m`.
It also differs from the errors that a particular preparation circuit leaves, which approximation 3 [below](#approximations-and-uncertainty) concerns.

### Standard errors of a simulated calibration

The standard errors of `Consumer.calibrate` are posterior standard deviations under a Jeffreys prior.
A cell in which no sample changed, or a baseline without observed failures, does not show that a probability is zero.
These standard errors are therefore positive for every sampled cell and baseline, even for a single sample.
If `k` of the `K` samples of fault weight `g` fail, the baseline has the posterior `Beta(k + 1/2, K - k + 1/2)`.
If `n₊`, `n₀`, and `n₋` samples of a cell have the paired changes `1`, `0`, and `-1`, the probabilities `p₊`, `p₀`, and `p₋` of these changes have the posterior `Dirichlet(n₊ + 1/2, n₀ + 1/2, n₋ + 1/2)`.
The excess of the cell is `p₊ - p₋`.
With `a₊ = n₊ + 1/2`, `a₋ = n₋ + 1/2`, and `A = n₊ + n₀ + n₋ + 3/2`, its posterior variance is

```text
Var(p₊ - p₋) = [a₊ (A - a₊) + a₋ (A - a₋) + 2 a₊ a₋] / (A² (A + 1)).
```

The point estimates are the sample means: the mean paired change for an excess, and the failure fraction for the baseline.
The prior changes only the standard errors.

The posterior of a cell treats its samples as independent draws from the calibration ensemble of residual weight `m`.
This is approximate, because the calibration draws a fixed number of samples for each physical error weight `w`, rather than a random `w` for each sample.
The fixed numbers remove the variation in the mix of physical weights, so the standard error is typically conservative.
Within one physical weight `w`, the numbers of samples that reach the residual weights are multinomial.
Excess cells of different residual weights are therefore not exactly independent, and the baseline shares their samples.

The direct check `Consumer.estimate_excess_failure_probability`, described [below](#how-to-validate), uses the same posterior for the paired changes of its accepted shots.
Each of its shots is rejected, or accepted with a paired change of `1`, `0`, or `-1`.
The Jeffreys prior `Dirichlet(1/2, 1/2, 1/2, 1/2)` over these four outcomes gives the probabilities of the changes, conditioned on acceptance, the posterior `Dirichlet(n₊ + 1/2, n₀ + 1/2, n₋ + 1/2)`.
Here `n₊`, `n₀`, and `n₋` count the accepted shots with each change.
Three components of a Dirichlet distribution, divided by their sum, have the Dirichlet distribution of their parameters, independently of the fourth component.
Rejected shots therefore do not affect this posterior.

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
   Each excess `c_s(m) - c(0)` averages over the [calibration's errors](#how-it-is-sampled) of residual weight `m`, not over those of the preparation circuit.
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
   Calibrating a consumer built at the target noise avoids this approximation.

The standard errors cover the sampling of the calibration and of the histograms, propagated to first order.
They exclude the approximations above.
The reported standard error of `Φ` combines three contributions in quadrature:

- the sampling errors of the histograms, which are independent between Pauli types;
- the standard errors of the cells of the excess tables, treated as independent;
- the error of the baseline.

Every certain failure depends on the baseline, whether it lies beyond a table or in a cell with a sample count of zero.
An unsampled cell has the excess `1 - f(0, g)`, so its uncertainty is that of `f(0, g)`.
The error contributions of each baseline `f(0, g)` therefore add linearly across the cells, residual weights, and Pauli types that use it, before the quadrature sum.
The baseline estimates for distinct `g` are treated as independent.
A calibration supplies only marginal standard errors.
The calculation therefore omits covariances between excess cells and the pooled baseline, and any covariance among excess cells.
Its reported uncertainty is approximate when those terms matter.
Importance-sampled histogram errors can be optimistic when few samples reach the outcomes that matter.
Check the histogram's `num_samples` when an error looks too small.

## Supplying a consumer's response

- **A Poisson profile.**
  `ConsumerProfile.poisson(expected_fault_weight, t)` models a consumer that fails exactly when `m + g > t` for a Poisson-distributed `g`.
  `estimate_downstream_fault_weight` estimates the expected fault weight from an operation-count preset.
  Score the profile with `ResidualWeightHistogram.get_excess_failure_probability` on a histogram computed with `pauli=None`.
  Its hard-threshold response and Poisson approximation make it an approximate score rather than a calibrated one or a bound.
- **A simulated consumer.**
  `Consumer(block_preparation, circuit, decoder).calibrate(...)` estimates [stratified tables](#how-it-is-sampled).
  The consumer can be any noisy Clifford circuit that uses a block in a CSS stabilizer state.
  Its decoder can be an observable decoder or a decoder specification such as `decoders.mwpm(decompose_errors=True)`.
  A specification is built from the detector error model of the consumer, which keeps its detector coordinates and Stim's suggested decompositions of errors.
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

- **Direct simulation.**
  `Consumer.estimate_excess_failure_probability(circuit, ...)` runs the consumer on the errors that the preparation circuit actually leaves.
  It pairs each error with one shot of the consumer's noise, decoded with and without the error.
  Its standard error is a Jeffreys-prior posterior standard deviation, like those of a calibration (see [above](#standard-errors-of-a-simulated-calibration)).
  It samples the noise channels of both circuits with Stim, and keeps the joint X and Z parts of each error and the circuit's own errors.
  It therefore probes approximations 1, 2, and 3 together, and a discrepancy does not identify its cause.
  Its `pauli` option keeps one part of each error.
  Within their reported sampling errors, compare the whole-error estimate with the sum of the X-only and Z-only estimates.
  Their difference estimates the X–Z interaction neglected by approximation 2.
  Compare the calibrated estimate with direct simulation when introducing a new consumer, code family, or noise regime.
  Direct simulation must be repeated for every circuit, and its cost grows as failures become rarer.
- **Fault tolerance of the consumer.**
  For a consumer that tolerates `t` faults in total, the calibrated excesses `Δ_s(m, g)` with `m + g ≤ t` should vanish.
- **Truncation and coverage.**
  `truncated_probability` should be small compared with the excesses of interest.
  `excess_num_samples` should be nonzero for every reachable cell that a histogram gives weight.
- **Exact small-instance comparison.**
  Compare calibrations, direct estimates, and histograms with exact calculations on a small controlled instance.
