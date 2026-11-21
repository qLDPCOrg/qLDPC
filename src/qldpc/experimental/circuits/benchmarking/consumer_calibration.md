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
One calibration of a consumer then scores any preparation circuit of the same ideal block, whatever its noise and post-selection.
Use the figure of merit to compare preparation circuits for one fixed consumer.
It is not a logical error rate, a fault-tolerance certificate, or a bound on the failure of a real computation.

## Two-call workflow

The [built-in consumer](#the-built-in-consumer) runs a short sequence of steps of swap-type Steane quantum error correction (QEC).
With it, two calls score a preparation circuit:

```python
calibration = get_steane_swap_consumer(code, noise_model).calibrate()
phi, standard_error = calibration.estimate_excess_failure_probability(preparation_circuit)
```

The first call calibrates the consumer by simulation, which is the expensive step.
The second call scores a preparation circuit from the [residual weights](#residual-weights) of the errors that it leaves, without simulating the consumer again.
Steane QEC consumes freshly prepared code blocks, and the prepared block enters the consumer in place of one of these fresh blocks.
We call the prepared block the **block under test**.
By default, it is a fresh logical `|0>` block, and with `basis="X"` it is a fresh logical `|+>` block (see [The built-in consumer](#the-built-in-consumer)).
The rest of this guide defines `phi`, written `Φ` below, and explains the calibration and its approximations.
The [state-preparation benchmarking notebook](../../../../../examples/experimental/logical_state_preparation.ipynb) demonstrates the workflow.

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
The `Consumer` docstring states the requirements on these observables, which need not be logical operators.
The consumer fails in a shot if the decoder mispredicts any of these observables.
For other consumers, the caller defines the failure event and supplies its probabilities.
A **profile** `c_s(m)` is the failure probability for a type-`s` block error of residual weight `m`.
It averages over the consumer's own noise and over the calibration ensemble of errors with that residual weight.
The **excess** `c_s(m) - c_s(0)` is the signed cost of residual weight `m`.
It is normally positive, but an error can occasionally turn a failure of the consumer into a success.
A measured or supplied profile therefore need not increase with `m`.

## The built-in consumer

`get_steane_swap_consumer(code, noise_model)` builds a consumer from swap-type Steane QEC, as in Figure 4d of [arXiv:2603.04584](https://arxiv.org/abs/2603.04584).
Swap-type Steane QEC teleports the logical qubits of a CSS code from block to block.
The **current block** holds the logical qubits.
Each step teleports them onto a fresh block, which becomes the current block of the next step.
Steps come in two kinds:

- A Z-correction step applies a transversal CNOT from the current block to a fresh logical `|0>` block, and measures the current block in the X basis.
  The X-type checks of the measured block define detectors, which detect Z errors.
- An X-correction step applies a transversal CNOT from a fresh logical `|+>` block to the current block, and measures the current block in the Z basis.
  The Z-type checks of the measured block define detectors, which detect X errors.

A **cycle** of the protocol has one step of each kind, so it consumes one fresh `|0>` block and one fresh `|+>` block.
We call these two places for fresh blocks the **slots** of the cycle.
The consumer runs a **window**, a short run of consecutive steps.
The block under test enters the window as the fresh block of its **entry step**.
The window starts with the logical qubits on an ideal **data block** in logical `|0>`.
It ends with a noiseless **final readout**, which measures the checks of the **last block** to define detectors.
The last block is the block that holds the logical qubits after the last step.
For `basis="Z"`, the following table numbers the steps of the window relative to the entry step.
The [window section](#the-window) explains its last row.

| Step | −2 | −1 | 0 (entry) | +1 | +2 |
| --- | --- | --- | --- | --- | --- |
| Kind | Z-correction | X-correction | Z-correction | X-correction | Z-correction |
| Fresh block | logical 0 | logical + | block under test | logical + | logical 0 |
| Errors of the block under test that this step decodes | none | none | Z, first decoding | X | Z, second decoding |

The steps do not apply the logical Pauli corrections that teleportation calls for.
Instead, each observable includes the logical outcomes that the steps measure, which tracks these corrections in software.
Each observable compares a logical X or Z operator of the data block, at the start of the window, with the same operator of the last block.
Without noise, the window acts as the identity on the logical qubits, so each observable is deterministic.

The consumer's `correctable_weight` defaults to `t = ⌊(d - 1) / 2⌋` for a code of distance `d`, so that `calibrate` tabulates residual weights up to `t + 1` by default.

### The window

A step decodes an error on the block under test when the block that it measures carries the error, or a copy that a CNOT made, and its measurement detects the error's type.
A transversal CNOT copies X errors from its control block to its target block, and Z errors from its target block to its control block.
So for `basis="Z"`, steps 0 and +2 both decode the block's Z errors, and step +1 decodes its X errors, as the last row of the table lists.
For `basis="X"`, X and Z are exchanged.
No error on the block under test reaches the steps before entry or the final readout.

The default decoder decodes each step on its own.
Without faults of the consumer, its two decodings of the twice-decoded errors see the same syndrome, so they make the same prediction, and the two predictions cancel in the observables.
The errors also flip the logical outcomes that the two steps measure, and these flips cancel too.
With the default decoder, Z errors on a fresh `|0>` block, and X errors on a fresh `|+>` block, therefore cause a failure only together with faults of the consumer that make the two decodings differ.

The errors that noise in a step leaves are decoded in that step, in one of the next two steps, or in the final readout.
The errors of the block under test are decoded in steps 0 to +2, so noise in steps −2 to +2 can leave errors that one of steps 0 to +2 also decodes, and noise in other steps cannot.
The window runs these five steps, each with the noise of the noise model.
A further noisy step before or after the window adds error mechanisms whose errors no step decodes together with the block's errors.
For a code of distance at least 3, the default decoder corrects every fault of weight one, so it corrects each such mechanism when it is the consumer's only fault, without changing the cost of the block's errors.

So with the default decoder and a code of distance at least 3, further steps change the excesses `c_s(m) - c_s(0)`, and so `Φ`, only at second order in the consumer's error probabilities.
For a code of distance less than 3, the default decoder can fail to correct a fault of weight one, and further steps can change the excesses at first order.
A decoder built from a decoder specification decodes the syndromes of the whole window together, so neither the cancellation of twice-decoded errors nor the second-order bound on the effect of further steps is guaranteed for it.

### What a calibration measures

A calibration of the consumer measures the cost of one imperfect fresh block in the slot of the block under test.
It makes these approximations of how the consumer stands in for a larger computation:

- The window is finite: it starts from an ideal data block and ends with a noiseless readout.
  [The window](#the-window) says when this matters only at second order in the consumer's error probabilities.
- The consumer prepares the data block and every fresh block other than the block under test without noise, with `circuits.get_encoding_circuit`, so they are ideal.
  The block under test also starts in its ideal state, and the calibration adds its errors.
- Only one fresh block is imperfect at a time, so the calibration neglects the interplay of errors on several fresh blocks.
- Noise acts on the CNOTs and measurements of each step, and any idle noise that the noise model specifies acts on the two blocks of the step.
  Noise outside the steps is neglected, such as the idling of the data block while a fresh block is prepared, or of a fresh block before its step.

To score a scheme that prepares blocks for both slots, calibrate both slots, with `basis="Z"` and `basis="X"`.
Score the `|0>` preparation circuit with the first calibration and the `|+>` preparation circuit with the second, then add the two scores.
Suppose that the code is self-dual, such as the Steane code, and that transversal Hadamard gates leave the noise model unchanged, as for `DepolarizingNoiseModel`.
Then, with the default decoder, the two slots have the same costs, with X and Z exchanged.
For comparison, the [Steane-QEC notebook](../../../../../examples/logical_error_rates/6_steane_qec.ipynb) simulates a full cycle, with noisy flagged preparations of all its fresh blocks at once.

### Decoders

The default decoder decodes the syndrome of each step, and the final syndrome, separately, with a minimum-weight lookup table for each type of check.
A table has an entry for every syndrome of the checks of one type, so the default decoder suits small codes.
The docstring of `get_steane_swap_consumer` gives its size limits and the layout of the window's detectors.
Other decoders are passed as decoder specifications, which the consumer builds from the detector error model of its window.
The detectors of the window come from the code's own checks, so these decoders see the weight and locality of those checks.
For example, `decoders.mwpm(decompose_errors=True)` suits a code whose checks give a matching graph, such as a surface code.
Such a decoder decodes the whole window together, so neither the cancellation of twice-decoded errors nor the second-order bound on the effect of further steps, which [the window section](#the-window) describes, is guaranteed for it.

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
The properties `fault_weight_probabilities` and `truncated_probability` of a `ConsumerCalibration` give `Pr[g]` for `g ≤ G` and `Pr[g > G]`.
Its method `get_failure_probabilities(s)` gives `c_s(0), c_s(1), ...` for the rows of the table of type `s`.
The values are not clipped, so sampling noise in estimated cells, or rounding, can put them slightly outside [0, 1].

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

`Φ` estimates the change in the failure probability of the calibrated consumer, per accepted preparation, relative to an ideal block, under the approximations below.
How well a consumer stands in for a larger computation is a separate question.
For the built-in consumer, [What a calibration measures](#what-a-calibration-measures) lists the approximations of this second kind.

1. **Independent error mechanisms.**
   The consumer's and the preparation's noise channels are replaced by independent DEM mechanisms, as described [above](#error-mechanisms-faults-and-acceptance).
2. **Separate Pauli types.**
   The X and Z contributions are added, which neglects their interaction when both parts of an error occur in one shot.
   Examples are the two parts of a Y error, and X and Z errors that one fault leaves on different qubits.
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

- the sampling errors of the histograms, treated as independent between Pauli types, as they are when each histogram has its own seed;
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

`ConsumerCalibration.get_standard_error_parts` splits the reported standard error into a histogram part, from the sampling errors of the histograms, and a calibration part, from the errors of the cells and the baseline.
To first order, more histogram samples reduce only the histogram part, and more calibration samples only the calibration part.
Neither part includes the bias of cells with a sample count of zero, which more calibration samples remove (see approximation 4).

To compare circuits `a` and `b` scored with one calibration, use `get_excess_failure_probability_difference` on the histograms of each circuit, one per calibrated Pauli type from `estimate_residual_weight_histogram`, as for `get_excess_failure_probability`.
It returns `Φ(a) - Φ(b)` and its standard error.
An error of a cell or a baseline moves both scores, so it largely cancels from their difference when the two circuits have similar residual-weight distributions.
The two standard errors, added in quadrature, can therefore greatly overstate the standard error of the difference.
The method keeps this cancellation, and adds the histogram parts of the two circuits in quadrature.
It treats every histogram of either circuit as independent of the others, so give each histogram its own seed, or `seed=None`.
The certain failures of approximation 4 are a bias that no standard error includes.
Apart from the statistical error of the baseline, this bias can only raise each score, but it can move a difference either way.

## Supplying a consumer's response

- **The built-in consumer.**
  `get_steane_swap_consumer` builds a simulated consumer from a window of swap-type Steane QEC steps; see [the built-in consumer](#the-built-in-consumer) and the [two-call workflow](#two-call-workflow).
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
  Direct simulation runs the same consumer as the calibration, so it does not test how well the consumer models a larger computation.
  Compare the calibrated estimate with direct simulation when introducing a new consumer, code family, or noise regime.
  Direct simulation must be repeated for every circuit, and its cost grows as failures become rarer.
- **Interaction of the X and Z parts.**
  The `pauli` option of `Consumer.estimate_excess_failure_probability` keeps one part of each error.
  Compare the whole-error estimate with the sum of the X-only and Z-only estimates.
  Their difference estimates the X–Z interaction neglected by approximation 2.
  With the same `seed`, the three runs share their shots, so the difference has a much smaller error than the quadrature sum of their reported standard errors.
- **Fault tolerance of the consumer.**
  For a consumer that tolerates `t` faults in total, the calibrated excesses `Δ_s(m, g)` with `m + g ≤ t` should vanish.
- **Truncation and coverage.**
  `truncated_probability` should be small compared with the excesses of interest.
  `excess_num_samples` should be nonzero for every reachable cell that a histogram gives weight.
- **Exact small-instance comparison.**
  Compare calibrations, direct estimates, and histograms with exact calculations on a small controlled instance.
