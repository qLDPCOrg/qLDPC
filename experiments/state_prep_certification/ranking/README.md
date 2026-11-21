# Ranking certified preparations against direct simulation

This experiment asks whether two decoder-free scores of state preparation circuits order certified circuits as direct simulation does, and at what cost relative to it.
Both scores combine a circuit's residual-weight histogram, the distribution of the residual weight R of an accepted shot, with a model of the consumer (see [Consumer and noise](#consumer-and-noise)).
Phi_unc assumes that the consumer fails when R plus its own fault weight exceeds the weight t that the code corrects, and Phi_cal uses a calibration of the consumer by simulation (qLDPC's `ResidualWeightHistogram` and `ConsumerCalibration`).
The reference, Phi_true, is the excess failure probability that an accepted preparation causes in one fixed noisy consumer with one fixed decoder, estimated by stratified direct simulation.
Two trivial baselines are scored beside them: the probability alpha that post-selection accepts a shot, and Pr[R > 0 | A], the probability that an accepted shot has a nonzero residual weight.
The analysis is descriptive, and its first-order standard errors are not confidence bounds.

## Candidates

The candidates are the circuits of the certification experiment's `circuits.json` of families A-ft and C-goto whose codes have t = 1: 19 circuits in six groups, a group being the circuits that prepare the same signed state of the same code.
Every task runs for every candidate, and the analysis admits a candidate only if the certification archive certifies its bounded fault propagation up to fault weight t: its verdict, checked up to fault weight t + 1, is "certified" or "violation at t + 1", and the certification arms and checks do not disagree.
A group with fewer than two admitted candidates is dropped.
Groups are named by their code file, such as `steane` for the Steane code's circuits and Goto's.
The experiment reads certification's inputs and code files from `../certification/circuit_data/`, which the data repository provides, and changes nothing there.

## Consumer and noise

The consumer of a group is qLDPC's Steane-swap consumer (`get_steane_swap_consumer`) of the group's code, built from its code file: five steps of swap-type Steane error correction around the entry of a fresh logical |0> block, which is the block under test ([`consumer.py`](consumer.py)).
Its block's qubits follow the output qubits' order, and its ideal state is the group's signed state, which `consumer.py` checks.
Both X and Z errors on the block reach its detectors, so Phi_cal uses a calibration of both, and the reference passes each preparation's output error, X and Z parts together, to the consumer.
The decoder is qLDPC's default for this consumer, which decodes the syndrome of each step on its own with a table of lightest errors.
At each rate p of `config.json`, the consumer gets `DepolarizingNoiseModel(p)`, and each preparation the noise model of its options in `circuits.json` at rate p, which for every candidate is also `DepolarizingNoiseModel(p)`.

## Arms

| Arm | Tasks | Content |
| --- | --- | --- |
| `histogram` | candidate and rate | the total-weight residual-weight histogram, and one of each Pauli type |
| `calib` | group and rate | the calibration of the consumer, its fault distance, the consumer circuit with its SHA-256, and the time of the scores and baselines |
| `direct` | candidate and rate | the stratified reference of Phi_true ([`direct.py`](direct.py)) |
| `exact` | circuits of the steane group | the exact leading-order coefficient of Phi_true, beside the reference at p = 10^-4 |
| `mc_check` | two circuits at p = 5 x 10^-3 | Phi_true by qLDPC's plain paired Monte Carlo, against `direct` |
| `repro` | one `histogram` and one `direct` task, twice each | the same outputs from the same seed |

[`experiment.py`](experiment.py) describes the arms and their seeds, and [`config.json`](config.json) holds the rates, sample sizes, limits, and the circuits of `exact`, `mc_check`, and `repro`.
The `histogram` and `direct` tasks record the CPU seconds of their setup, their exactly listed strata, and their sampled strata, and `calib` those of its setup and its sampled cells, for the cost comparison.
The scores are computed by the analysis from the stored histograms and calibrations.
Every arm reaches galois's GF(2) arithmetic, which galois compiles in each process without caching it on disk, so every task first compiles it, untimed (`_compile_galois` in [`experiment.py`](experiment.py)).

## Commands

```bash
python experiments/state_prep_certification/ranking/acceptance.py --local
python experiments/state_prep_certification/run_node.py ranking --local --subset smoke --approved-cpu-hours 32
python experiments/state_prep_certification/ranking/analyze.py --label-dir DIR --certification-archive PATH
```

On the node, `node.sh ranking --approved-cpu-hours H` runs the experiment, as the [shared README](../README.md) describes; it first runs the acceptance tests that check the inputs, the consumers, and the node's modules (RT1, RT2, and RT8 of [`acceptance.py`](acceptance.py)).
A bundle for ranking must also name `certification`, whose entry carries the inputs and code files: `bundle.py certification ranking --out DIR`.
`analyze.py --archive PATH --certification-archive PATH` reads the ranking archive and the certification archive of subset `main` that come back, checks the certification archive's provenance, and writes CSV and JSON tables; its docstring lists them.
`analyze.py`'s docstring says how it treats a row with `compiled_in_timing` set (see the [shared README](../README.md#runs)).
