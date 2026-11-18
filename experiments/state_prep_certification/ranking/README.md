# Ranking certified preparations against direct simulation

This experiment asks whether two decoder-free scores of state preparation circuits order certified circuits as direct simulation does, and at what cost relative to it.
Both scores combine a circuit's residual-weight histogram, the distribution of the residual weight R of an accepted shot, with a model of the consumer, defined below.
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

The consumer of a group is one noisy round of syndrome measurement followed by a noisy Z-basis readout of all data qubits: qLDPC's Z-basis memory experiment of the group's code, built from its code file, without the resets of the data qubits.
A noiseless block preparation, Stim's circuit for the group's signed state, puts data qubit i in the state of output qubit i, and the observables are the code file's logical Z operators.
The decoder is BP-OSD (`OSD_CS`, order 7), built from the consumer's detector error model at each rate.
At each rate p of `config.json`, the consumer gets `DepolarizingNoiseModel(p)`, and each preparation its own noise model of `circuits.json` at rate p, which is that model for every candidate.
The X checks of this consumer have no detectors, so Z errors on the block change nothing, and Phi_cal uses a calibration of X-type errors only.

## Arms

| Arm | Tasks | Content |
| --- | --- | --- |
| `histogram` | candidate and rate | the total-weight and X-type residual-weight histograms |
| `calib` | group and rate | the X-type calibration of the consumer, its fault distance, the consumer circuit with its SHA-256, and the time of the scores and baselines |
| `direct` | candidate and rate | the stratified reference of Phi_true ([`direct.py`](direct.py)) |
| `exact` | circuits of the steane group | the exact leading-order coefficients of Phi_true and Phi_cal, with the decoder frozen at p = 10^-4 |
| `mc_check` | two circuits at p = 5 x 10^-3 | Phi_true by qLDPC's plain paired Monte Carlo, against `direct` |
| `repro` | one `histogram` and one `direct` task, twice each | the same outputs from the same seed |

[`experiment.py`](experiment.py) describes the arms and their seeds, and [`config.json`](config.json) holds the rates, sample sizes, limits, and the circuits of `exact`, `mc_check`, and `repro`.
The `histogram` and `direct` tasks record the CPU seconds of their setup, their exactly listed strata, and their sampled strata, and `calib` those of its setup and its sampled cells, for the cost comparison.
The scores are computed by the analysis from the stored histograms and calibrations.

## Commands

```bash
python experiments/state_prep_certification/ranking/acceptance.py --local
python experiments/state_prep_certification/run_node.py ranking --local --subset smoke --approved-cpu-hours 20
python experiments/state_prep_certification/ranking/analyze.py --label-dir DIR --certification-archive PATH
```

On the node, `node.sh ranking --approved-cpu-hours H` runs the experiment, as the [shared README](../README.md) describes; it first runs the acceptance tests that check the inputs, the consumers, and the node's modules (RT1, RT2, and RT8 of [`acceptance.py`](acceptance.py)).
A bundle for ranking must also name `certification`, whose entry carries the inputs and code files: `bundle.py certification ranking --out DIR`.
`analyze.py --archive PATH --certification-archive PATH` reads the ranking archive and the certification archive of subset `main` that come back, checks the certification archive's provenance, and writes CSV and JSON tables; its docstring lists them.
