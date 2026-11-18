# Ranking certified preparations against direct simulation

This experiment asks whether two decoder-free scores of state preparation circuits order certified circuits as direct simulation does, and at what cost relative to it.
The scores are Phi_unc, the excess failure probability of a residual-weight histogram for a consumer that corrects t errors of any kind, and Phi_cal, the same for a calibrated consumer (qLDPC's `ResidualWeightHistogram` and `ConsumerCalibration`).
The reference, Phi_true, is the excess failure probability that an accepted preparation causes in one fixed noisy consumer with one fixed decoder, estimated by stratified direct simulation.
Two trivial baselines, the acceptance probability alpha and Pr[R > 0 | A], are scored beside them.
The analysis is descriptive, and its first-order standard errors are not confidence bounds.

## Candidates

The candidates are the circuits of the certification experiment's `circuits.json` of families A-ft and C-goto whose codes have t = 1: 19 circuits in six groups, a group being the circuits that prepare the same signed state of the same code.
Every task runs for every candidate, and the analysis admits a candidate only if the certification archive certifies it at K = t, that is, its verdict at K = t + 1 is "certified" or "violation at t + 1", with no disagreement.
A group with fewer than two admitted candidates is dropped.
The experiment reads certification's inputs and code files from `../certification/circuit_data/`, which the data repository provides, and changes nothing there.

## Consumer and noise

The consumer of a group is one noisy round of syndrome measurement followed by a noisy Z-basis readout of all data qubits: qLDPC's Z-basis memory experiment of the group's code, built from its code file, without the resets of the data qubits.
A noiseless block preparation, Stim's circuit for the group's signed state, puts data qubit i in the state of output qubit i, and the observables are the code file's logical Z operators.
The decoder is BP-OSD (`OSD_CS`, order 7), built from the consumer's detector error model at each rate.
Preparation and consumer get `DepolarizingNoiseModel(p)` at each rate of `config.json`.
The X checks of this consumer have no detectors, so Z errors on the block change nothing, and Phi_cal uses one X-type table.

## Arms

| Arm | Tasks | Content |
| --- | --- | --- |
| `histogram` | candidate and rate | the total-weight and X-type residual-weight histograms |
| `calib` | group and rate | the X-type calibration of the consumer, its fault distance, the consumer circuit with its SHA-256, and the time of the four scores |
| `direct` | candidate and rate | the stratified reference of Phi_true ([`direct.py`](direct.py)) |
| `exact` | circuits of the steane group | the exact leading-order coefficients of Phi_true and Phi_cal, with the decoder frozen at p = 10^-4 |
| `mc_check` | two circuits at p = 5 x 10^-3 | Phi_true by qLDPC's plain paired Monte Carlo, against `direct` |
| `repro` | one `histogram` and one `direct` task, twice each | the same outputs from the same seed |

[`experiment.py`](experiment.py) describes the arms and their seeds, and [`config.json`](config.json) holds the rates, sample sizes, limits, and the circuits of `exact`, `mc_check`, and `repro`.
Each task records the CPU seconds of its setup, of its exactly listed strata, and of its sampled strata or cells, for the cost comparison.
The scores are computed by the analysis from the stored histograms and calibrations.

## Commands

```bash
python experiments/state_prep_certification/ranking/acceptance.py --local
python experiments/state_prep_certification/run_node.py ranking --local --subset smoke --approved-cpu-hours 20
python experiments/state_prep_certification/ranking/analyze.py --label-dir DIR --certification-archive PATH
```

On the node, `node.sh ranking --approved-cpu-hours H` runs the experiment, as the [shared README](../README.md) describes; it runs the acceptance tests RT1, RT2, and RT8 first.
A bundle for ranking must also name `certification`, whose entry carries the inputs and code files: `bundle.py certification ranking --out DIR`.
`analyze.py --archive PATH --certification-archive PATH` reads the ranking archive and the certification archive of subset `main` that come back, checks the certification archive's provenance, and writes CSV and JSON tables; its docstring lists them.
