# Exact certification of bounded fault propagation

This experiment shows what exact certification of bounded fault propagation makes possible on published state preparation circuits, and how its cost behaves against the size of the problem.
It measures which circuits certify within the time limit, the number of faults that the listing considers against all accepted fault combinations, the smallest counterexamples where violations exist, and the cost of the two weight oracles on identical queries.
Its one baseline is internal: brute force that queries every accepted fault combination with the certifier's own oracle.
It compares with no other tool; its internal checks are the agreement of the Brouwer-Zimmermann (BZ) and connected-cluster (CC) oracles, the agreement of the certifier with the baseline, and brute force on small circuits.

## Circuits

The circuits are fixed by rules and public sources, and no timing selects any of them.
`circuits.py --build` makes from the sources a certification input for each circuit that it accepts: the Stim circuit that the experiment certifies on a Linux node without network access.
`circuits.json` lists every circuit, with its source, derivation, certify options, and the SHA-256 of its input, or the reason that `--build` excluded it.
`circuits.json` also lists the parameters of each circuit's detector error model (DEM), which `--build` gets by running the certifier with its fault search replaced by a recorder, so that no oracle runs: the error mechanisms N, the post-selection detectors p, the reduced mechanisms N_0 (one per distinct nonzero signature of flags and syndrome), the unflagged ones z, the flag rank r, K' = min(K, r + 1), and U, the sum over 1 <= m <= K of C(N_0, m).

No circuit is committed to qLDPC.
Every circuit lives in `circuit_data/`, which Git ignores: the circuits fetched from MQT QECC in `mqt/`, the hand-translated circuits in `translated/`, the synthesized circuits in `synthesized/`, and the inputs with `circuits.json` in `inputs/`.
The offline bundle carries `inputs/` and `synthesized/`.
The experiment runs with whatever `circuits.json` and inputs `--build` wrote: `circuits.json` enters the label of each run, and each label keeps a copy of it, from which the analysis reads the circuits.

| Family | Circuits | Source |
| --- | --- | --- |
| A-ft, A-unv | MQT QECC's verified and unverified \|0...0> preparations of ten codes | MQT QECC, `scripts/ft_stateprep/eval/circuits/`, at the commit of `config.json` |
| B-cat | MQT QECC's cat-state preparations `ft_ghz_<w>_<t>` | MQT QECC, `scripts/cat_states/circuits/` |
| C-goto | Goto's flagged Steane preparation | `circuit_data/translated/goto_steane.stim` |
| D-synth | verified circuits synthesized with MQT QECC, made outside qLDPC | `circuit_data/synthesized/` |
| E-chen | Chen et al.'s 8-qubit cat preparation, translated literally | `circuit_data/translated/chen_cat8.stim` |
| W | one weakened variant of each circuit of A-ft, C-goto, and D-synth | derived by `circuits.py --build` |

The synthesized circuits are made outside qLDPC, by the tooling that keeps them, because the synthesis needs MQT QECC and qiskit, which qLDPC does not depend on.
`circuit_data/synthesized/provenance.json` records each synthesis call, with its code's [[n, k, d]] and the release of MQT QECC that it used; `--build` takes every circuit that it records as synthesized, and `circuits.json` labels each D-synth circuit with its release.

Each certification input prepends resets to qubits that start without one.
Each B-cat circuit gets a detector for each pair of consecutive measurement results, on which it is post-selected; E-chen's verification measurements are flags, as in the other families.
The node applies `DepolarizingNoiseModel(1e-3)`, CSS weight, and K = t + 1.

## Arms

| Arm | What it does |
| --- | --- |
| `cert_bz`, `cert_cc` | `certify_bounded_fault_propagation` with the BZ or CC oracle, instrumented by `instrument.py` |
| `brute_bz` | the same BZ certification with the fault search replaced by the brute-force baseline of `baseline.py`, only where U is within the cap of `config.json`, 10^9 |
| `shadow` | BZ certification in which a CC oracle also answers every query, each timed |
| `count_accepted`, `count_listing` | the accepted fault combinations A_m for m <= K, and the counts of the listing, by `counting.py` |
| `ground_truth` | brute force with residual weights from the whole stabilizer group, on the small circuits that `circuits.json` marks |

The baseline enumerates every accepted fault combination of weight at most K over the N_0 reduced mechanisms, and queries each with the certifier's own oracle, built from the same arguments; only the fault search differs from `cert_bz`.
U bounds both the combinations and the rows that it queries, and is fixed by the circuit's parameters, so that a circuit over the cap gets no task and is reported as "not run (U > 1e9)", never as a timeout.

## Workflow

On the workstation, with `circuit_data/` holding the hand-translated and synthesized circuits:

```bash
python experiments/state_prep_certification/certification/circuits.py --fetch
python experiments/state_prep_certification/certification/circuits.py --build
python experiments/state_prep_certification/certification/circuits.py --check
python experiments/state_prep_certification/certification/acceptance.py --local
python experiments/state_prep_certification/run_node.py certification --local --subset smoke --approved-cpu-hours 22
```

`--fetch` downloads MQT QECC's circuits, and `--build` writes the inputs, with the licenses and `circuits.json`, to `circuit_data/inputs/`.
`--check` checks the inputs' hashes and DEM parameters against `circuits.json`, and the states of the D-synth inputs; where the fetched sources are present, as on the workstation, it also rebuilds the inputs from them in a scratch directory and compares.
On the node, the experiment runs the same check before any task, copies `circuits.json` into the label directory, and never imports MQT or qiskit.

The subset `smoke` has a ceiling of about 22 CPU-hours, which `--approved-cpu-hours` must cover, although its runs take minutes; `run_node.py certification --subset smoke --dry-run` prints it.

On the node, `node.sh certification --approved-cpu-hours H` runs the subset `main` (certification by both oracles, the brute-force baseline where U is within the cap, the shadow run, and the counts), and `--subset check` the ground truth by brute force, in a label directory and archive of its own.
The ceiling of `main` follows from `circuits.json`: each run costs at most its start-up cap, its limit, its post-processing allowance, and the grace, and a fast run of `cert_bz`, `cert_cc`, or `brute_bz` gets two repeats.
`run_node.py certification --dry-run` prints the tasks and the ceiling of the current `circuits.json`.
On the workstation, `analyze.py --archive PATH --check-archive PATH` writes the tables and counterexamples; without the archive of `check`, the tables report the ground truth as not run.
It takes the circuits and their parameters from the label's copy of `circuits.json`, and stops if the label of `check` ran with another one.
If a label does not match its `circuits.json` and the working tree's `config.json` and `pins.json`, it prints a warning and analyzes the label anyway.
Its outputs include `listing_table.csv`, the listed faults and oracle rows against all accepted combinations per family and weight; and `counterexamples.json`, every counterexample and the box of three example counterexamples.
It reports every task without a row as missing, and leaves every circuit with a disagreement out of the verdict counts, medians and maxima, `listing_table.csv`, and the box.
Where A_m was not computed, `listing_table.csv` divides by C(N_0, m) instead, so that its ratios are lower bounds on those over A_m, as its column `ratio_is_lower_bound` marks.
The acceptance test AT8 runs the analysis on synthetic rows: a listing count that timed out, disagreements with the baseline and the ground truth, and a task without a row.

## Attributions and licenses

MQT QECC (MIT) provides the A and B circuits and synthesized the D-synth circuits; its license travels with the inputs in `circuit_data/inputs/MQT_QECC_LICENSE`.
Chen et al.'s cat preparation comes from [vftqecc](https://github.com/vftqc/vftqecc), under the MIT license, whose notice lies beside it as `circuit_data/translated/VFTQECC_LICENSE` and travels with the inputs; and Goto's circuit from Goto, Sci. Rep. 6, 19578 (2016).
