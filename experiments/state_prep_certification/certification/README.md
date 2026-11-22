# Exact certification of bounded fault propagation

This experiment measures exact certification of bounded fault propagation on published state preparation circuits, and how its cost grows with the size of the problem.
It measures which circuits certify within the time limit, the number of faults that the listing considers against all accepted fault combinations, the smallest counterexamples where violations exist, and the cost of the two weight oracles on identical queries.
Its one baseline is internal: brute force that queries every accepted fault combination with the certifier's own oracle.
It compares with no other tool; its internal checks are the agreement of the Brouwer-Zimmermann (BZ) and connected-cluster (CC) oracles (qLDPC's [BZ guide](../../../src/qldpc/codes/distance/brouwer_zimmermann.md) and [connected-cluster guide](../../../src/qldpc/codes/distance/connected_cluster.md)), the agreement of the certifier with the baseline, and brute force on small circuits.
Two applications of the certifier use the same circuits: the exact margin of each circuit, and the deletion of verification measurements under exact certification.

## Circuits

The circuits are fixed by rules and public sources, and no timing selects any of them.
`circuits.py --build` makes from the sources a certification input for each circuit that it accepts: the Stim circuit that the experiment certifies on a Linux node without network access.
`circuits.json` lists every circuit, with its source, derivation, certify options, and the SHA-256 of its input, or the reason that `--build` excluded it.
`circuits.json` also lists the parameters of each circuit's detector error model (DEM), which `--build` gets by running the certifier with its fault search replaced by a recorder, so that no oracle runs: the error mechanisms N, the post-selection detectors p, the reduced mechanisms N_0 (one per distinct nonzero signature of flags and syndrome), the unflagged ones z, the flag rank r, K' = min(K, r + 1), and U, the sum over 1 <= m <= K of C(N_0, m).

No circuit is committed.
Every circuit lives in the experiment's data directory, `circuit_data/`, which Git ignores:

- `mqt/`: the circuits that `circuits.py --fetch` copies from MQT QECC at the commit of `config.json`, with MQT QECC's license;
- `translated/`: E-chen's circuit, translated by hand, with vftqecc's license (see [Attributions](#attributions-and-licenses)), and C-goto's, as qLDPC's notebook writes it;
- `synthesized/`: circuits synthesized with MQT QECC, with the code files of their codes.
  Its `provenance.json` records each synthesis call: the code, its [[n, k, d]], and the MQT QECC release;
- `synthesized_mqt_1.9.0/`: naive verified circuits synthesized with MQT QECC 1.9.0 (its `provenance.json` describes the naive verification), with their code files, in the same form;
- `codes/`: the code files of the A-ft codes;
- `inputs/`: the inputs and `circuits.json`.

The experiment's scripts make `mqt/` (`circuits.py --fetch`) and `inputs/` (`circuits.py --build`).
No script of the experiment makes `translated/`, `synthesized/`, `synthesized_mqt_1.9.0/`, or `codes/`: they are inputs that the experiment takes as given, and checks, as `circuits.py --check` does.
`circuits.py --build` reads `mqt/`, `translated/`, both folders of synthesized circuits, and `codes/`, and writes `inputs/`.
It takes every verified circuit, from every synthesis call but `prep`, that a folder's `provenance.json` records as synthesized, and `circuits.json` labels each synthesized circuit with its MQT QECC release.
The offline bundle carries `inputs/`, `synthesized/`, `synthesized_mqt_1.9.0/`, and `codes/`.
The experiment runs with whatever `circuits.json` and inputs `--build` wrote: `circuits.json` enters the label of each run, and each label keeps a copy of it, from which the analysis reads the circuits.

| Family | Circuits | Source |
| --- | --- | --- |
| A-ft, A-unv | MQT QECC's verified and unverified \|0...0> preparations of ten codes | MQT QECC, `scripts/ft_stateprep/eval/circuits/`, at the commit of `config.json` |
| B-cat | MQT QECC's cat-state preparations `ft_ghz_<w>_<t>` | MQT QECC, `scripts/cat_states/circuits/` |
| C-goto | Goto's flagged Steane preparation | `circuit_data/translated/goto_steane.stim` |
| D-synth | verified circuits synthesized with MQT QECC | `circuit_data/synthesized/` |
| D-synth (MQT 1.9.0) | naive verified circuits synthesized with MQT QECC 1.9.0 | `circuit_data/synthesized_mqt_1.9.0/` |
| E-chen | Chen et al.'s 8-qubit cat preparation, translated literally | `circuit_data/translated/chen_cat8.stim` |
| W | one weakened variant of each circuit of A-ft, C-goto, and both D-synth families | derived by `circuits.py --build` |

C-goto's circuit is the flagged preparation of the Steane code's |0> state with one ancilla qubit for verification, Figure 1c of H. Goto, "Minimizing resource overheads for fault-tolerant preparation of encoded states of the Steane code", Scientific Reports 6, 19578 (2016), doi:10.1038/srep19578, as qLDPC's notebook `examples/logical_error_rates/5_fault_tolerance_certificates.ipynb` writes it.
E-chen's circuit is the 8-qubit cat preparation of K. Chen, Y. Liu, W. Fang, J. Paykin, X.-C. Wu, A. Schmitz, S. Zdancewic, and G. Li, "Verifying Fault-Tolerance of Quantum Error Correction Codes", Computer Aided Verification (CAV 2025), pp. 3-27, doi:10.1007/978-3-031-98685-7_1, arXiv:2501.14380.

Each A-ft, C-goto, and synthesized entry of `circuits.json` also names its code file, `code_file`, with its SHA-256, `code_sha256`.
A code file `<code>_code.json` holds the code's `distance` and its matrices `Hx`, `Hz`, `Lx`, and `Lz`, with one column per output qubit, in output-qubit order.
`circuits.py --check` checks each input's state against these matrices, which split the state's Z-type stabilizers into checks Hz and logical operators Lz as the code's definition in MQT QECC does; the state alone does not determine that split, since the Z-type stabilizers of |0...0>_L span both Hz and Lz.
The code files of the A-ft codes, in `codes/`, follow MQT QECC's definitions of the codes at the commit of `config.json`, and those of the synthesized codes lie beside their circuits; C-goto uses the Steane code's file.
MQT's `RotatedSurfaceCode(5)` at that commit has its X checks where the shipped d = 5 circuits have their Z checks, so its file exchanges Hx with Hz and Lx with Lz.

Each certification input prepends resets to qubits that start without one.
Each B-cat circuit gets a detector for each pair of consecutive measurement results, on which it is post-selected; E-chen's verification measurements are flags, as in the other families.
The node applies `DepolarizingNoiseModel(1e-3)` and K = t + 1.
It leaves `css_weight` at its default, None, under which qLDPC chooses CSS weight for a CSS state.
Every input prepares one, as the top-level `css_weight` of each input in `circuits.json` records.
The ground truth (`counting.brute_force`) requires a CSS state.

## Arms

| Arm | What it does |
| --- | --- |
| `cert_bz`, `cert_cc` | `certify_bounded_fault_propagation` with the BZ or CC oracle, instrumented by `instrument.py` |
| `brute_bz` | the same BZ certification with the fault search replaced by the brute-force baseline of `baseline.py`, only where U is within the cap of `config.json`, 10^9 |
| `shadow` | BZ certification in which a CC oracle also answers every query, each timed |
| `count_accepted`, `count_listing` | the accepted fault combinations A_m for m <= K, and the counts of the listing, by `counting.py` |
| `ground_truth` | brute force with residual weights from the whole stabilizer group, on the small circuits that `circuits.json` marks |
| `margin` | BZ certification with no limit on the fault weight, on the circuits whose flag rank r exceeds t, by `applications.py` |
| `prune`, `prune_sensitivity` | the deletion of the measured qubits of each A-ft, C-goto, and synthesized circuit under exact certification at K = t, by `applications.py`; `prune_sensitivity` tries another order, on the circuits of `prune.sensitivity_inputs` of `config.json` |

The baseline enumerates every accepted fault combination of weight at most K over the N_0 reduced mechanisms, and queries each with the certifier's own oracle, built from the same arguments; only the fault search differs from `cert_bz`.
U bounds both the combinations and the rows that it queries, and is fixed by the circuit's parameters, so that a circuit over the cap gets no task and is reported as "not run (U > 1e9)", never as a timeout.

The margin m* of a circuit is the smallest fault weight of an accepted fault whose residual weight exceeds its fault weight, so that the circuit has bounded fault propagation up to exactly m* - 1.
It is "all" if no such fault exists at any fault weight, which the search proves once it completes fault weight r + 1, or earlier if no flagged mechanism is heavy.
"All" does not mean that the state is protected against any number of faults: every accepted fault still has a residual weight up to its fault weight.
`cert_bz` at K = t + 1 gives m* where it finds a violation or r <= t, and `margin` gives it for the other circuits.
A `margin` run that reaches its limit gives only the largest fault weight k that it completed without a violation, so that m* > k.

A `prune` task first certifies its circuit, the parent, at K = t with BZ, and stops if the parent violates.
It then tries each measured qubit once, by decreasing index of the instruction of its last measurement, with qubits tied by one instruction in the order of their first measurement.
For each, it deletes every instruction target group that involves the qubit, and keeps the deletion if and only if the circuit still prepares the parent's signed state on the parent's output qubits, qLDPC accepts its detectors, and BZ certifies it at K = t.
The final circuit, which can equal the parent, is then certified at K = t by CC, the acceptance probabilities of parent and final circuit are estimated from 10^6 Stim shots each, and the final circuit gets its m*.
`prune_sensitivity` does the same by increasing index of the instruction of the first measurement.
`margin` and the deletion arms send each completed step, a fault weight or a tried qubit, to the runner as a partial result, so that a run killed at its limit keeps its steps; the circuit of a killed deletion run is not a result.

The runner retries any run whose measured call compiled, as the [shared README](../README.md#runs) describes.
No run warms up before timing, apart from `ground_truth`, whose task process first compiles galois's GF(2) arithmetic, untimed (`_compile_galois` in [`counting.py`](counting.py)).
The count and ground-truth arms take their DEM from an untimed certification at fault weight 1.
This certification is preparation, not a warm-up, but it also compiles, or loads from the cache, the code that their measured calls share with it.
The experiment's own compiled code, in `counting.py` and `baseline.py`, is cached on disk.

`analyze.py`'s docstring says how it treats a row with `compiled_in_timing` set (see the [shared README](../README.md#runs)).

## Workflow

On the workstation, with `circuit_data/` holding the inputs that no script of the experiment makes (see [Circuits](#circuits)):

```bash
python experiments/state_prep_certification/certification/circuits.py --fetch
python experiments/state_prep_certification/certification/circuits.py --build
python experiments/state_prep_certification/certification/circuits.py --check
python experiments/state_prep_certification/certification/acceptance.py --local
python experiments/state_prep_certification/run_node.py certification --local --subset smoke --approved-cpu-hours 79
```

`--fetch` downloads MQT QECC's circuits, and `--build` writes the inputs, with the licenses and `circuits.json`, to `circuit_data/inputs/`.
`--check` recomputes each input's hash, DEM parameters, prepared state, and output qubits, which must match `circuits.json`, and refuses an input that does not prepare a CSS state.
It also checks each code file as `code_files.check_code_file` describes, such as that the code's |0...0> is the input's state.
Where the fetched sources are present, as on the workstation, it also rebuilds the inputs from them in a scratch directory and compares.
On the node, the experiment runs the same check before any task, copies `circuits.json` into the label directory, and never imports MQT or qiskit.

The subset `smoke` has a ceiling of about 78 CPU-hours, which `--approved-cpu-hours` must cover, although its runs take minutes; `run_node.py certification --subset smoke --dry-run` prints it.

On the node, `node.sh certification --approved-cpu-hours H` runs the subset `main` (certification by both oracles, the brute-force baseline where U is within the cap, the shadow run, the counts, the margins, and the deletions), and `--subset check` the ground truth by brute force, in a label directory and archive of its own.
The ceiling of `main` follows from `circuits.json`: each run costs at most its start-up cap, its limit, its post-processing allowance, and the grace, and a fast run of `cert_bz`, `cert_cc`, or `brute_bz` gets two repeats.
The runner's retries add to this, as [`common/tasks.py`](../common/tasks.py) explains.
`run_node.py certification --dry-run` prints the tasks and the ceiling of the current `circuits.json`.
On the workstation, run `analyze.py --archive PATH --check-archive PATH`.
Its docstring lists the outputs, among them `summary.csv`, `families.csv`, and `listing_table.csv`.
`listing_table.csv` compares the listed faults and oracle rows with all accepted combinations, per family and weight.
`counterexamples.json` holds every counterexample under `all`, and up to three examples under `examples`, one for each group of families that has one; `analyze.py`'s docstring names the groups and the rule.
Without the archive of `check`, the outputs report the ground truth as not run.
The analysis takes the circuits and their parameters from the label's copy of `circuits.json`, and stops if the label of `check` ran with another one.
If a label does not match its `circuits.json` and the working tree's `config.json` and `pins.json`, it prints a warning and analyzes the label anyway.
It reports every task without a row as missing, and leaves every circuit with a disagreement out of the verdict counts, medians and maxima, `listing_table.csv`, and the example counterexamples.
Where A_m was not computed, `listing_table.csv` divides by C(N_0, m) instead, so that its ratios are lower bounds on those over A_m, as its column `ratio_is_lower_bound` marks.
The acceptance test AT8 runs the analysis on synthetic rows: a listing count that timed out, disagreements with the baseline and the ground truth, a task without a row, rows with `compiled_in_timing` set, and rows of `--local` runs, which the analysis refuses without `--allow-local`, as the [shared README](../README.md#workflow) says.

`analyze_applications.py` adds the margins and the deletions.
`summary.csv` gives each circuit's combined margin: the column `margin` holds the m* or "all" that `cert_bz`, `cert_cc`, and the `margin` arm determine; if none determines it, the column `margin_lower_bound` holds the largest certified fault weight k, meaning m* > k.
The determined values must agree and exceed every certified fault weight; otherwise the circuit is a disagreement, as it is when a deletion parent's verdict differs from `cert_bz`'s or CC does not certify a final circuit.
`families.csv` gets the margin counts: per family, the circuits with m* <= t, m* = t + 1, m* >= t + 2, "all", m* > k, and no information on m*.
`margin_violations.csv` lists the circuits with m* <= t, with the residual weights of the X and Z parts of their counterexamples, since the property certified here bounds both parts, whereas the verification of a circuit's source may target only one part.
`deletion_table.csv` has one row per finished deletion task whose parent is certified at K = t, on a circuit without a disagreement, with the gate counts, measured qubits, measurements, and acceptance probabilities of parent and final circuit; it compares each final circuit only with its parent.
`deletion_notes.json` lists the deletion orders, the parents not certified at K = t, the tasks that did not finish, the disagreements, and the final circuits whose rebuild from the parent's input does not match; `deletion.csv` lists every step, `deletion_tasks.csv` every finished task whose parent is certified at K = t, disagreements included, and `pruned/` holds the final circuits of `deletion_table.csv`, and nothing else: the analysis deletes it with everything in it first.

## Attributions and licenses

MQT QECC (MIT) provides the A and B circuits and synthesized the circuits of both D-synth families; its license travels with the inputs in `circuit_data/inputs/MQT_QECC_LICENSE`.
The code files are MQT QECC's definitions of its codes.
Chen et al.'s cat preparation (cited under [Circuits](#circuits)) comes from [vftqecc](https://github.com/vftqc/vftqecc), under the MIT license, whose notice lies beside it as `circuit_data/translated/VFTQECC_LICENSE` and travels with the inputs.
Goto's circuit (cited under [Circuits](#circuits)) comes through qLDPC's notebook `examples/logical_error_rates/5_fault_tolerance_certificates.ipynb`.
