# Exact certification of bounded fault propagation

**Prototype.** Everything in `state_prep_certification/` is a prototype, subject to change: its code, interfaces, inputs, outputs, and documentation.

A noisy circuit that prepares a stabilizer state has bounded fault propagation up to fault weight K if no accepted fault of fault weight at most K has a residual weight above its fault weight ([Terms](#terms) defines these words).
qLDPC's `certify_bounded_fault_propagation` decides this exactly, as its guide [`fault_propagation.md`](../../../src/qldpc/circuits/certification/fault_propagation.md) explains.
This experiment measures that exact certification on published state preparation circuits, and how its cost grows with the size of a circuit and its detector error model (DEM).
It measures which circuits certify within the time limit, the number of faults that the listing considers against all accepted fault combinations, the smallest counterexamples where violations exist, and the cost of the two weight oracles on identical queries.
Its one baseline is internal: brute force that queries the accepted fault combinations, weight by weight, with the certifier's own oracle, until it finds a violation.
It compares with no other tool; its internal checks are the agreement of the Brouwer-Zimmermann (BZ) and connected-cluster (CC) oracles (qLDPC's [BZ guide](../../../src/qldpc/codes/distance/brouwer_zimmermann.md) and [connected-cluster guide](../../../src/qldpc/codes/distance/connected_cluster.md)), the agreement of the certifier with the baseline, and, on small circuits, brute force with exact residual weights computed without either oracle (the ground truth); [Internal checks](#internal-checks) says what each covers.
Two applications of the certifier use the same circuits: the exact margin of each circuit, and the deletion of verification measurements under exact certification.

## Terms

These are the terms of qLDPC's guide [`fault_propagation.md`](../../../src/qldpc/circuits/certification/fault_propagation.md), in short:

- An **error mechanism** is one error of the circuit's DEM, and a **fault** is a set of error mechanisms; its **fault weight** is their number.
- The **flags** of a fault are the post-selection detectors that it fires: detectors on which the circuit's output is post-selected, such as those of its verification measurements, not flag qubits.
  A fault is **accepted** if it fires none.
- The **syndrome** of a fault is the syndrome, against the stabilizers of the prepared state, of the Pauli error that the fault leaves on the output qubits.
  Its **residual weight** is the smallest weight of a Pauli error with that syndrome.
  For a CSS state, as here, the weight is the CSS weight max(|x|, |z|), with x and z the X and Z supports of the error; the residual weights of the X and Z parts of the error are its **sectors**.
- A fault **violates** the bound if its residual weight exceeds its fault weight; a violating accepted fault is a **counterexample**.
  An error mechanism is **heavy** if it violates by itself: no Pauli error of weight at most 1 has its syndrome.
- An accepted fault is **irreducible** if it contains no smaller nonzero accepted fault; if any accepted fault violates, an irreducible one of no larger fault weight does, so these suffice.
  The **listing** is the certifier's enumeration, at each fault weight from 2 to the cutoff K' (below), of the irreducible accepted faults that contain a heavy mechanism, the **listed faults**; at fault weight 1, the certifier checks each mechanism with zero flags.
- A **weight oracle** decides whether a syndrome has a Pauli error of weight at most a threshold: qLDPC has a BZ and a CC oracle.
  The fault search of a certification submits the syndromes of listed faults to the oracle in batches, whose entries are **rows**.
- A circuit's **design t** is the number of faults that its source designed it to tolerate, a trusted input (see [Circuits](#circuits)).
  Certification runs at K = t + 1, so that one run decides both K = t and the next fault weight; the deletions keep K = t.
- The **margin** m* of a circuit is the smallest fault weight of a counterexample, or "all" if none exists (see [Arms](#arms)).

The outputs use these symbols for counts of a circuit's DEM, whose values `circuits.json` and the count arms give:

| Symbol | Meaning |
| --- | --- |
| N | the error mechanisms |
| N_0 | the reduced mechanisms: one per distinct nonzero signature (flags, syndrome); every count below is over them |
| z | the reduced mechanisms with zero flags |
| p | the post-selection detectors, the bits of the flags |
| r | the GF(2) rank of the flags; no irreducible accepted fault has fault weight above r + 1 |
| K' | min(K, r + 1), the cutoff of the listing |
| U | the sum over 1 <= m <= K of C(N_0, m), which bounds the work of the baseline |
| h | the heavy mechanisms with nonzero flags |
| q | the flag classes: the reduced mechanisms with nonzero flags, grouped by equal flags |
| q_h | the start classes: the flag classes with a heavy member |
| A_m | the accepted fault combinations of fault weight m: sets of m reduced mechanisms whose flags sum to zero |
| C(N_0, m) | all sets of m reduced mechanisms, accepted or not |
| E_m | the class codewords of m >= 3 classes whose first class is a start class (see [`counting.py`](counting.py)) |
| Phi_m | the listed faults of fault weight m |
| Pi_(m-1) | the prefixes of m - 1 classes of a naive enumeration of class codewords, most of which qLDPC's walk by shared flags does not visit (`naive_prefixes_m_minus_1` of `per_weight.csv`) |
| Omega_m | the rows that the fault search submits to the oracle at fault weight m, each the syndrome of a listed fault ([`instrument.py`](instrument.py)) |

## Circuits

The circuits are fixed by rules and public sources, and no timing selects any of them.
`circuits.py --build` makes from the sources a certification input for each circuit that it does not exclude: the Stim circuit that the experiment certifies on a Linux node without network access.
`circuits.json` lists every circuit, with its source, derivation, certify options, and the SHA-256 of its input, or the reason that `--build` excluded it.
`circuits.json` also lists the parameters of each circuit's DEM: N, p, N_0, z, r, K', and U of [Terms](#terms).
`--build` gets them by running the certifier with its fault search replaced by a recorder, so that no oracle runs.

No circuit is committed.
Every circuit lives in the experiment's data directory, `circuit_data/`, which Git ignores:

- `mqt/`: the circuits that `circuits.py --fetch` copies from MQT QECC at the commit that `config.json` pins (`workstation_sources`), with MQT QECC's license;
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
| A-ft, A-unv | MQT QECC's verified and unverified \|0...0> preparations of the codes of `circuits.A_CODES` | MQT QECC, `scripts/ft_stateprep/eval/circuits/`, at the commit that `config.json` pins |
| B-cat | MQT QECC's cat-state preparations `ft_ghz_<w>_<t>` | MQT QECC, `scripts/cat_states/circuits/` |
| C-goto | Goto's flagged Steane preparation | `circuit_data/translated/goto_steane.stim` |
| D-synth | verified circuits synthesized with MQT QECC | `circuit_data/synthesized/` |
| D-synth (MQT 1.9.0) | naive verified circuits synthesized with MQT QECC 1.9.0 | `circuit_data/synthesized_mqt_1.9.0/` |
| E-chen | Chen et al.'s 8-qubit cat preparation, translated literally | `circuit_data/translated/chen_cat8.stim` |
| W | one weakened variant of each circuit of A-ft, C-goto, and both D-synth families | derived by `circuits.py --build` |

C-goto's circuit is the flagged preparation of the Steane code's |0> state with one ancilla qubit for verification, Figure 1c of H. Goto, "Minimizing resource overheads for fault-tolerant preparation of encoded states of the Steane code", Scientific Reports 6, 19578 (2016), doi:10.1038/srep19578, as qLDPC's notebook `examples/logical_error_rates/5_fault_tolerance_certificates.ipynb` writes it.
E-chen's circuit is the 8-qubit cat preparation of K. Chen, Y. Liu, W. Fang, J. Paykin, X.-C. Wu, A. Schmitz, S. Zdancewic, and G. Li, "Verifying Fault-Tolerance of Quantum Error Correction Codes", Computer Aided Verification (CAV 2025), pp. 3-27, doi:10.1007/978-3-031-98685-7_1, arXiv:2501.14380.

A W variant of a circuit, its parent (as distinct from the parent of a deletion task, below), deletes every instruction target group that involves one measured qubit of the parent.
`circuits.py --build` tries the measured qubits from the last target of the last measurement backwards, and keeps the first variant that qLDPC's certifier does not reject with the parent's output qubits and that prepares the parent's state (`circuits._make_weakened`).
The variant keeps its parent's code and t.
It need not violate at K = t + 1, since the verification that remains may still bound fault propagation, so the W variants are not negative controls.

Each circuit's design t sets K = t + 1, the margins' groups, and the deletions' K = t.
It is a trusted input, which `circuits.py` takes from each family's source (`circuits._sources`):

- A-ft, A-unv: (d - 1) // 2, with d from the [[n, k, d]] that `circuits.A_CODES` lists for each of MQT QECC's codes;
- B-cat: the t of the file name `ft_ghz_<w>_<t>`;
- C-goto: 1, and E-chen: 2, as `circuits.py` sets them;
- D-synth, D-synth (MQT 1.9.0): (d - 1) // 2, with d from the synthesis call's record in `provenance.json`;
- W: the parent's t.

`circuits.py --check` checks that K = t + 1, and checks t against the `distance` of the code file of each circuit of A-ft and both D-synth families; nothing checks these distances, or the d of `circuits.A_CODES` and `provenance.json`, themselves.

Each A-ft, C-goto, and synthesized entry of `circuits.json` also names its code file, `code_file`, with its SHA-256, `code_sha256`.
A code file `<code>_code.json` holds the code's `distance` and its matrices: the X- and Z-type checks `Hx` and `Hz`, and the X- and Z-type logical operators `Lx` and `Lz`, with one column per output qubit, in output-qubit order.
`circuits.py --check` checks each input's state against these matrices, which split the state's Z-type stabilizers into checks Hz and logical operators Lz as the code's definition in MQT QECC does; the state alone does not determine that split, since the Z-type stabilizers of |0...0>_L span both Hz and Lz.
The code files of the A-ft codes, in `codes/`, follow MQT QECC's definitions of the codes at the commit that `config.json` pins, and those of the synthesized codes lie beside their circuits; C-goto uses the Steane code's file.
MQT's `RotatedSurfaceCode(5)` at that commit has its X checks where the shipped d = 5 circuits have their Z checks, so its file exchanges Hx with Hz and Lx with Lz.

Each certification input prepends resets to qubits that start without one.
Each B-cat circuit gets a detector for each pair of consecutive measurement results, on which it is post-selected; E-chen's verification measurements are flags, as in the other families.
The node applies K = t + 1 and the noise model of `config.json` (`noise`), qLDPC's `DepolarizingNoiseModel` with the probability p that it gives.
This noise model puts faults only on gates, resets, and measurements: one- or two-qubit depolarization after each gate, a reset to the orthogonal state, and a flipped measurement result, each with probability p.
Idle qubits get no errors, so a verdict covers only these fault locations.
For a fixed set of fault mechanisms, each with nonzero probability, the value of p changes no verdict, count, or margin, which depend only on where the model puts faults; it enters only the acceptance probabilities that the deletion tasks estimate.
Setting p = 0 removes every fault mechanism, so it does not pose the same certification problem.
The certify options of every circuit leave `css_weight` at its default, None, under which qLDPC chooses CSS weight for a CSS state.
Every input prepares one: the top-level `css_weight` of each input in `circuits.json` records qLDPC's choice, true.
The ground truth (`counting.ground_truth`) requires a CSS state.

## Arms

| Arm | What it does |
| --- | --- |
| `cert_bz`, `cert_cc` | `certify_bounded_fault_propagation` with the BZ or CC oracle, instrumented by `instrument.py` |
| `brute_bz` | the same BZ certification with the fault search replaced by the brute-force baseline of `baseline.py`, only where U is within the cap of `config.json` (`brute_force.max_combinations`) |
| `shadow` | BZ certification in which a CC oracle also answers every query, each timed |
| `count_accepted`, `count_listing` | A_m and C(N_0, m) for m <= K, and the counts of the listing at each fault weight m, E_m, Phi_m, and Pi_(m-1), by `counting.py` |
| `ground_truth` | brute force with residual weights from the whole stabilizer group, on the small circuits that `circuits.json` marks |
| `margin` | BZ certification with no limit on the fault weight, on the circuits whose flag rank r exceeds t, by `applications.py` |
| `prune`, `prune_sensitivity` | the deletion of the measured qubits of each A-ft, C-goto, and synthesized circuit under exact certification at K = t, by `applications.py`; `prune_sensitivity` tries another order, on the circuits of `prune.sensitivity_inputs` of `config.json` |

[`experiment.py`](experiment.py) says what T includes for each arm, which runs warm up before timing, and which runs get repeats: a run of `cert_bz`, `cert_cc`, or `brute_bz` that finishes in under `repeats.below_s` of `config.json`.

The baseline enumerates the accepted fault combinations over the N_0 reduced mechanisms, fault weight by fault weight up to K, and queries them in batches with the certifier's own oracle, built from the same arguments, until the first batch that holds a violation; only the fault search differs from `cert_bz`.
U bounds both the combinations and the rows that it queries, and is fixed by the circuit's parameters, so that a circuit over the cap gets no task and is reported with the cap's value, as in "not run (U > 1e9)", never as a timeout.

The margin m* of a circuit is the smallest fault weight of a counterexample, so that the circuit has bounded fault propagation up to exactly m* - 1.
It is "all" if no such fault exists at any fault weight, which the search proves once it completes fault weight r + 1, or earlier if no flagged mechanism is heavy.
"All" does not mean that the state is protected against any number of faults: every accepted fault still has a residual weight up to its fault weight.
`cert_bz` at K = t + 1 gives m* where it finds a violation or r <= t, and `margin` gives it for the other circuits.

A deletion task starts from its circuit, its parent, which it first certifies at K = t with BZ, and it stops if the parent violates.
It then tries to delete each measured qubit once, and keeps a deletion if and only if the circuit still prepares the parent's state, qLDPC's certifier does not reject its detectors, and BZ certifies it at K = t.
The final circuit, which can equal the parent, is then certified at K = t by CC, the acceptance probabilities of parent and final circuit are estimated from `prune.acceptance_shots` Stim shots each (`config.json`), and the final circuit gets its m*.
`prune` and `prune_sensitivity` try the measured qubits in two orders.
[`applications.py`](applications.py) gives these orders, what a deletion removes, and what a `margin` or deletion run that reaches its limit keeps: each run sends its completed steps, fault weights or tried qubits, to the runner as partial results.

The runner retries any run whose measured call compiled, as the [shared README](../README.md#runs) describes.
The experiment's own compiled code, in `counting.py` and `baseline.py`, is cached on disk by qLDPC's `always_compiled`, as qLDPC's is; a comment near the top of `counting.py` says what this code may call and read.

`analyze.py`'s docstring says how it treats a row with `compiled_in_timing` set (see the [shared README](../README.md#runs)).

## Internal checks

Each internal check compares computations that share some of their inputs and code, and cannot detect an error in what they share.
None of them can detect an error in what they all share: the DEM, the flags and syndromes of its mechanisms, and the state's stabilizers, as qLDPC builds them from the circuit.

- BZ against CC (`cert_bz`, `cert_cc`, and the `shadow` run, in which a CC oracle answers every query of a BZ certification): the same certification with two weight oracles.
  They share the reduction to N_0 mechanisms and the fault search, so their agreement tests the oracles.
- The baseline (`brute_bz`) against the certifier: the same certification, with the fault search replaced by queries of every accepted fault combination to the same BZ oracle.
  They share the rule of the reduction and the oracle, so their agreement tests the fault search (its listing, its classification of heavy mechanisms, the cutoff K', and its skipping), not the oracle.
- The ground truth (`ground_truth`) against the certifier, the baseline, and the counters: it takes the DEM from an untimed certification at fault weight 1, as the counters do, and reduces it by the same rule.
  Its residual weights come from an enumeration of every X error and every Z error on the output qubits (`counting.residual_weight_table`), independent of both oracles, and it enumerates every accepted fault, so its agreement tests the oracles, the fault search, and the counts on the circuits that it covers.
  Its counts and the counters' share the definition of the flag classes (`counting.get_classes`).
- The analysis checks a counterexample's exact residual weight against its fault weight, the certifier's lower bound, and, for CSS weight, the residual weights of its X and Z parts.
  The residual weights of the counterexample and of its parts come from a BZ oracle (`describe_certificate` in `applications.py`), so this check does not test BZ itself.

## Workflow

### Running

On the workstation, with `circuit_data/` holding the inputs that no script of the experiment makes (see [Circuits](#circuits)):

```bash
python experiments/state_prep_certification/certification/circuits.py --fetch
python experiments/state_prep_certification/certification/circuits.py --build
python experiments/state_prep_certification/certification/circuits.py --check
python experiments/state_prep_certification/certification/acceptance.py --local
python experiments/state_prep_certification/run_node.py certification --local --subset smoke --approved-cpu-hours 79
```

`--fetch` downloads MQT QECC's circuits, and `--build` writes the inputs, with the licenses and `circuits.json`, to `circuit_data/inputs/`.
`--check` checks every input and its code file against `circuits.json` and `config.json`, as `_check_input` in [`circuits.py`](circuits.py) lists, and refuses an input that does not prepare a CSS state.
Where the fetched sources are present, as on the workstation, it also rebuilds the inputs from them in a scratch directory and compares.
On the node, the experiment runs the same check before any task, copies `circuits.json` into the label directory, and never imports MQT or qiskit.

The subset `smoke` runs every arm that applies to C-goto, E-chen, and the A-ft circuits of the Steane code.
Its runs take minutes, but its ceiling is about 78 CPU-hours at the current `circuits.json`, which the approval of 79 above covers.
To see the number of tasks of a subset and their ceiling, run `run_node.py certification --local --subset S --dry-run` without `--approved-cpu-hours`: it prints them, without the tasks, and exits with status 1, and `--local` keeps a workstation from running the acceptance tests first.
Approve the printed ceiling rounded up.

On the node, `node.sh certification --approved-cpu-hours H` runs the subset `main` (certification by both oracles, the brute-force baseline where U is within the cap, the shadow run, the counts, the margins, and the deletions), and `--subset check` the ground truth by brute force, in a label directory and archive of its own.
The ceiling of `main` follows from `circuits.json`: each run costs at most its start-up cap, its limit, its post-processing allowance, and the grace period after a kill, and a fast run of `cert_bz`, `cert_cc`, or `brute_bz` gets `repeats.count` repeats.
The runner's retries add to this, as [`common/tasks.py`](../common/tasks.py) explains.

### Analysis

On the workstation, run `analyze.py --archive PATH --check-archive PATH --out DIR`, with the archives of `main` and `check`.
It checks each archive against its `.sha256` file and its inner `SHA256SUMS`, and extracts it into a temporary directory, which it removes when it ends.
So `--archive` requires `--out`: the default output directory, `analysis/` in the label directory, would be removed with the extraction.
Without the archive of `check`, the outputs report the ground truth as not run.
The analysis takes the circuits and their parameters from the label's copy of `circuits.json`, and stops if the label of `check` ran with another one.
If a label does not match its `circuits.json` and the working tree's `config.json` and `pins.json`, it prints a warning and analyzes the label anyway.
It reports every task without a row as missing.

[`analyze.py`](analyze.py)'s docstring lists the outputs and their columns, and [`analyze_applications.py`](analyze_applications.py)'s those of the margins and the deletions:

- `summary.csv`: one row per circuit, with its parameters, verdicts, times, statuses, agreements, contradictions, margin, and counterexample;
- `families.csv`: one row per family, with the counts of verdicts, statuses, agreements, disagreements, and margins, and the medians and maxima of T;
- `per_weight.csv`: one row per circuit and fault weight m, with A_m, C(N_0, m), Pi_(m-1), E_m, Phi_m, Omega_m, the times of each weight, and the work of the baseline;
- `listing_table.csv`: Phi_m and Omega_m against A_m, per family and fault weight;
- `shadow.csv`: the cost of the two weight oracles on identical queries, from the `shadow` runs;
- `counterexamples.json`: the counterexamples of the first runs of `cert_bz` and `cert_cc` that finished, and up to three examples, one for each of three groups of families;
- `margin_violations.csv`: the circuits whose exact m* is at most t;
- `deletion.csv`, `deletion_tasks.csv`, `deletion_table.csv`, `deletion_notes.json`, and `pruned/`: the steps, the tasks, and the final circuits of the deletions.

The acceptance test AT8 runs the analysis on synthetic rows: a listing count that timed out, each kind of contradiction, shadow runs, a task without a row, rows with `compiled_in_timing` set, and rows of `--local` runs, which the analysis refuses without `--allow-local`, as the [shared README](../README.md#local-runs) says.

### Exact results, bounds, and estimates

The verdicts, the margins m*, the counts of [Terms](#terms), and the residual weights of counterexamples are exact.
Some outputs are bounds:

- `margin_lower_bound` and `final_margin_lower_bound` hold a fault weight k where only m* > k is known;
- a counterexample's `residual_weight_lower_bound` is the certifier's lower bound on its residual weight, whose exact value `residual_weight` gives;
- the ratios of `listing_table.csv` over C(N_0, m), in place of A_m, are lower bounds on those over A_m, as its column `ratio_is_lower_bound` marks.

The acceptance probabilities of the deletions are estimates from Stim shots, with their standard errors (`parent_acceptance_se`, `final_acceptance_se`).
Times are measurements of runs on the node (see the shared README's [Runs](../README.md#runs)).

### Disagreements

A circuit is a disagreement if its exact results contradict each other, between two runs or within one, as where a counterexample's residual weight does not exceed its fault weight.
The analysis compares, as claims on the circuit's margin m*, every result that a run of `cert_bz`, `cert_cc`, `shadow`, `brute_bz`, `margin`, or `ground_truth` kept, repeats included and whatever the status of its row, each fault weight that such a run completed or found a violation at, and the claims of the deletion tasks, as `analyze_applications.py` lists.
A `timeout` row can keep the result of a call that returned at or after the limit, and a `memout` row the result of a call whose memory exceeded the cap before the result arrived, as found afterward (see the shared README's [Runs](../README.md#runs)).
The runner keeps no result of a run that it discarded because it compiled, so the analysis compares no claim of such a run.
It also checks the counts against each other, and the counterexamples and deletions against themselves; `check_consistency` in `analyze.py` lists every contradiction.
The column `contradictions` of `summary.csv` names a circuit's contradictions, and the column `disagreements` of `families.csv` counts these circuits.
A disagreement enters no count of verdicts or margins, no median or maximum, nor `listing_table.csv`, `shadow.csv`, `margin_violations.csv`, `deletion_table.csv`, or the example counterexamples.
The verdicts, T, and `counterexamples.json` instead use only rows with status `ok`, so `margin` can be exact where the verdict is `undecided`.

The agreement columns of `summary.csv`, `agree_bz_cc`, `baseline_counts_match`, `agree_baseline`, `agree_ground_truth`, and `agree_prune`, compare first runs, as `analyze.py`'s docstring says.
The columns `agreement_bz_cc`, `agreement_baseline`, and `agreement_ground_truth` of `families.csv` count the circuits where `agree_bz_cc`, `agree_baseline`, and `agree_ground_truth` are true.
`agree_margin` instead compares the claims on m* of every run that the runner kept, as `contradictions` does, so a circuit can agree on its first runs and still be a disagreement.

## Attributions and licenses

MQT QECC (MIT) provides the A and B circuits and synthesized the circuits of both D-synth families; its license travels with the inputs in `circuit_data/inputs/MQT_QECC_LICENSE`.
The code files are MQT QECC's definitions of its codes.
Chen et al.'s cat preparation (cited under [Circuits](#circuits)) comes from [vftqecc](https://github.com/vftqc/vftqecc), under the MIT license, whose notice lies beside it as `circuit_data/translated/VFTQECC_LICENSE` and travels with the inputs.
Goto's circuit (cited under [Circuits](#circuits)) comes through qLDPC's notebook `examples/logical_error_rates/5_fault_tolerance_certificates.ipynb`.
