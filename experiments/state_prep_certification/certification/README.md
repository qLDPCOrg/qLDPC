# Exact certification of bounded fault propagation

**Prototype.** Everything in `state_prep_certification/` is a prototype, subject to change: its code, interfaces, inputs, outputs, and documentation.

A noisy circuit that prepares a stabilizer state has bounded fault propagation up to fault weight K if every accepted fault of fault weight at most K has a residual weight of at most its fault weight ([Terms](#terms) defines these words).
qLDPC's `certify_bounded_fault_propagation` decides this exactly, as its guide [`fault_propagation.md`](../../../src/qldpc/circuits/certification/fault_propagation.md) explains.
This experiment measures that exact certification on published state preparation circuits, and how its cost grows with the size of a circuit and its detector error model (DEM).
It measures which circuits certify within the time limit, how many faults the certifier lists compared with all accepted fault combinations, the smallest counterexamples where violations exist, and the cost of the two weight oracles on identical queries: Brouwer-Zimmermann (BZ) and connected-cluster (CC), as qLDPC's [BZ guide](../../../src/qldpc/codes/distance/brouwer_zimmermann.md) and [CC guide](../../../src/qldpc/codes/distance/connected_cluster.md) describe them.
Its one baseline is internal: brute force with the certifier's own oracle (see [Arms](#arms)).
It compares with no other tool.
Its internal checks compare the two oracles, the certifier with the baseline, and both with a ground truth on small circuits ([Internal checks](#internal-checks)).
Two applications of the certifier use the same circuits: the exact margin of each circuit, and the deletion of measured qubits under exact certification.

## Terms

These are the terms of qLDPC's guide [`fault_propagation.md`](../../../src/qldpc/circuits/certification/fault_propagation.md), in short:

- An **error mechanism** is one error of the circuit's DEM, and a **fault** is a set of error mechanisms; its **fault weight** is their number.
- The **flags** of a fault are the post-selection detectors that it fires: detectors on which the circuit's output is post-selected, such as those of its verification measurements, not flag qubits.
  A fault is **accepted** if it fires none.
- The **syndrome** of a fault is the syndrome, against the stabilizers of the prepared state, of the Pauli error that the fault leaves on the output qubits.
  Its **residual weight** is the smallest weight of a Pauli error with that syndrome.
  For a CSS state, as here, the weight is the CSS weight max(|x|, |z|), with x and z the X and Z supports of the error.
  The outputs call the residual weights of the X and Z parts of a counterexample's error its **sector**.
- A fault **violates** the bound if its residual weight exceeds its fault weight; a violating accepted fault is a **counterexample**.
  An error mechanism is **heavy** if it violates by itself: no Pauli error of weight at most 1 has its syndrome.
- A nonzero accepted fault is **irreducible** if it contains no smaller nonzero accepted fault.
  If any accepted fault violates, an irreducible one of no larger fault weight does, so the certifier need check only irreducible faults.
  The **listing** is the certifier's enumeration, at each fault weight from 2 to the cutoff K' (below), of the irreducible accepted faults that contain a heavy mechanism, the **listed faults**; at fault weight 1, the certifier checks each mechanism with zero flags.
- A **weight oracle** decides whether a syndrome has a Pauli error of weight at most a threshold: qLDPC has a BZ and a CC oracle.
  The fault search of a certification submits the syndromes of listed faults to the oracle in batches, whose entries are **oracle rows**.
- A circuit's **design t** is the number of faults that its source designed it to tolerate, a trusted input (see [Circuits](#circuits)).
  Certification runs at K = t + 1, so that one run decides both K = t and the next fault weight; the deletions keep K = t.
- The **margin** m* of a circuit is the smallest fault weight of a counterexample, or "all" if none exists (see [Arms](#arms)).

The outputs use these symbols for counts of a circuit's DEM, whose values `circuits.json`, the count arms, and the certification runs give:

| Symbol | Meaning |
| --- | --- |
| N | the error mechanisms |
| N_0 | the reduced mechanisms: one per distinct nonzero signature (flags, syndrome); every count of mechanisms below is over them |
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
| E_m | the class codewords whose first class is a start class: sets of m >= 3 flag classes whose flags sum to zero, and every proper nonempty subset of which has independent flags (see [`counting.py`](counting.py)) |
| Phi_m | the listed faults of fault weight m |
| Pi_(m-1) | the work of the naive enumeration of class codewords, which `fault_propagation.md` contrasts with qLDPC's listing: its tuples of m - 1 classes (`naive_prefixes_m_minus_1` of `per_weight.csv`; see [`counting.py`](counting.py)) |
| Omega_m | the oracle rows that the fault search submits in batches at fault weight m >= 2, each the syndrome of a listed fault ([`instrument.py`](instrument.py)); at m = 1, `per_weight.csv` counts the certifier's threshold-1 queries instead |

## Circuits

The circuits are fixed by rules and public sources, and no timing selects any of them.
`circuits.py --build` makes from the sources a certification input for each circuit that it does not exclude: the Stim circuit that the experiment certifies on a Linux node without network access.
`circuits.json` lists every circuit, with its source, derivation, certify options, and the SHA-256 of its input, or the reason that `--build` excluded it.
`circuits.json` also lists the parameters of each circuit's DEM: N, p, N_0, z, r, K', and U of [Terms](#terms).
`--build` gets them by running the certifier with its fault search replaced by a recorder, so that no oracle runs.

| Family | Circuits | Source |
| --- | --- | --- |
| A-ft, A-unv | MQT QECC's verified and unverified \|0...0> preparations of the codes of `circuits.A_CODES` | MQT QECC, `scripts/ft_stateprep/eval/circuits/`, at the commit that `config.json` pins |
| B-cat | MQT QECC's cat-state preparations `ft_ghz_<w>_<t>` | MQT QECC, `scripts/cat_states/circuits/` |
| C-goto | Goto's flagged Steane preparation | `circuit_data/translated/goto_steane.stim` |
| D-synth | verified circuits synthesized with MQT QECC | `circuit_data/synthesized/` |
| D-synth (MQT 1.9.0) | circuits synthesized and verified with MQT QECC 1.9.0 | `circuit_data/synthesized_mqt_1.9.0/` |
| E-chen | Chen et al.'s 8-qubit cat preparation, translated literally | `circuit_data/translated/chen_cat8.stim` |
| W | one weakened variant of each circuit of A-ft, C-goto, and both D-synth families | derived by `circuits.py --build` |

C-goto's circuit is the flagged preparation of the Steane code's |0> state with one ancilla qubit for verification, Figure 1c of H. Goto, "Minimizing resource overheads for fault-tolerant preparation of encoded states of the Steane code", Scientific Reports 6, 19578 (2016), doi:10.1038/srep19578, as qLDPC's notebook `examples/logical_error_rates/5_fault_tolerance_certificates.ipynb` writes it.
E-chen's circuit is the 8-qubit cat preparation of K. Chen, Y. Liu, W. Fang, J. Paykin, X.-C. Wu, A. Schmitz, S. Zdancewic, and G. Li, "Verifying Fault-Tolerance of Quantum Error Correction Codes", Computer Aided Verification (CAV 2025), pp. 3-27, doi:10.1007/978-3-031-98685-7_1, arXiv:2501.14380.

No circuit is committed.
Every circuit lives in the experiment's data directory, `circuit_data/`, which Git ignores:

- `mqt/`: the circuits that `circuits.py --fetch` copies from MQT QECC at the commit that `config.json` pins (`workstation_sources`), with MQT QECC's license;
- `translated/`: E-chen's circuit, translated by hand, with vftqecc's license (see [Attributions](#attributions-and-licenses)), and C-goto's, as qLDPC's notebook writes it;
- `synthesized/`: circuits synthesized with MQT QECC, with the code files of their codes.
  Its `provenance.json` records each synthesis call: the code, its [[n, k, d]], and the MQT QECC release;
- `synthesized_mqt_1.9.0/`: circuits synthesized with MQT QECC 1.9.0 and verified by its `naive_verification_circuit`, which their `provenance.json` describes, with their code files, in the same form;
- `codes/`: the code files of the A-ft codes;
- `inputs/`: the inputs and `circuits.json`.

The experiment's scripts make `mqt/` (`circuits.py --fetch`) and `inputs/` (`circuits.py --build`).
No script of the experiment makes `translated/`, `synthesized/`, `synthesized_mqt_1.9.0/`, or `codes/`: they are inputs that the experiment takes as given, and checks, as `circuits.py --check` does.
`circuits.py --build` reads `mqt/`, `translated/`, both folders of synthesized circuits, and `codes/`, and writes `inputs/`.
It takes every verified circuit that a folder's `provenance.json` records as synthesized, leaving out the unverified preparations of the step `prep`, and `circuits.json` labels each synthesized circuit with its MQT QECC release.
The offline bundle carries `inputs/`, `synthesized/`, `synthesized_mqt_1.9.0/`, and `codes/`.
The experiment runs with whatever `circuits.json` and inputs `--build` wrote: `circuits.json` enters the label of each run, and each label keeps a copy of it, from which the analysis reads the circuits.

A W variant is its original circuit with every instruction target group that involves one measured qubit deleted.
`circuits.py --build` tries the measured qubits from the last target of the last measurement backwards, and keeps the first variant that qLDPC's certifier accepts as input with the original's output qubits and that prepares the original's state (`circuits._make_weakened`).
The variant keeps its original's code and t.
It need not violate at K = t + 1, since the verification that remains may still bound fault propagation, so the W variants are not negative controls.

Each circuit's design t sets K = t + 1, the deletions' K = t, and the margin counts of `families.csv`, which compare m* with t.
It is a trusted input, which `circuits.py` takes from each family's source (`circuits._sources`):

- A-ft, A-unv: (d - 1) // 2, with d from the [[n, k, d]] that `circuits.A_CODES` lists for each of MQT QECC's codes;
- B-cat: the t of the file name `ft_ghz_<w>_<t>`;
- C-goto: 1, and E-chen: 2, as `circuits.py` sets them;
- D-synth, D-synth (MQT 1.9.0): (d - 1) // 2, with d from the synthesis call's record in `provenance.json`;
- W: the original's t.

`circuits.py --check` checks that K = t + 1, and checks t against the `distance` of the code file of each circuit of A-ft and both D-synth families; nothing checks these distances, or the d of `circuits.A_CODES` and `provenance.json`, themselves.

Each A-ft, C-goto, and synthesized entry of `circuits.json` also names its code file, `code_file`, with its SHA-256, `code_sha256`.
A code file `<code>_code.json` holds the code's `distance` and its matrices: the X- and Z-type checks `Hx` and `Hz`, and the X- and Z-type logical operators `Lx` and `Lz`, with one column per output qubit, in output-qubit order.
`circuits.py --check` checks each input's state against these matrices.
They split the state's Z-type stabilizers into checks Hz and logical operators Lz, as the code's definition in MQT QECC does.
The state alone does not determine that split, since the Z-type stabilizers of |0...0>_L span both Hz and Lz.
The code files of the A-ft codes, in `codes/`, follow MQT QECC's definitions of the codes at the commit that `config.json` pins, and those of the synthesized codes lie beside their circuits; C-goto uses the Steane code's file.
MQT's `RotatedSurfaceCode(5)` at that commit has its X checks where the shipped d = 5 circuits have their Z checks, so its file exchanges Hx with Hz and Lx with Lz.

Each certification input prepends resets to qubits that start without one.
Each B-cat circuit gets a detector for each pair of consecutive measurement results, on which it is post-selected; E-chen's verification measurements are flags, as in the other families.
The node applies K = t + 1 and the noise model of `config.json` (`noise`), qLDPC's `DepolarizingNoiseModel` with the noise probability `noise.p`.
This noise model puts faults only on gates, resets, and measurements: one- or two-qubit depolarization after each gate, a reset to the orthogonal state, and a flipped measurement result, each with that probability.
Idle qubits get no errors, so a verdict covers only these fault locations.
For a fixed set of fault mechanisms, each with nonzero probability, the noise probability changes no verdict, count, or margin, which depend only on where the model puts faults; it enters only the acceptance probabilities that the deletion tasks estimate.
A noise probability of 0 removes every fault mechanism, so it does not pose the same certification problem.
The certify options of every circuit leave `css_weight` at its default, None, under which qLDPC chooses CSS weight for a CSS state.
Every input prepares a CSS state, and the top-level `css_weight` of each input in `circuits.json` records qLDPC's choice, true.
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

[`experiment.py`](experiment.py) says what T includes for each arm, which runs warm up before timing, and which runs get repeats.

The baseline enumerates the accepted fault combinations over the N_0 reduced mechanisms, fault weight by fault weight up to K, and queries them in batches with the certifier's own oracle, built from the same arguments, until the first batch that holds a violation.
Only the fault search differs from `cert_bz`.
U bounds both the combinations and the oracle rows that it queries, and is fixed by the circuit's parameters, so that a circuit over the cap gets no task.

A circuit with margin m* has bounded fault propagation up to exactly m* - 1.
m* is "all" if no counterexample exists at any fault weight, which the search proves once it completes fault weight r + 1, or earlier if no flagged mechanism is heavy.
"All" does not mean that the state is protected against any number of faults: every accepted fault still has a residual weight up to its fault weight.
`cert_bz` at K = t + 1 gives m* where it finds a violation or r <= t, and `margin` gives it for the other circuits.

A deletion task calls its starting circuit the parent, and first certifies it at K = t with BZ; it stops if the parent violates.
It then tries to delete each measured qubit once, and keeps a deletion if and only if the circuit still prepares the parent's state, qLDPC's certifier accepts its detectors, and BZ certifies it at K = t.
The final circuit, which can equal the parent, is then certified at K = t by CC.
Next, the acceptance probabilities of parent and final circuit are estimated from `prune.acceptance_shots` Stim shots each (`config.json`).
Last, the final circuit gets its m*.
[`applications.py`](applications.py) gives the two deletion orders, what a deletion removes, and what a `margin` or deletion run that reaches its limit keeps: each run sends its completed steps, fault weights or tried qubits, to the runner as partial results.

The experiment's own compiled code, in `counting.py` and `baseline.py`, is cached on disk by qLDPC's `always_compiled`, as qLDPC's is, so that the runner's retry of a run that compiled loads it (see the shared README's [Runs](../README.md#runs)); a comment near the top of `counting.py` says what this code may call and read.

## Internal checks

Each internal check compares computations that share some of their inputs and code, and cannot detect an error in what they share.
None of them can detect an error in what they all share: the DEM, the flags and syndromes of its mechanisms, and the state's stabilizers, as qLDPC builds them from the circuit.

- BZ against CC (`cert_bz`, `cert_cc`, and `shadow`): the same certification with two weight oracles.
  They share the reduction to N_0 mechanisms and the fault search, so their agreement tests the oracles.
- The baseline (`brute_bz`) against the certifier: the same certification, with the fault search replaced by the baseline's queries to the same BZ oracle.
  They share the rule of the reduction and the oracle, so their agreement tests the fault search (its listing, its classification of heavy mechanisms, the cutoff K', and the listed faults that it skips without a query), not the oracle.
- The ground truth (`ground_truth`) against the certifier, the baseline, and the count arms: it takes the DEM from an untimed certification at fault weight 1, as the count arms do, and reduces it by the same rule.
  Its residual weights come from an enumeration of every X error and every Z error on the output qubits (`counting.residual_weight_table`), independent of both oracles, and it enumerates every accepted fault, so its agreement tests the oracles, the fault search, and the counts on the circuits that it covers.
  Its counts and those of the count arms share the definition of the flag classes (`counting.get_classes`).
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
A dry run prints the ceiling of a subset, as the shared README's [Scheduling](../README.md#scheduling) describes; approve it rounded up.

On the node, `node.sh certification --approved-cpu-hours H` runs the subset `main` (certification by both oracles, the brute-force baseline where U is within the cap, the shadow run, the counts, the margins, and the deletions), and `--subset check` the ground truth by brute force, in a label directory and archive of its own.

### Analysis

On the workstation, run `analyze.py --archive PATH --check-archive PATH --out DIR`, with the archives of `main` and `check`; without the archive of `check`, the outputs report the ground truth of `main` as not run.
[`analyze.py`](analyze.py)'s docstring says what the analysis reads and checks, and lists the outputs and their columns, and [`analyze_applications.py`](analyze_applications.py)'s those of the margins and the deletions:

- `summary.csv`: one row per circuit, with its parameters, verdicts, times, statuses, agreements, contradictions, margin, and counterexample;
- `families.csv`: one row per family, with the counts of verdicts, statuses, agreements, disagreements, and margins, and the medians and maxima of T;
- `per_weight.csv`: one row per circuit and fault weight m, with A_m, C(N_0, m), Pi_(m-1), E_m, Phi_m, Omega_m, the times of each weight, and the work of the baseline;
- `listing_table.csv`: Phi_m and Omega_m against A_m, per family and fault weight;
- `shadow.csv`: the cost of the two weight oracles on identical queries, from the `shadow` runs;
- `counterexamples.json`: the counterexamples of the first runs of `cert_bz` and `cert_cc` that finished, and up to three examples, one for each of the groups of families that `analyze.py`'s docstring lists;
- `margin_violations.csv`: the circuits whose exact m* is at most t;
- `deletion.csv`, `deletion_tasks.csv`, `deletion_table.csv`, `deletion_notes.json`, and `pruned/`: the steps, the tasks, and the final circuits of the deletions.

### Exact results, bounds, and estimates

The verdicts, the margins m*, the counts of [Terms](#terms), and the residual weights of counterexamples are exact.
Some outputs are bounds:

- `margin_lower_bound` and `final_margin_lower_bound` hold a fault weight k where only m* > k is known;
- a counterexample's `residual_weight_lower_bound` is the certifier's lower bound on its residual weight, whose exact value `residual_weight` gives;
- where A_m is unknown, `listing_table.csv` divides by C(N_0, m) instead, which gives lower bounds on the ratios over A_m; its column `ratio_is_lower_bound` marks these.

The acceptance probabilities of the deletions are estimates from Stim shots, with their standard errors (`parent_acceptance_se`, `final_acceptance_se`).
Times are measurements of runs on the node (see the shared README's [Runs](../README.md#runs)).

### Disagreements

A circuit is a disagreement if its exact results contradict each other, between two runs or within one, as where a counterexample's residual weight does not exceed its fault weight.
The analysis compares every claim on the circuit's margin m* that a run kept, repeats included and whatever its row's status, and checks the counts, the counterexamples, and the deletions against themselves; `check_consistency` in `analyze.py` and `analyze_applications.py` list every contradiction.
A disagreement enters no count of verdicts or margins, and no median, maximum, or ratio; `analyze.py`'s docstring says which results each column uses, such as the agreement columns, which compare first runs.

## Attributions and licenses

MQT QECC (MIT) provides the A and B circuits and synthesized the circuits of both D-synth families; its license travels with the inputs in `circuit_data/inputs/MQT_QECC_LICENSE`.
The code files are MQT QECC's definitions of its codes.
Chen et al.'s cat preparation (cited under [Circuits](#circuits)) comes from [vftqecc](https://github.com/vftqc/vftqecc), under the MIT license, whose notice lies beside it as `circuit_data/translated/VFTQECC_LICENSE` and travels with the inputs.
Goto's circuit (cited under [Circuits](#circuits)) comes through qLDPC's notebook `examples/logical_error_rates/5_fault_tolerance_certificates.ipynb`.
