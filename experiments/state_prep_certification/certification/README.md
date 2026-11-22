# Exact certification of bounded fault propagation

This experiment measures exact certification of bounded fault propagation on published state preparation circuits, and how its cost grows with the size of the problem.
It measures which circuits certify within the time limit, the number of faults that the listing considers against all accepted fault combinations, the smallest counterexamples where violations exist, and the cost of the two weight oracles on identical queries.
Its one baseline is internal: brute force that queries every accepted fault combination with the certifier's own oracle.
It compares with no other tool; its internal checks are the agreement of the Brouwer-Zimmermann (BZ) and connected-cluster (CC) oracles (qLDPC's [BZ guide](../../../src/qldpc/codes/distance/brouwer_zimmermann.md) and [connected-cluster guide](../../../src/qldpc/codes/distance/connected_cluster.md)), the agreement of the certifier with the baseline, and, on small circuits, brute force with exact residual weights computed without either oracle (the ground truth); [Internal checks](#internal-checks) says what each covers.
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

A W variant of a circuit, its parent, deletes every instruction target group that involves one measured qubit of the parent.
`circuits.py --build` tries the measured qubits from the last target of the last measurement backwards, and keeps the first variant that qLDPC accepts with the parent's output qubits and that prepares the parent's state (`circuits._make_weakened`).
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
A code file `<code>_code.json` holds the code's `distance` and its matrices `Hx`, `Hz`, `Lx`, and `Lz`, with one column per output qubit, in output-qubit order.
`circuits.py --check` checks each input's state against these matrices, which split the state's Z-type stabilizers into checks Hz and logical operators Lz as the code's definition in MQT QECC does; the state alone does not determine that split, since the Z-type stabilizers of |0...0>_L span both Hz and Lz.
The code files of the A-ft codes, in `codes/`, follow MQT QECC's definitions of the codes at the commit of `config.json`, and those of the synthesized codes lie beside their circuits; C-goto uses the Steane code's file.
MQT's `RotatedSurfaceCode(5)` at that commit has its X checks where the shipped d = 5 circuits have their Z checks, so its file exchanges Hx with Hz and Lx with Lz.

Each certification input prepends resets to qubits that start without one.
Each B-cat circuit gets a detector for each pair of consecutive measurement results, on which it is post-selected; E-chen's verification measurements are flags, as in the other families.
The node applies qLDPC's `DepolarizingNoiseModel(1e-3)` and K = t + 1.
This noise model puts faults only on gates, resets, and measurements: one- or two-qubit depolarization after each gate, a reset to the orthogonal state, and a flipped measurement result, each with probability p.
Idle qubits get no errors, so a verdict covers only these fault locations.
For a fixed set of fault mechanisms, each with nonzero probability, the value of p changes no verdict, count, or margin, which depend only on where the model puts faults; it enters only the acceptance probabilities that the deletion tasks estimate.
Setting p = 0 removes every fault mechanism, so it does not pose the same certification problem.
It leaves `css_weight` at its default, None, under which qLDPC chooses CSS weight for a CSS state.
Every input prepares one, as the top-level `css_weight` of each input in `circuits.json` records.
The ground truth (`counting.brute_force`) requires a CSS state.

## Arms

| Arm | What it does |
| --- | --- |
| `cert_bz`, `cert_cc` | `certify_bounded_fault_propagation` with the BZ or CC oracle, instrumented by `instrument.py` |
| `brute_bz` | the same BZ certification with the fault search replaced by the brute-force baseline of `baseline.py`, only where U is within the cap of `config.json`, 10^9 |
| `shadow` | BZ certification in which a CC oracle also answers every query, each timed |
| `count_accepted`, `count_listing` | the accepted fault combinations A_m for m <= K, and the counts of the listing at each fault weight m, by `counting.py`: the class codewords E_m, the listed faults Phi_m, and the prefixes Pi_(m-1) of a naive enumeration of the class codewords (`naive_prefixes_m_minus_1` of `per_weight.csv`), most of which qLDPC's walk by shared flags does not visit |
| `ground_truth` | brute force with residual weights from the whole stabilizer group, on the small circuits that `circuits.json` marks |
| `margin` | BZ certification with no limit on the fault weight, on the circuits whose flag rank r exceeds t, by `applications.py` |
| `prune`, `prune_sensitivity` | the deletion of the measured qubits of each A-ft, C-goto, and synthesized circuit under exact certification at K = t, by `applications.py`; `prune_sensitivity` tries another order, on the circuits of `prune.sensitivity_inputs` of `config.json` |

`instrument.py` counts Omega_m, the rows that the fault search of a certification submits to the weight oracle at fault weight m.

The baseline enumerates every accepted fault combination of weight at most K over the N_0 reduced mechanisms, and queries each with the certifier's own oracle, built from the same arguments; only the fault search differs from `cert_bz`.
U bounds both the combinations and the rows that it queries, and is fixed by the circuit's parameters, so that a circuit over the cap gets no task and is reported as "not run (U > 1e9)", never as a timeout.

The margin m* of a circuit is the smallest fault weight of an accepted fault whose residual weight exceeds its fault weight, so that the circuit has bounded fault propagation up to exactly m* - 1.
It is "all" if no such fault exists at any fault weight, which the search proves once it completes fault weight r + 1, or earlier if no flagged mechanism is heavy.
"All" does not mean that the state is protected against any number of faults: every accepted fault still has a residual weight up to its fault weight.
`cert_bz` at K = t + 1 gives m* where it finds a violation or r <= t, and `margin` gives it for the other circuits.
A `margin` run that reaches its limit gives the largest fault weight k that it completed without a violation, so that m* > k, or the weight at which it found a violation, which is m*.

A `prune` task first certifies its circuit, the parent, at K = t with BZ, and stops if the parent violates.
It then tries each measured qubit once, by decreasing index of the instruction of its last measurement, with qubits tied by one instruction in the order of their first measurement.
For each, it deletes every instruction target group that involves the qubit, and keeps the deletion if and only if the circuit still prepares the parent's signed state on the parent's output qubits, qLDPC accepts its detectors, and BZ certifies it at K = t.
The final circuit, which can equal the parent, is then certified at K = t by CC, the acceptance probabilities of parent and final circuit are estimated from 10^6 Stim shots each, and the final circuit gets its m*.
`prune_sensitivity` does the same by increasing index of the instruction of the first measurement.
`margin` and the deletion arms send each completed step, a fault weight or a tried qubit, to the runner as a partial result, so that a run killed at its limit keeps its steps.
A deletion run sends its final circuit with its final checks, before the search of its final m*, which sends its completed fault weights too.

The runner retries any run whose measured call compiled, as the [shared README](../README.md#runs) describes.
No run warms up before timing, apart from `ground_truth`, whose task process first compiles galois's GF(2) arithmetic, untimed (`_compile_galois` in [`counting.py`](counting.py)).
The count and ground-truth arms take their DEM from an untimed certification at fault weight 1.
This certification is preparation, not a warm-up, but it also compiles, or loads from the cache, the code that their measured calls share with it.
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
  The residual weights of the counterexample and of its parts come from a BZ oracle (`_describe` in `experiment.py`), so this check does not test BZ itself.

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
It checks that each record's K is t + 1, and that its noise is that of `config.json`.
It also checks each code file as `code_files.check_code_file` describes, such as that the code's |0...0> is the input's state, and that the code's n, and t where the family takes t from the distance, agree with the code file.
Where the fetched sources are present, as on the workstation, it also rebuilds the inputs from them in a scratch directory and compares.
On the node, the experiment runs the same check before any task, copies `circuits.json` into the label directory, and never imports MQT or qiskit.

The subset `smoke` has a ceiling of about 78 CPU-hours, which `--approved-cpu-hours` must cover, although its runs take minutes; `run_node.py certification --subset smoke --dry-run` prints it.

On the node, `node.sh certification --approved-cpu-hours H` runs the subset `main` (certification by both oracles, the brute-force baseline where U is within the cap, the shadow run, the counts, the margins, and the deletions), and `--subset check` the ground truth by brute force, in a label directory and archive of its own.
The ceiling of `main` follows from `circuits.json`: each run costs at most its start-up cap, its limit, its post-processing allowance, and the grace, and a fast run of `cert_bz`, `cert_cc`, or `brute_bz` gets two repeats.
The runner's retries add to this, as [`common/tasks.py`](../common/tasks.py) explains.
`run_node.py certification --dry-run` prints the tasks and the ceiling of the current `circuits.json`.
On the workstation, run `analyze.py --archive PATH --check-archive PATH --out DIR`.
It extracts each archive into a temporary directory, which it removes when it ends.
So `--archive` requires `--out`: the default output directory, `analysis/` in the label directory, would be removed with the extraction.
Its docstring lists the outputs, among them `summary.csv`, `families.csv`, and `listing_table.csv`.
`listing_table.csv` compares Phi_m and Omega_m with A_m, per family and weight.
`shadow.csv` gives the cost of the two weight oracles on identical queries: for each circuit and threshold of the queries of its `shadow` run, the calls, the rows that they submit, and the seconds of BZ and of CC, from the run's result or, for a run that did not finish, from the calls that it wrote to `oracle_calls/` in the label directory; it leaves out runs with `compiled_in_timing` set and circuits with a disagreement.
A run that wrote no calls has no row of `shadow.csv`; `summary.csv` gives its status.
The seconds of `shadow.csv` are those of the queries themselves, not of building either oracle, and BZ answers each query first.
`counterexamples.json` holds under `all` the counterexamples of the first runs of `cert_bz` and `cert_cc` that finished, and up to three examples under `examples`, one for each group of families that has one, by a rule that `analyze.py`'s docstring gives.
The groups are A-unv with the W variants of A-ft and C-goto, B-cat, and both D-synth families with their W variants, so that every W variant of `circuits.json` is in a group.
Without the archive of `check`, the outputs report the ground truth as not run.
The analysis takes the circuits and their parameters from the label's copy of `circuits.json`, and stops if the label of `check` ran with another one.
If a label does not match its `circuits.json` and the working tree's `config.json` and `pins.json`, it prints a warning and analyzes the label anyway.
It reports every task without a row as missing, and leaves every circuit with a disagreement out of the counts of verdicts and margins, medians and maxima, `listing_table.csv`, `shadow.csv`, `margin_violations.csv`, `deletion_table.csv`, and the example counterexamples.
A circuit is a disagreement if its exact results contradict each other, between two runs or within one, as where a counterexample's residual weight does not exceed its fault weight.
The column `contradictions` of `summary.csv` names the contradictions (`check_consistency` in `analyze.py`), and the column `disagreements` of `families.csv` counts these circuits.
The analysis compares, as claims on the circuit's margin m* (see [Arms](#arms)):

- the result of every run of `cert_bz`, `cert_cc`, `shadow`, `brute_bz`, `margin`, and `ground_truth` that returned one, repeats included;
- each fault weight that a run of these arms completed or found a violation at, even if it did not finish or returned no result;
- each deletion task's parent verdict at K = t;
- the final m* of a deletion task that accepted no deletion, with the fault weights that its search reported.

A result counts whatever the status of its row.
A `timeout` row can keep the result of a call that returned at or after the limit.
A `memout` row can keep the result of a call whose memory exceeded the cap before the result arrived, as found afterward (see the shared README's [Runs](../README.md#runs)).
The runner keeps no result of a run that it discarded because it compiled, so the analysis compares no claim of such a run.
The verdicts, T, and `counterexamples.json` instead use only rows with status `ok`, so `margin` can be exact where the verdict is `undecided`.
The analysis also compares A_m, from `count_accepted` or else from the ground truth, with the combinations that each run of the baseline, repeats included, enumerated on each weight.
They must be equal on a weight that the run completed.
On a weight where the run found a violation, which stops its enumeration early, they must be at most A_m.
The analysis also compares the ground truth's counts and heavy mechanisms with those of the counters.
It checks that Phi_m <= A_m, for Phi_m of `count_listing` and of the ground truth, since each listed fault is an accepted combination.
It checks that Omega_m <= Phi_m for every run of `cert_bz`, `cert_cc`, `shadow`, and `margin`, since each row that the fault search submits is a listed fault.
This holds also on a weight where the run found a violation, which only stops the search early.
It also treats as contradictions:

- a `shadow` run that ended in `ShadowDisagreement`, whether the runner kept it or discarded it because it compiled;
- a counterexample whose exact residual weight does not exceed its fault weight, by its field `residual_exceeds_fault_weight` or against the fault weight of its result;
- a counterexample whose exact residual weight is below the certifier's lower bound on it;
- for CSS weight, a counterexample whose exact residual weight is not the larger of the residual weights of its X and Z parts;
- a deletion task's final circuit that CC does not certify at K = t;
- a deletion task whose final m*, the fault weights that its search reported, and its certification at K = t disagree with each other;
- two deletion tasks that end at the same final circuit, by its SHA-256, whose final m*, fault weights, or certifications at K = t disagree.

These agreement columns of `summary.csv` compare first runs:

- `agree_bz_cc`: the verdicts of the first runs of `cert_bz` and `cert_cc`, where both finished.
- `baseline_counts_match`: the combinations that the first run of the baseline recorded, whether it finished or stopped, against A_m as above.
  It is false where they differ on a completed weight, or exceed A_m on a weight where the run found a violation; otherwise true where a completed weight has a known A_m, and empty where none has.
- `agree_baseline`: false where `baseline_counts_match` is, and equal to it where no certification finished.
  Otherwise it compares the verdict and smallest violating weight of the baseline's first run, or, if that run stopped, its recorded violations and completed weights, with those of each certification whose first run finished.
- `agree_ground_truth`: the smallest violating weight of the ground truth against that of each certification whose first run finished, and its counts against those of the counters (`ground_truth_counts_match`).
  It uses the result of the first ground-truth run that kept one, whatever its status.
- `agree_prune`: the first run of each deletion task, by its parent verdict at K = t against the first run of `cert_bz`, or of `cert_cc` where `cert_bz` did not finish, and by whether CC certifies its final circuit at K = t (`agree_prune` in `analyze_applications.py`).

The columns `agreement_bz_cc`, `agreement_baseline`, and `agreement_ground_truth` of `families.csv` count the circuits where these agree.
`agree_margin` instead compares the claims on m* of every run that the runner kept, as `contradictions` does, so a circuit can agree on its first runs and still be a disagreement.

Where `count_accepted` did not compute A_m, `listing_table.csv` divides by C(N_0, m) instead, if z >= m, so that its ratios are lower bounds on those over A_m, as its column `ratio_is_lower_bound` marks.
Where z < m, A_m may be 0, and the column `ratio_unknown` counts the circuit instead, as it does a circuit with A_m = 0.
The acceptance test AT8 runs the analysis on synthetic rows: a listing count that timed out, each kind of contradiction, shadow runs, a task without a row, rows with `compiled_in_timing` set, and rows of `--local` runs, which the analysis refuses without `--allow-local`, as the [shared README](../README.md#workflow) says.

`analyze_applications.py` adds the margins and the deletions.
`summary.csv` gives each circuit's combined margin: the column `margin` holds the m* or "all" that the claims listed above determine; if none determines it, the column `margin_lower_bound` holds the largest fault weight k up to which a run found no violation, meaning m* > k.
`families.csv` gets the margin counts: per family, the circuits with m* <= t, m* = t + 1, m* >= t + 2, "all", m* > k, and no information on m*.
`margin_violations.csv` lists the circuits without a disagreement whose exact m* is at most t.
Where the first run of `cert_bz`, `cert_cc`, or `margin` described a counterexample of weight m*, the columns `counterexample_*` of `summary.csv` and `margin_violations.csv` describe one: the arm that described it (`counterexample_arm`), the residual weights of its X and Z parts (`counterexample_sector_x`, `counterexample_sector_z`), and the parts whose own residual weight exceeds the fault weight (`counterexample_violating_parts`).
An empty cell says nothing about a counterexample, as where m* comes from a deletion parent or a run that was stopped.
These columns describe that counterexample only: they do not show whether another fault of weight m* violates in the other part, so they do not tell whether the circuit meets a criterion on one part alone.
`deletion_table.csv` has one row per deletion task whose final checks record the final circuit and its SHA-256, on a circuit without a disagreement, with the gate counts, measured qubits, measurements, and acceptance probabilities of parent and final circuit, and the final m*, with the parts of its counterexample whose residual weight exceeds the fault weight (`final_violating_parts`); it compares each final circuit only with its parent.
A task whose final checks do not record these has no row, and `deletion_notes.json` lists it instead.
A task killed during the search of its final m* keeps its row, with its status.
Its `final_margin` holds the weight at which the search found a violation, which is m*, or "all" where the task accepted no deletion and the search completed the parent's cutoff r + 1.
Otherwise its `final_margin_lower_bound` holds k, the largest of t and the fault weights that the search completed, meaning m* > k.
`deletion_notes.json` lists the deletion orders, the parents not certified at K = t, the other tasks without a row of `deletion_table.csv`, such as those that did not reach their final checks or whose final checks do not record the final circuit, with the number of their completed steps, the disagreements, and the final circuits whose rebuild from the parent's input does not match; `deletion.csv` lists every step, `deletion_tasks.csv` every task whose final checks record the final circuit, disagreements included, and `pruned/` holds the final circuits of `deletion_table.csv`, and nothing else: the analysis deletes it with everything in it first.

## Attributions and licenses

MQT QECC (MIT) provides the A and B circuits and synthesized the circuits of both D-synth families; its license travels with the inputs in `circuit_data/inputs/MQT_QECC_LICENSE`.
The code files are MQT QECC's definitions of its codes.
Chen et al.'s cat preparation (cited under [Circuits](#circuits)) comes from [vftqecc](https://github.com/vftqc/vftqecc), under the MIT license, whose notice lies beside it as `circuit_data/translated/VFTQECC_LICENSE` and travels with the inputs.
Goto's circuit (cited under [Circuits](#circuits)) comes through qLDPC's notebook `examples/logical_error_rates/5_fault_tolerance_certificates.ipynb`.
