# Exact code distances

**Prototype.** Everything in `state_prep_certification/` is a prototype, subject to change: its code, interfaces, inputs, outputs, and documentation.

This experiment measures how much four extensions of the exact Brouwer-Zimmermann (BZ) distance search that qLDPC implements together change the wall time of the search.
The "on" arm runs qLDPC's BZ search with everything on, at its defaults.
The "prior" arm runs the same search with all four extensions turned off at once by qLDPC's tuning fields.
Both arms search the same codes, with the same time limit.
Since the extensions are turned off together, the experiment measures their joint effect, not each one's.
The four extensions, and the fields that turn them off, are:

- overlapping families of search bases and their coverage certificate, off by `overlapping_minimum_work` (Section 3.1 of qLDPC's guide [`brouwer_zimmermann.md`](../../../src/qldpc/codes/distance/brouwer_zimmermann.md));
- matroid partitioning into families of coverage two or more, off by `partition_max_coverage = 1`, which keeps only disjoint information sets (Section 3.1 of the guide, and Section 4.4 of [`brouwer_zimmermann_code.md`](../../../src/qldpc/codes/distance/brouwer_zimmermann_code.md));
- completion tables, off by `completion_bound_minimum_layer_work` and `completion_bound_max_table_bytes` (Section 3.9 of the guide);
- the images of an information set under automorphisms, off by `automorphism_minimum_layer_work` (Section 3.7 of the guide).

Both arms keep qLDPC's other techniques, among them the search of one sector of a CSS code through a sector map ([`problems.md`](../../../src/qldpc/codes/distance/problems.md#searching-one-sector-of-a-css-code)), rounding the certificate to a proven weight multiple (Section 3.2 of the guide), witnesses shared between CSS sectors (Section 3.3), quotient labels for the stabilizer test (Section 3.4), and packed enumeration with pivot-column deletion (Section 3.5).
The prior arm is thus an ablation of the four extensions within qLDPC; its name labels this ablation, not earlier or published tools, and the guide's sections say which parts of each technique are published.

## Arms

| Arm | Call | Codes |
| --- | --- | --- |
| `q_bz_on` | `code.get_distance_exact(method="brouwer_zimmermann")` | all |
| `q_bz_prior` | the same call inside `use_tuning(**BZ_PRIOR)` | all |

`config.json` holds the prior tuning (`bz_prior`, which [`arms.py`](arms.py) reads as `BZ_PRIOR` and says what each setting turns off), the time limit, the memory cap, and the rule for repeats (`repeat_below_s` and `repeats`; see [`experiment.py`](experiment.py)).
Every arm builds a plain `CSSCode` or `QuditCode` from the stored matrices inside its timer.
It does not build a family's class, such as `BBCode`, which could give the search extra structure, such as symmetries.
[`arms.py`](arms.py) says which sectors the arms search, and why the connected-cluster (CC) search is not compared here.

## Codes

[`codes.py`](codes.py) generates every code from fixed rules and seeds, with no timing of any kind:

- the families of `FAMILIES` in `codes.py`, each at up to four sizes, with at most `CODES_PER_SIZE` codes per size, and possibly none;
- the anchors of `ANCHORS` in `codes.py`: named codes, most with a reference distance from an independent source.

Four targets are spaced geometrically from a family's endpoint n_lo to its endpoint n_hi (below).
Each size is the length, among those that the family's construction allows, nearest to a target, and can lie outside the endpoints.
The docstring of `codes.py` gives the rule that draws the codes of the families: its sizes, draws, merges, and shortfalls.
The analysis reports each anchor on its own, in `anchors.csv`, never pooled with the families.

The data directory `code_data/`, which Git ignores and the offline bundle carries, holds two files that are not committed.
`code_data/endpoints.json` records each family's endpoints with their source, as in the table below; it is an input that no script of the experiment makes.
`code_data/sample.json` fixes every code: a drawn code by its name, family, parameters, draw, seed, and hash, and an anchor by its name, source, hash, and reference distance, if any.
`codes.py --write-sample` writes it from `endpoints.json` and the rules of `codes.py`, overwriting the frozen sample, so run it only to refreeze the sample.
`codes.py --check` compares a regenerated sample with it, as every start does before it runs any task (`codes.compare_with_sample`).
For each family but `css_walk` and `rstab`, n_lo and n_hi are the smallest and largest lengths among the published instances of its source.
For `css_walk` and `rstab`, n_lo is the smallest n whose lower bound on d is at least 6, and n_hi the largest n <= 256 whose upper bound on d is at most 20, in Grassl's table of qubit stabilizer codes with k = 2.

| Family | Codes | n_lo, n_hi | Source |
| --- | --- | --- | --- |
| `bb` | bivariate bicycle | 72, 756 | S. Bravyi, A. W. Cross, J. M. Gambetta, D. Maslov, P. Rall, and T. J. Yoder, "High-threshold and low-overhead fault-tolerant quantum memory", Nature 627, 778-782 (2024), doi:10.1038/s41586-024-07107-7, arXiv:2308.07915, Table 3 of arXiv:2308.07915v2 |
| `gb` | generalized bicycle | 46, 900 | P. Panteleev and G. Kalachev, "Degenerate Quantum LDPC Codes With Good Finite Length Performance", Quantum 5, 585 (2021), doi:10.22331/q-2021-11-22-585, arXiv:1904.02703, Appendix B |
| `hgp` | hypergraph products of random (3, 4)-regular LDPC codes | 240, 1922 | A. Aydin, N. Delfosse, and E. Tham, "Cyclic Hypergraph Product Code" (2025), arXiv:2511.09683, Table 3 |
| `lp` | lifted products | 544, 1428 | Q. Xu, J. P. Bonilla Ataides, C. A. Pattison, N. Raveendran, D. Bluvstein, J. Wurtz, B. Vasić, M. D. Lukin, L. Jiang, and H. Zhou, "Constant-overhead fault-tolerant quantum computation with reconfigurable atom arrays", Nature Physics 20, 1084-1090 (2024), doi:10.1038/s41567-024-02479-z, arXiv:2308.08648, Methods |
| `pp` | pair-partition codes of published arrays | 110, 3122 | Okada and Kasai (2026) and Lee et al. (2026), cited under [Attributions](#attributions) |
| `qt` | quantum Tanner | 144, 576 | A. Leverrier, W. Rozendaal, and G. Zémor, "Small quantum Tanner codes from left-right Cayley complexes" (2025), arXiv:2512.20532, Table 1(b) |
| `css_walk`, `rstab` | dense CSS codes from random walks of CNOT gates, and dense random stabilizer codes, both with k = 2 | 16, 61 | M. Grassl, "Bounds on the minimum distance of linear codes and quantum codes", http://www.codetables.de, qubit stabilizer codes with k = 2 |


## Commands

```bash
python experiments/state_prep_certification/distances/codes.py --out /tmp/codes --check
python experiments/state_prep_certification/distances/acceptance.py --local
python experiments/state_prep_certification/run_node.py distances --local --subset smoke --approved-cpu-hours 25
python experiments/state_prep_certification/distances/analyze.py --label-dir experiments/state_prep_certification/results/distances/<label>-local-smoke --allow-local
```

`codes.py` writes the regenerated codes and `draws.jsonl` to `--out`, a scratch directory.
The subset `smoke` runs both arms on four small anchors.
Its runs take minutes, but its ceiling is about 24 CPU-hours, which the approval of 25 covers.
The local run's label directory, which `run_node.py` prints, ends in `-local-smoke`, and the analysis of its rows needs `--allow-local` (see the shared README's [Local runs](../README.md#local-runs)).

On the node, `node.sh distances --approved-cpu-hours H` runs the experiment, as the [shared README](../README.md) describes.
`analyze.py --archive PATH --out DIR` reads the archive that comes back, and writes `summary.csv`, `families.csv`, `anchors.csv`, `distances.pdf`, and `figure_notes.json`.
The docstring of [`analyze.py`](analyze.py) says how to read the timings and the outputs, including a median that a timed-out run determines, an arm without a first run, and a row with `compiled_in_timing` set (see the [shared README](../README.md#runs)).

## Exact results and timings

The distances that the arms report are exact, from qLDPC's exact BZ search.
A code is a disagreement if, among its rows with status `ok`, two report different distances, or one reports a distance other than the code's reference distance; `analyze.py` reports it (case 1 of its docstring).
The anchors' reference distances and the families' endpoints are trusted inputs, from the sources that this README and `codes.py` cite.
The times are measurements of runs on the node.

## Attributions

The pair-partition arrays of the `pp` family and several anchors come from [`experiments/distance_tuning/workloads.py`](../../distance_tuning/workloads.py), whose instances come from J. Y. Lee, K. Okada, N. Maskara, K. Kasai, and H. Zhou, "Design principles for ultra-high-rate quantum codes" (2026), arXiv:2609.30069, and K. Okada and K. Kasai, "Pair-partition constructions for CPM-based quantum LDPC codes" (2026), arXiv:2607.14091.
The [Codes](#codes) section cites the sources of the families' endpoints, and `codes.py` cites those of the anchors.
