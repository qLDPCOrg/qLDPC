# Exact code distances

This experiment measures how much four extensions of the exact Brouwer-Zimmermann (BZ) distance search that qLDPC implements change the wall time of the search, together.
The "on" arm runs qLDPC's BZ search with everything on, at its defaults.
The "prior" arm runs the same search with all four extensions turned off at once by qLDPC's tuning fields.
Both arms search the same codes, with the same time limit.
Since the extensions are turned off together, the experiment measures their joint effect, not each one's.
The four extensions, and the fields that turn them off, are:

- overlapping families of search bases and their coverage certificate, off by `overlapping_minimum_work` (Section 3.1 of qLDPC's guide [`brouwer_zimmermann.md`](../../../src/qldpc/codes/distance/brouwer_zimmermann.md));
- matroid partitioning into families of coverage two or more, restricted to disjoint information sets by `partition_max_coverage = 1` (Section 3.1 of the guide, and Section 4.4 of [`brouwer_zimmermann_code.md`](../../../src/qldpc/codes/distance/brouwer_zimmermann_code.md));
- completion tables, off by `completion_bound_minimum_layer_work` and `completion_bound_max_table_bytes` (Section 3.9 of the guide);
- the images of an information set under automorphisms, off by `automorphism_minimum_layer_work` (Section 3.7 of the guide).

Both arms keep qLDPC's other techniques, among them the search of one sector of a CSS code through a sector map ([`problems.md`](../../../src/qldpc/codes/distance/problems.md#searching-one-sector-of-a-css-code)), rounding the certificate to a proven weight multiple (Section 3.2 of the guide), witnesses shared between CSS sectors (Section 3.3), quotient labels for the stabilizer test (Section 3.4), and packed enumeration with pivot-column deletion (Section 3.5).
The prior arm is thus an ablation of the four extensions within qLDPC; its name labels this ablation, not earlier or published tools, and the guide's sections say which parts of each technique are published.

## Arms

| Arm | Call | Codes |
| --- | --- | --- |
| `q_bz_on` | `code.get_distance_exact(method="brouwer_zimmermann")` | all |
| `q_bz_prior` | the same call inside `use_tuning(**BZ_PRIOR)` | all |

`config.json` holds the prior tuning (`bz_prior`, which [`arms.py`](arms.py) reads as `BZ_PRIOR` and says what each setting turns off), the time limit, the memory cap, and the repeat rule.
Every arm builds a plain `CSSCode` or `QuditCode` from stored matrices inside its timer, so that no arm benefits from a family class.
Both arms make the same call, so they search the same sectors: one sector of a plain `CSSCode` whenever qLDPC finds a sector map, and both otherwise.
qLDPC finds one for codes with equal check matrices and, with bliss, which the bundle installs (`qldpc_extras` of [`pins.json`](../pins.json)), also for some codes whose check matrices differ, such as the anchors `surface9` and `toric10`.
The connected-cluster (CC) search is not compared here: the certification experiment measures it as an oracle of the certifier, on the certifier's own queries.

## Codes

[`codes.py`](codes.py) generates every code from fixed rules and seeds, with no timing of any kind:

- the families of `FAMILIES` in `codes.py`, each at sizes from published endpoints by geometric spacing, with a fixed number of codes per size (see the docstring of `codes.py`);
- the named anchors of `ANCHORS` in `codes.py`, most with a reference distance.

The data directory `code_data/` holds two files, which are not committed: the experiment reads them from that directory, which Git ignores, and the offline bundle carries it.
`code_data/endpoints.json` records each family's endpoints with their source, as in the table below; it is an input that no script of the experiment makes, which the experiment takes as given.
`code_data/sample.json` fixes the name, parameters, seed, and hashes of every code and anchor; `codes.py --write-sample` writes it from `endpoints.json` and the rules of `codes.py`, and `codes.py --check` checks a regenerated sample against it.
For each family, n_lo and n_hi are the smallest and largest lengths among the published instances of its source.
For `css_walk` and `rstab`, n_lo is the smallest n whose lower bound on d is at least 6, and n_hi the largest n <= 256 whose upper bound on d is at most 20, in Grassl's table of qubit stabilizer codes with k = 2.

| Family | n_lo, n_hi | Source |
| --- | --- | --- |
| `bb` | 72, 756 | S. Bravyi, A. W. Cross, J. M. Gambetta, D. Maslov, P. Rall, and T. J. Yoder, "High-threshold and low-overhead fault-tolerant quantum memory", Nature 627, 778-782 (2024), doi:10.1038/s41586-024-07107-7, arXiv:2308.07915, Table 3 of arXiv:2308.07915v2 |
| `gb` | 46, 900 | P. Panteleev and G. Kalachev, "Degenerate Quantum LDPC Codes With Good Finite Length Performance", Quantum 5, 585 (2021), doi:10.22331/q-2021-11-22-585, arXiv:1904.02703, Appendix B |
| `hgp` | 240, 1922 | A. Aydin, N. Delfosse, and E. Tham, "Cyclic Hypergraph Product Code" (2025), arXiv:2511.09683, Table 3 |
| `lp` | 544, 1428 | Q. Xu, J. P. Bonilla Ataides, C. A. Pattison, N. Raveendran, D. Bluvstein, J. Wurtz, B. Vasić, M. D. Lukin, L. Jiang, and H. Zhou, "Constant-overhead fault-tolerant quantum computation with reconfigurable atom arrays", Nature Physics 20, 1084-1090 (2024), doi:10.1038/s41567-024-02479-z, arXiv:2308.08648, Methods |
| `pp` | 110, 3122 | Okada and Kasai (2026) and Lee et al. (2026), cited under [Attributions](#attributions) |
| `qt` | 144, 576 | A. Leverrier, W. Rozendaal, and G. Zémor, "Small quantum Tanner codes from left-right Cayley complexes" (2025), arXiv:2512.20532, Table 1(b) |
| `css_walk`, `rstab` | 16, 61 | M. Grassl, "Bounds on the minimum distance of linear codes and quantum codes", http://www.codetables.de, qubit stabilizer codes with k = 2 |

`codes.py --write-sample` overwrites the frozen `sample.json`, so run it only to refreeze the sample; use `--check` otherwise.
The node regenerates the sample and stops on any difference from the frozen one, in the codes or in the metadata of their draws, such as the endpoints and each code's family, parameters, and seed (`codes.compare_with_sample`).
It ignores the name of the class that built each code and the qLDPC source tree that `sample.json` records.
Each size has 4 target codes and at most 100 draws.
A size where fewer than 4 draws are kept (see the docstring of `codes.py`) keeps fewer codes and is reported as a shortfall, such as a bivariate bicycle size where every draw has k = 0.
Sizes whose target lengths land on the same parameters merge (see the docstring of `codes.py`); in the sample's `ladders`, a target that repeats an earlier one has `merged` set to true.

## Commands

```bash
python experiments/state_prep_certification/distances/codes.py --out /tmp/codes --check
python experiments/state_prep_certification/distances/acceptance.py --local
python experiments/state_prep_certification/run_node.py distances --local --subset smoke --approved-cpu-hours 25
python experiments/state_prep_certification/distances/analyze.py --label-dir experiments/state_prep_certification/results/distances/<label>-local-smoke --allow-local
```

The local run's label directory, which `run_node.py` prints, ends in `-local-smoke`, and its rows were not timed as on the node, so the analysis refuses them without `--allow-local`, and with it marks every output with their number.

On the node, `node.sh distances --approved-cpu-hours H` runs the experiment, as the [shared README](../README.md) describes.
`analyze.py --archive PATH --out DIR` reads the archive that comes back, and writes `summary.csv`, `families.csv`, `anchors.csv`, `distances.pdf`, and `figure_notes.json`.
It extracts the archive into a temporary directory, which it removes when it ends, so `--archive` requires `--out`, since the default output directory, `analysis/` in the label directory, would be removed with the extraction.
The docstring of [`analyze.py`](analyze.py) says how to read the timings and the outputs, including a median that a timed-out run determines, an arm without a first run, and a row with `compiled_in_timing` set (see the [shared README](../README.md#runs)).

## Attributions

Pair-partition arrays and several anchors come from [`experiments/distance_tuning/workloads.py`](../../distance_tuning/workloads.py), whose instances come from J. Y. Lee, K. Okada, N. Maskara, K. Kasai, and H. Zhou, "Design principles for ultra-high-rate quantum codes" (2026), arXiv:2609.30069, and K. Okada and K. Kasai, "Pair-partition constructions for CPM-based quantum LDPC codes" (2026), arXiv:2607.14091.
The [Codes](#codes) section cites the sources of the families' endpoints, and `codes.py` cites those of the anchors.
