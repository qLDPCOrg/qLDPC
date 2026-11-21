# Exact code distances

This experiment measures what four of qLDPC's extensions of the exact Brouwer-Zimmermann (BZ) distance search give.
It runs qLDPC's BZ search with everything on, at its defaults, against the same search with these four extensions turned off by qLDPC's tuning fields (the "prior" arm), on the same codes and with the same time limit:

- overlapping families of search bases and their coverage certificate, off by `overlapping_minimum_work` (Section 3.1 of qLDPC's guide [`brouwer_zimmermann.md`](../../../src/qldpc/codes/distance/brouwer_zimmermann.md));
- matroid partitioning into families of coverage two or more, restricted to disjoint information sets by `partition_max_coverage = 1` (Section 3.1 of the guide, and Section 4.4 of [`brouwer_zimmermann_code.md`](../../../src/qldpc/codes/distance/brouwer_zimmermann_code.md));
- completion tables, off by `completion_bound_minimum_layer_work` and `completion_bound_max_table_bytes` (Section 3.9);
- the images of an information set under automorphisms, off by `automorphism_minimum_layer_work` (Section 3.7).

Both arms keep qLDPC's other techniques, among them the search of one sector of a CSS code through a sector map ([`problems.md`](../../../src/qldpc/codes/distance/problems.md#searching-one-sector-of-a-css-code)), rounding the certificate to a proven weight multiple (Section 3.2 of the guide), witnesses shared between CSS sectors (Section 3.3), quotient labels for the stabilizer test (Section 3.4), and packed enumeration with pivot-column deletion (Section 3.5).
The prior arm is thus an ablation of the four extensions within qLDPC.
The experiment reports the wall time of each search.

## Arms

| Arm | Call | Codes |
| --- | --- | --- |
| `q_bz_on` | `code.get_distance_exact(method="brouwer_zimmermann")` | all |
| `q_bz_prior` | the same call inside `use_tuning(**BZ_PRIOR)` | all |

`config.json` holds the prior tuning (`bz_prior`, which [`arms.py`](arms.py) reads as `BZ_PRIOR` and says what each setting turns off), the time limit, the memory cap, and the repeat rule.
Every arm builds a plain `CSSCode` or `QuditCode` from stored matrices inside its timer, so that no arm benefits from a family class.
Both arms make the same call, so they search the same sectors: only the X sector when a plain `CSSCode` has equal check matrices.
The connected-cluster (CC) search is not compared here: the certification experiment measures it as an oracle of the certifier, on the certifier's own queries.

## Codes

[`codes.py`](codes.py) generates every code from fixed rules and seeds, with no timing of any kind:

- the families of `FAMILIES` in `codes.py`, each at sizes from published endpoints by geometric spacing, with a fixed number of codes per size (see the docstring of `codes.py`);
- the named anchors of `ANCHORS` in `codes.py`, most with a reference distance.

`code_data/endpoints.json` records each family's endpoints with its source, and `code_data/sample.json` fixes the name, parameters, seed, and hashes of every code and anchor.
The data in `code_data/` are not committed: the experiment reads them from that directory, which Git ignores, and the offline bundle carries it.
`endpoints.json` is written by hand.
`codes.py --write-sample` overwrites the frozen `sample.json`, so run it only to refreeze the sample; use `--check` otherwise.
The node regenerates the codes and stops on any difference in the codes from the sample.
Each size has 4 target codes and at most 100 draws.
A size where fewer than 4 draws are kept (see the docstring of `codes.py`) keeps fewer codes and is reported as a shortfall, such as a bivariate bicycle size where every draw has k = 0.
Sizes whose target lengths land on the same parameters merge (see the docstring of `codes.py`); in the sample's `ladders`, a target that repeats an earlier one has `merged` set to true.

## Commands

```bash
python experiments/state_prep_certification/distances/codes.py --out /tmp/codes --check
python experiments/state_prep_certification/distances/acceptance.py --local
python experiments/state_prep_certification/run_node.py distances --local --subset smoke --approved-cpu-hours 25
python experiments/state_prep_certification/distances/analyze.py --label-dir experiments/state_prep_certification/results/distances/<label>
```

On the node, `node.sh distances --approved-cpu-hours H` runs the experiment, as the [shared README](../README.md) describes.
`analyze.py --archive PATH --out DIR` reads the archive that comes back, and writes `summary.csv`, `families.csv`, `anchors.csv`, `distances.pdf`, and `figure_notes.json`.
`analyze.py`'s docstring says how it treats a row with `compiled_in_timing` set (see the [shared README](../README.md#runs)).

## Attributions

Pair-partition arrays and several anchors come from [`experiments/distance_tuning/workloads.py`](../../distance_tuning/workloads.py), whose instances come from Lee et al. and Okada and Kasai.
`code_data/endpoints.json` cites the sources of the size endpoints, and `codes.py` cites those of the anchors.
