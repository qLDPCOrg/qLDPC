# Exact code distances

This experiment shows what qLDPC's extensions of the exact Brouwer-Zimmermann (BZ) distance search give beyond the techniques that public exact tools already use.
It runs qLDPC's BZ search with everything on, at its defaults, against the same search configured with only the techniques that some public exact tool shares with it (the "prior" arm), on the same codes and with the same time limit.
It reports wall times and deterministic work counts side by side.
The work counts need qLDPC's opt-in work counters: without them, the experiment runs only locally, and its node acceptance tests fail.
The prior arm is a configuration of qLDPC, not an external tool: public techniques that qLDPC lacks cannot be switched on, so its times are not those of any public tool.

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
Git ignores `code_data/`, which lives with the experiment's other data outside qLDPC, and which the offline bundle carries.
The node regenerates the codes and stops on any difference in the codes from the sample.
Each size has 4 target codes and at most 100 draws.
A size with fewer structurally valid draws keeps fewer codes and is reported as a shortfall, such as a bivariate bicycle size where every draw has k = 0.
Target lengths that the size rule maps to the same parameters merge, and the merge is reported.

## Commands

```bash
python experiments/state_prep_certification/distances/codes.py --out /tmp/codes --check
python experiments/state_prep_certification/distances/acceptance.py --local
python experiments/state_prep_certification/run_node.py distances --local --subset smoke --approved-cpu-hours 38
python experiments/state_prep_certification/distances/analyze.py --label-dir experiments/state_prep_certification/results/distances/<label>
```

On the node, `node.sh distances --approved-cpu-hours H` runs the experiment, as the [shared README](../README.md) describes.
`analyze.py --archive PATH --out DIR` reads the archive that comes back, and writes `summary.csv`, `families.csv`, `anchors.csv`, `distances.pdf`, and `figure_notes.json`.
A row with `compiled_in_timing` set (see the [shared README](../README.md#runs)) stops the analysis with an error that names it.

## Attributions

The constructions of the families, the 6.6.6 color codes, and the quadratic-residue codes are adapted from an earlier unpublished version of this experiment by its authors.
Pair-partition arrays and several anchors come from [`experiments/distance_tuning/workloads.py`](../../distance_tuning/workloads.py), whose instances come from Lee et al. and Okada and Kasai.
`code_data/endpoints.json` cites the sources of the size endpoints, and `codes.py` cites those of the anchors.
