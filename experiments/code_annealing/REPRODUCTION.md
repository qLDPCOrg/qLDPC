# Reproducing the paper's figures and tables

This file maps each figure and table of [the paper](https://arxiv.org/abs/2607.19574) to the command that rebuilds it from [`paper_data/`](paper_data/).
[README.md](README.md#sec-rebuild-figures) gives the running times and quick-look options.
Every artifact reproduces.
The figures come from random simulations, and match within their statistical error.

## Map

Each command runs as `python -m experiments.code_annealing.` followed by the command below, and `PAPER` stands for `experiments/code_annealing/results/paper`.

| Paper artifact | Command | Output, under `PAPER` |
| --- | --- | --- |
| Table of the best codes (`tab:codes`), and the appendix listings | `reproduce_paper --skip-figures` | `data/table_{css,swel}.tex`, `data/listing_{css,swel}.txt` |
| Logical error and discard rates (`fig:error_rates`) | `figures.plot_error_rates_by_code_type --results-dir PAPER`, after the table | `figures/error_rates_{css,swel}.pdf` |
| Simulated and estimated logical error rates (`fig:error_estimates`) | `figures.plot_error_estimates --results-dir PAPER`, after the table | `figures/error_estimates.pdf` |
| Headline claims: exact distances and counts, the Gilbert-Varshamov comparisons, the comparison with qecdb.org, and the logical Clifford gates of the automorphisms of the [[20, 6, 4]] SWEL code | `check_claims` | printed |

`reproduce_paper` without `--skip-figures` runs the first three rows.
The pseudocode, the proofs, and the circuit of the elementary SWEL-preserving gate are not computational, so no code reproduces them.

## What kind of evidence each result is

- **Exact:** the distances and the counts of minimum-weight logical operators, which are recounted rather than read from the code files.
  Also the comparisons with the Gilbert-Varshamov bounds, which compare these distances with the bounds' formulas, and the order of the group of logical Clifford gates, given its generators.
- **Trusted input:** the best known codes at qecdb.org, whose parameters are those that the database listed on the date of the scrape, and are not recomputed.
- **Estimates:** the logical error rates and discard rates of the figures, which are Monte Carlo estimates with statistical error bars and a bound on the error from errors too heavy to sample.
  Also the estimates of logical error rates from counts of minimum-weight logical operators, which are leading-order approximations.
