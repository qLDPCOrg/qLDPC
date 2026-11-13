# Reproducing the paper's figures and tables

This file maps every figure and table of [Quantum codes from classical annealing](https://arxiv.org/abs/2607.19574) to the command that reproduces it, and records whether it reproduces.
Set up the environment as in [ENVIRONMENT.md](ENVIRONMENT.md), and run every command from the root of the repository.
The inputs are the files of [`paper_data/`](paper_data/), which its README describes.
Outputs go to `experiments/code_annealing/results/paper/`, which Git ignores.
The status is `reproduced` if the output matches the paper, and `runs-but-differs` if the command runs but its output differs, for the reasons in the notes.

## Map

| Paper artifact | Command | Inputs | Output | Status | Notes |
| --- | --- | --- | --- | --- | --- |
| Table of the best CSS and SWEL codes (`tab:codes`) | `python -m experiments.code_annealing.reproduce_paper --skip-figures` | `paper_data/annealed_codes*.txt`, `paper_data/scrape_qecdb_2026-05-21.txt` | `results/paper/data/table_css.tex`, `table_swel.tex`, for LaTeX's `\input` | `runs-but-differs` | Takes seconds.  Every cell matches the paper except the mean stabilizer weights of the SWEL codes [[22, 6, 4]] and [[24, 4, 4]], which the paper gives as 7.0 and the codes give as 7.50 and 7.20. |
| Figure of logical error rates and discard rates (`fig:error_rates`, CSS and SWEL panels) | `python -m experiments.code_annealing.figures.plot_error_rates_by_code_type --results-dir experiments/code_annealing/results/paper`, after the table command | the code files, copied by the table command | `results/paper/figures/error_rates_css.pdf`, `error_rates_swel.pdf` | `runs-but-differs` | Heavy: the default of 10^7 sampled errors per code takes many hours on one core; pass a small `--num-samples` and `--max-qubits` for a quick look.  The paper's curves were computed before qLDPC corrected its logical error rate for decoders that discard shots, which lowered the rates, so the figures differ from the paper.  Simulations are cached in `results/paper/cache/` by code and settings, not by qLDPC version, so delete that directory after updating qLDPC. |
| Figure of simulated and estimated logical error rates (`fig:error_estimates`) | `python -m experiments.code_annealing.figures.plot_error_estimates --results-dir experiments/code_annealing/results/paper`, after the table command | the code files, and `results/paper/data/half_supports.txt` from the table command | `results/paper/figures/error_estimates.pdf` | `runs-but-differs` | Heavy, as above.  The simulated points differ from the paper for the same reason.  The estimates of the paper are unchanged, apart from the refined estimate, which now also leaves out the errors that the decoder discards, as in the revised paper. |
| Appendix of the best codes (CSS and SWEL listings) | `python -m experiments.code_annealing.reproduce_paper --skip-figures` | `paper_data/annealed_codes*.txt` | `results/paper/data/listing_css.txt`, `listing_swel.txt` | `reproduced` | The listings are the code files with the paper's headers.  `check_claims` recounts their distances and counts exactly in seconds. |
| Appendix on automorphisms of the [[20, 6, 4]] SWEL code (2304 logical Clifford gates) | `python -m experiments.code_annealing.check_claims` | `paper_data/annealed_codes_swel.txt`, `paper_data/automorphisms_swel_n20_k06.txt` | printed | `reproduced` | Finds a group of 2304 logical Clifford gates modulo logical Pauli gates, as the paper reports. |

`reproduce_paper` is a grouped wrapper of the first three rows: it copies `paper_data/`, recounts every code, writes the table from the paper's snapshot of qecdb.org, and plots both figures, together with `error_rates.pdf`, which shows both code types in one figure and is not in the paper.
It accepts `--num-samples` and `--max-qubits` for a quick look.
With `--revised`, it rebuilds the table, the listings, and both figures of the revised version of the paper from `paper_data/revised/`, in `results/paper_revised/`.

Not computational, so not reproduced by code: the pseudocode of adaptive annealing, the proofs, and the circuit diagram of the elementary SWEL-preserving gate.

## What kind of evidence each result is

- Exact: the distances and counts of minimum-weight logical operators, which are recounted rather than read from the code files; the comparisons with the Gilbert-Varshamov bounds, which compare these exact distances with bounds computed from their formulas; and the order of the group of logical Clifford gates, given its generators.
- Trusted input: the best known codes at qecdb.org, whose parameters are those that the database listed on the date of the scrape, and are not recomputed.
- Estimates: the logical error rates and discard rates of the figures, which are Monte Carlo estimates with statistical error bars, and an upper bound on the error from errors too heavy to sample; and the estimates of logical error rates from counts of minimum-weight logical operators, which are leading-order approximations.

## Quick checks of the headline claims

`python -m experiments.code_annealing.check_claims` reads `paper_data/`, takes seconds, changes no file, and prints:

| Claim of the paper | What the command reports |
| --- | --- |
| The codes have the distances and numbers of minimum-weight logical operators that the paper lists | each code recounted exactly against its file |
| The distances frequently meet or exceed the Gilbert-Varshamov bounds | for each code type, how many codes meet the stabilizer bound and the CSS bound (the columns delta_QGV and delta_CSS of the table): 22 and 26 of 26 CSS codes, 17 and 26 of 26 SWEL codes |
| Many codes improve on the best codes known before the paper | for each code type, how many codes no code that qecdb.org listed on 2026-05-21 matches or betters in all of `n`, `k`, and `d`: 12 of 26 CSS codes, 14 of 26 SWEL codes |
| The [[20, 6, 4]] SWEL code has at least 2304 SWAP-transversal logical Clifford gates modulo logical Pauli gates | the order of the group that the five generators generate: 2304 |

With `--revised`, it checks the same claims for the revised paper from `paper_data/revised/`: 24 and 33 of 33 CSS codes and 18 and 33 of 33 SWEL codes meet the stabilizer and CSS bounds, 10 of 33 CSS codes and 19 of 33 SWEL codes improve on qecdb.org as of 2026-11-13, and the generators of the revised [[20, 6, 4]] SWEL code generate 2304 logical Clifford gates.
