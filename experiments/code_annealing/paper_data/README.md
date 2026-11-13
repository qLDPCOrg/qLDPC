# Data of the paper

The files of this directory are the data behind the paper's table and figures.
No script writes them: `reproduce_paper`, `check_claims`, and `latex_tables` read them, and `reproduce_paper` copies them to `results/paper/data/` before working on the copies.

| File | What it is | How it was produced | Used by |
| --- | --- | --- | --- |
| `annealed_codes.txt` | The 26 best CSS codes of the paper, one block per code | Found by an earlier implementation of the paper's code search, and listed in the paper's appendix of best codes; stabilizer weights lowered with `reduce_weights` of `core/code_files.py` | `reproduce_paper`, `check_claims`, the table, and the figures |
| `annealed_codes_swel.txt` | The 26 best SWEL codes of the paper, in the same format | As above | As above |
| `scrape_qecdb_2026-05-21.txt` | For each code, the best known neighbors at [qecdb.org](https://qecdb.org) before the paper: the smallest `n`, largest `k`, and largest `d` among codes that are at least as good in the other two parameters | Scraped from the qecdb.org API by `analysis/scrape_qecdb.py`, with self-dual codes as the neighbors of SWEL codes; the date is that of the scrape.  The parameters are those that the database listed, a trusted input that is not recomputed | `latex_tables` (the comparison columns of the table), `check_claims` |
| `scrape_qecdb_2026-11-13.txt` | The same neighbors on 2026-11-13, for the 66 codes of the revised table: the best code of each size of the campaign of `write_jobs`, except CSS [[24, 4, 5]], where the paper's code has fewer minimum-weight logical operators | Scraped as above, on that date.  By then the database also listed codes found after the snapshot of 2026-05-21, and some of the paper's own | `latex_tables`, given both scrapes with `--data-file`, oldest first.  Each row of the revised table is compared with the snapshot of 2026-05-21, taken before the paper, if that snapshot has a line for the row's exact `(n, k, d)`, and with this scrape otherwise; those rows are marked with a dagger.  They are the 14 sizes that the paper does not have, and SWEL [[38, 10, 6]], whose distance rose from 5 |
| `automorphisms_swel_n20_k06.txt` | Five Stim circuits that generate SWAP-transversal logical Clifford gates of the [[20, 6, 4]] SWEL code | Found by a non-exhaustive search for automorphisms, and listed in the paper's appendix on automorphisms | `check_claims` |

## Formats

A code file has one block per code, separated by blank lines.
A block starts with a header `# (n, k, d) m min-weight logicals`, where `m` counts the minimum-weight X-only plus Z-only logical operators, followed by one line per stabilizer generator: X-type generators first, then Z-type, with `_` for the identity.
`core/code_files.py` documents and reads the format.

A scrape file has one line `(n, k, d) n_best k_best d_best` per code, in one block per code file (CSS first, then SWEL), with `None` for a neighbor that the database did not have.
`analysis/scrape_qecdb.py` documents and reads the format.

## Loading the codes

Loading checks each code's numbers of qubits and logical qubits against its header, and `recount` checks its distance and count exactly.
Every code is a `qldpc.codes.CSSCode`:

```python
from experiments.code_annealing.core.code_files import read_code_file, recount

entries = read_code_file("experiments/code_annealing/paper_data/annealed_codes_swel.txt")
code = entries[0].code  # the first SWEL code, a qldpc.codes.CSSCode
counts = recount(entries[0])  # an exact recount, which raises if the header is wrong
swel = code.to_swel()  # with a basis of logical operators whose X and Z parts are equal
```

## Data that is not shipped

The searches write their checkpoints, records, collected codes, and simulations under `results/`, which Git ignores; see [What the scripts write](../README.md#what-the-scripts-write) in the experiment's README.
qecdb.org grows, and may come to list the codes of the paper, so a new scrape is saved under its own date instead of replacing the snapshot here.
