# Data of the paper

The files of this directory are the data behind the paper's table and figures.
No script writes them.

| File | What it is | How it was produced | Used by |
| --- | --- | --- | --- |
| `annealed_codes.txt` | The 33 best CSS codes of the paper, from 20 to 56 qubits, one block per code | The best code of each size from the campaign of `write_jobs.py`, collected by `collect_data` with minimum-weight generators.  The paper's appendix lists them | `reproduce_paper`, `check_claims`, the table, and the figures |
| `annealed_codes_swel.txt` | The 33 best SWEL codes of the paper, in the same format | The best code of each size of the campaign, collected as above | As above |
| `scrape_qecdb_2026-11-14.txt` | For each code, the best known neighbors at [qecdb.org](https://qecdb.org): the smallest `n`, largest `k`, and largest `d` among codes that are at least as good in the other two parameters, each with the ID of a code that has it, as in `https://qecdb.org/codes/ID` | Scraped from the qecdb.org API by `analysis/scrape_qecdb.py`, with self-dual codes as the neighbors of SWEL codes.  The parameters are those that the database listed, a trusted input that is not recomputed.  The name carries the date of the scrape, because qecdb.org changes | `latex_tables` (the comparison columns of the table), `check_claims` |
| `searches.txt` | The search that found each code of the two code files, by its name, such as `css_n24_k04_s016`, which gives the code type, `n`, `k`, and seed | Found by matching the stabilizer group of each code with those of the codes that the campaign's searches saved; each code matches exactly one search | None: it records where the codes come from.  `write_jobs` writes the command of each search, under its name |
| `automorphisms_swel_n20_k06.txt` | Five Stim circuits that generate SWAP-transversal logical Clifford gates of the [[20, 6, 4]] SWEL code | Found by an exhaustive search for automorphisms of the code, with every linear combination of its stabilizer generators added to its parity check matrix, and listed in the paper's appendix on automorphisms | `check_claims` |

## Formats

A code file has one block per code, separated by blank lines.
A block starts with a header `# (n, k, d) m min-weight logicals`, where `m` counts the minimum-weight X-only plus Z-only logical operators, followed by one line per stabilizer generator: X-type generators first, then Z-type, with `_` for the identity.
`core/code_files.py` documents and reads the format.

A scrape file has one line `(n, k, d) n_best k_best d_best id_n id_k id_d` per code, in one block per code file (CSS first, then SWEL), with `None` for a neighbor that the database did not have.
Each ID names a code at qecdb.org with the value of its neighbor, the first that the database listed, since several codes may share it.
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
