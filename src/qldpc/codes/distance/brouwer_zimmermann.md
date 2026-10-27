# Brouwer-Zimmermann distance search in qLDPC

## 1. Purpose and scope

The Brouwer-Zimmermann (BZ) algorithm computes the distance of a classical code ([arXiv:1603.06757](https://arxiv.org/abs/1603.06757)), and has been adapted to quantum codes ([arXiv:2408.10743](https://arxiv.org/abs/2408.10743)).
This guide explains how qLDPC computes the exact distance of a binary code using BZ.
It is written for contributors who need to understand, debug, or modify the BZ code in this directory.
It is organized as follows:

- [Section 2](#2-the-brouwer-zimmermann-algorithm) explains the BZ algorithm as published, including its adaptation to quantum codes.
- [Section 3](#3-extensions-and-optimizations) explains what qLDPC adds to the published algorithm: extensions that strengthen the lower certificate or improve the upper bound, and optimizations that make the search cheaper to run.
- [Section 4](#4-guide-to-the-source-code) is a guide to the source code: how the modules are organized, how a computation flows through them, which code names correspond to the concepts of [Section 2](#2-the-brouwer-zimmermann-algorithm) and [Section 3](#3-extensions-and-optimizations), and which behavior spans several files.
- [Section 5](#5-correctness-requirements) summarizes the correctness requirements that apply when changing the implementation.

Sections [2](#2-the-brouwer-zimmermann-algorithm) and [3](#3-extensions-and-optimizations) have been carefully reviewed to help human readers understand the BZ algorithm and its implementation in qLDPC at a high level.
Sections [4](#4-guide-to-the-source-code) and [5](#5-correctness-requirements) are highly technical summaries written to help AI agents navigate the BZ implementation in qLDPC.

## 2. The Brouwer-Zimmermann algorithm

This section describes BZ as presented in [arXiv:1603.06757](https://arxiv.org/abs/1603.06757) and [arXiv:2408.10743](https://arxiv.org/abs/2408.10743).

### 2.1 The problem

BZ works over the binary field, where addition is bitwise XOR.
`wt(v)` denotes the Hamming weight of a binary word `v`, meaning its number of nonzero coordinates.

For a classical code with generator matrix `G`, the distance is the smallest weight of a nonzero codeword:

$$
d = \min_{v \in V,\;v \ne 0} \operatorname{wt}(v),
\qquad V = \operatorname{rowspan}(G).
$$

For a quantum code, the distance is the smallest weight of a nontrivial logical operator.
Let `S` be the span of the stabilizers (or, for a subsystem code, the stabilizers and gauge operators), and let `V` be the span of `S` together with the logical operators.
Words in `S` act trivially on the encoded information, so the nontrivial logical operators are the words in `V` that are outside `S`.
If the code encodes at least one logical qubit (`V ≠ S`), its distance is

$$
d = \min_{v \in V \setminus S} \operatorname{wt}(v).
$$

The classical and quantum definitions are instances of one problem: given binary spaces `S ⊆ V` with `V ≠ S`, find the lightest word in `V \ S`.
For a classical code, `V` is the code and `S = {0}`.

A word in `V \ S` is called **eligible**, because it can determine the distance.
For a classical code, the eligible words are the nonzero codewords.
For a quantum code, they are the nontrivial logical operators.

### 2.2 The idea: upper bounds and lower certificates

Every codeword is a sum of rows of a basis of `V`.
Throughout this guide, a **basis** of `V` is a matrix whose rows are linearly independent and span `V`.
Let `r = dim(V)` be the number of rows in a basis of `V`, and let `N` be its number of columns, which is the number of **coordinates** of a word.
A coordinate at which every word of `V` is zero carries no information, so we assume that there are none.

For a basis `G` of `V`, a **coefficient vector** `a` is a binary vector of length `r` that selects rows, and `aG` is the XOR of the selected rows.
The **coefficient weight** of `aG` is `wt(a)`, the number of rows selected.
Since the rows of `G` are independent, each word of `V` has exactly one coefficient vector `a` in a given basis.
A **layer** is the set of all words with a given coefficient weight `w`, taken over all choices of `w` rows.

BZ keeps track of two quantities:

- A **witness** is an eligible word whose weight is known.
  The lightest known witness has weight `U`, which is an **upper bound** on the distance.
  The distance is at most `U`, but a lighter eligible word might exist.
- A **lower certificate** `L` is a proven statement that every eligible word that has not yet been examined has weight at least `L`.

Once `L ≥ U`, no unexamined word can be lighter than the best witness.
Then `U` is the distance, and the search stops.

The search examines words layer by layer, starting with coefficient weight one.
Examining a layer can only lower `U`.
Completing layers also strengthens `L`, because the words that remain unexamined all need more rows.
The remaining question is why unexamined words must be heavy, which is the role of information sets.

### 2.3 Systematic bases and information sets

An **information set** is a set `I` of `r` coordinate indices (column indices) such that the columns of the basis indexed by `I` form a rank-`r` matrix.
Row reduction on `I` produces a **systematic basis** `G_I`, whose restriction to `I` is the identity matrix:

$$
G_I[:,I] = I_r.
$$

The columns indexed by `I` are the **pivot columns** of `G_I`.

For a word `v`, write `v|_I` for the restriction of `v` to the coordinates indexed by `I`.
Let `v` be any word in `V`, and let `a` be its coefficient vector in `G_I`, so that `v = aG_I`.
Then `v|_I = a`, because the columns of `G_I` indexed by `I` form the identity matrix:

$$
v|_I = a\,G_I[:,I] = a\,I_r = a.
$$

Hence `wt(v|_I) = wt(a)`:

> For every word `v` in `V`, the Hamming weight `wt(v|_I)` equals the coefficient weight of `v` in `G_I`, which is a lower bound for the Hamming weight `wt(v)`.

**Searching a basis through weight `t`** means examining every word with coefficient weight at most `t` with respect to `G_I`; that is, every XOR of at most `t` rows of `G_I`.
Each word in `V` has exactly one coefficient vector in `G_I`, so a word that is not examined by this search has coefficient weight at least `t+1`.
Since `wt(v|_I)` equals the coefficient weight of `v`, such a word `v` satisfies `wt(v) ≥ wt(v|_I) ≥ t+1`.
[Section 2.5](#25-the-stopping-certificate) uses this consequence to lower-bound the weight of every word that the search has not examined.

### 2.4 Search bases and rank-deficient blocks

BZ searches `m` bases `G_1, ..., G_m` of `V`, which we call the **search bases**.
Each `G_i` is row reduced on a **block** `B_i`, a set of column indices, and the blocks `B_1, ..., B_m` are pairwise disjoint.
We call such a collection of search bases a **disjoint family of search bases**.
These blocks are chosen greedily: `B_1` is an information set, `B_2` is an information set among the columns not in `B_1`, and so on, until the remaining columns no longer contain an information set.
If any columns remain at that point, they form one last block, which has rank less than `r`.

Each block `B_i` that is an information set gives a systematic basis, as in [Section 2.3](#23-systematic-bases-and-information-sets).
Only the last block, `B_m`, can have rank `r_m < r`, and then it is not an information set.
In general, if `B_i` has rank `r_i`, row reducing on `B_i` gives a basis `G_i` of the whole space `V` in which:

- the first `r_i` rows have distinct pivot columns in `B_i`, and
- the remaining `r - r_i` rows are zero on the columns indexed by `B_i`.

Suppose a word `v` has coefficient weight at least `t+1` in `G_i`.
At most `r - r_i` of its selected rows can be zero on `B_i`, so at least `t + 1 - (r - r_i)` of them are pivot rows.
Each pivot row contributes a distinct nonzero coordinate to `v|_{B_i}`, the restriction of `v` to `B_i`, so

$$
\operatorname{wt}(v|_{B_i}) \ge b_i(t) := \max(0,\;t+1-(r-r_i)).
$$

For a full information set (`r_i = r`, `B_i = I`), this is `b_i(t) = t + 1`, matching [Section 2.3](#23-systematic-bases-and-information-sets).

### 2.5 The stopping certificate

Suppose that each of the search bases `G_1, ..., G_m` of [Section 2.4](#24-search-bases-and-rank-deficient-blocks) has been searched through coefficient weight `t`, in the sense of [Section 2.3](#23-systematic-bases-and-information-sets).
Consider an eligible word `v` that was not examined in any of them.
For each `i`, the coefficient weight of `v` in `G_i` is at least `t+1`.
By Sections [2.3](#23-systematic-bases-and-information-sets) and [2.4](#24-search-bases-and-rank-deficient-blocks), `wt(v|_{B_i}) ≥ b_i(t)` for every `i`.

Because the blocks are pairwise disjoint, they share no column indices, so these weights add up:

$$
\operatorname{wt}(v) \ge \sum_i b_i(t) =: L(t).
$$

If all `m` blocks are information sets, `L(t) = m(t+1)`.

`L(t)` is the lower certificate: every eligible word that has not been examined has weight at least `L(t)`.
It makes no claim about the words that have been examined, so a witness that was already found can be lighter than `L(t)`, and `L(t)` can exceed the distance.
When `L(t) ≥ U`, no unexamined eligible word is lighter than `U`, which is enough to stop.

BZ is therefore the following loop:

1. Build the search bases `G_1, ..., G_m` from disjoint blocks, as in [Section 2.4](#24-search-bases-and-rank-deficient-blocks).
2. Find the lightest eligible row among the rows of the search bases, which gives an initial `U`.
3. For `t = 1, 2, ...`, examine every selection of `t` rows in each search basis, lowering `U` whenever a lighter eligible word is found.
   Then compute `L(t)`, and stop if `L(t) ≥ U`.

The search always terminates, because after `t = r` every nonzero word has been examined.
The result is then the exact distance.
Algorithm 1 of [arXiv:1603.06757](https://arxiv.org/abs/1603.06757v2) gives this loop and this certificate for disjoint and rank-deficient blocks.

The cost of this loop is measured in **candidate/set evaluations**.
One candidate/set evaluation is the examination of one selection of rows in one basis.
Searching the `m` search bases through coefficient weight `t` therefore costs `m` times the number of selections of at most `t` rows, which is `m * sum_{w=1}^{t} binom(r, w)`.

### 2.6 Excluding stabilizers

For a quantum code, the lightest word in `V` may lie in `S`.
Such a word is not eligible, so it must not lower `U`.
Section 2.1 of [arXiv:2408.10743](https://arxiv.org/abs/2408.10743v2) therefore checks whether a candidate word of weight below `U` lies in `S` before updating `U`.
qLDPC performs this membership test with quotient labels ([Section 3.4](#34-quotient-labels-for-the-stabilizer-test)).
The lower certificate of [Section 2.5](#25-the-stopping-certificate) is unchanged, because it holds for all unexamined words, eligible or not.

### 2.7 Pauli weight as Hamming weight

An operator that contains only X (or only Z) Pauli operators on `n` qubits is a binary vector of length `n`, and its Hamming weight is its physical weight.

A general Pauli operator is a symplectic row `[x | z]` of length `2n`.
Its physical weight is the number of qubits on which `x` or `z` is nonzero, which differs from the Hamming weight of `[x | z]`.

To search for the minimum physical weight with a Hamming-weight algorithm, Section 2.4 of [arXiv:2408.10743](https://arxiv.org/abs/2408.10743v2) uses the `Saved_isometry` construction, which follows earlier work of White and Grassl.
It maps each symplectic row to a binary word of length `3n`:

$$
T(x,z) = (x \mid z \mid x+z).
$$

On each qubit, the pair `(x_i, z_i)` is `(0,0)` for the identity, and `(1,0)`, `(0,1)`, or `(1,1)` for a nontrivial Pauli operator.
The identity contributes Hamming weight zero to `T`, and each nontrivial Pauli operator contributes weight exactly two.
Hence

$$
\operatorname{wt}_{H}(T(v)) = 2\operatorname{wt}_{P}(v),
$$

where `wt_P` is the physical Pauli weight.

BZ therefore searches words with `3n` coordinates, and the quantum distance is half of the minimum Hamming weight that BZ finds.
Because `T` is linear and injective, it maps the spaces `S ⊆ V` to spaces `T(S) ⊆ T(V)`, and a word of `T(V)` is eligible exactly when its preimage lies outside `S`.

## 3. Extensions and optimizations

qLDPC adds extensions and optimizations to the published BZ algorithm.
Extensions strengthen the lower certificate or the upper bound, while optimizations make the search cheaper to run.

The extensions are:

- [Section 3.1](#31-overlapping-search-bases) strengthens the lower certificate by allowing for search bases with overlapping blocks.
- [Section 3.2](#32-rounding-the-certificate-to-a-proven-weight-multiple) strengthens the lower certificate by rounding it up when the weight of every eligible word is a multiple of some integer.
- [Section 3.3](#33-sharing-witnesses-between-css-sectors) tightens the upper bound by sharing witnesses between the two sectors of a CSS code.

The optimizations are:

- [Section 3.4](#34-quotient-labels-for-the-stabilizer-test) tests whether a candidate word lies in `S` by XORing precomputed labels.
- [Section 3.5](#35-packed-enumeration-with-pivot-column-deletion) packs words into machine words, reuses partial XORs, and avoids storing pivot columns.
- [Section 3.6](#36-exhaustive-enumeration-as-a-fallback) enumerates every word when that is cheaper than running BZ.

[Section 3.7](#37-worked-example) illustrates [Section 3.1](#31-overlapping-search-bases) and [Section 3.2](#32-rounding-the-certificate-to-a-proven-weight-multiple) with an example.

### 3.1 Overlapping search bases

#### Motivation

For a disjoint family of search bases, [Section 2.5](#25-the-stopping-certificate) gives the lower certificate as a sum over the search bases:

$$
L(t) = \sum_i b_i(t).
$$

The term `b_i(t)` for the search basis `G_i` is at most `t+1`, and it equals `t+1` exactly when the block `B_i` is an information set ([Section 2.4](#24-search-bases-and-rank-deficient-blocks)).
So `L(t)` grows with the number of search bases whose blocks are information sets.

An information set has `r` column indices, and the blocks of a disjoint family share no column indices.
A disjoint family therefore has at most `floor(N / r)` blocks that are information sets.
When `N / r` is not an integer, the remaining coordinates cannot form another information set, so they form at most one rank-deficient block, whose term `b_i(t)` is smaller.

For example, take `N = 64` and `r = 41`.
A disjoint family has one block that is an information set, with term `b_1(t) = t+1`.
The other 23 coordinates form a rank-deficient block whose term `b_2(t)` is zero for `t ≤ 17`.
Hence `L(t) = t+1` for `t ≤ 17`, and the 23 other coordinates do not help the certificate.

An **overlapping family of search bases** is a collection of search bases whose blocks may share column indices.
Because its blocks need not be disjoint, such a family is not limited to `floor(N / r)` blocks that are information sets.
However, a coordinate that lies in several blocks is counted in several of the terms `b_i(t)`.
The certificate derived below accounts for this duplication by dividing the sum of the terms by the maximum coverage `C`, defined below.
For `m` blocks that are information sets, this yields a certificate `L(t)` of about `(m/C)(t+1)`, which exceeds the `k(t+1)` of a disjoint family with `k` information-set blocks whenever `m/C > k`.

#### The certificate

Let `C` be the **maximum coverage** of an overlapping family, which is the largest number of blocks `B_i` that contain any single column index.
Suppose that each search basis of the family has been searched through coefficient weight `t`, and let `v` be an eligible word that was not examined in any of them.
As in [Section 2.5](#25-the-stopping-certificate), `wt(v|_{B_i}) ≥ b_i(t)` for every `i`.
Each nonzero coordinate `j` of `v` is counted in `wt(v|_{B_i})` for every block `B_i` that contains `j`, and there are at most `C` such blocks.
Summing over blocks therefore gives

$$
\sum_i b_i(t) \le \sum_i \operatorname{wt}(v|_{B_i}) \le C\operatorname{wt}(v).
$$

Because `wt(v)` is an integer, this gives the lower certificate

$$
L(t) =
\left\lceil \frac{\sum_i b_i(t)}{C} \right\rceil.
$$

If all `m` blocks are information sets, it becomes

$$
L(t) = \left\lceil \frac{m(t+1)}{C} \right\rceil.
$$

A disjoint family is the special case `C = 1`, which recovers [Section 2.5](#25-the-stopping-certificate).
The formula holds whether or not each block is an information set, through the terms `b_i(t)`.

[arXiv:1603.06757](https://arxiv.org/abs/1603.06757v2) mentions in its Section 2 that overlapping information sets require a different formula for the certificate, citing Grassl, but does not give one.
The coverage-divided certificate above is derived for qLDPC.

#### A small example

Take the code generated by the rows `101` and `011`, whose distance is two.
It has `N = 3` coordinates and `r = 2`, so a disjoint family contains at most `floor(3/2) = 1` search basis whose block is an information set.
An overlapping family can use all three pairs of column indices, `{0,1}`, `{0,2}`, and `{1,2}`, each of which is an information set.
Then `m = 3`, and each column index lies in two of the blocks, so `C = 2`.
After all three search bases have been searched through coefficient weight `t = 1`, the formula gives `L(1) = ceil(3*2/2) = 3`.
This exceeds the distance, which is allowed because `L(t)` is a statement only about words that have not been examined.
Every row has weight two, so `U = 2 ≤ L(1)`, and the search stops with distance two.

### 3.2 Rounding the certificate to a proven weight multiple

Suppose that every eligible word has a weight that is a multiple of an integer `D`.
An unexamined eligible word then has a weight that is at least `L(t)` and a multiple of `D`, so it has a weight of at least `D * ceil(L(t)/D)`, the smallest multiple of `D` that is at least `L(t)`.
The certificate can therefore be rounded up to that value.

In qLDPC, `D` comes from two sources:

- For general Pauli problems, the map of [Section 2.7](#27-pauli-weight-as-hamming-weight) doubles every weight, which gives `D = 2`.
- For any problem, qLDPC checks whether every row of the search basis has even Hamming weight.
  The XOR of two even-weight words has even weight, so if all rows are even, every word in the space is even.
  When this holds, `D = 2` is combined with any existing divisor by taking the least common multiple.

Parity is the only property checked in this way, because evenness of the rows carries over to all of their sums.
A larger divisor does not: the rows `11110` and `01111` each have weight four, but their XOR `10001` has weight two.

### 3.3 Sharing witnesses between CSS sectors

The distance of a CSS code is `min(d_x, d_z)`, where `d_x` and `d_z` are the distances of its X and Z sectors.
qLDPC runs one BZ search for each sector, and advances the searches in round-robin order, one layer at a time.
A witness found in either sector is a nontrivial logical operator of the code, so it lowers the upper bound `U` used by both searches.

Each search keeps its own lower certificate, because a certificate bounds only the unexamined words of its own sector.
The sectors can also differ in their search bases, coverage, and weight divisor.
For example, suppose that the X sector finds a witness of weight seven, so that `U = 7` for both sectors.
If the Z sector contains only even-weight words, it can have no witness of weight seven.
Its search still stops as soon as its own certificate reaches `U = 7`.
At that point it has shown only that `d_z ≥ 7`, and hence `d_z ≥ 8`, without determining the exact value of `d_z`.
A joint search therefore determines `min(d_x, d_z)`, while the individual sector distances may remain unknown.

### 3.4 Quotient labels for the stabilizer test

[Section 2.6](#26-excluding-stabilizers) requires deciding whether a candidate word lies in `S`.
Testing membership directly would reduce each candidate against a basis of `S`.
Quotient labels avoid that reduction, because they turn the test into an XOR of precomputed rows.

For the problem of [Section 2.1](#21-the-problem) with nested spaces `S ⊆ V`, let `s = dim(S)` and `q = dim(V/S)`, so that `r = s + q`.
Take a basis `G` of `V` of the form

```text
G = [ s independent rows spanning S                       ]
    [ q independent rows, whose cosets form a basis of V/S ]
```

Such a basis, whose first rows span the excluded space `S`, is called an **adapted basis**.
Each row of `G` is given a binary **label**.
The labels form a matrix `M` with `r` rows and `q` columns:

```text
M = [ zero matrix with s rows     ]
    [ identity matrix, q x q      ]
```

For any coefficient vector `a`,

$$
aG \in S \quad\Longleftrightarrow\quad aM = 0.
$$

A candidate word `aG` is therefore eligible exactly when the XOR of its selected row labels, `aM`, is nonzero.
For a classical search, `S = {0}`, so no labels are needed and every nonempty selection of rows is eligible.

Each search basis of [Section 2.4](#24-search-bases-and-rank-deficient-blocks) is obtained by row reducing `G`, so every row operation on `G` must be applied identically to `M`.
If an invertible row transformation replaces `G` by `G' = EG`, the labels must become `M' = EM`.
Then `aM'` is still the label of the word `aG'`.
Omitting the transformation of `M` would either accept stabilizers as witnesses or reject nontrivial logical operators.

### 3.5 Packed enumeration with pivot-column deletion

Packing stores a word of `N` coordinates into `ceil(N/64)` 64-bit machine words.
Then one XOR of two words costs one operation per machine word, and the weight of a word is a population count of its machine words.
[arXiv:1603.06757](https://arxiv.org/abs/1603.06757v2) describes packing bits into machine words.

Enumerating a layer uses prefix XORs.
Describe a selection of `w` rows by its **support**, the selected row indices in increasing order, and visit the supports of a layer in lexicographic order.
Consecutive supports then share their first rows, so the XOR of each such shared prefix is computed once and reused.
Extending a support by one row costs one row XOR per search basis instead of a new XOR of all selected rows.
Reusing partial XORs follows the idea of saved additions in [arXiv:1603.06757](https://arxiv.org/abs/1603.06757v2).

Pivot-column deletion is a qLDPC choice.
In a search basis `G_i` with block `B_i`, each of the first `r_i` rows has a 1 in exactly one pivot column of `B_i`, in distinct columns, and the other rows are zero on `B_i`.
Hence for a support of `w` selected rows, the number of nonzero coordinates on the pivot columns is the number of selected row indices below `r_i` (counting from 0), which is `w` when `B_i` is an information set.
This number is determined by the support alone, so the pivot columns need not be stored.
qLDPC deletes them before packing, and it computes the weight of a word as this known count plus the population count of the packed XOR of the remaining columns.
For example, take `r = 2` and rows `1 0 | 1 1` and `0 1 | 1 0`, with pivot columns `{0, 1}`.
The stored rows are `1 1` and `1 0`.
The support `{0, 1}` gives the stored XOR `0 1`, so the weight is `2 + 1 = 3`, which is the weight of the full word `1 1 0 1`.

Let `N_stored` be the number of columns in the stored basis before deletion.
This equals `N` under the assumption of [Section 2.2](#22-the-idea-upper-bounds-and-lower-certificates), but the implementation also accepts and retains all-zero columns.
A block of rank `r_i` has `r_i` pivot columns, so deletion shrinks each stored row from `ceil(N_stored/64)` to `ceil((N_stored - r_i)/64)` machine words.
For an information set, `r_i = r`.
The saving depends on crossing a 64-bit boundary: if `N_stored ≤ 64` and at least one column remains, a row occupies one word with or without deletion.
This is the situation in [Section 3.7](#37-worked-example).
If every column is a pivot column, the stored remainder is empty.

When both widths span many machine words, the ratio of word counts is approximately `N_stored / (N_stored - r_i)`.
Rounding can make the actual ratio larger or smaller than this approximation.
For example, deleting 65 pivot columns from a 129-column basis reduces the word count from three to one, a factor of three rather than about two.
For the X sector of a CSS stabilizer code, `r = N_stored - rank(H_z)`.
In a balanced code with `rank(H_z)` near `N_stored/2`, the large-width approximation therefore suggests about a factor of two for full blocks.
These are counts of packed words, not measured speedups: label XORs, comparisons with `U`, and loop overhead do not shrink.

Weights are compared with `U` before labels are examined.
A candidate whose weight is at least `U` cannot lower `U`, so its label is not examined, and labels are examined only for lighter candidates.

### 3.6 Exhaustive enumeration as a fallback

For small or low-dimensional problems, the setup cost of BZ exceeds the cost of enumerating every eligible word.
Neither [arXiv:1603.06757](https://arxiv.org/abs/1603.06757v2) nor [arXiv:2408.10743](https://arxiv.org/abs/2408.10743v2) describes switching to exhaustive enumeration, so it is a qLDPC choice.
With `q = dim(V/S)` as in [Section 3.4](#34-quotient-labels-for-the-stabilizer-test), the number of eligible words is `(2^q - 1) * 2^(r-q)`.

The estimated cost of BZ is a work estimate in candidate/set evaluations.
Let `m` be the number of search bases, and let `t*` be the smallest layer after which the certificate reaches `U`.
The certificate is the one of [Section 3.1](#31-overlapping-search-bases), rounded as in [Section 3.2](#32-rounding-the-certificate-to-a-proven-weight-multiple), for the ranks and maximum coverage of the family of search bases under consideration.
The estimate is

$$
W = m\sum_{w=1}^{t_*} \binom{r}{w}.
$$

The estimate is pessimistic, because it assumes that no lighter witness is found before the certificate reaches `U`.
The sum starts at `w = 1`, so it includes the systematic rows.
When the comparison with exhaustive enumeration is made, layer one has already been checked, so `W` slightly overestimates the remaining BZ work.
Actual runtime also depends on basis construction, the width of packed words, and early improvements of the witness, so `W` is not a runtime guarantee.

Exhaustive enumeration is chosen at three points, which [Section 4.2](#42-flow-and-exit-points) lists.
It enumerates words in vectorized, packed blocks and updates them with Gray-code steps, and it does not use the BZ certificate.

### 3.7 Worked example

This example uses one CSS sector of the QC64 code from the test suite ([_brouwer_zimmermann_test.py](_brouwer_zimmermann_test.py)).
The sector has these properties:

- It has 64 coordinates and dimension `r = 41`.
- The lightest eligible row of its basis has weight eight, so initially `U = 8`.
- Every word in it has even weight, so `D = 2` ([Section 3.2](#32-rounding-the-certificate-to-a-proven-weight-multiple)).

A disjoint family can contain only `floor(64/41) = 1` search basis whose block is an information set.
The leftover columns hold a rank-deficient block, which contributes nothing to `L(t)` for `t ≤ 17`.
Hence `L(t) = t + 1` for `t ≤ 17`, and `L(6) = 7` rounds up to eight.
The search stops after layer six, so it enumerates layers two through six (layer one consists of the rows already checked).

An overlapping family of `m = 3` search bases, each with an information set as its block, can be chosen with maximum coverage `C = 2`.
Completing layer four gives

$$
L(4)=\left\lceil\frac{3(4+1)}{2}\right\rceil=8,
$$

which reaches eight even before rounding to an even weight.
The search stops after layer four.

| Family | Last layer | Candidate/set evaluations in layers two and above |
| --- | ---: | ---: |
| Disjoint, one search basis | 6 | $\sum_{w=2}^{6}\binom{41}{w} = 5{,}358{,}536$ |
| Overlapping, three search bases, `C = 2` | 4 | $3\sum_{w=2}^{4}\binom{41}{w} = 338{,}250$ |

The overlapping family needs fewer evaluations because it avoids layers five and six.
The counts cover only enumeration; building and packing the search bases costs extra.

## 4. Guide to the source code

**Audience note:** This section is a highly technical summary written to help AI agents navigate the BZ implementation in qLDPC.
For a carefully reviewed, high-level explanation intended for human readers, see Sections [2](#2-the-brouwer-zimmermann-algorithm) and [3](#3-extensions-and-optimizations).

This section is a guide to the source files in this directory.
It connects the concepts of Sections [2](#2-the-brouwer-zimmermann-algorithm) and [3](#3-extensions-and-optimizations) to modules, identifiers, and behavior that spans files.
Heuristic tunables are identified by parameter name so that their current values can be found in the source.

### 4.1 Module map and layering

| File | Responsibility | Key functions |
| --- | --- | --- |
| [core.py](core.py) | Public interfaces, input validation, adapted bases, method selection, validation of exact-distance method and distance-bound backend selectors, and conversion between symplectic and Hamming units | `get_distance_classical`, `get_distance_quantum`, `_validate_distance_method`, `_validate_distance_backend` |
| [_brouwer_zimmermann.py](_brouwer_zimmermann.py) | The BZ distance search: preparation exits, witness discovery, scanning, sharing between CSS sectors, and exhaustive cost models | `get_distance_css_brouwer_zimmermann`, `get_distance_brouwer_zimmermann_many`, `_prepare_brouwer_zimmermann_search`, `_BrouwerZimmermannSearch`, `_scan_brouwer_zimmermann` |
| [_information_sets.py](_information_sets.py) | Families of search bases and their certificates, shared by distance and counting and used by the Pauli-coset oracle: disjoint packing, pruning, overlapping replacement, parity inference, the lower certificate, and the work estimate | `get_disjoint_family`, `prune_family`, `get_overlapping_family`, `prepare_family`, `BasisFamily`, `Certificate` |
| [_brute_force.py](_brute_force.py) | Exhaustive enumeration | `get_distance_quantum_brute_force` |
| [_binary.py](_binary.py) | Shared binary helpers: row reduction with labels, independent and adapted bases, packing, weight functions, and the map of [Section 2.7](#27-pauli-weight-as-hamming-weight) | `row_reduce_binary`, `get_independent_rows`, `get_nested_code_basis`, `symplectic_to_hamming` |
| [_compilation.py](_compilation.py) | Compiled functions: the disk cache of the always-compiled functions, named by a hash of the source files of the package | `always_compiled` |

[`__init__.py`](__init__.py) exports names from [core.py](core.py) and [logical_counts.py](logical_counts.py).
[logical_counts.py](logical_counts.py) counts the logical operators of a given weight exactly.
The code that only counting uses is in the `_counting` package, and its scan of the words is in [_counting/scan.py](_counting/scan.py).
It builds its families and certificates with the same stages of [_information_sets.py](_information_sets.py) as the distance search, under the constraint in [Section 5](#5-correctness-requirements).
The family policy lives only in that module, so a change to it applies to both consumers.
[core.py](core.py) calls [_brouwer_zimmermann.py](_brouwer_zimmermann.py) and [_brute_force.py](_brute_force.py), and [_brouwer_zimmermann.py](_brouwer_zimmermann.py) calls [_brute_force.py](_brute_force.py) for its exhaustive routes.
All three modules, and [_information_sets.py](_information_sets.py), use [_binary.py](_binary.py).
Every module with compiled functions uses [_compilation.py](_compilation.py).
`get_distance_css_brouwer_zimmermann` builds an adapted basis ([Section 3.4](#34-quotient-labels-for-the-stabilizer-test)) for each nontrivial sector, then passes the resulting problems to `get_distance_brouwer_zimmermann_many`.

### 4.2 Flow and exit points

A public entry point validates inputs and chooses a route.
For the BZ route, `get_distance_brouwer_zimmermann_many` prepares each problem and coordinates any searches that remain.
Preparation either returns a result immediately or produces a `_BrouwerZimmermannSearch` whose `advance` method searches one coefficient layer.
For a joint CSS search, the coordinator advances the prepared searches in round-robin order and shares the best witness weight.

The tables below index the public routes, the preparation exits, and the subsequent search.
An integer returned by preparation can incorporate a witness from another sector, so it need not be the exact distance of the sector being prepared.

| Public route in [core.py](core.py) | Outcome |
| --- | --- |
| Trivial input | Handles an empty classical code, a quantum problem with no logical rows, or `V = S` using the entry point's return convention |
| Direct exhaustive shortcut in `get_distance_quantum` with `method="brouwer_zimmermann"` | Runs exhaustive enumeration if `exhaustive_is_cheaper` approves the number of supplied rows and those rows are linearly independent |
| `method="brute_force"` | Runs exhaustive enumeration over an independent basis, adapted to `S ⊆ V` for a quantum problem |

`_prepare_brouwer_zimmermann_search` considers the following exits in order.
The disjoint family, its pruning, and any overlapping replacement come from the stages of [_information_sets.py](_information_sets.py); the exits between those stages belong to the distance search.

| Preparation stage | Outcome |
| --- | --- |
| Initial row witness | Returns if `U ≤ cutoff`, where `U` includes the lightest eligible row and any supplied bound from another sector |
| Exhaustive by size | Runs exhaustive enumeration if `exhaustive_is_cheaper(dimension, length)` approves |
| Rows of the disjoint family | Returns if checking the rows of every search basis gives `U ≤ cutoff` |
| Certificate of the disjoint family | Returns if `L(1) ≥ U` |
| Rows of a replacement overlapping family, if selected | Returns if `U ≤ cutoff` |
| Certificate of the replacement family | Returns if `L(1) ≥ U` |
| Exhaustive by work | Runs exhaustive enumeration if `_brute_force_is_cheaper` finds that the eligible-word count is at most the estimated BZ work of [Section 3.6](#36-exhaustive-enumeration-as-a-fallback) |
| Ready search | Returns a `_BrouwerZimmermannSearch` that scans from layer two |

Exhaustive enumeration is chosen automatically at three points: the direct shortcut, preparation's size check, and preparation's work check.

| Search location | Outcome |
| --- | --- |
| `_BrouwerZimmermannSearch.advance` | Returns after one layer, or earlier if `U ≤ cutoff`; a finished search does no further work |
| `_BrouwerZimmermannSearch.finished` | Marks a search finished if `U ≤ cutoff`, `L ≥ U`, or every layer has been enumerated (`next_weight > dimension`) |
| `get_distance_brouwer_zimmermann_many` | Returns the shared best bound when it is at most `cutoff` after an advance, or when no search can advance |

The tests `test_brouwer_zimmermann_setup_dispatches`, `test_brouwer_zimmermann_setup_cutoffs`, `test_brouwer_zimmermann_overlap_setup_certification`, and `test_brouwer_zimmermann_exhaustive_dispatch` exercise these exits.

### 4.3 Concept-to-code glossary

| Concept | Code |
| --- | --- |
| Upper bound `U` | `best`, or `upper_bound` as a parameter |
| Lower certificate `L` | `lower_bound` of a search, computed by `Certificate.lower_bound(completed_weight)` from the certificate's `dimension`, `ranks`, `weight_divisor`, and `max_coverage` |
| Family of search bases | `BasisFamily`, which holds the search bases and their `Certificate`; a search keeps only the certificate |
| Last completed layer `t`, next layer | `completed_weight`, `next_weight` |
| Divisor `D` | `Certificate.weight_divisor`, in Hamming units of the searched basis |
| Maximum coverage `C` | `Certificate.max_coverage` |
| Dimension `r`, block rank `r_i` | `dimension`, `ranks` (`InformationSet.rank`); a search's `ranks` array is the native copy of its certificate's ranks |
| Search basis and its block | `InformationSet`, with fields `generators`, `labels`, `rank`, and `pivots`; the recorded block is the pivot subset, as explained below |
| Number of coordinates `N` | The length of `columns` (or `nonzero_columns`), the array of nonzero columns, in the functions that build families of search bases |
| Stored width `N_stored` | `basis.shape[1]` before pivot deletion, including any all-zero columns |
| Labels `M` | `labels`, which is `None` for a classical problem |
| Eligible rows | Every row of a classical basis; rows with nonzero labels for a quantum problem |
| Support, prefix XORs | `support` and `prefixes` in `advance` and `_scan_brouwer_zimmermann` |
| Packed generators | `generators` of `_BrouwerZimmermannSearch`: `uint64` words after deleting pivot columns; `widths` counts the valid words for each search basis |
| Packed labels | `labels` of `_BrouwerZimmermannSearch`: `uint64` quotient-label words, without pivot-column deletion; `label_widths` counts the valid words |
| Chunk | `batch_size`, equal to `1 << min(block_size, 15)` supports |
| Candidate/set evaluation | The unit counted by `Certificate.work` |

For the rank-deficient remainder of [Section 2.4](#24-search-bases-and-rank-deficient-blocks), the implementation records only the independent pivot columns in `InformationSet.pivots`.
The remaining columns of that remainder stay in the generator matrix.
This smaller block has the same rank and gives the same certificate term `b_i(t)`.
Coverage and pivot-column deletion both use the recorded pivot subset.

### 4.4 Heuristics and cost models

The certificate is computed from verified ranks and the actual maximum coverage of whichever family of search bases is chosen.
The heuristics below change search work while preserving certificate validity.
With an early-exit cutoff above one, a different search order can find a different first witness at or below the cutoff, while preserving the upper-bound contract of [Section 4.5](#45-behavior-spanning-files).

- **`get_disjoint_family`** builds the disjoint family greedily in natural column order, through `get_information_set_generators`.
  If that does not reach the maximum number of blocks that are information sets, it tries a small fixed number of fixed-seed random column orders and keeps the family with the lexicographically largest tuple of ranks.
  Fixed seeds keep results reproducible, and the bounded retries keep setup cheap.
  It also infers the divisor `2` when every basis row has even weight, and combines it with any divisor supplied by the caller.
- **`prune_family`** is exact, not heuristic.
  A search basis whose block has rank deficiency `d = r - r_i` has a term `b_i(t)` that is zero for `t < d`.
  The function removes such a search basis when the blocks that are information sets reach `U` by layer `d`, so the removal leaves the result unchanged and saves enumeration.
  It removes nothing if no block is an information set.
- **`get_overlapping_family`** tries to lower the work estimate of the disjoint family, with tunables `max_sets` and `minimum_work`.
  It returns `None` if the existing work is below `minimum_work`, if the blocks already use every nonzero column, if the family is at `max_sets`, or if even the optimistic coverage `ceil(m*r/N)` cannot reduce the estimate.
  Otherwise it builds search bases by row reducing on column orders that put the least-used columns first, with seeded tie-breaking, and keeps the cheapest family built.
  A repeated pivot set is discarded, but it still consumes one of a bounded number of attempts.
- **`Certificate.work`, `_brute_force_is_cheaper`, and `exhaustive_is_cheaper`** are cost models, which [Section 3.6](#36-exhaustive-enumeration-as-a-fallback) explains.
  They are not runtime guarantees.
  `exhaustive_is_cheaper` compares the setup cost of BZ with packed enumeration of all words, and its rule is in the source.

### 4.5 Behavior spanning files

- **Cutoff.**
  `get_distance_classical` and `get_distance_quantum` default to `cutoff=1`, and `get_distance_css_brouwer_zimmermann` requires a `cutoff`.
  If a witness reaches `cutoff`, the computation can return its weight immediately as an upper bound, which can exceed the exact distance.
  Otherwise, the search continues and returns the exact distance, which may be larger than `cutoff`.
  The default of one is still exact for a valid nontrivial binary problem, because a distance is at least one.
  A `cutoff` of zero or less never triggers an early exit for such a problem, and `advance` clamps negative values to `-1` before it calls the compiled scan.
  [Section 4.2](#42-flow-and-exit-points) lists where the `cutoff` is checked.
- **Units.**
  Homogeneous input represents a pure-X or pure-Z problem with one support bit per qubit.
  Non-homogeneous input (`homogeneous=False`) uses general symplectic Pauli rows, as in [Section 2.7](#27-pauli-weight-as-hamming-weight).
  For non-homogeneous input, the BZ route maps rows to Hamming coordinates, doubles `cutoff`, passes `weight_divisor=2`, and halves the result.
  The exhaustive routes inside preparation receive the same Hamming-coordinate basis, so they use the doubled `cutoff`.
  Only the direct shortcut and `method="brute_force"` use native symplectic weights and the original `cutoff`.
- **Independence.**
  Callers need not supply independent generators.
  The classical path uses `get_independent_rows`, and the quantum path uses `get_nested_code_basis`.
  The direct shortcut verifies independence itself, so the packed enumerators always receive independent bases.
- **Resumable layers.**
  The compiled scan returns after at most `batch_size` supports.
  Within one `advance` call, successive compiled batches reuse the local `support` and `prefixes` arrays.
  `advance` changes `lower_bound` and `next_weight` only after a layer is complete in every search basis.
  Aborting `advance`, for example with an exception or a cutoff return, leaves those completed-layer fields unchanged.
  A subsequent call that is still permitted to advance restarts the unfinished layer from its first support.
- **CSS sectors.**
  A joint search runs one search per sector in round-robin order and shares the best bound.
  Each sector keeps its own `lower_bound` and `certificate`, including its coverage and divisor.

## 5. Correctness requirements

**Audience note:** This section is a highly technical summary written to help AI agents navigate the BZ implementation in qLDPC.
For a carefully reviewed, high-level explanation intended for human readers, see Sections [2](#2-the-brouwer-zimmermann-algorithm) and [3](#3-extensions-and-optimizations).

Changes to the BZ algorithm can use different representations, certificates, and search strategies while preserving the following requirements for the nontrivial problems of [Section 2.1](#21-the-problem).

- **Search the intended space and recognize eligible witnesses.**
  Transformations must preserve `V` and the excluded subspace `S`, or provide an equivalent representation of `V \ S`.
  With quotient labels, this means preserving the membership test of [Section 3.4](#34-quotient-labels-for-the-stabilizer-test).
- **Preserve the meaning of weights.**
  Candidate weights, certificates, cutoffs, and returned results must use consistent units.
  For the current implementation, this includes restoring the deleted pivot weight ([Section 3.5](#35-packed-enumeration-with-pivot-column-deletion)) and converting between Pauli and Hamming weights ([Section 4.5](#45-behavior-spanning-files)).
- **Use only justified certificates.**
  A certificate must hold for every eligible word still unexamined in the search where it is used.
  The current formula relies on block ranks, overlap coverage, and completed layers (Sections [2.4](#24-search-bases-and-rank-deficient-blocks), [2.5](#25-the-stopping-certificate), and [3.1](#31-overlapping-search-bases)).
  Coverage may be conservatively overestimated; underestimating it can make the formula unsound.
  Optional rounding requires a divisor valid for every eligible weight ([Section 3.2](#32-rounding-the-certificate-to-a-proven-weight-multiple)).
  A certificate from another CSS sector needs justification for the sector where it is applied, just as any other certificate does.
- **Keep the strict certificate available to fixed-weight counting.**
  [_counting/scan.py](_counting/scan.py) builds its family with `prepare_family` of [_information_sets.py](_information_sets.py), which runs the same stages as the distance search, and stops with that family's `Certificate` to count every word of a target weight `w`, which defaults to the distance.
  Distance may stop at `L(t) ≥ U`, but counting must also exhaust words of weight exactly `w`, so it stops only at `L(t) > w`.
  Its bound must use the ranks and actual maximum coverage of the family it scans; it passes `upper_bound=w+1` to the family stages only to aim them at that stopping point.
- **Preserve the return contract.**
  Return the exact minimum unless taking an early exit permitted by `cutoff`.
  Such an early exit returns the weight of a known eligible witness, an upper bound between the distance and `cutoff` ([Section 4.5](#45-behavior-spanning-files)).

Representative correctness tests check certificate validity against exhaustive words, including rank-deficient blocks, inferred parity, and overlapping coverage (`test_family_certificates` and `test_overlapping_family_certificates` in [_information_sets_test.py](_information_sets_test.py)), and joint-search results with distinct sector certificates (`test_brouwer_zimmermann_sector_certificates` in [_brouwer_zimmermann_test.py](_brouwer_zimmermann_test.py)).
`test_counting_continues_at_an_equal_lower_bound` in [logical_counts_test.py](logical_counts_test.py) checks the strict counting stop with a count that a stop at `L(t) = w` would miss.
In [core_test.py](core_test.py), `test_brouwer_zimmermann_random_cross_checks` forces the BZ route and compares both homogeneous and general Pauli problems with exhaustive results.

Performance tests also check broad work budgets, and helper tests exercise the current implementation's interfaces.
Those tests can evolve with the implementation; they do not make a particular information-set family, packing order, or optimization a correctness requirement.

## References

- [arXiv:1603.06757v2](https://arxiv.org/abs/1603.06757v2) (Algorithm 1): Section 2 and Algorithm 1 give the classical disjoint and rank-deficient bound.
- [arXiv:2408.10743v2](https://arxiv.org/abs/2408.10743v2): Section 2.4 describes the `Saved_isometry` symplectic-to-Hamming construction.
- [core.py](core.py), [_brouwer_zimmermann.py](_brouwer_zimmermann.py), [_information_sets.py](_information_sets.py), [_brute_force.py](_brute_force.py), and [_binary.py](_binary.py): the implementation.
- [_brouwer_zimmermann_test.py](_brouwer_zimmermann_test.py), [_information_sets_test.py](_information_sets_test.py), and [core_test.py](core_test.py): executable invariants and independent small-instance oracles.
