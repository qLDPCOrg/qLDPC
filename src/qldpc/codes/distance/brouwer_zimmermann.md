# Brouwer-Zimmermann distance search in qLDPC

## 1. Purpose and scope

The Brouwer-Zimmermann (BZ) algorithm computes the distance of a classical code ([arXiv:1603.06757](https://arxiv.org/abs/1603.06757)), and has been adapted to quantum codes ([arXiv:2408.10743](https://arxiv.org/abs/2408.10743)).
This guide explains how qLDPC computes the exact distance of a binary code using BZ.
It is written for contributors who need to understand, debug, or modify the BZ code in this directory.
It is organized as follows:

- [Section 2](#2-the-brouwer-zimmermann-algorithm) explains the BZ algorithm as published, including its adaptation to quantum codes.
- [Section 3](#3-extensions-and-optimizations) explains what qLDPC uses beyond the algorithm of Section 2, and where each part comes from: extensions that strengthen the lower certificate or improve the upper bound, and optimizations that make the search cheaper to run.
- [Section 4](#4-guide-to-the-source-code) is a guide to the source code: how the modules are organized, how a computation flows through them, which code names correspond to the concepts of [Section 2](#2-the-brouwer-zimmermann-algorithm) and [Section 3](#3-extensions-and-optimizations), and which behavior spans several files.
- [Section 5](#5-correctness-requirements) summarizes the correctness requirements that apply when changing the implementation.

[connected_support.md](connected_support.md) describes another exact search for codes with sparse checks, which searches supports that are connected in the Tanner graph of the checks.

Sections [2](#2-the-brouwer-zimmermann-algorithm) and [3](#3-extensions-and-optimizations) are intended to help readers understand the BZ algorithm and its implementation in qLDPC at a high level.
Sections [4](#4-guide-to-the-source-code) and [5](#5-correctness-requirements) are highly technical summaries for contributors who need to locate identifiers in the BZ implementation of qLDPC, or check a change against its requirements.

## 2. The Brouwer-Zimmermann algorithm

This section describes BZ as presented in [arXiv:1603.06757](https://arxiv.org/abs/1603.06757) and [arXiv:2408.10743](https://arxiv.org/abs/2408.10743), and it also describes the certificate within a layer, from Grassl's chapter and from Lisoněk and Trummer (see [References](#references)).

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

The certificate can also advance within a layer.
Suppose that the first `j` search bases have been searched through coefficient weight `t`, and the others through `t-1`.
An unexamined eligible word then has coefficient weight at least `t+1` in each of the first `j` bases, and at least `t` in the others, so

$$
\operatorname{wt}(v) \ge \sum_{i \le j} b_i(t) + \sum_{i > j} b_i(t-1) =: L(t, j).
$$

`L(t, 0) = L(t-1)` and `L(t, m) = L(t)`.
Completing layer `t` in search basis `j` is step `(t, j)`, and the steps run in the order `(1, 1), ..., (1, m), (2, 1), ...`.
When a layer starts, qLDPC finds the fewest leading bases whose completion of the layer makes `L(t, j) ≥ U`, and it searches only those bases in that layer.
This can save most of the last layer, which often dominates the cost.
Grassl notes this improvement in footnote 2 of his chapter, on page 295 of *Discovering Mathematics with Magma*, and Magma uses it.
Equation (5) of Lisoněk and Trummer gives the same certificate for these disjoint blocks.

The cost of this loop is measured in **candidate/set evaluations**.
One candidate/set evaluation is the examination of one selection of rows in one basis.
Searching the `m` search bases through coefficient weight `t` therefore costs `m` times the number of selections of at most `t` rows, which is `m * sum_{w=1}^{t} binom(r, w)`.
Stopping after the first `j` bases of layer `t` costs `m * sum_{w=1}^{t-1} binom(r, w) + j * binom(r, t)`.

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

qLDPC uses extensions and optimizations beyond the algorithm of [Section 2](#2-the-brouwer-zimmermann-algorithm).
Several of them are published, as the sections below say; the others are choices of this implementation.
Extensions strengthen the lower certificate or the upper bound, while optimizations make the search cheaper to run.

The extensions are:

- [Section 3.1](#31-overlapping-search-bases) strengthens the lower certificate by allowing for search bases with overlapping blocks, chosen heuristically or by matroid partitioning, and certified by coverage or by linear programming.
- [Section 3.2](#32-rounding-the-certificate-to-a-proven-weight-multiple) strengthens the lower certificate by rounding it up when the weight of every eligible word is a multiple of some integer.
- [Section 3.3](#33-sharing-witnesses-between-css-sectors) tightens the upper bound by sharing witnesses between the two sectors of a CSS code.

The optimizations are:

- [Section 3.4](#34-quotient-labels-for-the-stabilizer-test) tests whether a candidate word lies in `S` by XORing precomputed labels.
- [Section 3.5](#35-packed-enumeration-with-pivot-column-deletion) packs words into machine words, reuses partial XORs, and avoids storing pivot columns.
- [Section 3.6](#36-exhaustive-enumeration-as-a-fallback) enumerates every word when that is cheaper than running BZ.
- [Section 3.9](#39-skipping-completions-that-cannot-lower-u) skips the supports whose last two rows provably cannot complete a word lighter than `U`.

[Section 3.7](#37-images-of-an-information-set-under-automorphisms) is both an extension and an optimization, for problems that a group of coordinate permutations preserves.
It certifies with the images of one information set under the group, and scans only that set.
[Section 3.8](#38-worked-example) illustrates [Section 3.1](#31-overlapping-search-bases) and [Section 3.2](#32-rounding-the-certificate-to-a-proven-weight-multiple) with an example.

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
Each nonzero coordinate `c` of `v` is counted in `wt(v|_{B_i})` for every block `B_i` that contains `c`, and there are at most `C` such blocks.
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
Within a layer, the same argument divides the sum of [Section 2.5](#25-the-stopping-certificate) that defines `L(t, j)` by `C`.

Overlapping information sets are part of the published algorithm.
Grassl's chapter (page 291 and Algorithm 2.5, see [References](#references)) credits Zimmermann with a certificate for an ordered list of blocks that are information sets, `sum_i max(0, t+1 - o_i)`, where `o_i` is the number of columns of the `i`-th block that also lie in an earlier block; Grassl calls `r - o_i` the relative rank of the block.
Proposition 5 of Lisoněk and Trummer gives the coverage-divided certificate above when every block has the same bound `b_i(t) = t+1`, and its proof covers different bounds, which qLDPC uses for rank-deficient blocks and for steps within a layer.
Neither Zimmermann's certificate nor the coverage-divided one dominates the other, and the linear program below dominates both.
[arXiv:1603.06757](https://arxiv.org/abs/1603.06757v2) mentions the overlapping case in its Section 2, citing Grassl, without giving a formula.

#### A small example

Take the code generated by the rows `101` and `011`, whose distance is two.
It has `N = 3` coordinates and `r = 2`, so a disjoint family contains at most `floor(3/2) = 1` search basis whose block is an information set.
An overlapping family can use all three pairs of column indices, `{0,1}`, `{0,2}`, and `{1,2}`, each of which is an information set.
Then `m = 3`, and each column index lies in two of the blocks, so `C = 2`.
After all three search bases have been searched through coefficient weight `t = 1`, the formula gives `L(1) = ceil(3*2/2) = 3`.
This exceeds the distance, which is allowed because `L(t)` is a statement only about words that have not been examined.
Every row has weight two, so `U = 2 ≤ L(1)`, and the search stops with distance two.

#### A certificate from linear programming

The coverage-divided certificate treats every column as if it lay in `C` blocks, so it is weak when coverage is uneven.
An unexamined eligible word `v` satisfies `wt(v|_{B_i}) ≥ b_i` for every block, where `b_i` is `b_i(t)` after layer `t`, or the mix of `b_i(t)` and `b_i(t-1)` of [Section 2.5](#25-the-stopping-certificate) within a layer.
Relaxing the coordinates of `v` to `0 ≤ x_c ≤ 1` gives the linear program

$$
\text{minimize} \sum_c x_c \quad \text{subject to} \quad \sum_{c \in B_i} x_c \ge b_i \text{ for each } i, \quad 0 \le x_c \le 1,
$$

whose optimum is a lower bound on `wt(v)`.
By weak duality, every `y ≥ 0` gives the bound

$$
\operatorname{wt}(v) \ge \sum_i b_i y_i - \sum_c \max\Big(0, \sum_{i : c \in B_i} y_i - 1\Big).
$$

The coverage-divided certificate is the value of `y_i = 1/C`, and Zimmermann's certificate is at most the value of `y_i = 1` on the blocks that contribute to it, so the optimum is at least both, and it can be larger.
For example, take the columns `e1, e2, e3, e1+e2+e3, e1, e2, e3`, the four 3-subsets of the first four columns, and the last three columns as blocks, with `b_i = 2`: the coverage-divided certificate and Zimmermann's certificate, in every order of the blocks, are four, and the linear program gives `ceil(14/3) = 5`.

qLDPC solves the linear program with `scipy.optimize.linprog`, and takes as its bound the ceiling of the dual value of the solver's `y` in exact rational arithmetic, so that rounding in the solver can only weaken the bound.
For disjoint blocks, the linear program separates by block and gives the certificate of [Section 2.5](#25-the-stopping-certificate), so only overlapping families use it.
Because each linear program takes milliseconds, only searches whose estimated work reaches the threshold `lp_minimum_work` of [Section 4.4](#44-heuristics-and-cost-models) use it.
To decide how many bases of a layer to search, a search solves a linear program only when the coverage-divided certificate falls short of `U` and a feasible point of the linear program shows that its optimum could reach `U`.
It also solves one at the end of each layer, to update its lower certificate.

### 3.2 Rounding the certificate to a proven weight multiple

Suppose that every eligible word has a weight that is a multiple of an integer `D`.
An unexamined eligible word then has a weight that is at least `L(t)` and a multiple of `D`, so it has a weight of at least `D * ceil(L(t)/D)`, the smallest multiple of `D` that is at least `L(t)`.
The certificate can therefore be rounded up to that value.
Grassl's chapter notes this rounding on page 299, and Magma uses it for even binary and quaternary codes, doubly even binary codes, and ternary codes whose weights are all multiples of three.

In qLDPC, `D` comes from two sources:

- For general Pauli problems, the map of [Section 2.7](#27-pauli-weight-as-hamming-weight) doubles every weight, which gives `D = 2`.
- For any problem, qLDPC checks whether every row of the search basis has even Hamming weight.
  The XOR of two even-weight words has even weight, so if all rows are even, every word in the space is even.
  When this holds, `D = 2` is combined with any existing divisor by taking the least common multiple.

qLDPC checks only parity, which carries over from the rows to all of their sums.
Divisibility by four does not carry over from the row weights alone: the rows `11110` and `01111` each have weight four, but their XOR `10001` has weight two.
Since `wt(u + v) = wt(u) + wt(v) - 2|u ∩ v|`, every word in the space has a weight divisible by four exactly when every row does and every two rows overlap in an even number of coordinates.
That stronger test mainly helps doubly even classical codes: in a CSS sector, it requires every logical operator of the sector to lie, as a binary vector, in the span of the stabilizers of the other type.

The same identity shows what carries over to the words of a coset `u + V`, which the coset searches of [Section 4.1](#41-module-map-and-layering) search.
A divisor of the weights of `V` need not divide the weights of a coset: every word of `V = {0000, 1111}` has a weight divisible by four, while the coset `1000 + V` has the weights one and three.
Parity does carry over: if every word of `V` is even, every word `u + v` of the coset has the parity of `wt(u)`.
So a coset search of an even code rounds its certificate up to the least integer of that parity, which it reads from the first word of the coset that it builds, and to no other divisor.
The searches of X and Z parts do this, and so does a search of Pauli errors weighed by their CSS weight, which bounds the CSS weight by half the Hamming weight of `[x | z]`, rounded up, after rounding to parity.
A search of Pauli errors weighed by their physical weight does not round: every word `T(x, z)` of [Section 2.7](#27-pauli-weight-as-hamming-weight) is even, and rounding a bound up to an even integer does not change its half.

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
This is the situation in [Section 3.8](#38-worked-example).
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
Algorithm 2.11 of Grassl's chapter (page 300) makes a similar choice: it compares the work estimate of BZ with the number of codewords of the code or of its dual, and computes the weight distribution when that is cheaper.
With `q = dim(V/S)` as in [Section 3.4](#34-quotient-labels-for-the-stabilizer-test), the number of eligible words is `(2^q - 1) * 2^(r-q)`.

The estimated cost of BZ is a work estimate in candidate/set evaluations.
Let `m` be the number of search bases, and let `(t*, j*)` be the first step, in the order of [Section 2.5](#25-the-stopping-certificate), after which the certificate reaches `U`: the first `j*` bases have completed layer `t*`.
The certificate is the one of [Section 3.1](#31-overlapping-search-bases), rounded as in [Section 3.2](#32-rounding-the-certificate-to-a-proven-weight-multiple), for the ranks and maximum coverage of the family of search bases under consideration.
The estimate is

$$
W = m\sum_{w=1}^{t_* - 1} \binom{r}{w} + j_* \binom{r}{t_*}.
$$

The estimate is pessimistic, because it assumes that no lighter witness is found before the certificate reaches `U`.
The sum starts at `w = 1`, so it includes the systematic rows.
When the comparison with exhaustive enumeration is made, layer one has already been checked, so `W` slightly overestimates the remaining BZ work.
Actual runtime also depends on basis construction, the width of packed words, and early improvements of the witness, so `W` is not a runtime guarantee.

Exhaustive enumeration is chosen at three points, which [Section 4.2](#42-flow-and-exit-points) lists.
It enumerates words in vectorized, packed blocks and updates them with Gray-code steps, and it does not use the BZ certificate.

### 3.7 Images of an information set under automorphisms

Many code constructions come with a group of qubit permutations that preserve the code.
For example, translating both blocks of qubits of a multivariate bicycle code by an element of its group maps its checks of each type onto themselves.
Such a group lets a search certify with a large family of information sets while scanning only one of them, which removes most of the cost of the layers that dominate a search.

#### The certificate of an automorphism family

Call a permutation `g` of the coordinates an **automorphism** of a problem if it maps both `V` and `S` onto themselves.
An automorphism maps each eligible word to an eligible word of the same weight, and each information set `I` to another information set `g(I)`.
Let `G` be a group of automorphisms, and consider the family of the images `g(I)` for all `g` in `G`, which we call an automorphism family.

Suppose that the search has examined every word `v` with `wt(v|_I) ≤ t`, as searching the systematic basis `G_I` through weight `t` does ([Section 2.3](#23-systematic-bases-and-information-sets)).
If an eligible word `c` has at most `t` ones on an image `g(I)`, then `g⁻¹(c)` is an eligible word of the same weight with at most `t` ones on `I`, which the search has examined.
So if no word of the orbit of `c` under `G` has been examined, `c` has at least `t+1` ones on every image, and the certificate of the automorphism family bounds its weight.
Every other eligible word weighs as much as an examined word, and hence at least `U`.
So once that certificate reaches `U`, no eligible word is lighter than `U`, and the search stops with the exact distance, just as in [Section 2.5](#25-the-stopping-certificate).

The search therefore scans only `I`, and every image completes each layer when `I` does.
Each layer costs `binom(r, t)` candidate/set evaluations however many images the family has, so the work estimate of [Section 3.6](#36-exhaustive-enumeration-as-a-fallback) becomes `W = sum_{w=1}^{t*} binom(r, w)`, where `t*` is the first layer after which the certificate reaches `U`.
Every image completes a layer together, so the certificate advances once per layer, rather than in the steps within a layer of [Section 2.5](#25-the-stopping-certificate).

A column `c` in an orbit `O` of `G` lies in `g(I)` for `|I ∩ O| |G| / |O|` elements `g`, since `G` maps each column of `I ∩ O` to `c` equally often.
The family of the `|G|` images therefore has maximum coverage `C = |G| max_O |I ∩ O| / |O|`, and the coverage-divided certificate of [Section 3.1](#31-overlapping-search-bases) is

$$
L(t) = \left\lceil (t+1) \min_O \frac{|O|}{|I \cap O|} \right\rceil,
$$

where the minimum is over the orbits that meet `I`, rounded up to a proven weight divisor as in [Section 3.2](#32-rounding-the-certificate-to-a-proven-weight-multiple).
This bound is strongest when `I` meets every orbit in an even share `|I ∩ O| / |O|`.
If every orbit meets `I` in the share `r/N`, as for a single orbit of all `N` coordinates, or for the two blocks of a multivariate bicycle code and a set with `r/2` columns in each, then `L(t) = ceil(N(t+1)/r)`.
No family of information sets has a larger coverage-divided certificate, because a family of `m` sets with maximum coverage `C` has `m r ≤ C N`.

#### Choosing the scanned set

qLDPC row reduces on the nonzero columns in their natural order.
It then exchanges pivots between orbits while that lowers the largest share.
Pivoting the row of a pivot column on another column with a one in that row gives another information set.
For two orbits, these exchanges reach the fewest pivots in the fuller orbit that any information set has.
If no exchange moves a pivot out of that orbit, every column of the other orbit lies in the span of the pivots in it.
With more orbits, the exchanges are a heuristic.

#### When a search uses an automorphism family

Finding automorphisms can take milliseconds, as long as many whole searches take.
Before a search finds a light witness, its work estimate of [Section 3.6](#36-exhaustive-enumeration-as-a-fallback) is too pessimistic to tell such easy searches apart.
A search therefore starts with the families of Sections [2.4](#24-search-bases-and-rank-deficient-blocks) and [3.1](#31-overlapping-search-bases).
Before the first layer whose scan would cost at least `automorphism_minimum_layer_work` candidate/set evaluations ([Section 4.4](#44-heuristics-and-cost-models)), it asks for automorphisms once.
If their automorphism family has a smaller work estimate for the current `U`, the search continues with it, from layer two.

This switch keeps both certificates.
An eligible word lighter than `U` is unexamined by both families, since examining it would have lowered `U`.
So the bounds that both families have reached hold for it, and the search takes the larger.
Words that the first family examined still contribute their witnesses to `U`.

#### Where automorphisms come from

A binary quantum code finds automorphisms by itself if the optional `pybliss` package is installed.
Bliss finds the automorphisms of the Tanner graph of the code, whose edges are colored by the Pauli operators of the checks.
Their permutations of qubits map each check to a check, so they preserve the stabilizers, gauge operators, and logical operators.
For a CSS code, the colors keep X-type and Z-type checks apart, so the permutations also preserve the logical operators and stabilizers of each sector.
These automorphisms are the symmetries of the given parity checks, a subgroup of the automorphisms of the code.
This subgroup suffices, since every group of automorphisms gives a sound certificate.

Without `pybliss`, or if bliss finds none, code classes can supply generators through a `_get_qudit_automorphisms` method.
`MBCode` and `PairPartitionCode` supply the translations of their groups, which act on the qubits of each block by a regular permutation.
`GALACode`, `LPCode`, and `SLPCode` supply them when their rings are group algebras of abelian groups that lift to permutation matrices.
A wrong generator would make the certificate unsound.
So before a search uses the generators, it checks that each maps `V` and `S` onto themselves, and raises an error if one does not.
Only distance searches use automorphisms: counting the logical operators of a given weight ([Section 5](#5-correctness-requirements)) must visit every word of that weight, not one word per orbit.

Generators from either source act on qubits.
The search of a general Pauli problem runs over the coordinates `(x | z | x+z)` of [Section 2.7](#27-pauli-weight-as-hamming-weight).
A permutation of qubits acts on these coordinates by moving each of the three blocks alike.
The map `T` commutes with these lifted permutations, so a lifted permutation preserves `T(V)` and `T(S)` exactly if the permutation preserves `V` and `S`.
Verification of a lifted permutation therefore checks the original one.

### 3.8 Worked example

This example uses one CSS sector of the MB64 code from the test suite ([_search/brouwer_zimmermann_test.py](_search/brouwer_zimmermann_test.py)).
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

### 3.9 Skipping completions that cannot lower U

The layers that a search scans last dominate its cost, and most of their words weigh far more than `U`.
For example, certifying distance eighteen for the X sector of the [[288, 12, 18]] code of [arXiv:2308.07915](https://arxiv.org/abs/2308.07915) with an automorphism family scans layer eight of a basis with `r = 150` rows, about `5 × 10^12` supports.
A search can skip whole groups of these supports if a cheap lower bound proves that none of their words is lighter than `U`.
qLDPC can do this when a layer scans one search basis whose block is an information set, which includes every layer of a search of an automorphism family ([Section 3.7](#37-images-of-an-information-set-under-automorphisms)).
Counting and the coset-leader search ([Section 4.1](#41-module-map-and-layering)) do not skip supports.

#### The bound

Write the search basis as `G = [I_r | P]`, with the pivot columns first.
A word that sums the rows of a support of size `w` weighs `w` plus the weight of the sum of their **tails**, the rows of `P` ([Section 3.5](#35-packed-enumeration-with-pivot-column-deletion)).
Consider the supports that share their first `w - 2` rows, whose tails sum to `p`, and whose next row is at least `s`.
Each is completed by a pair of rows `s ≤ i < j < r`, and its word weighs `w + wt(p + P_i + P_j)`.

Partition the tail columns into disjoint blocks `J_b`.
On each block, let

$$
m_b = \min_{s \le i < j < r} \operatorname{wt}\big((p + P_i + P_j)|_{J_b}\big).
$$

Every completion uses the same pair on every block, and its weight on each block is at least the minimum there, so

$$
\operatorname{wt}(p + P_i + P_j) \ge B = \sum_b m_b
$$

for every pair.
If `w + B ≥ U`, no completion is lighter than `U`, and the search skips them all.
Different blocks may attain their minima at different pairs, which only weakens `B`.
The bound ignores eligibility, which only weakens it too.

The bound must minimize over every pair: a minimum over fewer pairs, or over sampled pairs, can exceed the weight of a real completion.
The blocks must be disjoint: a column in two blocks can be counted twice.
The pivot ones must number exactly `w`, which holds only for a block that is an information set ([Section 2.4](#24-search-bases-and-rank-deficient-blocks)).
The bound uses the rows in the order in which the search sums them, since the pairs from row `s` on depend on that order.

#### Tables and lookups

For each block and each `s`, a table stores the distance from every value of `p` on the block to the nearest sum of a pair from row `s` on.
The tables for `s` are those for `s + 1`, lowered by the pairs of row `s`, and a breadth-first search from those new sums lowers them.
With blocks of `β` columns, the tables take `(r - 1) Σ_b 2^|J_b|` bytes, and building them costs about `2^β β^2` steps for each block, plus writing each table.
A bound then costs one table entry per block.
The search looks up the bound each time it tries a new row `s` as the first of the last pair, and if it skips the completions, it skips the rest of the prefix's subtree.
A bound for a larger `s` is stronger, so a support is skipped exactly if the bound of its own `s` reaches `U - w`.

#### Certificates and contracts

Skipped supports are not examined, but they weigh at least the value of `U` when they were skipped, and hence at least any later `U`.
An eligible word lighter than `U` is therefore neither examined nor skipped, so every argument of [Section 2.5](#25-the-stopping-certificate) still holds for it.
The certificate `L` of a search bounds only such words: no eligible word weighs less than `min(L, U)`.
`L` alone can exceed the weights of skipped words, just as it can exceed the weights of words whose images an automorphism family has examined ([Section 3.7](#37-images-of-an-information-set-under-automorphisms)), so a bound on the distance is `min(L, U)`.
No skipped word is lighter than `U` at any later time, so a scan with tables finds the same witnesses in the same order as a scan without them, and an early exit at a cutoff returns the same weight.
A search stops exactly when it would without skipping, because it stops once its certificate reaches `U`.
Words that weigh exactly `U` can be skipped, so counting the words of a given weight, which must visit them all ([Section 5](#5-correctness-requirements)), does not use these tables.

#### When a search uses the tables

Every layer of two or more rows that scans one information set uses `scan_information_set` of [_search/affine_search.py](_search/affine_search.py), a scan specialized to that case, with or without tables.
Tables of a basis with `r = 150` rows and 138 tail columns, in blocks of sixteen columns, take 75 MiB, and about a twentieth of a second to build on the machine described below.
That is worthwhile for a layer of `10^10` supports, but not for an easy search.
A search considers tables for a layer of at least `completion_bound_minimum_layer_work` supports ([Section 4.4](#44-heuristics-and-cost-models)).
It then estimates the share of supports that the tables would skip, from the bounds of at most `completion_bound_samples` random supports computed directly from the rows, and takes a conservative lower bound on that share.
It predicts the cost of the layer from this share, the number of lookups, and the cost of building tables that it does not have yet.
Each lookup costs a few plain support evaluations for each block of the tables, and more for larger tables, which miss the processor's caches more often.
The search uses tables if the predicted cost is at most `completion_bound_max_cost_percent` percent of the cost without them.
By default, tables use blocks of sixteen columns, or of eight columns if the tables would otherwise exceed `completion_bound_max_table_bytes`; the fields `completion_bound_block_bits` and `completion_bound_fallback_block_bits` set these widths.
A search keeps its tables for later layers, and drops them when its basis changes, as when it switches to an automorphism family.
It frees them when it finishes, but the search of each sector of a joint search keeps its own tables until then, so a joint search can hold twice `completion_bound_max_table_bytes`.

Measurements on one Apple M3 Max processor, single-threaded, illustrate these trade-offs.
The cost ratio of a layer is its time with tables, including their construction, divided by its time without them, with `completion_bound_max_table_bytes` set to zero.
The coefficients of the cost model are fields of `SearchTuning` in [_tuning.py](_tuning.py), whose module docstring says on what layers they were fitted and how to refit them.
For the X sector of the [[288, 12, 18]] code above, with `U = 18` and the scanned set and row order that qLDPC chooses, layer six had a cost ratio of 0.22 under the default tuning, and its tables skipped 87% of its supports.
On nine layers of `5 × 10^8` to `1.5 × 10^10` supports of that code and of three pair-partition codes, the scan of one information set without tables took 70% to 95% of the time of the scan of any family of sets.
These two scans were measured as the kernels `_scan_one_information_set` and `_scan_brouwer_zimmermann`, before the distance and coset searches shared their scans; they are now `scan_information_set` and `scan_family` of [_search/affine_search.py](_search/affine_search.py), which scan in the same order.
With tables forced, by `completion_bound_minimum_layer_work` set to zero and `completion_bound_max_cost_percent` set to `MAX_VALUE`, three of four layers of `2 × 10^7` to `5 × 10^7` supports, of that code and of two pair-partition codes, were slower, with cost ratios up to 1.6, and the cost model predicted ratios above one for all four.
Such small layers gain little even when tables pay, and estimating the share of skipped supports takes milliseconds, so a search considers tables only for layers of at least `completion_bound_minimum_layer_work` supports.
The share of skipped supports depends on the code, the scanned set, and the order of its rows, and other codes may gain less, or nothing when a search declines the tables.
Skipping does not reduce the number of layers that a search must complete, and the number of prefixes still grows like that of supports for a fixed ratio `w/r`.

## 4. Guide to the source code

**Audience note:** This section is a highly technical summary for contributors who need to locate identifiers in the BZ implementation of qLDPC.
For a high-level explanation, see Sections [2](#2-the-brouwer-zimmermann-algorithm) and [3](#3-extensions-and-optimizations).

This section is a guide to the source files in this directory.
It connects the concepts of Sections [2](#2-the-brouwer-zimmermann-algorithm) and [3](#3-extensions-and-optimizations) to modules, identifiers, and behavior that spans files.
The thresholds, limits, and coefficients of the heuristics are fields of `SearchTuning` in [_tuning.py](_tuning.py), whose comments give their meanings and defaults and say how each was chosen, and [Section 4.4](#44-heuristics-and-cost-models) says where each is used.
Section 3.5 of [connected_support.md](connected_support.md) describes the field of the connected-support search.

### 4.1 Module map and layering

[README.md](README.md) maps the modules of this directory, says where each kind of change goes, and lists the levels in which the modules import each other.
The rest of this section describes how the searches use those modules.

[`__init__.py`](__init__.py) exports names from [core.py](core.py) and [logical_counts.py](logical_counts.py).
[logical_counts.py](logical_counts.py) counts the logical operators of a given weight exactly.
It builds its families and certificates with the same stages of [_search/information_sets.py](_search/information_sets.py) as the distance search, under the constraint in [Section 5](#5-correctness-requirements).
It runs the stages in `_select_basis_family` and `count_words` of [_counting/scan.py](_counting/scan.py), and its distance search starts from the disjoint basis families that it keeps for counting.
The set-up functions of both searches can be compiled on request, through [_compilation.py](../../_compilation.py).
The code that only counting uses is in the `_counting` package.
`get_overlapping_family_policy` in [_search/information_sets.py](_search/information_sets.py) gives every caller of the overlapping stage its thresholds and its fixed-seed generator, so a change to either applies to both consumers.
[core.py](core.py) calls [_search/brouwer_zimmermann.py](_search/brouwer_zimmermann.py), [_search/brute_force.py](_search/brute_force.py), and [_search/connected_support.py](_search/connected_support.py), which [connected_support.md](connected_support.md) describes, and [_search/brouwer_zimmermann.py](_search/brouwer_zimmermann.py) calls [_search/brute_force.py](_search/brute_force.py) for its exhaustive routes.
All four modules use [_binary.py](_binary.py), and they and [_search/information_sets.py](_search/information_sets.py) use the bit-packed linear algebra over GF(2) in [qldpc/_binary.py](../../_binary.py).
Every module with compiled functions uses [_compilation.py](../../_compilation.py).
`_get_distance_css_brouwer_zimmermann` of [core.py](core.py) builds an adapted basis ([Section 3.4](#34-quotient-labels-for-the-stabilizer-test)) for each nontrivial sector, then passes the resulting problems to `get_distance_brouwer_zimmermann_many`.
It also passes a function that returns automorphisms of the code.
`get_distance_exact` of `CSSCode` in `src/qldpc/codes/common.py` passes the code's `_found_qudit_automorphisms`, which uses bliss from `src/qldpc/external/_bliss.py` or the code's `_get_qudit_automorphisms` method.
`get_distance_exact` of `QuditCode` passes the same function to `_get_distance_quantum`, a private form of `get_distance_quantum` that accepts automorphisms.
`_get_distance_quantum` lifts them to the coordinates of a general Pauli problem with `_get_lifted_automorphisms`, and passes its problem to `get_distance_brouwer_zimmermann_many`.
[_search/automorphism_families.py](_search/automorphism_families.py) verifies automorphisms and builds the automorphism families of [Section 3.7](#37-images-of-an-information-set-under-automorphisms), using [_search/information_sets.py](_search/information_sets.py) for their certificates.
The distance search uses [_search/completion_bounds.py](_search/completion_bounds.py) to choose and build the tables of [Section 3.9](#39-skipping-completions-that-cannot-lower-u).
The distance search and `CosetLeaderSearch` scan their layers with the compiled entry points of [_search/affine_search.py](_search/affine_search.py), which run one body, `_scan`, with constant flags.
`scan_family` scans every set of a family in lockstep, `scan_information_set` scans one information set with or without completion tables, and `scan_coset_layer` and `scan_coset_layer_by_css_weight` scan one layer of one set of a coset and keep its lightest word.
Numba compiles the body separately for each entry point, so no flag costs a branch for each word.
Each search builds an `AffineInstance` of [_search/affine_instance.py](_search/affine_instance.py) for its problem before it scans, rather than for each layer or query.
The instance holds the packed rows, labels, and pivots of the sets that the search scans, the ranks of those sets, and the certificate of their steps.
The weight divisor of the certificate rounds its Hamming bounds up to multiples of a proven divisor of eligible weights.
`get_distance_instance` packs a basis family with its pivot columns deleted ([Section 3.5](#35-packed-enumeration-with-pivot-column-deletion)), and takes the certificate that the distance search selects.
The distance search builds another instance when it switches to an automorphism family, whose certificate covers every image of the one set that it scans.

`CosetLeaderSearch` in [_search/coset_leader_search.py](_search/coset_leader_search.py), which the fault-propagation certifier in `qldpc.circuits.certification` uses, applies the same search to a coset `e + V` instead of `V`, unless it is built with `method="connected_support"` ([connected_support.md](connected_support.md)).
There a layer adds `w` rows to an **offset**, a word of the coset that is zero on the pivot columns ([Section 2.3](#23-systematic-bases-and-information-sets)), so layer zero is the offset itself.
A **query** of `CosetLeaderSearch` asks whether the coset of a syndrome has a word of weight at most a given maximum weight.
`CosetLeaderSearch` answers each query with one or two **coset searches**: a search of Pauli errors, or, for the CSS weight on a CSS state, searches of their X and Z parts, as described at the end of this section.
Each coset search is built from a **coset instance**, which `get_coset_instance` packs from the full rows of the disjoint sets of its code.
The certificate of a coset instance applies unchanged, except that it takes no weight divisor, because the words of a coset need not share the divisibility of `V`, even when `get_disjoint_family` infers a divisor for `V`.
If every word of `V` has even weight, and some coset of the search has odd words, the instance instead rounds its bounds up to the parity of each coset ([Section 3.2](#32-rounding-the-certificate-to-a-proven-weight-multiple)); `AffineInstance.parity_rounding` records this.
In a search of Pauli errors, an error weighs at least half the Hamming weight of the binary word that represents it.
So the instance of such a search has a **weight factor** of two: `AffineInstance.get_weight_bounds` divides the bounds of its certificate by two, rounding up, after any rounding to parity.
A query of a coset search asks the same of a coset of its code, whose words have syndromes for the checks of that code.
It starts from the **word of its syndrome**, a word of the coset that sums the rows of a right inverse of the checks for the set bits of the syndrome.
It stops at its **stopping step**, the first step whose bound exceeds the maximum weight, unless it finds a light enough word sooner.
It skips each rank-deficient search basis whose deficiency keeps it out of the bound of that step.
`find_stopping_step` of [_search/information_sets.py](_search/information_sets.py) finds the stopping step by a binary search of steps whose bounds never decrease.
Each coset search keeps the bounds of its steps in a **planning buffer**, with a table for each parity of its cosets if it rounds to parity, and one table otherwise.
A query takes the table of the parity of the word of its syndrome, and each table has one more row that caches a stopping step.
For a nonnegative maximum weight, `_get_stopping_step` reuses the cached step of a table exactly while it stays the stopping step of that weight, and replaces it otherwise.
A query below maximum weight two looks up the words of weight at most one instead of searching, and without one bounds the weight of the coset by two, or by three for an odd coset of a search of X or Z parts that rounds to parity.
The distance search also uses `find_stopping_step`, through `Certificate.get_num_sets_to_certify`, when its certificate solves no linear programs.
A distance certificate with linear programs instead checks its steps in order with `certifies`, which solves each program only when the coverage bound falls short.
For the CSS weight `max(|x|, |z|)` on a CSS state, whose stabilizer group is generated by stabilizers with no Z part and stabilizers with no X part, `CosetLeaderSearch` instead runs this search on two classical codes, spanned by the X parts of the first kind and by the Z parts of the second, with words weighed by their Hamming weight.

### 4.2 Flow and exit points

A public entry point validates inputs and chooses a route.
For the BZ route, `get_distance_brouwer_zimmermann_many` prepares each problem and coordinates any searches that remain.
Preparation either returns a result immediately or produces a `_BrouwerZimmermannSearch` whose `advance` method searches one coefficient layer, or only the leading search bases of a layer whose completion of it certifies `U`.
For a joint CSS search, the coordinator advances the prepared searches in round-robin order and shares the best witness weight.

The tables below index the public routes, the preparation exits, and the subsequent search.
An integer returned by preparation can incorporate a witness from another sector or the bound of a weight limit, so it need not be the exact distance of the sector being prepared.

| Public route in [core.py](core.py) | Outcome |
| --- | --- |
| Trivial input | Returns `n+1`, capped at `max_weight+1`, for an empty classical code, a quantum problem with no logical rows, or `V = S` |
| `method="brute_force"` | Runs exhaustive enumeration over an independent basis, adapted to `S ⊆ V` for a quantum problem |

`_prepare_brouwer_zimmermann_search` considers the following exits in order.
The disjoint family, its pruning, and any overlapping or partitioned replacement come from the stages of [_search/information_sets.py](_search/information_sets.py); the exits between those stages belong to the distance search.

| Preparation stage | Outcome |
| --- | --- |
| Initial row witness | Returns if `U ≤ cutoff`, where `U` includes the lightest eligible row and any supplied bound (from another sector or a weight limit) |
| Exhaustive by size | Runs exhaustive enumeration if the tuning field `allow_enumeration_by_size` is true and `enumeration_is_cheaper_than_setup(dimension, length)` approves |
| Rows of the disjoint family | Returns if checking the rows of every search basis gives `U ≤ cutoff` |
| Certificate of the disjoint family | Returns if `L(1) ≥ U` |
| Rows of each replacement family, overlapping and then partitioned, if selected | Returns if `U ≤ cutoff` |
| Certificate of each replacement family | Returns if `L(1) ≥ U` |
| Exhaustive by work | Runs exhaustive enumeration if the tuning field `allow_enumeration_by_work` is true and `_enumeration_is_cheaper_than_search` finds that the eligible-word count is at most the estimated BZ work of [Section 3.6](#36-exhaustive-enumeration-as-a-fallback) |
| Ready search | Returns a `_BrouwerZimmermannSearch` that scans from layer two |

Exhaustive enumeration is chosen automatically at three points: the direct shortcut, preparation's size check, and preparation's work check.

| Search location | Outcome |
| --- | --- |
| `_BrouwerZimmermannSearch.advance` | Returns after one layer, or after the leading search bases of a layer that certify `U`, or earlier if `U ≤ cutoff`; a search may first switch to an automorphism family ([Section 3.7](#37-images-of-an-information-set-under-automorphisms)); a finished search does no further work |
| `_BrouwerZimmermannSearch.finished` | Marks a search finished if `U ≤ cutoff`, `L ≥ U`, or every layer has been enumerated (`next_weight > dimension`) |
| `get_distance_brouwer_zimmermann_many` | Returns the shared best bound when it is at most `cutoff` after an advance, or when no search can advance |

The tests `test_brouwer_zimmermann_setup_exits`, `test_brouwer_zimmermann_setup_cutoffs`, `test_rows_of_an_automorphism_family_can_reach_the_cutoff`, and `test_brouwer_zimmermann_exhaustive_dispatch` exercise these exits.

### 4.3 Concept-to-code glossary

| Concept | Code |
| --- | --- |
| Upper bound `U` | `best`, or `upper_bound` as a parameter |
| Lower certificate `L` | `lower_bound` of a search, computed by `Certificate.lower_bound(completed_weight)` from the certificate's `dimension`, `ranks`, `weight_divisor`, and `max_coverage`; it bounds the eligible words lighter than `U`, so a bound on the distance is `min(L, U)` |
| Certificate within a layer `L(t, j)` | `Certificate.step_lower_bound(weight, num_completed)` and the table of `get_step_lower_bounds`. `Certificate.get_num_sets_to_certify` gives the smallest `j` that certifies a bound. `find_stopping_step` gives the first step whose bound exceeds a limit. `Certificate.get_weight_bounds` divides the table by a weight factor, rounding up, after rounding it up to a parity if it is given one. A coset search keeps the results in its planning buffer, `planning` of `_CosetSearchTables`: a table for each parity of its cosets if it rounds to parity, and one table otherwise, each with a cached stopping step. |
| Certificate from linear programming | `get_lp_lower_bound`, which a `Certificate` with `pivot_blocks` combines with the coverage-divided certificate; `BasisFamily.get_search_certificate` adds the blocks to costly searches of overlapping families |
| Family of search bases | `BasisFamily`, which stacks the search bases into arrays and gives their `Certificate`. A distance search keeps the `AffineInstance` of [_search/affine_instance.py](_search/affine_instance.py) that packs them, with the certificate. A coset search keeps its generators, pivots, and planning buffer in `_CosetSearchTables`. |
| Automorphisms, generators of `G` | `automorphisms`, whose row `i` maps column `j` to column `automorphisms[i, j]`, as returned by `get_automorphisms`; `_found_qudit_automorphisms` and `_get_qudit_automorphisms` of `QuditCode` |
| Automorphism family | `get_automorphism_family` of [_search/automorphism_families.py](_search/automorphism_families.py), which returns a `BasisFamily` of the scanned set and the `Certificate` of all of its images; `_switch_to_automorphism_family` adopts them, and sets `uses_automorphism_family` of the search |
| Last completed layer `t`, next layer | `completed_weight`, `next_weight` |
| Divisor `D` | `Certificate.weight_divisor`, in Hamming units of the searched basis. Coset searches of even codes round to the parity of each coset instead (`AffineInstance.parity_rounding`). |
| Maximum coverage `C` | `Certificate.max_coverage` |
| Dimension `r`, block rank `r_i` | `dimension`, `ranks` (`BasisFamily.ranks`). `AffineInstance.ranks` holds the ranks of the packed sets of a search, counted from their pivots, as a native array. |
| Search basis and its block | One set of a `BasisFamily`: its entries of `generators`, `labels`, `pivots`, and `ranks`; the recorded block is the pivot subset, as explained below |
| Number of coordinates `N` | The length of `columns` (or `nonzero_columns`), the array of nonzero columns, in the functions that build families of search bases |
| Stored width `N_stored` | `basis.shape[1]` before pivot deletion, including any all-zero columns |
| Labels `M` | `labels`, which has no columns for a classical problem |
| Eligible rows | `get_eligible_rows` of [_binary.py](_binary.py): every row of a classical basis; rows with nonzero labels for a quantum problem |
| Support, prefix XORs | `support` and `prefixes` in `advance` and in the scans of [_search/affine_search.py](_search/affine_search.py), which step through a layer with `advance_support` of [_search/information_sets.py](_search/information_sets.py) |
| Packed generators | `generators` of the `AffineInstance` of a distance search: `uint64` words, after deleting pivot columns. `widths` counts the valid words for each search basis. |
| Packed labels | `labels` of the `AffineInstance` of a distance search: `uint64` quotient-label words, without pivot-column deletion; `label_widths` counts the valid words |
| Chunk | `batch_size`, equal to `1 << information_set_batch_exponent` supports, or supports and lookups together in a scan with completion tables ([Section 4.5](#45-behavior-spanning-files)) |
| Candidate/set evaluation | The unit counted by `Certificate.work` |
| Completion tables ([Section 3.9](#39-skipping-completions-that-cannot-lower-u)) | `CompletionTables` of [_search/completion_bounds.py](_search/completion_bounds.py), chosen for each layer by `select_completion_tables` from the completion fields of the search's `SearchTuning` and kept in `completion_tables` of a search; the scan is `scan_information_set` of [_search/affine_search.py](_search/affine_search.py), which returns the numbers of supports that it weighs and skips and of bounds that it looks up |

For the rank-deficient remainder of [Section 2.4](#24-search-bases-and-rank-deficient-blocks), the implementation records only the independent pivot columns, in the first `rank` entries of the set's row of `BasisFamily.pivots`, and pads the rest with -1.
The remaining columns of that remainder stay in the generator matrix.
This smaller block has the same rank and gives the same certificate term `b_i(t)`.
Coverage and pivot-column deletion both use the recorded pivot subset.

### 4.4 Heuristics and cost models

The certificate is computed from verified ranks and the actual maximum coverage of whichever family of search bases is chosen.
The heuristics below change search work while preserving certificate validity.
With an early-exit cutoff above one, a different search order can find a different first witness at or below the cutoff, while preserving the upper-bound contract of [Section 4.5](#45-behavior-spanning-files).

The Tuning section of [README.md](README.md) explains how searches read the thresholds of these heuristics.

- **`get_disjoint_family`** builds the disjoint family greedily in natural column order.
  If that does not reach the maximum number of blocks that are information sets, it tries `disjoint_random_column_orders` fixed-seed random column orders and keeps the family with the lexicographically largest tuple of ranks.
  Fixed seeds keep results reproducible, and the bounded retries keep setup cheap.
  It also infers the divisor `2` when every basis row has even weight, and combines it with any divisor supplied by the caller.
- **`prune_family`** is exact, not heuristic.
  A search basis whose block has rank deficiency `d = r - r_i` has a term `b_i(t)` that is zero for `t < d`.
  The function removes such a search basis when the blocks that are information sets reach `U` by layer `d`, so the removal leaves the result unchanged and saves enumeration.
  It removes nothing if no block is an information set.
- **`get_overlapping_family`** tries to lower the work estimate of the disjoint family, with the thresholds `max_information_sets` and `overlapping_minimum_work`, which it receives as `max_sets` and `minimum_work`.
  It returns a basis family with no sets if the existing work is below `minimum_work`, if the blocks already use every nonzero column, if the family is at `max_sets`, or if even the optimistic coverage `ceil(m*r/N)` cannot reduce the estimate.
  Otherwise it builds search bases by row reducing on column orders that put the least-used columns first, breaking ties with a generator that `get_overlapping_family_policy` seeds afresh for every call, and keeps the cheapest family built.
  It makes `overlapping_attempts_per_set` attempts for each set that it may add, and a repeated pivot set is discarded but still consumes an attempt.
- **`get_partitioned_family`** builds families of information sets in which no column appears more than `b` times, for `b` from two to `max_partition_coverage`.
  It uses the matroid partitioning of Section 5.1 of Lisoněk and Trummer (see [References](#references)), in `_search/matroid_partition.py`, so each family has the most information sets, up to `max_information_sets`, that its coverage allows.
  A search uses the family with the least estimated work if it does less work than the family of the earlier stages.
  It tries this only if its estimated work is at least `partition_work_ratio` times an estimate of the work of partitioning, and the optimistic coverage `ceil(m*r/N)` does not rule out an improvement.
- **`get_search_certificate`** adds the certificate from linear programming ([Section 3.1](#31-overlapping-search-bases)) to searches of overlapping families whose estimated work reaches `lp_minimum_work`, since smaller searches cost less than their linear programs.
  The work estimates that choose families use only the coverage-divided certificate.
- **`get_automorphism_family`** row reduces in natural column order, then exchanges pivots between orbits in the compiled `_balance_orbit_shares`, as [Section 3.7](#37-images-of-an-information-set-under-automorphisms) explains.
  A search asks for automorphisms before the first layer whose scan would cost at least `automorphism_minimum_layer_work`, and switches to the automorphism family if `estimate_automorphism_family_work` is less than the `Certificate.work` of its family.
- **`select_completion_tables`** decides whether a layer of one information set uses completion tables, as [Section 3.9](#39-skipping-completions-that-cannot-lower-u) explains.
  It considers layers of at least `completion_bound_minimum_layer_work` supports, uses tables if it predicts that they cost at most `completion_bound_max_cost_percent` percent of the cost without them, and keeps tables within `completion_bound_max_table_bytes`.
  Random supports only estimate the value of tables; a search skips supports only when `w` plus their computed bound reaches `U`.
- **`Certificate.work`, `estimate_automorphism_family_work`, `_enumeration_is_cheaper_than_search`, and `enumeration_is_cheaper_than_setup`** are cost models, which Sections [3.6](#36-exhaustive-enumeration-as-a-fallback) and [3.7](#37-images-of-an-information-set-under-automorphisms) explain.
  They are not runtime guarantees.
  `enumeration_is_cheaper_than_setup` compares the setup cost of BZ with packed enumeration of all words, with the fields `enumeration_max_dimension` and `enumeration_setup_cost_factor`.

### 4.5 Behavior spanning files

- **Cutoff.**
  `get_distance_classical` and `get_distance_quantum` default to `cutoff=1`, and `_get_distance_css_brouwer_zimmermann` requires a `cutoff`.
  If a witness reaches `cutoff`, the computation can return its weight immediately as an upper bound, which can exceed the exact distance.
  Otherwise, the search continues and returns the exact distance, which may be larger than `cutoff`.
  The default of one is still exact for a valid nontrivial binary problem, because a distance is at least one.
  A `cutoff` of zero or less never triggers an early exit for such a problem, and `advance` clamps negative values to `-1` before it calls the compiled scan.
  [Section 4.2](#42-flow-and-exit-points) lists where the `cutoff` is checked.
- **Weight limit.**
  `get_distance_classical` and `get_distance_quantum` accept `max_weight`, which limits the search to weights at most `max_weight`: the result is `min(distance, max_weight+1)`.
  They pass `upper_bound=max_weight+1` to the BZ route, as though a witness of that weight were known, so the search stops once its certificate reaches that bound, and they clamp the results of the other routes to it.
  A limit below the distance therefore avoids the cost of certifying the distance itself.
  The limit also caps `cutoff` at `max_weight`.
  Without that cap, a cutoff at or above `max_weight+1` would end the search at once on the assumed witness, even if a lighter word exists.
  Passing `cutoff=max_weight` as well stops the search at the first word within the limit.
  Inputs without nontrivial words have no distance and give `n+1`, capped at `max_weight+1`.
  No word on `n` bits or qubits has weight `n+1`.
  For non-homogeneous input, the BZ route doubles the bound along with `cutoff`.
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
  A scan of one information set with completion tables ([Section 3.9](#39-skipping-completions-that-cannot-lower-u)) returns after at most `batch_size` supports and lookups together, so it stays responsive while it skips most supports.
  Within one `advance` call, successive compiled batches reuse the local `support` and `prefixes` arrays.
  `advance` changes `lower_bound` and `next_weight` only after a layer is complete in every search basis, or when the search switches to an automorphism family, which sets `next_weight` to two and keeps the larger `lower_bound`.
  Aborting `advance`, for example with an exception or a cutoff return, leaves those completed-layer fields unchanged.
  A subsequent call that is still permitted to advance restarts the unfinished layer from its first support.
- **CSS sectors.**
  A joint search runs one search per sector in round-robin order and shares the best bound.
  Each sector keeps its own `lower_bound` and the `certificate` of its `instance`, including its coverage and divisor.
- **Automorphisms.**
  `get_distance_exact` of `QuditCode` and of `CSSCode` pass the code's `_found_qudit_automorphisms` to their BZ searches, as a function.
  Each search calls the function only before a costly layer, and the code caches its result, so bliss runs at most once for the code.
  A search uses the generators only after `verify_automorphisms` accepts them for its problem.
  A generator preserves both sectors of a CSS code if it maps the row spaces of both parity check matrices onto themselves.
  For a general Pauli problem, `_get_distance_quantum` lifts each generator to the three blocks of coordinates of [Section 2.7](#27-pauli-weight-as-hamming-weight).
  Only these BZ searches of binary codes receive automorphisms.

## 5. Correctness requirements

**Audience note:** This section is a highly technical summary for contributors who need to check a change to the BZ implementation of qLDPC against its requirements.
For a high-level explanation, see Sections [2](#2-the-brouwer-zimmermann-algorithm) and [3](#3-extensions-and-optimizations).

Changes to the BZ algorithm can use different representations, certificates, and search strategies while preserving the following requirements for the nontrivial problems of [Section 2.1](#21-the-problem).

- **Search the intended space and recognize eligible witnesses.**
  Transformations must preserve `V` and the excluded subspace `S`, or provide an equivalent representation of `V \ S`.
  With quotient labels, this means preserving the membership test of [Section 3.4](#34-quotient-labels-for-the-stabilizer-test).
- **Preserve the meaning of weights.**
  Candidate weights, certificates, cutoffs, and returned results must use consistent units.
  For the current implementation, this includes restoring the deleted pivot weight ([Section 3.5](#35-packed-enumeration-with-pivot-column-deletion)) and converting between Pauli and Hamming weights ([Section 4.5](#45-behavior-spanning-files)).
- **Use only justified certificates.**
  A certificate `L` must hold for every eligible word lighter than `U` in the search where it is used, which are all unexamined.
  It need not hold for heavier words, such as skipped ones, so a bound that a search reports on the distance is `min(L, U)`, not `L`.
  The current formula relies on block ranks, overlap coverage, and completed layers (Sections [2.4](#24-search-bases-and-rank-deficient-blocks), [2.5](#25-the-stopping-certificate), and [3.1](#31-overlapping-search-bases)).
  Coverage may be conservatively overestimated; underestimating it can make the formula unsound.
  Optional rounding requires a congruence that every eligible weight satisfies, such as a common divisor, or the parity of a coset of an even code ([Section 3.2](#32-rounding-the-certificate-to-a-proven-weight-multiple)).
  A certificate from another CSS sector needs justification for the sector where it is applied, just as any other certificate does.
  The certificate of an automorphism family holds instead for every eligible word whose images under the automorphisms are all unexamined, so it requires automorphisms that map `V` and `S` onto themselves ([Section 3.7](#37-images-of-an-information-set-under-automorphisms)), which searches verify before they use them.
- **Skip only supports that a bound proves cannot lower `U`.**
  A search may skip supports only if a lower bound on the weight of every one of their words reaches `U` ([Section 3.9](#39-skipping-completions-that-cannot-lower-u)).
  The completion bounds of qLDPC rely on a minimum over every remaining pair of rows, on disjoint blocks of columns, on exactly `w` pivot ones, which hold only for a block that is an information set, and on the order in which the scan sums the rows.
- **Keep the strict certificate available to fixed-weight counting.**
  [logical_counts.py](logical_counts.py) builds its basis family with the same stages as the distance search, in `_select_basis_family` and `count_words` of [_counting/scan.py](_counting/scan.py), and stops at the first step after which that basis family's certificate exceeds a target weight `w`, to count every word of weight `w`, which defaults to the distance.
  Distance may stop at `L(t, j) ≥ U`, but counting must also exhaust words of weight exactly `w`, so it stops only at `L(t, j) > w`.
  For the same reason, counting does not skip supports, which may hold words of the target weight `w`.
  A word of weight `w` belongs to the first step that visits it, and the steps that the scan visits are the first ones in order, so stopping within a layer still counts every such word once.
  Its bound must use the ranks and actual maximum coverage of the family it scans; it passes `upper_bound=w+1` to the family stages only to aim them at that stopping point.
- **Preserve the return contract.**
  Return the exact minimum unless taking an early exit permitted by `cutoff`.
  Such an early exit returns the weight of a known eligible witness, an upper bound between the distance and `cutoff` ([Section 4.5](#45-behavior-spanning-files)).
  Under a weight limit, return the smaller of the distance and `max_weight+1`; an early exit must still return a witness within the limit.

Representative correctness tests check certificate validity against exhaustive words, including rank-deficient blocks, inferred parity, and overlapping coverage (`test_family_certificates` and `test_overlapping_family_certificates` in [_search/information_sets_test.py](_search/information_sets_test.py)), and joint-search results with distinct sector certificates (`test_brouwer_zimmermann_sector_certificates` in [_search/brouwer_zimmermann_test.py](_search/brouwer_zimmermann_test.py)).
`test_counting_continues_at_an_equal_lower_bound` in [logical_counts_test.py](logical_counts_test.py) checks the strict counting stop with a count that a stop at `L(t) = w` would miss.
`test_automorphism_families_certify_like_the_family_of_all_images` in [_search/automorphism_families_test.py](_search/automorphism_families_test.py) checks the certificate of an automorphism family against every image of the scanned set, and `test_automorphism_family_searches_scan_every_layer_they_certify` in [_search/brouwer_zimmermann_test.py](_search/brouwer_zimmermann_test.py) checks a search of an automorphism family whose lightest word appears only in its last layer.
`test_automorphism_family_searches_scan_their_new_set_from_layer_two` checks a distance that a search that switched to an automorphism family would miss if it skipped the early layers of its new set.
`test_searches_ask_for_automorphisms_only_before_costly_layers` checks that easy searches do not pay to find automorphisms.
`test_bounds_are_exact_relaxations` in [_search/completion_bounds_test.py](_search/completion_bounds_test.py) checks completion bounds against enumerated completions.
`test_scan_of_one_information_set` in [_search/affine_search_test.py](_search/affine_search_test.py) checks that the scan finds the lightest word of a layer and weighs or skips each support exactly once, and `test_completion_tables_keep_distances_exact` in [_search/brouwer_zimmermann_test.py](_search/brouwer_zimmermann_test.py) checks whole searches against brute force.
In [core_test.py](core_test.py), `test_brouwer_zimmermann_random_cross_checks` forces the BZ route and compares both homogeneous and general Pauli problems with exhaustive results.
`test_max_weight` checks weight-limited results against enumeration on every route, and `test_max_weight_caps_cutoff` checks a case that a weight limit without the `cutoff` cap would get wrong.

Performance tests also check broad work budgets, and helper tests exercise the current implementation's interfaces.
Those tests can evolve with the implementation; they do not make a particular information-set family, packing order, or optimization a correctness requirement.

## References

- [arXiv:1603.06757v2](https://arxiv.org/abs/1603.06757v2) (Algorithm 1): Section 2 and Algorithm 1 give the classical disjoint and rank-deficient bound.
- M. Grassl, "Searching for linear codes with large minimum distance," in *Discovering Mathematics with Magma*, Springer, 2006, pp. 287-313 ([doi:10.1007/978-3-540-37634-7_13](https://doi.org/10.1007/978-3-540-37634-7_13)): Algorithm 2.5 gives the certificate for overlapping information sets, footnote 2 on page 295 the certificate within a layer, page 299 the rounding to a weight divisor, and Algorithm 2.11 the choice between BZ and the weight distribution.
- P. Lisoněk and L. Trummer, "Algorithms for the minimum weight of linear codes," *Advances in Mathematics of Communications* 10(1), 195-207, 2016 ([doi:10.3934/amc.2016.10.195](https://doi.org/10.3934/amc.2016.10.195)): equation (5) gives the certificate within a layer, Proposition 5 a coverage-divided certificate for any blocks, and Section 5.1 builds families of information sets with bounded coverage by matroid partitioning.
- [arXiv:2408.10743v2](https://arxiv.org/abs/2408.10743v2): Section 2.4 describes the `Saved_isometry` symplectic-to-Hamming construction.
- [core.py](core.py), [_search/brouwer_zimmermann.py](_search/brouwer_zimmermann.py), [_search/coset_leader_search.py](_search/coset_leader_search.py), [_search/affine_search.py](_search/affine_search.py), [_search/affine_instance.py](_search/affine_instance.py), [_search/information_sets.py](_search/information_sets.py), [_search/automorphism_families.py](_search/automorphism_families.py), [_search/completion_bounds.py](_search/completion_bounds.py), [_search/matroid_partition.py](_search/matroid_partition.py), [_search/brute_force.py](_search/brute_force.py), [_binary.py](_binary.py), and [qldpc/_binary.py](../../_binary.py): the implementation.
- [_search/brouwer_zimmermann_test.py](_search/brouwer_zimmermann_test.py), [_search/coset_leader_search_test.py](_search/coset_leader_search_test.py), [_search/affine_search_test.py](_search/affine_search_test.py), [_search/affine_instance_test.py](_search/affine_instance_test.py), [_search/information_sets_test.py](_search/information_sets_test.py), [_search/automorphism_families_test.py](_search/automorphism_families_test.py), [_search/completion_bounds_test.py](_search/completion_bounds_test.py), and [core_test.py](core_test.py): executable invariants and independent small-instance oracles.
