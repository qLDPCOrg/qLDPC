# Brouwer-Zimmermann distance search in qLDPC

## 1. Purpose and scope

The Brouwer-Zimmermann (BZ) algorithm computes the distance of a classical code ([arXiv:1603.06757](https://arxiv.org/abs/1603.06757)), and has been adapted to quantum codes ([arXiv:2408.10743](https://arxiv.org/abs/2408.10743)).
This guide explains how qLDPC computes exact distances of binary codes with BZ, and how it counts logical operators ([Section 3.11](#311-counting-logical-operators)), for contributors who need to understand, debug, or modify the BZ code in this directory.
[connected_cluster.md](connected_cluster.md) describes another exact search, for codes with sparse checks.

- [Section 2](#2-the-brouwer-zimmermann-algorithm) explains BZ as published, including its adaptation to quantum codes.
- [Section 3](#3-extensions-and-optimizations) explains the extensions and optimizations of qLDPC, and where each comes from.
- [Section 4](#4-guide-to-the-source-code) maps these concepts to the source code, for contributors who need to locate identifiers.
- [Section 5](#5-correctness-requirements) lists the requirements that a change must preserve.

## 2. The Brouwer-Zimmermann algorithm

This section describes BZ as presented in [arXiv:1603.06757](https://arxiv.org/abs/1603.06757) and [arXiv:2408.10743](https://arxiv.org/abs/2408.10743), together with a stopping rule within a layer from Grassl's chapter and from Lisoněk and Trummer (see [References](#references)).

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

Both definitions are instances of one problem: given binary spaces `S ⊆ V` with `V ≠ S`, find the lightest word in `V \ S`, where a classical code has `S = {0}`.
A word in `V \ S`, a nonzero codeword or a nontrivial logical operator, is called **eligible**.

### 2.2 The idea: upper bounds and lower certificates

A **basis** `G` of `V` is a matrix of `r = dim(V)` independent rows that span `V`, and its `N` columns are the **coordinates** of a word.
We ignore coordinates that are zero on every word of `V`.
Each word of `V` is `aG` for exactly one **coefficient vector** `a`, a binary vector of length `r` that selects rows; `wt(a)` is the word's **coefficient weight**.
A **layer** is the set of words of one coefficient weight.

BZ keeps track of two quantities:

- A **witness** is an eligible word whose weight is known, and the lightest has weight `U`, an **upper bound** on the distance.
- A **lower certificate** is a proven bound `L` on the weight of every eligible word not yet examined.

BZ examines words layer by layer, from coefficient weight one.
Each layer can only lower `U`, and raises `L`, since the words that remain unexamined need more rows.
Once `L ≥ U`, no unexamined word is lighter than the best witness, so `U` is the distance.
Information sets explain why the unexamined words must be heavy.

### 2.3 Systematic bases and information sets

An **information set** is a set `I` of `r` coordinate indices (column indices) such that the columns of the basis indexed by `I` form a rank-`r` matrix.
Row reduction on `I` produces a **systematic basis** `G_I`, whose restriction to `I` is the identity matrix:

$$
G_I[:,I] = I_r.
$$

The columns indexed by `I` are the **pivot columns** of `G_I`.

Write `v|_I` for the restriction of a word `v` to `I`.
If `v = aG_I`, then

$$
v|_I = a\,G_I[:,I] = a\,I_r = a,
$$

so `wt(v) ≥ wt(v|_I) = wt(a)`: the coefficient weight of a word in `G_I` bounds its weight.
**Searching a basis through weight `t`** means examining every XOR of at most `t` of its rows.
After searching `G_I` through weight `t`, every unexamined word `v` therefore has `wt(v) ≥ wt(v|_I) ≥ t+1`.

### 2.4 Search bases and rank-deficient blocks

BZ searches a **disjoint family** of `m` **search bases** `G_1, ..., G_m` of `V`, each row reduced on its own **block** `B_i` of columns, with the blocks pairwise disjoint.
The blocks are chosen greedily: `B_1` is an information set, `B_2` an information set among the other columns, and so on, and any columns that contain no further information set form one last block, of rank less than `r`.
Once qLDPC has taken `2^r - 1` information sets ([Section 4.4](#44-heuristics-and-cost-models)), it leaves any remaining columns out.

If `B_i` has rank `r_i`, row reducing on `B_i` gives a basis `G_i` of `V` whose first `r_i` rows, the **pivot rows**, have distinct pivot columns in `B_i`, and whose other rows are zero on `B_i`.
A word `v` of coefficient weight at least `t+1` in `G_i` selects at least `t + 1 - (r - r_i)` pivot rows, each of which contributes a distinct one to `v|_{B_i}`, so

$$
\operatorname{wt}(v|_{B_i}) \ge b_i(t) := \max(0,\;t+1-(r-r_i)),
$$

which is `t + 1` for an information set, as in [Section 2.3](#23-systematic-bases-and-information-sets).

### 2.5 The stopping certificate

Suppose that every search basis of [Section 2.4](#24-search-bases-and-rank-deficient-blocks) has been searched through weight `t`.
An eligible word `v` that none of them examined has `wt(v|_{B_i}) ≥ b_i(t)` for every `i`, and since the blocks are disjoint,

$$
\operatorname{wt}(v) \ge \sum_i b_i(t) =: L(t),
$$

which is `m(t+1)` if every block is an information set.
`L(t)` is the lower certificate.
It bounds only unexamined words, so it can exceed the distance, and once `L(t) ≥ U` the search can stop.

BZ is therefore the following loop:

1. Build the search bases `G_1, ..., G_m` from disjoint blocks, as in [Section 2.4](#24-search-bases-and-rank-deficient-blocks).
2. Find the lightest eligible row among the rows of the search bases, which gives an initial `U`.
3. For `t = 1, 2, ...`, examine every selection of `t` rows in each search basis, lowering `U` whenever a lighter eligible word is found.
   Then compute `L(t)`, and stop if `L(t) ≥ U`.

The search terminates with the exact distance, at the latest after `t = r`, when every nonzero word has been examined.
Algorithm 1 of [arXiv:1603.06757](https://arxiv.org/abs/1603.06757v2) gives this loop and this certificate for disjoint and rank-deficient blocks.

The certificate can also advance within a layer.
If the first `j` search bases have been searched through weight `t`, and the others through `t-1`, an unexamined eligible word has

$$
\operatorname{wt}(v) \ge \sum_{i \le j} b_i(t) + \sum_{i > j} b_i(t-1) =: L(t, j).
$$

`L(t, 0) = L(t-1)` and `L(t, m) = L(t)`.
Completing layer `t` in search basis `j` is step `(t, j)`, and the steps run in the order `(1, 1), ..., (1, m), (2, 1), ...`.
When layer `t` starts, qLDPC finds the least `j` with `L(t, j) ≥ U`, or `m` if there is none, and searches only the first `j` bases in that layer.
It searches these bases one at a time, and stops sooner once a word lowers `U` to at most the bound of a completed step.
Stopping within a layer can save most of the last layer, which often dominates the cost of a search.
Grassl notes this improvement in footnote 2 of his chapter, on page 295 of *Discovering Mathematics with Magma*, and Magma uses it.
Equation (5) of Lisoněk and Trummer gives the same certificate.

The cost of a search is counted in **candidate/set evaluations**, examinations of one selection of rows in one basis ([Section 3.6](#36-work-estimates-and-exhaustive-enumeration)).

### 2.6 Excluding stabilizers

For a quantum code, the lightest word in `V` may lie in `S`, and must not lower `U`.
Section 2.1 of [arXiv:2408.10743](https://arxiv.org/abs/2408.10743v2) therefore checks whether a candidate lighter than `U` lies in `S`, which qLDPC tests with quotient labels ([Section 3.4](#34-quotient-labels-for-the-stabilizer-test)).
The certificate of [Section 2.5](#25-the-stopping-certificate) is unchanged, since it holds for all unexamined words.

### 2.7 Pauli weight as Hamming weight

An X-type or Z-type Pauli operator on `n` qubits is a binary vector of length `n`, whose Hamming weight is its physical weight.
A general Pauli operator is a row `[x | z]` of length `2n`, whose physical weight `wt_P` counts the qubits on which `x` or `z` is nonzero.
Section 2.4 of [arXiv:2408.10743](https://arxiv.org/abs/2408.10743v2) maps it, by the `Saved_isometry` construction that follows earlier work of White and Grassl, to

$$
T(x,z) = (x \mid z \mid x+z),
$$

which has Hamming weight two on each qubit with a nontrivial Pauli operator and zero on the others, so

$$
\operatorname{wt}_{H}(T(v)) = 2\operatorname{wt}_{P}(v).
$$

`T` is linear and injective, so BZ searches `T(S) ⊆ T(V)` on `3n` coordinates, and the distance is half the Hamming weight that it finds.

## 3. Extensions and optimizations

qLDPC goes beyond [Section 2](#2-the-brouwer-zimmermann-algorithm) as follows; each section says which parts are published.

- Extensions that strengthen the bounds of a search: overlapping search bases ([Section 3.1](#31-overlapping-search-bases)), rounding the certificate up to a proven weight multiple or to the parity of a coset ([Section 3.2](#32-rounding-the-certificate-to-a-proven-weight-multiple)), and witnesses shared between CSS sectors ([Section 3.3](#33-sharing-witnesses-between-css-sectors)).
- Optimizations: quotient labels for the stabilizer test ([Section 3.4](#34-quotient-labels-for-the-stabilizer-test)), packed enumeration without pivot columns ([Section 3.5](#35-packed-enumeration-with-pivot-column-deletion)), skipping words that precomputed tables prove too heavy ([Section 3.9](#39-skipping-completions-by-completion-tables)), and minimum weights of cosets in one search ([Section 3.10](#310-minimum-weights-of-cosets)).
- Both: certifying with the images of one information set under automorphisms, while scanning only that set ([Section 3.7](#37-images-of-an-information-set-under-automorphisms)).

[Section 3.6](#36-work-estimates-and-exhaustive-enumeration) defines the work estimate with which searches choose their search bases, and [Section 3.8](#38-worked-example) works an example.
[Section 3.11](#311-counting-logical-operators) adapts the search to counts of logical operators, and explains the MacWilliams method of counting.

### 3.1 Overlapping search bases

An information set has `r` columns, so a disjoint family has at most `floor(N / r)` of them, and the other columns form at most one rank-deficient block, which can add little or nothing to `L(t) = Σ_i b_i(t)` ([Section 3.8](#38-worked-example) gives an example).
An **overlapping family of search bases** lets blocks share columns, so it can hold more information sets, but a column in several blocks then counts in several terms `b_i(t)`.
Distance searches and counts build overlapping families in `get_overlapping_family` and `get_partitioned_family` ([Section 4.4](#44-heuristics-and-cost-models)).
In [_tuning.py](_tuning.py), `overlapping_minimum_work` at `MAX_VALUE` turns off the first, and `partition_max_coverage` at one restricts the second to disjoint families.

#### The certificate

Let the **maximum coverage** `C` of a family be the largest number of its blocks that contain one column.
If every search basis of the family has been searched through weight `t`, an unexamined eligible word `v` has `wt(v|_{B_i}) ≥ b_i(t)` for every `i`, and each of its ones lies in at most `C` blocks, so

$$
\sum_i b_i(t) \le \sum_i \operatorname{wt}(v|_{B_i}) \le C\operatorname{wt}(v),
\qquad\text{hence}\qquad
L(t) = \left\lceil \frac{\sum_i b_i(t)}{C} \right\rceil.
$$

For `m` information sets, `L(t) = ceil(m(t+1)/C)`, which beats a disjoint family of `k` information sets when `m/C > k`.
`C = 1` recovers [Section 2.5](#25-the-stopping-certificate), and within a layer the same argument divides the sum that defines `L(t, j)` by `C`.
For example, the code generated by `101` and `011` has distance two, and a disjoint family holds one information set.
Its three pairs of columns are information sets with `C = 2`, so after layer one `L(1) = ceil(3·2/2) = 3`; this bound on unexamined words exceeds the distance, and the search stops with `U = 2`.

Grassl's chapter (page 291 and Algorithm 2.5, see [References](#references)) credits Zimmermann with a certificate for an ordered list of information sets, `sum_i max(0, t+1 - o_i)`, where `o_i` counts the columns of block `i` that lie in an earlier block (`r - o_i` is its relative rank).
Proposition 5 of Lisoněk and Trummer gives the coverage-divided certificate for blocks of equal bounds `t+1`, and its proof covers the different bounds of rank-deficient blocks and of steps within a layer.
[arXiv:1603.06757](https://arxiv.org/abs/1603.06757v2) mentions the overlapping case in its Section 2, citing Grassl, without a formula.
Neither Zimmermann's certificate nor the coverage-divided one dominates the other, but the linear program below dominates both.

#### A certificate from linear programming

The coverage-divided certificate is weak when coverage is uneven.
An unexamined eligible word `v` satisfies `wt(v|_{B_i}) ≥ b_i` for each `i`, with `b_i` the bound of the completed layer or step, so relaxing its coordinates to `0 ≤ x_c ≤ 1` bounds `wt(v)` by the optimum of

$$
\text{minimize} \sum_c x_c \quad \text{subject to} \quad \sum_{c \in B_i} x_c \ge b_i \text{ for each } i, \quad 0 \le x_c \le 1.
$$

By weak duality, every `y ≥ 0` gives the bound

$$
\operatorname{wt}(v) \ge \sum_i b_i y_i - \sum_c \max\Big(0, \sum_{i : c \in B_i} y_i - 1\Big).
$$

The coverage-divided certificate is the value of `y_i = 1/C`, and Zimmermann's is at most the value of `y_i = 1` on the blocks that contribute to it, so the optimum is at least both.
It can be larger: for the seven columns `e1, e2, e3, e1+e2+e3, e1, e2, e3`, with `b_i = 2` on the four 3-subsets of the first four columns and on the last three, both certificates are four in every order of the blocks, but the linear program gives `ceil(14/3) = 5`.
Its optimum is `14/3`, the value both of the point `x_c = 2/3` and of the dual weights `y_i = 1/3` on the four subsets and `y_i = 1` on the block of the last three columns.

qLDPC solves the program with `scipy.optimize.linprog`, evaluates the dual bound at the solver's `y` in exact rational arithmetic, and takes its ceiling, so rounding in the solver can only weaken the certificate.
For disjoint blocks the program separates into the certificate of [Section 2.5](#25-the-stopping-certificate), so only overlapping families use it, and only in costly searches ([Section 4.4](#44-heuristics-and-cost-models)).
A search solves one at the end of each layer, and within a layer only when the coverage-divided certificate falls short of `U` and the value of a feasible point does not show the optimum to be below `U`.

#### Families for coset searches

The coset searches of [Section 4.1](#41-module-map-and-layering) search a coset `u + V` to answer **queries**: whether the coset has a word of weight at most a given maximum weight.
For each search basis `G_i`, fix an **offset**, a word of the coset that is zero on the pivot columns of `B_i`.
Every word of the coset is the offset plus a unique sum of rows of `G_i`, so once `G_i` has completed layer `t`, an unexamined word has at least `b_i(t)` ones on `B_i`, as a word of `V` does.
So every certificate of this section holds for cosets, and any family of search bases of `V` serves every coset of `V`.
A coset search can thus use an overlapping family, or a disjoint family that matroid partitioning finds ([Section 4.4](#44-heuristics-and-cost-models)).

### 3.2 Rounding the certificate to a proven weight multiple

If every eligible weight is a multiple of an integer `D`, an unexamined eligible word weighs at least `D * ceil(L(t)/D)`, so the certificate can be rounded up to that value.
Grassl's chapter notes this rounding on page 299, and Magma uses it for even binary and quaternary codes, doubly even binary codes, and ternary codes whose weights are all multiples of three.
In qLDPC, the distance functions pass `D = 2` for general Pauli problems, whose map `T` of [Section 2.7](#27-pauli-weight-as-hamming-weight) doubles every weight, and qLDPC infers `D = 2` when every row of the basis of `V` has even weight, since sums of even words are even.

qLDPC checks only parity.
Since `wt(u + v) = wt(u) + wt(v) - 2|u ∩ v|`, every word of `V` has a weight divisible by four exactly when every row does and every two rows overlap evenly: the rows `11110` and `01111` weigh four, but their sum `10001` weighs two.
That stronger test mainly helps doubly even classical codes, since a CSS sector passes it only if every logical operator of the sector lies, as a binary vector, in the span of the stabilizers of the other type.

By the same identity, a divisor of the weights of `V` need not divide those of a coset `u + V`: every word of `V = {0000, 1111}` has a weight divisible by four, but `1000 + V` has the weights one and three.
Parity does carry over: if `V` is even, every word of `u + V` has the parity of `wt(u)`.
So a coset search of an even code rounds each bound up to that parity, which it reads from the word of the coset's syndrome ([Section 4.1](#41-module-map-and-layering)), and uses no other rounding:

- A search of the X or Z parts of errors, which searches the cosets of a classical code, rounds when that code is even.
- A search of Pauli errors by their CSS weight `max(|x|, |z|)` rounds when the words `[x | z]` of its code are even, and bounds the CSS weight by half its bound on the Hamming weight of `[x | z]`, rounded up, after rounding to parity.
  Until it has visited every word, each step of [Section 2.5](#25-the-stopping-certificate) raises its Hamming bound by at most one, so the halved bound of the step at which a query stops early is exactly its maximum weight plus one, with or without parity.
  Rounding can still stop a query of an odd coset sooner: once its Hamming bound reaches twice the maximum weight, rather than exceeds it.
- A search of Pauli errors by their physical weight does not round, since every word `T(x, z)` is even, and rounding up to an even integer does not change the half, rounded up.

### 3.3 Sharing witnesses between CSS sectors

The distance of a CSS code is `min(d_x, d_z)`, the lesser of the distances of its X and Z sectors.
qLDPC runs one BZ search for each sector, advances them in round-robin order, one layer at a time, and lets a witness of either sector, a nontrivial logical operator, lower the `U` of both.
Each search keeps its own certificate, search bases, coverage, and weight divisor, since its certificate bounds only the unexamined words of its sector.
For example, if the X sector finds a witness of weight seven and the Z sector has only even words, the Z search stops once its own certificate reaches `U = 7`, having shown only `d_z ≥ 8`.
A joint search therefore determines `min(d_x, d_z)`, but not always both sector distances.

### 3.4 Quotient labels for the stabilizer test

[Section 2.6](#26-excluding-stabilizers) must decide whether a candidate lies in `S`, which quotient labels do without reducing the candidate against a basis of `S`.
Let `s = dim(S)` and `q = dim(V/S)`, and take an **adapted basis** `G` of `V`, whose first `s` rows span `S` and whose other `q` rows span `V` modulo `S`.
Give its rows the **labels** `M`, an `r × q` matrix whose first `s` rows are zero and whose last `q` rows form the identity:

```text
G = [ s rows spanning S          ]      M = [ zero, s x q     ]
    [ q rows spanning V modulo S ]          [ identity, q x q ]
```

Then `aG ∈ S` exactly when `aM = 0`, so a candidate is eligible exactly when the XOR of the labels of its selected rows is nonzero; a classical search, with `S = {0}`, needs no labels.
Every row operation `G' = EG` that builds a search basis must also give `M' = EM`, or the test would accept stabilizers or reject logical operators.

### 3.5 Packed enumeration with pivot-column deletion

Packing stores a word of `N` coordinates in `ceil(N/64)` 64-bit machine words, so that an XOR costs one operation per machine word and a weight is a population count ([arXiv:1603.06757](https://arxiv.org/abs/1603.06757v2)).
A search enumerates the **supports** of a layer, its sets of selected row indices, in lexicographic order.
Consecutive supports share their first rows, whose XOR is computed once, so extending a support costs one XOR of a row, following the saved additions of [arXiv:1603.06757](https://arxiv.org/abs/1603.06757v2).

Pivot-column deletion is a qLDPC choice.
In `G_i`, each pivot row has a one in its own pivot column of `B_i`, and the other rows are zero on `B_i`, so the word of a support of `w` rows has a **pivot one** for each selected row below `r_i` (counting from 0): `w` for an information set.
The support alone determines that count, so qLDPC deletes the pivot columns before packing, and weighs a word as its pivot ones plus the population count of its **tail**, its restriction to the remaining columns.
For example, with `r = 2` and rows `1 0 | 1 1` and `0 1 | 1 0`, the support `{0, 1}` has the tail `0 1` and the weight `2 + 1 = 3` of the full word `1 1 0 1`.

Deletion shrinks a stored row from `ceil(N_stored/64)` to `ceil((N_stored - r_i)/64)` machine words, where `N_stored`, unlike `N` ([Section 2.2](#22-the-idea-upper-bounds-and-lower-certificates)), counts every column of the stored basis, even all-zero ones.
With `N_stored ≤ 64`, as in [Section 3.8](#38-worked-example), a row takes at most one machine word either way.
For wide rows the ratio is about `N_stored / (N_stored - r_i)`: about two for an information set of an X sector whose Z checks `H_z` have rank near `N_stored/2`, since `r = N_stored - rank(H_z)`.
Rounding can move it either way: deleting 65 pivots of 129 columns leaves one machine word of three.
These are counts of packed words, not speedups, since label XORs, comparisons with `U`, and loop overhead do not shrink.

Coset searches by Hamming weight also weigh words by their tails, in the queries that they predict to examine many words ([Section 4.4](#44-heuristics-and-cost-models)).
The offset of a search basis is zero on its pivot columns ([Section 3.1](#31-overlapping-search-bases)), so the count of pivot ones still holds, and the tail keeps every remaining column, even one on which no row has a one, since an offset can have ones there.

### 3.6 Work estimates and exhaustive enumeration

To choose between families of search bases, and to decide whether costly stages pay ([Section 4.4](#44-heuristics-and-cost-models)), a search estimates its work in candidate/set evaluations.
If `(t*, j*)` is the first step after which the certificate of a family of `m` search bases, rounded as in [Section 3.2](#32-rounding-the-certificate-to-a-proven-weight-multiple), reaches `U`, the estimate is

$$
W = m\sum_{w=1}^{t_* - 1} \binom{r}{w} + j_* \binom{r}{t_*}.
$$

It is pessimistic, since it assumes that no lighter witness turns up, and it is not a runtime guarantee, since runtime also depends on building the bases and on the width of packed words.
The estimate of a coset query at maximum weight `ω` sums from layer zero, the offset, to the first step whose bound exceeds `ω`, and, like the query, omits the rank-deficient bases that add nothing to that bound.
If the coset has no word within `ω`, the query examines or skips exactly that many words besides the word of its syndrome ([Section 4.1](#41-module-map-and-layering)), and otherwise at most that many.

Exhaustive enumeration, which `method="brute_force"` runs instead of BZ, examines all `(2^q - 1) * 2^(r-q)` eligible words, for `q = dim(V/S)`, in vectorized, packed batches updated by Gray-code steps.
A batch holds at most `2^enumeration_batch_exponent` packed 64-bit integers, or one word if a word needs more.
For classical codes of low dimension whose distance is a large fraction of their length, it can cost less than the setup and work of BZ.
Algorithm 2.11 of Grassl's chapter (page 300) makes this choice itself, by comparing the work estimate of BZ with the number of codewords of the code or of its dual.

### 3.7 Images of an information set under automorphisms

Many codes come with a group of qubit permutations that preserve them, such as the translations of both halves of a multivariate bicycle code.
Such a group lets a search certify with many information sets while scanning only one, which removes most of the cost of the layers that dominate a search.

#### The certificate of an automorphism family

An **automorphism** of a problem is a permutation `g` of the coordinates that maps `V` and `S` onto themselves, and so maps eligible words to eligible words of the same weight, and an information set `I` to an information set `g(I)`.
For a group `Γ` of automorphisms, the images `g(I)` for `g ∈ Γ` form an **automorphism family**.

Suppose that a search has examined every word `v` with `wt(v|_I) ≤ t`, as searching `G_I` through weight `t` does.
If an eligible word `c` has at most `t` ones on some `g(I)`, then `g⁻¹(c)`, of the same weight, has at most `t` ones on `I`, and has been examined.
So a word whose orbit under `Γ` is wholly unexamined has at least `t+1` ones on every image, and the certificate of the family bounds its weight, while every other eligible word weighs as much as an examined word, at least `U`.
Once that certificate reaches `U`, the search therefore stops with the exact distance.
It scans only `I`, so the certificate advances once per layer, and each layer costs `binom(r, t)` evaluations however many images there are: `W = sum_{w=1}^{t*} binom(r, w)`.

A column in an orbit `O` of `Γ` lies in `|I ∩ O| |Γ| / |O|` images, so `C = |Γ| max_O |I ∩ O| / |O|`, and the certificate of [Section 3.1](#31-overlapping-search-bases), before the rounding of [Section 3.2](#32-rounding-the-certificate-to-a-proven-weight-multiple), is

$$
L(t) = \left\lceil (t+1) \min_O \frac{|O|}{|I \cap O|} \right\rceil
$$

over the orbits that meet `I`.
It is strongest when every orbit meets `I` in the same share, `r/N`, as a single orbit does, or the two halves of a multivariate bicycle code with `r/2` columns of `I` in each.
Then `L(t) = ceil(N(t+1)/r)`, which no coverage-divided certificate of a family of information sets beats, since `m` sets of maximum coverage `C` have `m r ≤ C N`.

#### Choosing the scanned set

qLDPC row reduces on the nonzero columns in natural order, then exchanges pivots between orbits while an exchange can move a pivot from an orbit of the largest share to one that stays emptier; pivoting a row on another column with a one in that row gives another information set.
For two orbits, the exchanges reach the least largest share of any information set: they stop once the emptier orbit's own pivot columns span all of its columns, or one more pivot would make it at least as full.
In the first case, no information set has more pivots in the emptier orbit, so none has fewer in the fuller one; in the second, any set with fewer there gives the emptier orbit at least the scanned set's largest share.
With more orbits, the exchanges are a heuristic.

#### When a search uses an automorphism family

Finding automorphisms, verifying them, and building their family can cost more than a whole easy search, and before a light witness turns up, the estimate of [Section 3.6](#36-work-estimates-and-exhaustive-enumeration) cannot tell whether a search is easy.
So a search starts with the families of Sections [2.4](#24-search-bases-and-rank-deficient-blocks) and [3.1](#31-overlapping-search-bases), and asks for automorphisms at most once:

- before matroid partitioning ([Section 4.4](#44-heuristics-and-cost-models)), if partitioning may try some coverage, the search's estimated work passes the gate `partition_work_ratio`, and some partitioned family could do less work than the search's current family;
- otherwise, before the first layer that would cost at least `automorphism_minimum_layer_work` evaluations.

A search asks before partitioning because switching would discard the partitioned family.
It then skips partitioning if the automorphism family would do no more work than any partitioned family could, and considers switching at the end of setup.
It switches if the work estimate of the automorphism family is less than the work that its current family has left from its next layer, and then scans the new set from layer two.
Both estimates depend on `U`, so a search that declines reconsiders before a later such layer once `U` has dropped; at the same `U`, the work left in its family only falls, so it would decline again.
The switch keeps both certificates: an eligible word lighter than `U` is unexamined by both families, so both bounds hold for it, and the search takes the larger, while the witnesses of the first family still count in `U`.

#### Where automorphisms come from

With the optional `pybliss` package, bliss searches the parity checks of a binary quantum code for automorphisms.
The class of the code can also supply other sets of checks that span the same space, through `QuditCode._get_automorphism_search_matrices`, so that bliss can find automorphisms that do not map parity checks to parity checks.
`QTCode`, for example, supplies its canonical local checks where they are few enough.
Bliss finds the automorphisms of the Tanner graph of each set, whose edges are colored by the Pauli operators of the checks.
These Tanner-graph automorphisms permute the checks and leave their Paulis unchanged, so they preserve the stabilizers, gauge operators, and logical operators, and for a CSS code those of each sector.
They form a subgroup of the automorphisms of the code, which suffices, since any group of automorphisms gives a sound certificate.
Code classes can also supply generators of automorphisms that their construction provides, through `QuditCode._get_qudit_automorphisms`, such as the translations of the group of an `MBCode`.
Searches use these generators together with those that bliss finds.
Bliss finds only automorphisms of the Tanner graphs of the sets of checks that it searches, so it can miss the generators of a construction.
A wrong generator would make the certificate unsound, so a search first checks that each maps `V` and `S` onto themselves, and raises an error if one does not.
When a BZ distance search uses automorphisms, it certifies a bound from the images of one information set.
The BZ count scan of [Section 3.11](#311-counting-logical-operators) does not use automorphisms: it counts each word individually and does not track orbit sizes.
For a general Pauli problem, a permutation of qubits moves the three thirds of the coordinates of `T` ([Section 2.7](#27-pauli-weight-as-hamming-weight)) alike, and `T` commutes with it, so verifying the lifted permutation checks the original one.

### 3.8 Worked example

One CSS sector of the MB64 code of the test suite ([_brouwer_zimmermann/distance_test.py](_brouwer_zimmermann/distance_test.py)) has 64 coordinates and dimension `r = 41`, a lightest eligible row of weight eight, so initially `U = 8`, and only even words, so `D = 2`.

A disjoint family holds `floor(64/41) = 1` information set, and the rank-deficient block of the other columns adds nothing to `L(t)` for `t ≤ 17`.
So `L(t) = t + 1`, and the search stops after layer six, when `L(6) = 7` rounds up to eight.
An overlapping family of `m = 3` information sets with `C = 2` reaches `L(4) = ceil(3·5/2) = 8` after layer four, even before rounding.
Within a layer ([Section 2.5](#25-the-stopping-certificate)), qLDPC stops sooner: after the first search basis of layer four, the bound `ceil((5+4+4)/2) = 7` rounds up to eight.

| Family | Last layer | Candidate/set evaluations in layers two and above |
| --- | ---: | ---: |
| Disjoint, one search basis | 6 | $\sum_{w=2}^{6}\binom{41}{w} = 5{,}358{,}536$ |
| Overlapping, three search bases, `C = 2` | 4 | $3\sum_{w=2}^{3}\binom{41}{w} + \binom{41}{4} = 135{,}710$ |

The counts leave out layer one, whose rows give the initial `U`, and the cost of building and packing the search bases.

### 3.9 Skipping completions by completion tables

The layers that a search scans last dominate its cost, and most of their words weigh far more than `U`.
For example, certifying distance eighteen for the X sector of the [[288, 12, 18]] code of [arXiv:2308.07915](https://arxiv.org/abs/2308.07915) with an automorphism family scans layer eight of a basis of `r = 150` rows, which has `binom(150, 8)`, about `5 × 10^12`, supports.
qLDPC skips whole groups of supports when a lower bound read from **completion tables** proves that none of their words is lighter than `U`.
A distance search can build tables for each search basis whose block is an information set, such as the one set of an automorphism family, if they pay (see "When a search uses the tables" below).
Coset searches by Hamming weight use them too, in their costly queries, even in search bases that are not information sets.

#### The bound

Write a search basis whose block is an information set as `G = [I_r | P]`, with the pivot columns first, so that the word of a support of `w` rows weighs `w` plus the weight of its tail, the sum of its rows of `P` ([Section 3.5](#35-packed-enumeration-with-pivot-column-deletion)).
Fix a **prefix**, the first `w - 2` rows of a support, whose rows of `P` sum to `p`, and a row `s`.
The supports with that prefix whose next row is at least `s` are completed by the pairs `s ≤ i < j < r`, and weigh `w + wt(p + P_i + P_j)`.
For disjoint blocks `J_b` of the tail columns, every pair satisfies

$$
\operatorname{wt}(p + P_i + P_j) \ge B = \sum_b m_b,
\qquad
m_b = \min_{s \le i < j < r} \operatorname{wt}\big((p + P_i + P_j)|_{J_b}\big),
$$

so if `w + B ≥ U`, the search skips all these completions.
`B` ignores eligibility, and lets different blocks take their minima at different pairs, which only weakens it.
It stays sound only if it minimizes over every pair, its blocks are disjoint, and it takes the rows in the search's order, which decides the pairs from row `s` on.
The pivot ones must number exactly `w`, as only in an information set ([Section 2.4](#24-search-bases-and-rank-deficient-blocks)), or be bounded as in "Coset searches" below.

#### Tables and lookups

For each block and each `s`, a table stores the distance from every value of `p` on the block to the nearest sum of a pair from row `s` on.
The table for `s` starts from that for `s + 1`: the new pairs of row `s` lower the distance at their sums to zero, and a breadth-first search from those sums lowers the distances around them.
The breadth-first search halts at each value whose distance it does not lower, since the distances beyond are already at most that distance plus their steps from it.
With blocks of `β` columns, the tables take `(r - 1) Σ_b 2^|J_b|` bytes, and building them costs about `2^β β^2` steps for each block, plus a step for each pair of rows on each block, and writing each table.
A lookup reads one entry per block.
The scan looks up the bound each time it tries a new first row `s` for the pair that completes a prefix, and if `w + B ≥ U`, skips every completion of the prefix from row `s` on.
A bound for a larger `s` is stronger, so a support is skipped exactly if the bound of its own `s` reaches `U - w`.

#### Certificates and contracts

A skipped word weighs at least the `U` of the moment it was skipped, and hence any later `U`, so an eligible word lighter than `U` has been neither skipped nor examined, and the arguments of [Section 2.5](#25-the-stopping-certificate) hold for it.
The certificate `L` bounds only such words, and can exceed the weights of skipped words, as it can those of words examined through their images ([Section 3.7](#37-images-of-an-information-set-under-automorphisms)), so the distance is at least `min(L, U)`, not `L`.
Since no skipped word is lighter than `U` at any later time, a scan with tables finds the same witnesses in the same order as one without, an early exit ([Section 4.5](#45-behavior-spanning-files)) returns the same weight, and the search stops when it would without skipping.
These arguments hold for each search basis, with its own tables, so a search can skip in each basis of a layer that it scans one at a time, and stop after any basis whose step `(t, j)` certifies `U` by the certificate of its family, which may need a linear program ([Section 3.1](#31-overlapping-search-bases)).

Words that weigh exactly `U` can be skipped, since they cannot lower `U`.
But a search whose answer depends on the words up to a weight may never skip a word of that weight, so it replaces `U` with the least bound that proves a word heavier than that weight.
A count of the words of weight `t` ([Section 5](#5-correctness-requirements)) skips only completions that all weigh more than `t`, as if its best weight were `t + 1`.
So none of its skipped words needs counting, and the certificate of its last step, which exceeds `t`, rules out any word of weight `t` that it neither visits nor skips.
A coset query of maximum weight `ω` skips with the limit `h(ω)` of "Coset searches" below.

#### Coset searches

The costly queries of a coset search by Hamming weight ([Section 4.4](#44-heuristics-and-cost-models)) skip completions by tables in layers of two or more rows.
The scan seeds every prefix with the tail `o` of the offset ([Section 3.1](#31-overlapping-search-bases)), so the tables bound `wt(o + p + P_i + P_j)`; they cover the tail columns up to the last on which a row has a one, and ones of `o` beyond it only weaken the bound.
The search basis need not be an information set, and its pivot rows come first, so a completion's pivot ones are those of its prefix's rows below the rank of the basis, plus at least `max(0, 2 - d)` from its last two rows, where `d` is the basis's rank deficiency.
The bound adds this count to `B` in place of `w`, which it equals for an information set.

A query of maximum weight `ω` skips the completions whose bound reaches `h(ω)`, the least bound that proves a word of the coset heavier than `ω`.
With the **weight factor** `η` of [Section 4.1](#41-module-map-and-layering), two in a search of Pauli errors and one in a search of X or Z parts, `h(ω)` is `η ω + 1`, or `η ω` if the search rounds to the parity of the coset and `η ω` has the other parity.
No skipped word lies within `ω`, and the query visits the other words in the order of a scan without tables, so it finds the same first word within `ω`, if there is one.
Otherwise, it returns the lightest word that it weighed, possibly heavier than without tables.
If it skipped any word, it also returns the lower bound `R`: `h(ω)`, rounded up to the coset's parity where the search rounds, then divided by `η` and rounded up ([Section 3.2](#32-rounding-the-certificate-to-a-proven-weight-multiple)).
Every word that the query skipped, weighed, or did not visit lies beyond `ω`, so it weighs at least `R`, the least weight beyond `ω` that a word of the coset can have.
A query that skipped nothing returns the bound of a scan without tables, which is never below `R`.

Until a query finds a word within `ω`, its lightest word has a Hamming weight of at least `h(ω)`, so skipping at `h(ω)` skips at least as much as skipping at that weight, as a distance search does.
It therefore weighs no more words, at the price of the heavier word and smaller bound above.
A search for the minimum weight of a coset skips at `h(ω)` too until it finds a word within `ω`, and then at a limit that falls as it finds lighter words ([Section 3.10](#310-minimum-weights-of-cosets)).

#### When a search uses the tables

A distance search scans each search basis of a layer by itself with `scan_set` of [_brouwer_zimmermann/affine_search.py](_brouwer_zimmermann/affine_search.py), with tables only for an information set; in a basis of deficient rank, it counts pivot ones only for rows below its rank, as coset searches do.
A count scans with its own kernels in [_counting/scan.py](_counting/scan.py), and chooses tables as a distance search does (below) for each of its steps of two or more rows of an information set, with `_StepTables`.

Tables cost memory, a build, and lookups, and save only the supports that they skip, so they pay on large layers where they skip many supports, but not on small layers or in easy searches.
The share that they skip depends on the code, the scanned set, and the order of its rows, so tables can pay on one layer and not on a layer of the same size of another code.
They save at most a constant factor in a layer, since the scan still looks up a bound for each prefix, and prefixes grow like supports for fixed `w/r`; nor do they reduce the number of layers.
So, just before scanning a search basis in a layer, a search:

1. considers tables only for a layer of at least `completion_bound_minimum_layer_work` supports ([Section 4.4](#44-heuristics-and-cost-models));
2. bounds at most `completion_bound_samples` random supports directly from the rows, and takes a conservative lower bound on the share that tables would skip;
3. predicts the cost of the layer from this share, the lookups, and any build, with rough coefficients of `SearchTuning`: a lookup costs a few scans of a support for each block, and more for larger tables, which miss the caches more often;
4. uses tables if the predicted cost is at most `completion_bound_max_cost_percent` percent of the cost without them.

Tables use blocks of `completion_bound_block_bits` columns, and a basis whose tables would exceed `completion_bound_max_table_bytes` gets none.
A distance search keeps the tables of one basis at a time, for its later layers, and drops them before another basis chooses tables or when its family changes, so a search whose layers scan several bases rebuilds their tables in each layer, which the prediction counts.
A joint search can hold twice `completion_bound_max_table_bytes`, half for each sector, until it finishes.
Within `completion_bound_max_table_bytes`, a count keeps the tables of the sets that it has used, besides those that it is building, and rebuilds them in each layer when they do not all fit.

### 3.10 Minimum weights of cosets

The residual-weight estimator of `qldpc.experimental` needs the minimum weight of each coset that it meets, up to a maximum weight `ω`, or `ω + 1` for a coset without a word within `ω`.
Queries at increasing maximum weights would each repeat the scan of the one before, so the coset searches find this weight with one search.
The search keeps `U`, the lesser of `ω + 1` and the weight of the lightest word found so far, starting from the word of its syndrome ([Section 4.1](#41-module-map-and-layering)).
It stops after the first step whose bound reaches `U`, or within a step once `U` is at most the bound of the step before, since every unvisited word then weighs at least `U`; `U` is then the minimum weight, if it is at most `ω`.
It finds that step as a query does, moves it earlier whenever `U` falls, and skips the search bases that add nothing to its bound, which add nothing to an earlier bound either.
It uses the family and the completion tables of a query at `min(ω, n)`, for `n` qubits ([Section 4.4](#44-heuristics-and-cost-models)), and since it stops no later than that query, those tables cover every layer that it scans.

In a search by Hamming weight, the scans keep only the words lighter than an **incumbent**: `h(ω)` of [Section 3.9](#39-skipping-completions-by-completion-tables) until the search finds a word within `ω`, and then `η U`, the Hamming weight of the lightest word.
In a search of Pauli errors by physical weight, whose words `T(x, z)` are all even, and of X or Z parts of an even code, every word of the coset has the parity of `η U`, so scans then lower the incumbent to `η U - 1`, keeping the same words and letting tables skip more.
Completion tables skip the completions whose bound reaches the incumbent, or, once a scan finds a lighter word, the Hamming weight of that word.
So the search skips no word lighter than its answer, and in a coset without a word within `ω`, it weighs and skips exactly the words of a query at `ω`.
A search of Pauli errors by CSS weight weighs words by that weight, and keeps those lighter than `U`.

For the CSS weight on a CSS state, `CosetLeaderSearch` searches the X part, then, if its weight `w_x` is at most `ω`, the Z part with the floor `w_x`, where any word within `w_x` ends the search ([README.md](README.md#the-coset-leader-search)).

### 3.11 Counting logical operators

This section describes qLDPC's implementation, apart from the MacWilliams identity, which it cites.
`CSSLogicalCounter` of [logical_counts.py](logical_counts.py) counts the nontrivial logical operators of a given weight `t` in a binary CSS code, where `t` is by default the distance `d`.
Its default method, BZ, scans each sector in layers of a family of search bases, but stops and skips only as Sections [3.9](#39-skipping-completions-by-completion-tables) and [5](#5-correctness-requirements) allow, so that it visits every eligible word of weight `t`.
A word lies in the layer of its coefficient weight in each search basis, so one step visits it first, and only that step counts it.
The X and Z sectors of a self-dual code, whose X and Z checks span the same space, are equal.
So a counter scans only one of them if it knows the code to be self-dual: if it was built from such checks, or deformed from such a counter by a gate that acts alike on X and Z.

#### Full Pauli counts

A nontrivial logical Pauli operator `(x, z)` of weight `d` and support `Q` takes one of two disjoint forms, since `x`, which commutes with the Z stabilizers, is a nontrivial X logical or an X gauge operator:

- `x` is a nontrivial X logical, which weighs at least `d` and so has support `Q`, and `z` is one of the `2^(d - rank(S_X[:, Q]))` words supported in `Q` that commute with the X stabilizers `S_X`.
- `x` is an X gauge operator, so `z` is a nontrivial Z logical, which has support `Q` for the same reason, and `x` is one of the `2^(rank(G_X) - rank(G_X[:, ~Q]))` words supported in `Q` in the span of the X gauge generators `G_X`.

So the scans of the X and Z sectors return histograms of these exponents over their words of weight `d`.
In a self-dual code, no nonzero gauge operator lies within the support `Q` of a nontrivial X logical `x` of weight `d`, since adding the gauge operator to `x` would give a lighter nontrivial logical.
So a nonzero word within `Q` that commutes with `S_X` is not a gauge operator but a nontrivial logical of weight at most `d`, which must be `x` itself.
Hence in the first form `z` is `0` or `x`, and in the second `x` is `0`: `Q` carries exactly three operators, of types X, Y, and Z, and the full count is three times the X count.
Above `d`, an operator need not have the support of a nontrivial X or Z logical, so a full count raises an error if its scans find a nontrivial X or Z logical lighter than `t`.

#### Deformed counters

A local CSS-preserving Clifford gate maps the X part of an operator on its target qubits by an invertible binary matrix `matrix_x`, and the Z part by `(matrix_x^-1)^T`, which preserves commutation.
So the gate maps the gauge operators, stabilizers, and sectors of a code onto those of the deformed code, and a basis of a sector, row by row, to a basis of the new sector with the same quotient labels ([Section 3.4](#34-quotient-labels-for-the-stabilizer-test)).
For each basis family that a counter keeps, the deformed counter reduces each mapped search basis again on the pivot columns of its block, with its labels.
If every search basis keeps its rank there, the family keeps its blocks, ranks, and coverage, and so its certificate; otherwise the counter builds the family again.
A gate can change the parity of words, so the counter infers the weight divisor of [Section 3.2](#32-rounding-the-certificate-to-a-proven-weight-multiple) again.
It can also change the distance and the weights of check words, so the counter searches for the distance again, and the MacWilliams method recomputes the weight distributions of the checks.

#### The MacWilliams method

For a stabilizer code on `n` qubits with checks `H_X` and `H_Z`, `method="macwilliams"` of [_counting/macwilliams.py](_counting/macwilliams.py) counts from the weight distributions `A_X` of `row(H_X)` and `A_Z` of `row(H_Z)`.
The nontrivial X logicals are the words of `ker(H_Z)` outside `row(H_X)`.
Since `ker(H_Z)` is the dual of `row(H_Z)`, the MacWilliams identity ([References](#references)) gives their number of weight `w`:

$$
N_X(w) = 2^{-\operatorname{rank}(H_Z)} \sum_j K_w(j)\,A_Z(j) - A_X(w),
$$

where the [binary Krawtchouk polynomial](https://en.wikipedia.org/wiki/Kravchuk_polynomials) `K_w(j)` is the sum of `(-1)^(x·y)` over the words `x` of weight `w`, for any `y` of weight `j`.
The Z counts exchange X and Z.
A Gray-code walk over the `2^r` words of a row space of rank `r` gives its distribution `A`, so the cost is exponential in the ranks of the checks rather than in the number of logical qubits.
Since `|K_w(j)| ≤ binom(n, w)` and `A` sums to `2^r`, no Krawtchouk value, entry of `A`, product `K_w(j) A(j)`, partial sum of these products, or divisor `2^r` exceeds `2^r binom(n, w)` in magnitude.
So the sum of these products and its division by `2^r` are exact in signed 64-bit integers at every weight `w` with `2^r binom(n, w) < 2^63`.
qLDPC uses 64-bit integers below the first weight that fails this bound, and Python integers from there on.

## 4. Guide to the source code

This section maps the concepts of Sections [2](#2-the-brouwer-zimmermann-algorithm) and [3](#3-extensions-and-optimizations) to modules and identifiers.
The thresholds, limits, and coefficients of the heuristics are fields of `SearchTuning` in [_tuning.py](_tuning.py): [Section 4.4](#44-heuristics-and-cost-models) gives the reasons for those of BZ searches.

### 4.1 Module map and layering

[README.md](README.md) maps the modules of this directory and the levels in which they import each other; this section says how the searches use them.

The distance functions of [core.py](core.py) run the BZ search of [_brouwer_zimmermann/distance.py](_brouwer_zimmermann/distance.py), the exhaustive enumeration of [_brute_force.py](_brute_force.py), or the connected-cluster search of [_connected_cluster/search.py](_connected_cluster/search.py).
For BZ, they build an adapted basis ([Section 3.4](#34-quotient-labels-for-the-stabilizer-test)) for each sector, and pass the sectors to `get_distance_brouwer_zimmermann`, both sectors of a CSS code at once in `get_distance_css`, or only the X sector if `_search_css` finds a qubit permutation that maps each sector onto the other.
`_search_css` finds only the identity and, with pybliss, permutations that map the nonzero X-type gauge generators, as listed, one to one onto the Z-type ones and back (section "Searching one sector of a CSS code" of [README.md](README.md)).
[_information_sets/automorphism_families.py](_information_sets/automorphism_families.py) verifies automorphisms and builds the families of [Section 3.7](#37-images-of-an-information-set-under-automorphisms), with the certificates of [_information_sets/certificates.py](_information_sets/certificates.py).

[logical_counts.py](logical_counts.py) builds its families with the stages of [_information_sets/basis_families.py](_information_sets/basis_families.py) that distance searches use ([Section 4.4](#44-heuristics-and-cost-models)), in `_select_basis_family` and `count_words` of [_counting/scan.py](_counting/scan.py), under the stricter stopping rule of [Section 5](#5-correctness-requirements).
To find its default weight, the distance, a count runs a distance search from the disjoint families that it keeps.
The setup functions of distance searches and counts can be compiled on request through [_compilation.py](../../_compilation.py).
Every caller of `get_overlapping_family` ([Section 4.4](#44-heuristics-and-cost-models)) gets its `OverlappingFamilyPolicy`, with a fixed-seed generator, from `build_overlapping_family_policy`: the distance search and the counts through `get_overlapping_family_policy`, which reads the tuning, and `CosetRoutes` of [_brouwer_zimmermann/coset_routes.py](_brouwer_zimmermann/coset_routes.py).
[_information_sets/completion_bounds.py](_information_sets/completion_bounds.py) chooses and builds the tables of [Section 3.9](#39-skipping-completions-by-completion-tables) for distance searches and counts, and sizes and builds them for the routes by tails of coset searches (below).

The distance search and `CosetLeaderSearch` scan with the compiled entry points of [_brouwer_zimmermann/affine_search.py](_brouwer_zimmermann/affine_search.py), which Numba compiles from one body, `_scan`, separately for each set of constant flags, so no flag costs a branch for each word:

- `scan_set` scans one set of a distance search, with or without completion tables.
- `scan_coset_layer` and `scan_coset_layer_by_css_weight` scan one layer of one set in a coset search, and keep its lightest word.
- `scan_coset_layer_by_tails` does so by the tails of the words ([Section 3.5](#35-packed-enumeration-with-pivot-column-deletion)), and keeps the support of the lightest word.
- `scan_coset_layer_by_tails_with_tables` also skips, in a layer of two or more rows, the completions whose bound reaches the lesser of a skip bound and the best weight so far.
  A query skips at `h(ω)` ([Section 3.9](#39-skipping-completions-by-completion-tables)), and a search for minimum weights at its incumbent ([Section 3.10](#310-minimum-weights-of-cosets)).

Each search packs its problem once, before it scans, in an `AffineInstance` of [_brouwer_zimmermann/affine_instance.py](_brouwer_zimmermann/affine_instance.py): the packed rows, labels, pivots, and ranks of its sets, and the certificate of their steps, whose weight divisor rounds its bounds.
`get_distance_instance` packs a basis family without its pivot columns ([Section 3.5](#35-packed-enumeration-with-pivot-column-deletion)), and a distance search builds another instance when it switches to an automorphism family.

`CosetLeaderSearch` of [coset_leaders.py](coset_leaders.py), which the fault-propagation certifier of `qldpc.circuits.certification` uses, applies BZ to cosets `u + V`, unless it is built with `method="connected_cluster"`.
A query of `CosetLeaderSearch`, whether the coset of a syndrome has a word within a maximum weight, runs one or two **coset searches**.
They are a search of Pauli errors or, for the CSS weight `max(|x|, |z|)` on a CSS state, searches by Hamming weight of the X and Z parts of the errors, each in a coset of a classical code ([README.md](README.md#the-coset-leader-search)).

Each family of search bases of a coset search has a **coset instance**: `get_coset_instance` packs the full rows of the disjoint sets of the code, and `get_coset_family_instance` those of any other family, with the certificate of its ranks and maximum coverage.
That certificate takes no weight divisor, even when `get_disjoint_family` infers one for `V`, since the words of a coset need not share it ([Section 3.2](#32-rounding-the-certificate-to-a-proven-weight-multiple)).
If `V` is even, `AffineInstance.get_weight_bounds` rounds the bounds up to a parity that it is given, and `get_search_tables` asks for that rounding when some coset is odd, which holds exactly when some word whose syndrome has one set bit is odd.
In a search of Pauli errors, an error weighs at least half the Hamming weight of its binary word, so the instance has the weight factor `η = 2`, by which `get_weight_bounds` divides its bounds, rounding up, after any rounding to parity; a search of X or Z parts has `η = 1`.

A query starts from the **word of its syndrome**, the sum of the words with unit syndromes (`get_unit_syndrome_words`) for the set bits of its syndrome.
It stops at its **stopping step**, the first step whose bound exceeds the maximum weight, unless it finds a light enough word sooner.
Each family of a coset search keeps the bounds of its steps in a **planning buffer**: one table for each parity of the cosets if the search rounds to parity, and one otherwise, and a query takes the table of its coset's parity.
Each table has an extra row that caches a stopping step; `get_stopping_step` reuses it for a nonnegative maximum weight exactly when it is that weight's stopping step.
Callers share the searches that `get_pauli_search` and `get_css_searches` cache, with the planning buffers of their families.
Numba compiles `get_stopping_step` without `nogil`, so its compiled calls hold the global interpreter lock, and threads interleave its calls only when it runs as plain Python or in free-threaded Python.
A cached step is written as two entries, so a read can pair the layer of one step with the number of completed sets of another.
Any such pair is a valid entry of the bounds, which never change, and `get_stopping_step` reuses it only if those bounds show that it is the stopping step.

`find_stopping_step` of [_information_sets/certificates.py](_information_sets/certificates.py) finds stopping steps by a binary search of the steps, whose bounds never decrease, and `plan_steps` also adds up their work ([Section 3.6](#36-work-estimates-and-exhaustive-enumeration)).
The distance search plans each layer with `Certificate.get_num_sets_to_certify`, which calls `Certificate.plan`; `plan` uses `find_stopping_step` when the certificate solves no linear programs, and otherwise checks every step of the layer but the last in order with `certifies`.

`BrouwerZimmermannCosetSearch.get_min_weight_bound` first looks up the words of weight at most one, and below a maximum weight of two returns the lower bound two without a search, or three for an odd coset in a search of X or Z parts that rounds to parity.
Otherwise a query takes the **route** of its maximum weight: a family of search bases, with its coset instance and planning buffer, which `CosetRoutes` of [_brouwer_zimmermann/coset_routes.py](_brouwer_zimmermann/coset_routes.py) chooses once for each maximum weight ([Section 4.4](#44-heuristics-and-cost-models)).
`find_first_not_within_max_weight` takes the route once for a batch, and for the CSS weight on a CSS state, `CosetLeaderSearch` gives the X parts and the Z parts of a batch to their own searches and routes.
`get_min_weights` of `CosetLeaderSearch` finds the minimum weights of [Section 3.10](#310-minimum-weights-of-cosets) through `BrouwerZimmermannCosetSearch.get_min_weights` and `_find_min_weights`, which finds its stopping steps with `find_stopping_step` and leaves the cached steps to the queries.

On a **route by tails** ([Section 4.4](#44-heuristics-and-cost-models)), the queries weigh words by their tails, and pass the family's **record of tails**, `CosetTails`, as a plain tuple of its fields to the compiled searches, such as `_search_coset`, in place of None; Numba compiles each search separately for a record and for None, so the choice costs nothing for each word.
The record holds the tails of the rows of each search basis and of the words with unit syndromes, from which a query sums the tail of its offset, and the completion tables that the routes of the family ask for.
`CosetRoutes` builds the tails of a family, and the tables of each search basis, once, for the first route that needs them (`get_coset_tails`, `CompletionTables.build`).
It builds new tables in the array of a new record, which copies the earlier tables; the old record keeps those until `CosetRoutes` replaces it.
A query by tails scans with `scan_coset_layer_by_tails_with_tables` the layers of two or more rows of a search basis that has tables, and the others with `scan_coset_layer_by_tails`.
It rebuilds its lightest word once, with the full rows, from the word's search basis and support.
The search of Pauli errors by CSS weight has no routes by tails, since the CSS weight of a word depends on which of its deleted pivot columns lie in the X part.

`BrouwerZimmermannCosetSearch`, `_search_coset`, and `_find_min_weights` are in [_brouwer_zimmermann/coset.py](_brouwer_zimmermann/coset.py), and the tables of steps and tails, such as `get_search_tables`, `get_stopping_step`, and `CosetTails`, in [_brouwer_zimmermann/coset_tables.py](_brouwer_zimmermann/coset_tables.py).

### 4.2 Flow and exit points

A public entry point validates inputs and chooses a method.
For BZ, `get_distance_brouwer_zimmermann` builds a `BrouwerZimmermannSearch`, which prepares a search of each sector.
Each call of its `advance` method advances the search of one remaining sector by one layer, in round-robin order for a joint CSS search ([Section 3.3](#33-sharing-witnesses-between-css-sectors)).
Preparation returns a result, or a `_SectorSearch` whose `advance` method searches one layer `t`, or only its first `j` search bases if step `(t, j)` certifies `U` ([Section 2.5](#25-the-stopping-certificate)).
A result of preparation can include a witness from another sector or the bound of a weight limit ([Section 4.5](#45-behavior-spanning-files)), so it need not be the distance of its sector.

| Input or method in [core.py](core.py) | Outcome |
| --- | --- |
| Trivial input | Returns `NO_DISTANCE`, whatever `max_weight` is, for an empty classical code, a quantum problem with no logical rows, or `V = S` |
| `method="brute_force"` | Runs exhaustive enumeration over an independent basis, adapted to `S ⊆ V` for a quantum problem |

`_prepare_sector_search` considers the following exits in order.
The disjoint family, its pruning ([Section 4.4](#44-heuristics-and-cost-models)), and any overlapping or partitioned replacement come from the stages of [_information_sets/basis_families.py](_information_sets/basis_families.py); the exits between those stages belong to the distance search.

| Preparation stage | Outcome |
| --- | --- |
| Initial row witness | Returns if `U ≤ cutoff`, where `U` includes the lightest eligible row and any supplied bound (from another sector or a weight limit) |
| Rows of the disjoint family | Returns if checking the rows of every search basis gives `U ≤ cutoff` |
| Certificate of the disjoint family | Returns if `L(1) ≥ U` |
| Rows of each replacement family, overlapping and then partitioned, if selected | Returns if `U ≤ cutoff` |
| Certificate of each replacement family | Returns if `L(1) ≥ U` |
| Switch at the end of setup | If setup asked for automorphisms before partitioning and found some, the search considers switching to their family ([Section 3.7](#37-images-of-an-information-set-under-automorphisms)); returns if then `U ≤ cutoff` or `L(1) ≥ U` |
| Ready search | Returns a `_SectorSearch` that scans from layer two |

| Search location | Outcome |
| --- | --- |
| `_SectorSearch.advance` | Returns after one layer, or after the leading search bases of a layer that certify `U`, or earlier if `U ≤ cutoff`; a search may first switch to an automorphism family ([Section 3.7](#37-images-of-an-information-set-under-automorphisms)); a finished search does no further work |
| `_SectorSearch.finished` | Marks a search finished if `U ≤ cutoff`, `L ≥ U`, or every layer has been enumerated (`next_weight > dimension`) |
| `BrouwerZimmermannSearch.finished` | Marks the search finished once the shared `U ≤ cutoff`, or the search of every sector has finished; `get_distance_brouwer_zimmermann` then returns `U` |

The tests `test_brouwer_zimmermann_setup_exits`, `test_brouwer_zimmermann_setup_cutoffs`, `test_rows_of_an_automorphism_family_can_reach_the_cutoff`, and `test_searches_that_setup_settles_are_finished_at_once` in [_brouwer_zimmermann/distance_test.py](_brouwer_zimmermann/distance_test.py) exercise these exits.

### 4.3 Concept-to-code glossary

| Concept | Code |
| --- | --- |
| Upper bound `U` | `best`, or `upper_bound` as a parameter |
| Lower certificate `L` | `lower_bound` of a search, computed by `Certificate.lower_bound(completed_weight)` from the certificate's `dimension`, `ranks`, `weight_divisor`, and `max_coverage` |
| Certificate within a layer `L(t, j)` | `Certificate.step_lower_bound(weight, num_completed)` and the table of `get_step_lower_bounds`, which `Certificate.get_weight_bounds` rounds and divides for coset searches; `get_num_sets_to_certify`, `plan`, `find_stopping_step`, and `plan_steps` find stopping steps ([Section 4.1](#41-module-map-and-layering)) |
| Certificate from linear programming | `get_lp_lower_bound`, which a `Certificate` with `pivot_blocks` combines with the coverage-divided certificate; `BasisFamily.get_search_certificate` adds the blocks to costly searches of overlapping families |
| Family of search bases | `BasisFamily` of [_information_sets/basis_families.py](_information_sets/basis_families.py), which stacks the search bases into arrays and gives their `Certificate`; packed in an `AffineInstance`, and, in coset searches, with the tables of `CosetSearchTables` and `CosetTails` of [_brouwer_zimmermann/coset_tables.py](_brouwer_zimmermann/coset_tables.py) |
| Automorphisms, generators of `Γ` | `automorphisms`, whose row `i` maps column `j` to column `automorphisms[i, j]`, as returned by `get_automorphisms`; `_found_qudit_automorphisms` and `_get_qudit_automorphisms` of `QuditCode` |
| Automorphism family | `get_automorphism_family` of [_information_sets/automorphism_families.py](_information_sets/automorphism_families.py), which returns a `BasisFamily` of the scanned set and the `Certificate` of all of its images; `_switch_to_automorphism_family` adopts them, and sets `uses_automorphism_family` of the search |
| Last completed layer `t`, next layer | `completed_weight`, `next_weight` |
| Divisor `D` | `Certificate.weight_divisor`, in Hamming units of the searched basis |
| Maximum coverage `C` | `Certificate.max_coverage` |
| Dimension `r`, block rank `r_i` | `dimension`, `ranks` (`BasisFamily.ranks`, `AffineInstance.ranks`) |
| Search basis and its block | One set of a `BasisFamily`: its entries of `generators`, `labels`, `pivots`, and `ranks`; the recorded block is the pivot subset, as explained below |
| Number of coordinates `N` | The length of `columns` (or `nonzero_columns`), the array of nonzero columns, in the functions that build families of search bases |
| Stored width `N_stored` | `basis.shape[1]` before pivot deletion, including any all-zero columns |
| Labels `M` | `labels`, which has no columns for a classical problem |
| Eligible rows | `get_eligible_rows` of [_binary.py](_binary.py): every row of a classical basis; rows with nonzero labels for a quantum problem |
| Support, prefix XORs | `support` and `prefixes` in `_SectorSearch._scan_set` and in the scans of [_brouwer_zimmermann/affine_search.py](_brouwer_zimmermann/affine_search.py), which step through a layer with `advance_support` of [_information_sets/support_stepping.py](_information_sets/support_stepping.py) |
| Packed generators | `generators` of the `AffineInstance` of a distance search: `uint64` words, after deleting pivot columns. `widths` counts the valid words for each search basis. |
| Packed labels | `labels` of the `AffineInstance` of a distance search: `uint64` quotient-label words, without pivot-column deletion |
| Batch of a compiled scan | `batch_size`, equal to `1 << information_set_batch_exponent` supports, or supports and lookups together in a scan with completion tables ([Section 4.5](#45-behavior-spanning-files)) |
| Candidate/set evaluation | The unit of the work of `plan_steps`, which `Certificate.work` counts |
| Completion tables ([Section 3.9](#39-skipping-completions-by-completion-tables)) | `CompletionTables` of [_information_sets/completion_bounds.py](_information_sets/completion_bounds.py), chosen by `select_completion_tables` and kept, with the index of their set, in `completion_tables` and `completion_tables_set` of a distance search, or in `CosetTails` of a coset search |

For the rank-deficient block of [Section 2.4](#24-search-bases-and-rank-deficient-blocks), the implementation records only its `rank` independent pivot columns in its row of `BasisFamily.pivots`, padded with -1, which give the same term `b_i(t)`, and which coverage and pivot-column deletion use.

### 4.4 Heuristics and cost models

Every family gets the certificate of its own verified ranks and maximum coverage, so the heuristics below change how much work a search does, not whether its certificate holds; with a cutoff above one, they can change which witness at or below the cutoff it returns ([Section 4.5](#45-behavior-spanning-files)).
Searches read the heuristics' fields as README.md's [Tuning](README.md#tuning) section says.
Some fields are thresholds of gates on optional steps, such as building a family of search bases, solving linear programs, asking for automorphisms, or predicting the cost of completion tables.
A step can save at most the work that it acts on, so its gate lets it run once an estimate of that work, in candidate/set evaluations, reaches the cost of the step, which the threshold need match only to an order of magnitude.
The Retuning section of [_tuning.py](_tuning.py) defines the other classes of fields, such as budgets and rough coefficients, and says how the defaults are set and when to change them.

- **`get_disjoint_family`** packs at most `2^r - 1` information sets greedily in natural column order.
  The layers of one set already reach every nonzero combination of rows, so a layer of more sets would examine more candidates than there are words, and the limit keeps the memory of the packing from growing with the square of the length.
  If it packs fewer information sets than `min(2^r - 1, floor(N / r))`, it tries up to `disjoint_random_column_orders` random orders from a fixed seed, a budget of setup effort since each costs another packing, and keeps the family with the lexicographically largest tuple of ranks.
  It also infers the divisor `D = 2` of [Section 3.2](#32-rounding-the-certificate-to-a-proven-weight-multiple).
- **`prune_family`** loses nothing: it removes a search basis of rank deficiency `d = r - r_i`, whose term `b_i(t)` is zero for `t < d`, when the information sets alone certify `U` by layer `d`.
- **`get_overlapping_family`** builds search bases on column orders that put the least-used columns first, with ties broken by a generator that `build_overlapping_family_policy` seeds afresh for every call, and keeps the cheapest family built.
  It makes up to `overlapping_attempts_per_set` attempts, each a row reduction, for each set that it may add, up to a family of `max_information_sets` sets, and a repeated pivot set still consumes an attempt; both fields are budgets of setup effort, since more sets can avoid costly layers but take more attempts.
  It builds nothing if the family that it is given has no information set, or at least `max_information_sets` of them, or information sets that use every nonzero column, or if no family of at most `max_information_sets` sets could do less work.
  Its gate, `overlapping_minimum_work` on the estimated work of the given family, breaks even with the cost of the attempts.
  `CosetRoutes` passes a minimum work of zero, since `coset_route_minimum_work` gates its families instead.
- **`get_partitioned_family`** finds the most information sets, up to `m = max_information_sets`, in which no column appears more than `b` times, by the matroid partitioning of Section 5.1 of Lisoněk and Trummer ([References](#references)) in `_information_sets/matroid_partition.py`.
  It may try each coverage `b` from two up to `ceil(m*r/N)`, the least at which `m` sets could fit by counting their pivots.
  A search takes the family of least work, if it does less work than the family of the earlier stages.
  Partitioning at `b` finds at most `m_b = min(m, floor(b*N/r))` sets, since at most `floor(b*N/r)` fit within coverage `b` by counting their pivots.
  A family of `j` information sets puts some column in at least `ceil(j*r/N)` of them, so at each step its coverage-divided certificate is at most that of `j` sets of that coverage.
  The work up to a step depends only on the step and the number of sets, so the family does at least the work of those `j` sets.
  `_estimate_optimistic_work` returns `W(k)`, the least such work over `j` from one to `k`.
  Partitioning skips each `b` at which `W(m_b)` is at least the work of the best family so far, at first the family of the earlier stages.
  No family that it finds at such a `b` could replace that family, so the search takes the same family as if partitioning tried every `b`.
  Trying every `b` from two up to `ceil(m*r/N)` costs roughly the sum of `b^2*r^2*N` over them, times a factor that varies between families of codes; the sum grows with the cube of `m`.
  So a search tries partitioning only if some family of at most `m` sets could do less work, and the search's estimated work is at least `partition_work_ratio` times that sum, a ratio that stands for the cost of one unit of the sum in evaluations.
  The gate charges that whole sum, though partitioning may skip some `b`.
  With `B = partition_max_coverage`, a limit for experiments, `m` is capped at `floor(B*N/r)`, the most sets that fit within coverage `B` by counting their pivots, so it tries no coverage above `B`.
  At `B = 1`, partitioning instead builds the maximum disjoint family of `get_maximum_disjoint_family` (see `CosetRoutes` below), pruned by `prune_family`.
  `m` does not limit that family, and the sum is `r^2*N`.
  The test of whether some family could do less work considers the disjoint families that partitioning can build.
  At coverage one, partitioning, like a packing, keeps a rank-deficient block only below `2^r - 1` information sets, so such a family holds at most `min(ceil(N/r), 2^r - 1)` sets, counting that block.
  It does no less work than as many information sets of coverage one, whose certificate is at least as large at each step, since the work up to a step depends only on the step and the number of sets.
  Let `f = min(floor(N/r), 2^r - 1)`, the most information sets that fit, up to the limit of a packing.
  The **disjoint rank bound** is `f` information sets and, if `f < 2^r - 1` and `N > f*r`, a rank-deficient block of rank `N - f*r`.
  By counting pivots, it bounds the ranks of every maximum disjoint family lexicographically.
  Partitioning skips coverage one if the family `F` of the earlier stages is disjoint and has the ranks of this bound, pruned for `U` as `prune_family` prunes a family.
  Every maximum disjoint family then has `f` information sets and a block of rank at most `N - f*r`, but at least that of any block of `F` (see `CosetRoutes`).
  Pruning drops a block of lower rank if it drops one of rank `N - f*r`, since the lower rank gives a larger deficiency `d`, and information sets that certify `U` by one layer also certify it by every later one.
  Once pruned, a maximum disjoint family therefore has the ranks of `F`, does the same work, and cannot replace `F`.
- **`get_search_certificate`** adds the linear programs of [Section 3.1](#31-overlapping-search-bases) to searches of overlapping families whose estimated work reaches `lp_minimum_work`, a gate that breaks even with their cost, since each costs far more than examining a word; the work estimates that choose families use only the coverage-divided certificate.
  The rational dual weights `y` have denominators of at most `lp_dual_max_denominator`, a convention: every positive limit is sound by weak duality, a smaller one can round away part of the bound, and a larger one lengthens the exact arithmetic.
- **`get_automorphism_family`** row reduces in natural column order and balances the shares of the orbits in `_balance_orbit_shares` ([Section 3.7](#37-images-of-an-information-set-under-automorphisms)).
  The gate `automorphism_minimum_layer_work` of [Section 3.7](#37-images-of-an-information-set-under-automorphisms) breaks even with the cost of finding automorphisms, verifying them, and building their family.
- **`CosetRoutes`** chooses the route of each maximum weight `ω` of a coset search ([Section 4.1](#41-module-map-and-layering)).
  The **predicted examinations** of a query at `ω` with a family, without completion tables, are one more than the largest work at `ω` over its planning buffer's tables, which bounds the words that the query examines or skips ([Section 3.6](#36-work-estimates-and-exhaustive-enumeration)).
  1. If the predicted examinations with the disjoint family are below `coset_route_minimum_work`, the route is the disjoint family.
  2. Otherwise, `CosetRoutes` tries once to build two families, at `ω_T`, the least weight from two at which the predicted examinations with the disjoint family reach the threshold, so that they do not depend on the order of queries.
     The first, which `get_maximum_disjoint_family` finds by matroid partitioning with coverage one, holds the most disjoint information sets, up to the limit of `get_disjoint_family`, and below that limit a rank-deficient block of the largest rank ([Section 2.4](#24-search-bases-and-rank-deficient-blocks)) on the other columns.
     It is kept if it has more information sets, or a rank-deficient block of larger rank, than the greedy packing.
     Edmonds' matroid partitioning algorithm maximizes the number of columns in its sets, all but the last of which are information sets, so no other choice of as many information sets leaves a block of larger rank.
     A greedy packing can miss sets: the columns `e1`, `e2`, `e1 + e2`, and `e1 + e2` hold two disjoint information sets, but a greedy packing in natural order takes `e1` and `e2`, and leaves a rank-deficient block of rank one.
     The second, an overlapping family of `get_overlapping_family` aimed at the Hamming weight `η ω_T + 1`, is kept if it does less work than the family that it starts from: the first, if kept, and otherwise the disjoint family.
  3. The route is the family of least work at `ω`, summed over its tables, and on ties the disjoint family, then the maximum disjoint one.
     A search for minimum weights up to `ω` takes the route of `ω`, or of the number of qubits if that is smaller, since no coset has a larger minimum weight.
  4. In a search by Hamming weight, the route is a route by tails if its own predicted examinations also reach the threshold.
     Its queries then weigh words by their tails, and skip completions by tables on blocks of `coset_completion_block_bits` columns, for every search basis that they scan in layers of two or more rows, if the tables fit in `completion_bound_max_table_bytes`; the threshold and this limit replace the minimum layer work and the cost model of `select_completion_tables`.

  Building the two families costs a matroid partitioning and the row reductions of an overlapping family; weighing by tails adds a fixed cost to each query, for the tail of its offset in each search basis and the rebuild of its lightest word; and tables cost memory and a build.
  Only queries that examine many words repay these costs.
  The threshold also sets the aim `ω_T`, which no break-even rule covers, so its default is chosen by hand.
- **`select_completion_tables`** decides, by the steps of [Section 3.9](#39-skipping-completions-by-completion-tables), whether a distance search uses tables on an information set in a layer, or a count on a step.
  Its gate `completion_bound_minimum_layer_work`, which `considers_tables` checks, breaks even with the cost of the prediction, whose sampling stays within the budget of setup effort `completion_bound_sampling_work`.
  The prediction takes the share that tables skip at the lower end of a Wilson score interval of `completion_bound_confidence_z` standard errors, or as zero with fewer than `completion_bound_min_samples` samples, conventions that keep sampling error from choosing tables that cost more than they save.
  The limit `completion_bound_max_cost_percent` compares two estimates in one unit, so it breaks even at 100 percent.
  The rough coefficients of `predict_completion_cost` are the costs `completion_bound_lookup_block_cost`, `completion_bound_lookup_block_cost_per_mib`, `completion_bound_table_byte_cost`, and `completion_bound_build_step_cost`, in scans of one support of a distance search, and the multiplier `completion_bound_kept_first_row_ratio`; a count scans full rows and labels, which cost more, so the model is conservative for it.
  The widths `completion_bound_block_bits` and `coset_completion_block_bits` default to sixteen, the widest that `CompletionTables.build` accepts, since a wider block never gives a weaker bound.
  Sampling only predicts the value of tables; every skip rests on a computed bound.
- **`Certificate.work` and `estimate_automorphism_family_work`** are the estimates of Sections [3.6](#36-work-estimates-and-exhaustive-enumeration) and [3.7](#37-images-of-an-information-set-under-automorphisms), both of which `plan_steps` computes.

### 4.5 Behavior spanning files

- **Cutoff.**
  The docstring of `get_distance_from_generators` in [core.py](core.py) states the contracts of `cutoff` and of an early exit, which the other distance functions share.
  The default `cutoff=1` gives the exact distance of a nontrivial problem, which is at least one, and a `cutoff` of zero or less never exits early; `_scan_cutoff` clamps negative values to `-1` for the compiled scan.
  [Section 4.2](#42-flow-and-exit-points) lists where `cutoff` is checked.
- **Weight limit.**
  The same docstring states the contract of `max_weight`.
  The functions pass `upper_bound=max_weight+1` to BZ, as though a witness of that weight were known, so that it stops once its certificate reaches that bound, and clamp the results of the other methods to `max_weight+1`.
  They also cap `cutoff` at `max_weight`, since a higher cutoff would end the search at once on the assumed witness.
  Inside the searches, a sector without nontrivial words counts as `n+1`, which no word on `n` bits or qubits weighs, so that it never lowers the minimum over sectors.
- **Units.**
  Binary input (`symplectic=False`) is a classical problem, or a pure-X or pure-Z problem with one bit per qubit, and symplectic input (`symplectic=True`) has general Pauli rows.
  For symplectic input, BZ maps rows by `T` ([Section 2.7](#27-pauli-weight-as-hamming-weight)), doubles `cutoff` and the bound of a weight limit, passes `weight_divisor=2`, and halves the result, while `method="brute_force"` uses symplectic weights and the original `cutoff`.
- **Independence.**
  `_get_nested_basis` in `core.py` makes the generators of a caller independent, with `get_independent_rows`, or with `get_nested_code_basis` when the caller gives `S`, for every BZ and brute-force search of `_search_generators` and `_search_checks`, including those of `search_by_cost`; `_search_css` calls `get_nested_code_basis` itself for its BZ searches and those of `search_by_cost`.
- **Resumable layers.**
  A compiled scan of a distance search returns after at most `batch_size` supports, or supports and lookups together with completion tables, so that it stays responsive while it skips most supports, and successive batches reuse its `support` and `prefixes` arrays.
  Once the search bases that `advance` scans have completed a layer, it may raise `lower_bound` ([Section 2.5](#25-the-stopping-certificate)), but it increments `next_weight` only once every search basis has.
  Before a costly layer, `advance` may switch to an automorphism family ([Section 3.7](#37-images-of-an-information-set-under-automorphisms)), which sets `next_weight` to two and may raise `lower_bound`.
  If a scan stops at the cutoff or raises an exception, `advance` leaves `lower_bound` and `next_weight` as they were before that scan, but keeps any earlier switch.
  A later call restarts the layer from its first support, unless the search has finished, as it has after a stop at the cutoff.
  If `BrouwerZimmermannSearch.advance` raises, a witness that the interrupted scan found can stay out of the shared `U`, so build a new search to search again.
- **CSS sectors.**
  Each sector of a joint search keeps its `lower_bound` and the `certificate` of its `instance` ([Section 3.3](#33-sharing-witnesses-between-css-sectors)).
- **Automorphisms.**
  The distance functions take automorphisms as an array or as a function, and pass them to their BZ searches as a function that calls the function of the caller at most once.
  `get_distance_exact` of `CSSCode` and of `QuditCode` in `src/qldpc/codes/common.py` pass a function that returns `_found_qudit_automorphisms` of the code, which the code caches, so bliss runs at most once for a code.
  It holds the automorphisms that bliss (`src/qldpc/external/_bliss.py`) finds among each set of checks from `_get_automorphism_search_matrices`, followed by the generators from `_get_qudit_automorphisms`, without repeats or the identity.
  The code verifies the generators from `_get_qudit_automorphisms` before it appends them.
  A search calls the function only at the points of [Section 3.7](#37-images-of-an-information-set-under-automorphisms), and uses the generators only after `verify_automorphisms` accepts them; for a CSS code, they must map the row spaces of both parity-check matrices onto themselves.
  For a general Pauli problem, `_get_lifted_automorphisms` lifts them to the coordinates of `T`, except for Pauli strings that `get_distance_pauli` searches as a CSS code.

## 5. Correctness requirements

A change may use other representations, certificates, and search strategies, but must keep these requirements for the nontrivial problems of [Section 2.1](#21-the-problem):

- **Search `V \ S`.**
  Transformations must preserve `V` and `S`, or an equivalent representation of `V \ S`, such as the label test of [Section 3.4](#34-quotient-labels-for-the-stabilizer-test).
- **Keep weights in one unit.**
  Candidate weights, certificates, cutoffs, and results must agree, which includes restoring the pivot ones ([Section 3.5](#35-packed-enumeration-with-pivot-column-deletion)) and converting between Pauli and Hamming weights ([Section 4.5](#45-behavior-spanning-files)).
- **Use only justified certificates.**
  A certificate `L` must hold for every eligible word lighter than `U`, all of which are unexamined, and need not hold for heavier words, such as skipped ones, so a search reports the bound `min(L, U)`, not `L`.
  It relies on block ranks, coverage, and completed layers (Sections [2.4](#24-search-bases-and-rank-deficient-blocks), [2.5](#25-the-stopping-certificate), and [3.1](#31-overlapping-search-bases)), and may overestimate coverage but never underestimate it; a coset search may use any family of its code, with that family's ranks and coverage.
  Rounding needs a congruence of every eligible weight: a common divisor, or, only in a coset search, where every word is eligible, the parity of a coset of an even code ([Section 3.2](#32-rounding-the-certificate-to-a-proven-weight-multiple)).
  A certificate from another CSS sector needs a justification in the sector where it is applied, and that of an automorphism family needs verified automorphisms ([Section 3.7](#37-images-of-an-information-set-under-automorphisms)).
- **Skip only words that a bound proves heavy enough:** at least `U` in a distance search ([Section 3.9](#39-skipping-completions-by-completion-tables)).
  Completion bounds need a minimum over every remaining pair of rows; disjoint blocks; exactly `w` pivot ones, as only in an information set, or the coset scans' bound on pivot ones; and the rows of their own search basis in scan order ([Section 3.9](#39-skipping-completions-by-completion-tables)).
  Coset tails must keep every column but the pivots ([Section 3.5](#35-packed-enumeration-with-pivot-column-deletion)).
  A coset query of maximum weight `ω` skips at `h(ω)`, never `h(ω) - 1`, and reports `R` only if it skipped a word.
- **Find minimum weights of cosets exactly up to `ω`.**
  A search may stop only at a step whose bound reaches `U`, or within a step once `U` is at most the bound of the step before or the floor of the search: `w_x` in the search of the Z part for the CSS weight ([Section 3.10](#310-minimum-weights-of-cosets)), and zero otherwise.
  Its scans may keep only words lighter than its incumbent, and its tables may skip only at the incumbent or at the weight of a lighter word that the scan has found ([Section 3.10](#310-minimum-weights-of-cosets)).
- **Keep the strict stop of counting.**
  A count of the words of weight `t` must also exhaust the words of weight exactly `t`, so it stops only at a step whose bound exceeds `t`, never at equality, and skips only completions heavier than `t` ([Section 3.9](#39-skipping-completions-by-completion-tables)).
  A word of weight `t` is counted at the first step that visits it, and steps run in order, so a count that stops within a layer counts each such word once.
  The certificate of a count uses the ranks and actual coverage of its family, and the `upper_bound=t+1` that it passes to the family stages only aims them.
- **Keep coset queries reproducible.**
  The family and tables that answer a query must depend only on the code, the maximum weight, and the tuning, not on earlier queries, since the certifier's counterexamples report a query's word and bound.
  A query therefore reads only the tables that its own route asks for (Sections [4.1](#41-module-map-and-layering) and [4.4](#44-heuristics-and-cost-models)).
- **Preserve the return contract.**
  Return `max_weight+1` if every nontrivial word is heavier than `max_weight`, and otherwise the exact minimum, unless `cutoff`, which `max_weight` caps, permits an early exit at a known eligible witness of weight at most `cutoff` ([Section 4.5](#45-behavior-spanning-files)).

These tests check the requirements, against exhaustive enumeration where they can:

- [_information_sets/basis_families_test.py](_information_sets/basis_families_test.py): `test_family_certificates` and `test_overlapping_family_certificates`, certificates with rank-deficient blocks, parity, and overlaps.
- [_information_sets/automorphism_families_test.py](_information_sets/automorphism_families_test.py): `test_automorphism_families_certify_like_the_family_of_all_images`.
- [_information_sets/completion_bounds_test.py](_information_sets/completion_bounds_test.py): `test_bounds_are_exact_relaxations`, completion bounds against enumerated completions.
- [_brouwer_zimmermann/distance_test.py](_brouwer_zimmermann/distance_test.py): `test_brouwer_zimmermann_sector_certificates`, joint searches with distinct sector certificates; `test_automorphism_family_searches_scan_every_layer_they_certify` and `test_automorphism_family_searches_scan_their_new_set_from_layer_two`, distances that skipping a layer would miss; `test_searches_ask_for_automorphisms_only_before_costly_layers`; and `test_completion_tables_keep_distances_exact`.
- [_brouwer_zimmermann/affine_search_test.py](_brouwer_zimmermann/affine_search_test.py): `test_scan_of_one_information_set`, which weighs or skips each support once, and `test_coset_tail_scans_with_tables_skip_only_completions_that_reach_the_skip_bound`, including rank deficiency three.
- [_brouwer_zimmermann/coset_test.py](_brouwer_zimmermann/coset_test.py): `test_searches_by_tails_answer_as_searches_of_whole_words`, the words, skips, and bounds `R` of queries with tables; `test_min_weights_of_cosets_match_brute_force`; `test_min_weight_searches_above_max_weight_examine_as_queries`; and `test_answers_do_not_depend_on_earlier_queries`.
- [_brouwer_zimmermann/coset_routes_test.py](_brouwer_zimmermann/coset_routes_test.py): the tables that routes by tails ask for and keep, also when threads choose routes at once.
- [coset_leaders_test.py](coset_leaders_test.py): `test_min_weights_match_exhaustive_minima`.
- [logical_counts_test.py](logical_counts_test.py) and [_counting/scan_test.py](_counting/scan_test.py): `test_counting_continues_at_an_equal_lower_bound` and `test_counts_with_completion_tables_match_enumeration`, counts that a stop or a skip at equality would get wrong.
- [core_test.py](core_test.py): `test_brouwer_zimmermann_random_cross_checks`, on binary and general Pauli problems; `test_max_weight`, on BZ and brute force; and `test_max_weight_caps_cutoff`.

Performance and helper tests check broad work budgets and current interfaces, and do not make a particular family, packing order, or optimization a requirement.

## References

- [arXiv:1603.06757v2](https://arxiv.org/abs/1603.06757v2) (Algorithm 1): Section 2 and Algorithm 1 give the classical disjoint and rank-deficient bound.
- M. Grassl, "Searching for linear codes with large minimum distance," in *Discovering Mathematics with Magma*, Springer, 2006, pp. 287-313 ([doi:10.1007/978-3-540-37634-7_13](https://doi.org/10.1007/978-3-540-37634-7_13)): Algorithm 2.5 gives the certificate for overlapping information sets, footnote 2 on page 295 the certificate within a layer, page 299 the rounding to a weight divisor, and Algorithm 2.11 the choice between BZ and the weight distribution.
- P. Lisoněk and L. Trummer, "Algorithms for the minimum weight of linear codes," *Advances in Mathematics of Communications* 10(1), 195-207, 2016 ([doi:10.3934/amc.2016.10.195](https://doi.org/10.3934/amc.2016.10.195)): equation (5) gives the certificate within a layer, Proposition 5 the coverage-divided certificate, and Section 5.1 builds families of information sets with bounded coverage by matroid partitioning.
- [arXiv:2408.10743v2](https://arxiv.org/abs/2408.10743v2): Section 2.4 describes the `Saved_isometry` symplectic-to-Hamming construction.
- J. MacWilliams, "A theorem on the distribution of weights in a systematic code," *Bell System Technical Journal* 42(1), 79-94, 1963 ([doi:10.1002/j.1538-7305.1963.tb04003.x](https://doi.org/10.1002/j.1538-7305.1963.tb04003.x)): the MacWilliams identity of Section 3.11.
- [core.py](core.py), [_brouwer_zimmermann/distance.py](_brouwer_zimmermann/distance.py), [coset_leaders.py](coset_leaders.py), [_brouwer_zimmermann/coset.py](_brouwer_zimmermann/coset.py), [_brouwer_zimmermann/coset_tables.py](_brouwer_zimmermann/coset_tables.py), [_brouwer_zimmermann/coset_routes.py](_brouwer_zimmermann/coset_routes.py), [_cosets.py](_cosets.py), [min_weight_bounds.py](min_weight_bounds.py), [_brouwer_zimmermann/affine_search.py](_brouwer_zimmermann/affine_search.py), [_brouwer_zimmermann/affine_instance.py](_brouwer_zimmermann/affine_instance.py), [_information_sets/basis_families.py](_information_sets/basis_families.py), [_information_sets/certificates.py](_information_sets/certificates.py), [_information_sets/support_stepping.py](_information_sets/support_stepping.py), [_information_sets/automorphism_families.py](_information_sets/automorphism_families.py), [_information_sets/completion_bounds.py](_information_sets/completion_bounds.py), [_information_sets/matroid_partition.py](_information_sets/matroid_partition.py), [logical_counts.py](logical_counts.py), [_counting/scan.py](_counting/scan.py), [_brute_force.py](_brute_force.py), [_binary.py](_binary.py), and [qldpc/_binary.py](../../_binary.py): the implementation.
- [_brouwer_zimmermann/distance_test.py](_brouwer_zimmermann/distance_test.py), [coset_leaders_test.py](coset_leaders_test.py), [_brouwer_zimmermann/coset_test.py](_brouwer_zimmermann/coset_test.py), [_brouwer_zimmermann/coset_tables_test.py](_brouwer_zimmermann/coset_tables_test.py), [_brouwer_zimmermann/coset_routes_test.py](_brouwer_zimmermann/coset_routes_test.py), [_cosets_test.py](_cosets_test.py), [_brouwer_zimmermann/affine_search_test.py](_brouwer_zimmermann/affine_search_test.py), [_brouwer_zimmermann/affine_instance_test.py](_brouwer_zimmermann/affine_instance_test.py), [_information_sets/basis_families_test.py](_information_sets/basis_families_test.py), [_information_sets/certificates_test.py](_information_sets/certificates_test.py), [_information_sets/support_stepping_test.py](_information_sets/support_stepping_test.py), [_information_sets/automorphism_families_test.py](_information_sets/automorphism_families_test.py), [_information_sets/completion_bounds_test.py](_information_sets/completion_bounds_test.py), [logical_counts_test.py](logical_counts_test.py), [_counting/scan_test.py](_counting/scan_test.py), and [core_test.py](core_test.py): executable invariants and independent small-instance oracles.
