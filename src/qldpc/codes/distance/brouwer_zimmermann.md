# Brouwer-Zimmermann distance search in qLDPC

<a name="sec-scope"></a>
## Purpose and scope

The Brouwer-Zimmermann (BZ) algorithm computes the distance of a classical code ([arXiv:1603.06757](https://arxiv.org/abs/1603.06757v2)), and has been adapted to quantum codes ([arXiv:2408.10743](https://arxiv.org/abs/2408.10743v2)).
This guide, in this file and [brouwer_zimmermann_code.md](brouwer_zimmermann_code.md), is for contributors who need to understand, debug, or modify the BZ code in this directory.
It explains how qLDPC computes exact distances of binary codes with BZ, how it counts logical operators ([Counting logical operators](#sec-counting)), and how the coset searches of `CosetLeaderSearch` use it ([Families for coset searches](#sec-coset-families)).
[connected_cluster.md](connected_cluster.md) describes another exact search, for codes with sparse checks.

- [The Brouwer-Zimmermann algorithm](#sec-algorithm) explains BZ as published, including its adaptation to quantum codes.
- [Extensions and optimizations](#sec-extensions) explains the extensions and optimizations of qLDPC, and where each comes from.
- [Guide to the source code](brouwer_zimmermann_code.md#sec-source-code), in [brouwer_zimmermann_code.md](brouwer_zimmermann_code.md), points to the modules that implement these concepts, and to their docstrings, and explains the coset searches, the heuristics and their cost models, and the behavior that spans files.
- [Correctness requirements](brouwer_zimmermann_code.md#sec-requirements), in the same file, lists the requirements that a change must preserve.

<a name="sec-algorithm"></a>
## The Brouwer-Zimmermann algorithm

This section describes BZ as presented in [arXiv:1603.06757](https://arxiv.org/abs/1603.06757v2) and [arXiv:2408.10743](https://arxiv.org/abs/2408.10743v2), together with a stopping rule within a layer from Grassl's chapter and from Lisoněk and Trummer (see [References](#sec-references)).

<a name="sec-problem"></a>
### The problem

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

<a name="sec-bounds-and-certificates"></a>
### The idea: upper bounds and lower certificates

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

<a name="sec-information-sets"></a>
### Systematic bases and information sets

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
**Searching a basis through weight `t`** means examining every XOR of at most `t` of its rows; each XOR that a search examines is a **candidate**.
After searching `G_I` through weight `t`, every unexamined word `v` therefore has `wt(v) ≥ wt(v|_I) ≥ t+1`.

<a name="sec-search-bases"></a>
### Search bases and rank-deficient blocks

BZ searches a **disjoint family** of `m` **search bases** `G_1, ..., G_m` of `V`, each row reduced on its own **block** `B_i` of columns, with the blocks pairwise disjoint.
The blocks are chosen greedily: `B_1` is an information set, `B_2` an information set among the other columns, and so on, and any columns that contain no further information set form one last block, of rank less than `r`.
Once qLDPC has taken `2^r - 1` information sets, it leaves any remaining columns out ([Heuristics and cost models](brouwer_zimmermann_code.md#sec-heuristics) gives the reason).

If `B_i` has rank `r_i`, row reducing on `B_i` gives a basis `G_i` of `V` whose first `r_i` rows, the **pivot rows**, have distinct pivot columns in `B_i`, and whose other rows are zero on `B_i`.
A word `v` of coefficient weight at least `t+1` in `G_i` selects at least `t + 1 - (r - r_i)` pivot rows, each of which contributes a distinct one to `v|_{B_i}`, so

$$
\operatorname{wt}(v|_{B_i}) \ge b_i(t) := \max(0,\;t+1-(r-r_i)),
$$

which is `t + 1` for an information set, as in [Systematic bases and information sets](#sec-information-sets).
The bound counts only the ones on the `r_i` pivot columns of the block, so it holds as well for the block of just those columns.
qLDPC records only these pivot columns as the block of a rank-deficient search basis, and computes coverage and deletes pivot columns with them.

<a name="sec-stopping-certificate"></a>
### The stopping certificate

Suppose that every search basis of [Search bases and rank-deficient blocks](#sec-search-bases) has been searched through weight `t`.
An eligible word `v` that none of them examined has `wt(v|_{B_i}) ≥ b_i(t)` for every `i`, and since the blocks are disjoint,

$$
\operatorname{wt}(v) \ge \sum_i b_i(t) =: L(t),
$$

which is `m(t+1)` if every block is an information set.
`L(t)` is the lower certificate.
It bounds only unexamined words, so it can exceed the distance, and once `L(t) ≥ U` the search can stop.

BZ is therefore the following loop:

1. Build the search bases `G_1, ..., G_m` from disjoint blocks, as in [Search bases and rank-deficient blocks](#sec-search-bases).
2. Examine layer one, the rows of the search bases, and let `U` be the weight of the lightest eligible row.
3. For `t = 1, 2, ...`, compute `L(t)`, and stop if `L(t) ≥ U`.
   Otherwise examine every selection of `t + 1` rows in each search basis, lowering `U` whenever a lighter eligible word is found.

The search terminates with the exact distance, at the latest after layer `r`, when every nonzero word has been examined.
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

The cost of a search is counted in **candidate/set evaluations**, examinations of one candidate in one search basis, which code comments and `brouwer_zimmermann_code.md` call a **set** for short ([Work estimates and exhaustive enumeration](#sec-work-estimates)).

<a name="sec-excluding-stabilizers"></a>
### Excluding stabilizers

For a quantum code, the lightest word in `V` may lie in `S`, and must not lower `U`.
Section 2.1 of [arXiv:2408.10743](https://arxiv.org/abs/2408.10743v2) therefore checks whether a candidate lighter than `U` lies in `S`, which qLDPC tests with quotient labels ([Quotient labels for the stabilizer test](#sec-quotient-labels)).
The certificate of [The stopping certificate](#sec-stopping-certificate) is unchanged, since it holds for all unexamined words.

<a name="sec-pauli-weight"></a>
### Pauli weight as Hamming weight

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

<a name="sec-extensions"></a>
## Extensions and optimizations

qLDPC goes beyond [The Brouwer-Zimmermann algorithm](#sec-algorithm) as follows; each section says which parts are published.

- Extensions that strengthen the bounds of a search: overlapping search bases ([Overlapping search bases](#sec-overlapping-bases)), rounding the certificate up to a proven weight multiple or to the parity of a coset ([Rounding the certificate to a proven weight multiple](#sec-weight-multiples)), and witnesses shared between CSS sectors ([Sharing witnesses between CSS sectors](#sec-shared-witnesses)).
- Optimizations: quotient labels for the stabilizer test ([Quotient labels for the stabilizer test](#sec-quotient-labels)), packed enumeration without pivot columns ([Packed enumeration with pivot-column deletion](#sec-packed-enumeration)), skipping words that precomputed tables prove too heavy ([Skipping completions by completion tables](#sec-completion-tables)), and minimum weights of cosets in one search ([Minimum weights of cosets](#sec-coset-min-weights)).
- Both: certifying with the images of one information set under automorphisms while scanning only that set ([Images of an information set under automorphisms](#sec-automorphisms)).

[Work estimates and exhaustive enumeration](#sec-work-estimates) defines the work estimate with which searches choose their search bases, and [Worked example](#sec-worked-example) works an example.
[Counting logical operators](#sec-counting) adapts the search to counts of logical operators, and [problems.md](problems.md#sec-macwilliams) explains the MacWilliams method of counting.

<a name="sec-overlapping-bases"></a>
### Overlapping search bases

An information set has `r` columns, so a disjoint family has at most `floor(N / r)` of them, and the other columns form at most one rank-deficient block, which can add little or nothing to `L(t) = Σ_i b_i(t)` ([Worked example](#sec-worked-example) gives an example).
An **overlapping family of search bases** lets blocks share columns, so it can hold more information sets, but a column in several blocks then counts in several terms `b_i(t)`.
Distance searches and counts build overlapping families in the overlapping and partitioning stages of [_information_sets/basis_families.py](_information_sets/basis_families.py) ([Heuristics and cost models](brouwer_zimmermann_code.md#sec-heuristics), which also names the tuning fields that decide whether they are built and how large they grow).

<a name="sec-overlapping-certificate"></a>
#### The certificate

Let the **maximum coverage** `C` of a family be the largest number of its blocks that contain one column.
If every search basis of the family has been searched through weight `t`, an unexamined eligible word `v` has `wt(v|_{B_i}) ≥ b_i(t)` for every `i`, and each of its ones lies in at most `C` blocks, so

$$
\sum_i b_i(t) \le \sum_i \operatorname{wt}(v|_{B_i}) \le C\operatorname{wt}(v),
\qquad\text{hence}\qquad
L(t) = \left\lceil \frac{\sum_i b_i(t)}{C} \right\rceil.
$$

For `m` information sets, `L(t) = ceil(m(t+1)/C)`, which beats a disjoint family of `k` information sets when `m/C > k`.
`C = 1` recovers [The stopping certificate](#sec-stopping-certificate), and within a layer the same argument divides the sum that defines `L(t, j)` by `C`.
For example, the code generated by `101` and `011` has distance two, and a disjoint family holds one information set.
Its three pairs of columns are information sets with `C = 2`, so after layer one `L(1) = ceil(3·2/2) = 3`; this bound on unexamined words exceeds the distance, and the search stops with `U = 2`.

Grassl's chapter (page 291 and Algorithm 2.5, see [References](#sec-references)) credits Zimmermann with a certificate for an ordered list of information sets, `sum_i max(0, t+1 - o_i)`, where `o_i` counts the columns of block `i` that lie in an earlier block (`r - o_i` is its relative rank).
Proposition 5 of Lisoněk and Trummer gives the coverage-divided certificate for blocks of equal bounds `t+1`, and its proof covers the different bounds of rank-deficient blocks and of steps within a layer.
[arXiv:1603.06757](https://arxiv.org/abs/1603.06757v2) mentions the overlapping case in its Section 2, citing Grassl, without a formula.
Neither Zimmermann's certificate nor the coverage-divided one dominates the other.
A linear program gives a third bound.
It minimizes `Σ_c x_c` over real `0 ≤ x_c ≤ 1` subject to `Σ_{c ∈ B_i} x_c ≥ b_i(t)` for each `i`.
The coordinates of an unexamined word `v` satisfy these constraints, so `wt(v)` is at least the optimum.
That optimum, rounded up to an integer, is at least both certificates, and can exceed them when coverage is uneven.
An earlier version of qLDPC solved the program, but qLDPC no longer does, since the program never changed the work or the lower bound of a layer in either group of searches measured:

- 14 distance searches and two counts at that version's defaults, which solved the program only in searches whose estimated work reached a threshold;
- 12 of those distance searches and one count, with the program solved in every search that used an overlapping family.

This guide at commit 0870f6a derives the program's bound and works an example in which it beats both certificates ([A certificate from linear programming](https://github.com/qLDPCOrg/qLDPC/blob/0870f6ac5273aa9544490a4835868c4da691a75c/src/qldpc/codes/distance/brouwer_zimmermann.md#a-certificate-from-linear-programming)).

Section 6 of Tillmann's preprint gives an integer program like the linear program above, with binary `x_c` and right-hand sides of one, for the size of the smallest set of linearly dependent columns of a matrix over the real numbers ([References](#sec-references)).
Tillmann's argument holds over any field.
Over GF(2), applied to a parity-check matrix, his program computes the distance of the code of that matrix, with one constraint for each information set.

<a name="sec-coset-families"></a>
#### Families for coset searches

The coset searches of `CosetLeaderSearch` ([Coset searches](brouwer_zimmermann_code.md#sec-coset-searches) in `brouwer_zimmermann_code.md`) search a coset `u + V` to answer **queries**: whether the coset has a word of weight at most a given maximum weight.
For each search basis `G_i`, fix an **offset**, a word of the coset that is zero on the pivot columns of `B_i`.
Every word of the coset is the offset plus a unique sum of rows of `G_i`, so once `G_i` has completed layer `t`, an unexamined word has at least `b_i(t)` ones on `B_i`, as a word of `V` does.
So every certificate of this section holds for cosets, and any family of search bases of `V` serves every coset of `V`.
A coset search can thus use an overlapping family, or a disjoint family that matroid partitioning finds ([Heuristics and cost models](brouwer_zimmermann_code.md#sec-heuristics)).
The GAP package GUAVA searches a coset of a linear code with offsets in the same way, but only over a disjoint family of information sets ([References](#sec-references)).
The distance program Qubitserf instead turns the minimum weight of a coset `u + V`, for `u` outside `V`, into a distance search ([References](#sec-references)).
In the terms of [The problem](#sec-problem), that search takes `V + span(u)` in the role of `V`, and `V` in the role of `S`, so its eligible words are those of the coset.

<a name="sec-weight-multiples"></a>
### Rounding the certificate to a proven weight multiple

If every eligible weight is a multiple of an integer `D`, an unexamined eligible word weighs at least `D * ceil(L(t)/D)`, so the certificate can be rounded up to that value.
Grassl's chapter notes this rounding on page 299, and Magma uses it for even binary and quaternary codes, doubly even binary codes, and ternary codes whose weights are all multiples of three.
In qLDPC, the distance functions pass `D = 2` for general Pauli problems, whose map `T` of [Pauli weight as Hamming weight](#sec-pauli-weight) doubles every weight, and qLDPC infers `D = 2` when every row of the basis of `V` has even weight, since sums of even words are even.

qLDPC checks only parity.
Since `wt(u + v) = wt(u) + wt(v) - 2|u ∩ v|`, every word of `V` has a weight divisible by four exactly when every row does and every two rows overlap evenly: the rows `11110` and `01111` weigh four, but their sum `10001` weighs two.
That stronger test mainly helps doubly even classical codes, since a CSS sector passes it only if every logical operator of the sector lies, as a binary vector, in the span of the stabilizers of the other type.

By the same identity, a divisor of the weights of `V` need not divide those of a coset `u + V`: every word of `V = {0000, 1111}` has a weight divisible by four, but `1000 + V` has the weights one and three.
Parity does carry over: if `V` is even, every word of `u + V` has the parity of `wt(u)`.
So a coset search of an even code rounds each bound up to that parity, which it reads from the word of the coset's syndrome ([Coset searches](brouwer_zimmermann_code.md#sec-coset-searches) in `brouwer_zimmermann_code.md`), and uses no other rounding:

- A search of the X or Z parts of errors, which searches the cosets of a classical code, rounds when that code is even.
- A search of Pauli errors by their CSS weight `max(|x|, |z|)` rounds when the words `[x | z]` of its code are even.
  It first rounds its bound on the Hamming weight of `[x | z]` up to the coset's parity, and then bounds the CSS weight by half that bound, rounded up.
  Until it has visited every word, each step of [The stopping certificate](#sec-stopping-certificate) raises its Hamming bound by at most one.
  So, with or without parity, the first step whose halved bound exceeds the query's maximum weight has a halved bound of exactly that maximum weight plus one, unless it is the step that visits every word.
  Rounding can still stop a query of an odd coset sooner: once its Hamming bound reaches twice the maximum weight, rather than exceeds it.
- A search of Pauli errors by their physical weight does not round, since every word `T(x, z)` is even, and rounding up to an even integer does not change the half, rounded up.

<a name="sec-shared-witnesses"></a>
### Sharing witnesses between CSS sectors

The distance of a CSS code is `min(d_x, d_z)`, the lesser of the distances of its X and Z sectors.
qLDPC runs one BZ search for each sector (only of the X sector if a qubit permutation maps each sector onto the other; [problems.md](problems.md#sec-one-sector)), advances them in round-robin order, one layer at a time, and lets a witness of either sector, a nontrivial logical operator, lower the `U` of both.
Each search keeps its own certificate, search bases, coverage, and weight divisor, since its certificate bounds only the unexamined words of its sector.
For example, if the X sector finds a witness of weight seven and the Z sector has only even words, the Z search stops once its own certificate reaches `U = 7`, having shown only `d_z ≥ 8`.
A joint search therefore determines `min(d_x, d_z)`, but not always both sector distances.

<a name="sec-quotient-labels"></a>
### Quotient labels for the stabilizer test

[Excluding stabilizers](#sec-excluding-stabilizers) must decide whether a candidate lies in `S`, which quotient labels do without reducing the candidate against a basis of `S`.
Let `s = dim(S)` and `q = dim(V/S)`, and take an **adapted basis** `G` of `V`, whose first `s` rows span `S` and whose other `q` rows span `V` modulo `S`.
Give its rows the **labels** `M`, an `r × q` matrix whose first `s` rows are zero and whose last `q` rows form the identity:

```text
G = [ s rows spanning S          ]      M = [ zero, s x q     ]
    [ q rows spanning V modulo S ]          [ identity, q x q ]
```

Then `aG ∈ S` exactly when `aM = 0`, so a candidate is eligible exactly when the XOR of the labels of its selected rows is nonzero; a classical search, with `S = {0}`, needs no labels.
Every row operation `G' = EG` that builds a search basis must also give `M' = EM`, or the test would accept stabilizers or reject logical operators.

<a name="sec-packed-enumeration"></a>
### Packed enumeration with pivot-column deletion

Packing stores a word in `ceil(N_stored/64)` 64-bit machine words, so that an XOR costs one operation per machine word and a weight is a population count ([arXiv:1603.06757](https://arxiv.org/abs/1603.06757v2)).
Here `N_stored`, unlike `N` ([The idea: upper bounds and lower certificates](#sec-bounds-and-certificates)), counts every column of the stored basis, even all-zero ones.
A search enumerates the **supports** of a layer, its sets of selected row indices, in lexicographic order.
Consecutive supports share their first rows, whose XOR is computed once, so extending a support costs one XOR of a row, following the saved additions of [arXiv:1603.06757](https://arxiv.org/abs/1603.06757v2).

Pivot-column deletion is a qLDPC choice.
In `G_i`, each pivot row has a one in its own pivot column of `B_i`, and the other rows are zero on `B_i`, so the word of a support of `w` rows has a **pivot one** for each selected row below `r_i` (counting from 0): `w` for an information set.
The support alone determines that count, so qLDPC deletes the pivot columns before packing, and weighs a word as its pivot ones plus the population count of its **tail**, its restriction to the remaining columns.
For example, with `r = 2` and rows `1 0 | 1 1` and `0 1 | 1 0`, the support `{0, 1}` has the tail `0 1` and the weight `2 + 1 = 3` of the full word `1 1 0 1`.

Deletion shrinks a stored row from `ceil(N_stored/64)` to `ceil((N_stored - r_i)/64)` machine words.
With `N_stored ≤ 64`, as in [Worked example](#sec-worked-example), a row takes at most one machine word either way.
Deletion saves packed words, not time in proportion, since label XORs, comparisons with `U`, and loop overhead do not shrink.

Coset searches by Hamming weight ([Terms](README.md#sec-terms) in README.md) also weigh words by their tails, in the queries that they predict to examine many words ([Heuristics and cost models](brouwer_zimmermann_code.md#sec-heuristics)).
The offset of a search basis is zero on its pivot columns ([Overlapping search bases](#sec-overlapping-bases)), so the count of pivot ones still holds, and the tail keeps every remaining column, even one on which no row has a one, since an offset can have ones there.

<a name="sec-work-estimates"></a>
### Work estimates and exhaustive enumeration

To choose between families of search bases, and to decide whether costly stages pay ([Heuristics and cost models](brouwer_zimmermann_code.md#sec-heuristics)), a search estimates its work in candidate/set evaluations.
If `(t*, j*)` is the first step after which the certificate of a family of `m` search bases, rounded as in [Rounding the certificate to a proven weight multiple](#sec-weight-multiples), reaches `U`, the estimate is

$$
W = m\sum_{w=1}^{t_* - 1} \binom{r}{w} + j_* \binom{r}{t_*}.
$$

It is pessimistic, since it assumes that no lighter witness turns up, and it is not a runtime guarantee, since runtime also depends on building the bases and on the width of packed words.
The estimate of a coset query at maximum weight `ω` sums from layer zero, the offset, to the first step whose bound exceeds `ω`, and, like the query, omits the rank-deficient bases that add nothing to that bound.
If the coset has no word within `ω`, the query examines or skips exactly that many words besides the word of its syndrome ([Coset searches](brouwer_zimmermann_code.md#sec-coset-searches) in `brouwer_zimmermann_code.md`), and otherwise at most that many.
A distance search on threads computes its estimates from `U` after whole layers, never from the measured counts of a layer split across threads ([Behavior spanning files](brouwer_zimmermann_code.md#sec-cross-file-behavior)).
This keeps the requirement "Decide only from what a layer leaves the same on any number of threads" of [Correctness requirements](brouwer_zimmermann_code.md#sec-requirements).

Exhaustive enumeration, which `method="brute_force"` runs instead of BZ, examines all `(2^q - 1) * 2^(r-q)` eligible words, for `q = dim(V/S)`, in vectorized, packed batches updated by Gray-code steps.
For classical codes of low dimension whose distance is a large fraction of their length, it can cost less than the setup and work of BZ.
Algorithm 2.11 of Grassl's chapter (page 300) makes this choice itself, by comparing the work estimate of BZ with the number of codewords of the code or of its dual.

<a name="sec-automorphisms"></a>
### Images of an information set under automorphisms

Many codes come with a group of qubit or bit permutations that preserve them, such as the translations of both halves of a multivariate bicycle code, or the cyclic shifts of a cyclic code.
Such a group lets a search certify with many information sets while scanning only one, which removes most of the cost of the layers that dominate a search.

<a name="sec-automorphism-certificate"></a>
#### The certificate of an automorphism family

An **automorphism** of a problem is a permutation `g` of the coordinates that maps `V` and `S` onto themselves.
It therefore maps eligible words to eligible words of the same weight, and an information set `I` to an information set `g(I)`.
qLDPC supplies only permutation automorphisms of a code, which permute bits or qubits and apply no single-qubit gates.
For a Pauli problem, it applies the same permutation of the qubits to each of the three parts `x`, `z`, and `x+z` of `T` in [Pauli weight as Hamming weight](#sec-pauli-weight).
A permutation that exchanges a qubit's coordinates among those parts applies a single-qubit Clifford gate; it can also satisfy the definition, and the certificate below holds for it, but qLDPC never supplies one.
For a group `Γ` of automorphisms, the images `g(I)` for `g ∈ Γ` form an **automorphism family**.

Suppose that a search has examined or skipped every word `v` with `wt(v|_I) ≤ t`, as searching `G_I` through weight `t` does.
If an eligible word `c` has at most `t` ones on some `g(I)`, then `g⁻¹(c)`, of the same weight, has at most `t` ones on `I`.
So the search has examined `g⁻¹(c)`, which then weighs at least `U`, or skipped it, which tables do only for words that weigh at least `U` ([Skipping completions by completion tables](#sec-completion-tables)).
So an eligible word lighter than `U` has at least `t+1` ones on every image, and the certificate of the family bounds its weight.
Once that certificate reaches `U`, the search therefore stops with the exact distance.
It scans only `I`, so the certificate advances once per layer, and each layer costs `binom(r, t)` evaluations however many images there are: `W = sum_{w=1}^{t*} binom(r, w)`.

A column in an orbit `O` of `Γ` lies in `|I ∩ O| |Γ| / |O|` images, so `C = |Γ| max_O |I ∩ O| / |O|`, and the certificate of [Overlapping search bases](#sec-overlapping-bases), before the rounding of [Rounding the certificate to a proven weight multiple](#sec-weight-multiples), is

$$
L(t) = \left\lceil (t+1) \min_O \frac{|O|}{|I \cap O|} \right\rceil
$$

over the orbits that meet `I`.
It is strongest when every orbit `O` has the same **share** `|I ∩ O| / |O|`, which is then `r/N`: for example, when there is a single orbit, or when the orbits are the two halves of a multivariate bicycle code and `I` has `r/2` columns in each.
Then `L(t) = ceil(N(t+1)/r)`, which no coverage-divided certificate of a family of information sets beats, since `m` sets of maximum coverage `C` have `m r ≤ C N`.

<a name="sec-scanned-set"></a>
#### Choosing the scanned set

By the formula above, the certificate grows as the **largest share** of `I`, `max_O |I ∩ O| / |O|`, falls.
qLDPC row reduces on the nonzero columns in natural order, and then exchanges pivots to lower the largest share.
An exchange pivots a row on another column with a one in that row, which gives another information set.
Let `O` be the orbit of the largest share, the one with the least column if several tie.
qLDPC looks for a row whose pivot lies in `O` and that has a one in a column of another orbit.
It pivots that row on that column only if the share of the other orbit after the exchange would be less than the current share of `O`.
After each exchange, it finds `O` again, and it repeats until no such exchange is possible.
For two orbits, the exchanges end at an information set whose largest share is the least of any information set.
Call the other orbit `E`, whose share is at most that of `O`.
The exchanges end in one of two cases, and in each no information set does better:

1. The pivot columns in `E` span all of its columns.
   Then no information set has more columns in `E`, so none has fewer in `O`.
2. One more pivot in `E` would give it a share at least that of `O`.
   Then any information set with fewer columns in `O` has more in `E`, whose share is then at least the largest share that the exchanges reached.

With more orbits, the exchanges are a heuristic.

<a name="sec-automorphism-gate"></a>
#### When a search uses an automorphism family

Finding automorphisms, verifying them, and building their family can cost more than a whole easy search, and before a light witness turns up, the estimate of [Work estimates and exhaustive enumeration](#sec-work-estimates) cannot tell whether a search is easy.
So the search of each sector starts with the families of [Search bases and rank-deficient blocks](#sec-search-bases) and [Overlapping search bases](#sec-overlapping-bases), and asks for automorphisms at most once:

- before matroid partitioning, if the search would try it ([Heuristics and cost models](brouwer_zimmermann_code.md#sec-heuristics));
- otherwise, before the first layer that would cost at least `automorphism_minimum_layer_work` evaluations.

A search asks before partitioning because switching would discard the partitioned family.
It then skips partitioning if the automorphism family would do no more work than any partitioned family could, and considers switching at the end of setup.
When it considers switching, it first weighs the rows of `G_I`, its layer one, which may lower `U`.
It switches if the work estimate of the automorphism family is less than the work that its current family has left from its next layer, and then scans `G_I` from layer two.
Both estimates depend on `U`.
At the same `U`, the work left in the current family only falls, so a search that declines would decline again.
So a search that declines reconsiders only before a later layer of at least `automorphism_minimum_layer_work` evaluations, and only if `U` has dropped since it declined.
The switch keeps the bound that the first family has reached.
An eligible word lighter than `U` is unexamined by both families, so both bounds hold for it, and the search takes the larger.
The witnesses that the first family found still count in `U`.
On threads, this decision must keep the requirement "Decide only from what a layer leaves the same on any number of threads" of [Correctness requirements](brouwer_zimmermann_code.md#sec-requirements).

Once a search has verified automorphisms, the images under `Γ` of the blocks of its current family could also bound the unexamined words lighter than `U`, through a linear program over the orbits of `Γ`.
Its optimum, rounded up to a multiple of the search's weight divisor, is at least the family's certificate.
After a switch, the images of the blocks of both families could give one such program, whose rounded optimum is at least either certificate.
Theorem 3.10 of White's thesis ([References](#sec-references)) gives a bound of this kind from the images of windows of cycles of automorphisms.
Searches solve neither program.
On 14 default distance searches, the bound from the program for the current family never changed the work of a layer.
On default searches of about 80 CSS codes, the bound from the program for both families never ended a search sooner than the larger certificate.
This guide at commit 0870f6a derives these bounds and compares them with White's theorem ([Images of the blocks of a family](https://github.com/qLDPCOrg/qLDPC/blob/0870f6ac5273aa9544490a4835868c4da691a75c/src/qldpc/codes/distance/brouwer_zimmermann.md#images-of-the-blocks-of-a-family)).

<a name="sec-automorphism-sources"></a>
#### Where automorphisms come from

With the optional `pybliss` package, the graph-automorphism tool bliss of Junttila and Kaski ([References](#sec-references)) searches the parity checks of a binary classical or quantum code for automorphisms.
For a quantum code, the class of the code can also supply other sets of checks that span the same space, through a method of `QuditCode` in [codes/common.py](../common.py) that its subclasses override.
Bliss can then find automorphisms that do not map parity checks to parity checks.
`QTCode`, for example, supplies its canonical local checks where they are few enough.
Bliss finds the automorphisms of the Tanner graph of each set, whose edges are colored by the Pauli operators of the checks.
These Tanner-graph automorphisms permute the checks and leave their Paulis unchanged, so they preserve the stabilizers, gauge operators, and logical operators, and for a CSS code those of each sector.
They form a subgroup of the automorphisms of the code, which suffices, since any group of automorphisms gives a sound certificate.
For a classical code, bliss searches only the parity check matrix, whose automorphisms permute checks and bits.
G. Lee (Appendix C.4) likewise finds the automorphisms of Tanner graphs automatically, but for the roots of a connected-cluster search ([Symmetries](connected_cluster.md#sec-symmetries); [References](#sec-references)).

With or without `pybliss`, the class of the code can supply generators of automorphisms that its construction provides.
Quantum code classes supply them through a method of `QuditCode` that they override, such as the translations of the group of an `MBCode`, and classical code classes through a method of `ClassicalCode`, both in [codes/common.py](../common.py).
Bliss finds only automorphisms of the Tanner graphs of the sets of checks that it searches, so it can miss these generators.
A cyclic code, for example, supplies its cyclic shift, which bliss misses for a `BCHCode`, whose parity checks hold only some shifts of one check.
Searches use these generators together with those that bliss finds.

A wrong generator would make the certificate unsound, so a search first checks that each maps `V` and `S` onto themselves, and raises an error if one does not.
For a general Pauli problem, a permutation of qubits moves the three thirds of the coordinates of `T` ([Pauli weight as Hamming weight](#sec-pauli-weight)) alike, and `T` commutes with it, so verifying the lifted permutation checks the original one.

<a name="sec-worked-example"></a>
### Worked example

Take the X sector of the 64-qubit `MBCode` of a test in [_brouwer_zimmermann/distance_test.py](_brouwer_zimmermann/distance_test.py).
It has 64 coordinates and dimension `r = 41`.
Its lightest eligible row has weight eight, so initially `U = 8`, and all its words are even, so `D = 2`.

A disjoint family holds `floor(64/41) = 1` information set, and the rank-deficient block of the other columns adds nothing to `L(t)` for `t ≤ 17`, so pruning drops it ([Heuristics and cost models](brouwer_zimmermann_code.md#sec-heuristics)).
Then `L(t) = t + 1`, and the search stops after layer six, when `L(6) = 7` rounds up to eight.
An overlapping family of `m = 3` information sets with `C = 2`, for example, reaches `L(4) = ceil(3·5/2) = 8` after layer four, even before rounding.
Within a layer ([The stopping certificate](#sec-stopping-certificate)), qLDPC stops sooner: after the first search basis of layer four, the bound `ceil((5+4+4)/2) = 7` rounds up to eight.

| Family | Last layer | Candidate/set evaluations in layers two and above |
| --- | ---: | ---: |
| Disjoint, one search basis | 6 | $\sum_{w=2}^{6}\binom{41}{w} = 5{,}358{,}536$ |
| Overlapping, three search bases, `C = 2` | 4 | $3\sum_{w=2}^{3}\binom{41}{w} + \binom{41}{4} = 135{,}710$ |

The counts leave out layer one, whose rows give the initial `U`, and the cost of building and packing the search bases.

<a name="sec-completion-tables"></a>
### Skipping completions by completion tables

The layers that a search scans last dominate its cost, and most of their words weigh far more than `U`.
For example, the X sector of the [[288, 12, 18]] code of [arXiv:2308.07915](https://arxiv.org/abs/2308.07915v2) has dimension `r = 150`.
So in each of its search bases, the layer of eight rows has `binom(150, 8)` supports, about `5 × 10^12`.
qLDPC skips whole groups of supports when a lower bound read from **completion tables** proves that none of their words is lighter than `U`.
A distance search can build tables for each search basis whose block is an information set, such as the one set of an automorphism family, if they pay (see [When a search uses the tables](#sec-completion-gate) below).
Coset searches by Hamming weight use them too, in the queries that they predict to examine many words ([Heuristics and cost models](brouwer_zimmermann_code.md#sec-heuristics)), even in search bases whose blocks are not information sets.

<a name="sec-completion-bound"></a>
#### The bound

Write a search basis whose block is an information set as `G = [I_r | P]`, with the pivot columns first, so that the word of a support of `w` rows weighs `w` plus the weight of its tail, the sum of its rows of `P` ([Packed enumeration with pivot-column deletion](#sec-packed-enumeration)).
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
The pivot ones must number exactly `w`, as only in an information set ([Search bases and rank-deficient blocks](#sec-search-bases)), or be bounded as in [Coset searches](#sec-completion-cosets) below.

<a name="sec-completion-lookups"></a>
#### Tables and lookups

For each block and each `s`, a table stores the distance from every value of `p` on the block to the nearest sum of a pair from row `s` on.
The table for `s` starts from that for `s + 1`: the new pairs of row `s` lower the distance at their sums to zero, and a breadth-first search from those sums lowers the distances around them.
The breadth-first search halts at each value whose distance it does not lower, since the distances beyond are already at most that distance plus their steps from it.
With blocks of `β` columns, the tables take `(r - 1) Σ_b 2^|J_b|` bytes, and building them costs about `2^β β^2` steps for each block, plus a step for each pair of rows on each block, and writing each table.
A lookup reads one entry per block.
The scan looks up the bound each time it tries a new first row `s` for the pair that completes a prefix, and if `w + B ≥ U`, skips every completion of the prefix from row `s` on.
Until it skips, the scan looks up the bounds of one prefix for consecutive `s`, so the tables store the entries of every `s` for one value on a block next to each other, where these lookups share cache lines.
A bound for a larger `s` is stronger, so a support is skipped exactly if the bound of its own `s` reaches `U - w`.
The threads of a split layer ([Behavior spanning files](brouwer_zimmermann_code.md#sec-cross-file-behavior)), and coset queries that run at once, each read shared tables, so a table must be complete before any scan reads it, and unchanged while scans do ([Correctness requirements](brouwer_zimmermann_code.md#sec-requirements)).
Heuristic search in artificial intelligence calls such tables additive pattern databases: their entries for disjoint parts of a problem add up to a lower bound.
Korf and Felner describe them, as do Felner, Korf, and Hanan ([References](#sec-references)).

<a name="sec-completion-contracts"></a>
#### Certificates and contracts

A skipped word weighs at least the `U` of the moment it was skipped, and hence any later `U`, so an eligible word lighter than `U` has been neither skipped nor examined, and the arguments of [The stopping certificate](#sec-stopping-certificate) hold for it.
The certificate `L` bounds only such words, and can exceed the weights of skipped words, as it can those of words examined through their images ([Images of an information set under automorphisms](#sec-automorphisms)), so the distance is at least `min(L, U)`, not `L`.
Since no skipped word is lighter than `U` at any later time, a scan with tables on one thread finds the same witnesses in the same order as one without, an early exit ([Behavior spanning files](brouwer_zimmermann_code.md#sec-cross-file-behavior)) returns the same weight, and the search stops when it would without skipping.
On several threads, the witnesses and their order can differ, and so can the result of an early exit, which still lies between the distance and the cutoff.
`U` after each layer, and so where the search stops, is the same as on one thread ([Behavior spanning files](brouwer_zimmermann_code.md#sec-cross-file-behavior)).
These arguments hold for each search basis, with its own tables, so a search can skip in each basis of a layer that it scans one at a time, and stop after any basis whose step `(t, j)` certifies `U` by the certificate of its family.

Words that weigh exactly `U` can be skipped, since they cannot lower `U`.
But a search whose answer depends on the words up to a weight may never skip a word of that weight, so it replaces `U` with the least bound that proves a word heavier than that weight.
A count of the words of weight `t` ([Correctness requirements](brouwer_zimmermann_code.md#sec-requirements)) skips only completions that all weigh more than `t`, as if its best weight were `t + 1`.
So none of its skipped words needs counting, and the certificate of its last step, which exceeds `t`, rules out any word of weight `t` that it neither visits nor skips.
A coset query of maximum weight `ω` skips with the limit `h(ω)` of [Coset searches](#sec-completion-cosets) below.

<a name="sec-completion-cosets"></a>
#### Coset searches

In a coset search by Hamming weight, the queries that the search predicts to examine many words ([Heuristics and cost models](brouwer_zimmermann_code.md#sec-heuristics)) skip completions by tables in layers of two or more rows.
The scan seeds every prefix with the tail `o` of the offset ([Overlapping search bases](#sec-overlapping-bases)), so the tables bound `wt(o + p + P_i + P_j)`.
The tables cover the tail columns up to the last on which a row has a one.
Ones of `o` beyond that column only weaken the bound.
The block of the search basis need not be an information set.
If the block has rank `r_i`, the first `r_i` rows of the search basis are its pivot rows ([Search bases and rank-deficient blocks](#sec-search-bases)).
The word of a support has one pivot one for each pivot row in the support, and none for the other rows of the support ([Packed enumeration with pivot-column deletion](#sec-packed-enumeration)).
So a completion has one pivot one for each pivot row of its prefix.
The pair of rows that completes the prefix adds at least `max(0, 2 - (r - r_i))` more, since only `r - r_i` rows are not pivot rows.
The bound adds the sum of these two counts to `B` in place of `w`; for an information set, that sum is `w`.

A query of maximum weight `ω` skips the completions whose bound reaches `h(ω)`, the least bound that proves a word of the coset heavier than `ω`.
With the **weight factor** `η` of [Coset searches](brouwer_zimmermann_code.md#sec-coset-searches) in `brouwer_zimmermann_code.md`, two in a search of Pauli errors and one in a search of X or Z parts, `h(ω)` is `η ω + 1`, or `η ω` if the search rounds to the parity of the coset and `η ω` has the other parity.
No skipped word lies within `ω`, and the query visits the other words in the order of a scan without tables, so it finds the same first word within `ω`, if there is one.
Otherwise, it returns the lightest word that it weighed, possibly heavier than without tables.
If it skipped any word, it also returns the lower bound `R`: `h(ω)`, rounded up to the coset's parity where the search rounds, then divided by `η` and rounded up ([Rounding the certificate to a proven weight multiple](#sec-weight-multiples)).
Every word that the query skipped, weighed, or did not visit lies beyond `ω`, so it weighs at least `R`, the least weight beyond `ω` that a word of the coset can have.
A query that skipped nothing returns the bound of a scan without tables, which is never below `R`.

Until a query finds a word within `ω`, its lightest word has a Hamming weight of at least `h(ω)`, so skipping at `h(ω)` skips at least as much as skipping at that weight, as a distance search does.
It therefore weighs no more words, at the price of the heavier word and smaller bound above.
A search for the minimum weight of a coset skips at `h(ω)` too until it finds a word within `ω`, and then at a limit that falls as it finds lighter words ([Minimum weights of cosets](#sec-coset-min-weights)).

<a name="sec-completion-gate"></a>
#### When a search uses the tables

Distance searches and counts use completion tables only on search bases whose blocks are information sets.
A distance search scans each search basis of a layer by itself with the compiled scans of [_brouwer_zimmermann/affine_search.py](_brouwer_zimmermann/affine_search.py), on one thread or, for a costly layer, on several ([Behavior spanning files](brouwer_zimmermann_code.md#sec-cross-file-behavior)).
A count scans with its own kernels in [_counting/scan.py](_counting/scan.py), and chooses tables for each of its steps `(t, j)` with `t ≥ 2`, by the same four rules as a distance search (below).

Tables cost memory, a build, and lookups, and save only the supports that they skip, so they pay on large layers where they skip many supports, but not on small layers or in easy searches.
The share that they skip depends on the code, the scanned search basis, and the order of its rows, so tables can pay on one layer and not on a layer of the same size of another code.
They save at most a constant factor in a layer, since the scan still looks up a bound for each prefix, and prefixes grow like supports for fixed `w/r`; nor do they reduce the number of layers.
So, just before scanning a search basis in a layer, a search:

1. considers tables only for a layer of at least `completion_bound_minimum_layer_work` supports ([Heuristics and cost models](brouwer_zimmermann_code.md#sec-heuristics));
2. bounds at most `completion_bound_samples` random supports directly from the rows, and takes a conservative lower bound on the share that tables would skip;
3. predicts the cost of the layer from this share, the lookups, and any build, with rough coefficients of `SearchTuning`: a lookup costs a few scans of a support for each block, and more for larger tables, which miss the caches more often;
4. uses tables if the predicted cost is at most `completion_bound_max_cost_percent` percent of the cost without them.

On threads, this choice must keep the requirement "Decide only from what a layer leaves the same on any number of threads" of [Correctness requirements](brouwer_zimmermann_code.md#sec-requirements).

Tables use blocks of `completion_bound_block_bits` columns, and a basis whose tables would exceed `completion_bound_max_table_bytes` gets none.
A distance search keeps the tables of one basis at a time, for its later layers, and drops them before another basis chooses tables or when its family changes, so a search whose layers scan several bases rebuilds their tables in each layer, which the prediction counts.
A joint search can hold twice `completion_bound_max_table_bytes`, half for each sector, until it finishes.
Within `completion_bound_max_table_bytes`, a count keeps the tables of the search bases that it has used, besides those that it is building, and rebuilds them in each layer when they do not all fit.

<a name="sec-coset-min-weights"></a>
### Minimum weights of cosets

The residual-weight estimator of `qldpc.experimental` needs the minimum weight of each coset that it meets, up to a maximum weight `ω`, or `ω + 1` for a coset without a word within `ω`.
Queries at increasing maximum weights would each repeat the scan of the one before, so the coset searches find this weight with one search.
The search keeps `U`, the lesser of `ω + 1` and the weight of the lightest word found so far, starting from the word of its syndrome ([Coset searches](brouwer_zimmermann_code.md#sec-coset-searches) in `brouwer_zimmermann_code.md`).
It stops after the first step whose bound reaches `U`.
It also stops within a step once `U` is at most the bound of the step before, since every unvisited word then weighs at least `U`.
`U` is then the minimum weight, if it is at most `ω`.
The search finds its stopping step, the first step whose bound reaches `U`, by a binary search of the steps, and moves that step earlier whenever `U` falls.
It skips the search bases that add nothing to its bound, and these add nothing to any earlier bound either.
It uses the family and the completion tables of a query at `min(ω, n)`, for `n` qubits ([Heuristics and cost models](brouwer_zimmermann_code.md#sec-heuristics)), and since it stops no later than that query, those tables cover every layer that it scans.
Batches run these searches on several threads at once, so they cache no stopping step in the planning buffer that queries share ([Coset searches](brouwer_zimmermann_code.md#sec-coset-searches) in `brouwer_zimmermann_code.md`), as "Keep coset queries reproducible" of [Correctness requirements](brouwer_zimmermann_code.md#sec-requirements) requires.

In a search by Hamming weight, the scans keep only the words lighter than an **incumbent**: `h(ω)` of [Skipping completions by completion tables](#sec-completion-tables) until the search finds a word within `ω`, and then `η U`, the Hamming weight of the lightest word.
In a search of Pauli errors by physical weight, whose words `T(x, z)` are all even, and of X or Z parts of an even code, every word of the coset has the parity of `η U`, so scans then lower the incumbent to `η U - 1`, keeping the same words and letting tables skip more.
Completion tables skip the completions whose bound reaches the incumbent, or, once a scan finds a lighter word, the Hamming weight of that word.
So the search skips no word lighter than its answer, and in a coset without a word within `ω`, it weighs and skips exactly the words of a query at `ω`.
A search of Pauli errors by CSS weight weighs words by that weight, and keeps those lighter than `U`.

For the CSS weight on a CSS state, `CosetLeaderSearch` searches the X part, then, if its weight `w_x` is at most `ω`, the Z part with the floor `w_x`, where any word within `w_x` ends the search ([problems.md](problems.md#sec-coset-leader-search)).

<a name="sec-counting"></a>
### Counting logical operators

`CSSLogicalCounter` of [logical_counts.py](logical_counts.py) counts the nontrivial logical operators of a given weight `t` in a binary CSS code, where `t` is by default the distance `d`.
Its BZ method, which `"auto"` takes, scans each sector in layers of a family of search bases, but stops and skips only as [Skipping completions by completion tables](#sec-completion-tables) and [Correctness requirements](brouwer_zimmermann_code.md#sec-requirements) allow, so that it visits every eligible word of weight `t`.
A word lies in the layer of its coefficient weight in each search basis, so one step visits it first, and only that step counts it.
Bouyuklieva and Bouyukliev (page 6) likewise list the words of a given weight without repetition, keeping each only in the first of their search bases that visits it ([References](#sec-references)).
A count uses no automorphisms, since it counts each word individually and does not track orbit sizes.
The X and Z sectors of a self-dual code, whose X and Z checks span the same space, are equal.
So a counter scans only one of them if it knows the code to be self-dual: if it was built from such checks, or deformed from such a counter by a gate that acts alike on X and Z.
[problems.md](problems.md#sec-macwilliams) explains the MacWilliams method, which counts from the weight distributions of the checks.

<a name="sec-pauli-counts"></a>
#### Full Pauli counts

A nontrivial logical Pauli operator `(x, z)` of weight `d` and support `Q` takes one of two disjoint forms, since `x`, which commutes with the Z stabilizers, is a nontrivial X logical or an X gauge operator:

- `x` is a nontrivial X logical, which weighs at least `d` and so has support `Q`, and `z` is one of the `2^(d - rank(S_X[:, Q]))` words supported in `Q` that commute with the X stabilizers `S_X`.
- `x` is an X gauge operator, so `z` is a nontrivial Z logical, which has support `Q` for the same reason, and `x` is one of the `2^(rank(G_X) - rank(G_X[:, ~Q]))` words supported in `Q` in the span of the X gauge generators `G_X`.

So the scans of the X and Z sectors return histograms of these exponents over their words of weight `d`.
In a self-dual code, no nonzero gauge operator lies within the support `Q` of a nontrivial X logical `x` of weight `d`, since adding the gauge operator to `x` would give a lighter nontrivial logical.
So a nonzero word within `Q` that commutes with `S_X` is not a gauge operator but a nontrivial logical of weight at most `d`, which must be `x` itself.
Hence in the first form `z` is `0` or `x`, and in the second `x` is `0`: `Q` carries exactly three operators, of types X, Y, and Z, and the full count is three times the X count.
Above `d`, an operator need not have the support of a nontrivial X or Z logical, so a full count raises an error if its scans find a nontrivial X or Z logical lighter than `t`.

<a name="sec-deformed-counters"></a>
#### Deformed counters

A local CSS-preserving Clifford gate maps the X part of an operator on its target qubits by an invertible binary matrix `matrix_x`, and the Z part by `(matrix_x^-1)^T`, which preserves commutation.
So the gate maps the gauge operators, stabilizers, and sectors of a code onto those of the deformed code, and a basis of a sector, row by row, to a basis of the new sector with the same quotient labels ([Quotient labels for the stabilizer test](#sec-quotient-labels)).
For each basis family that a counter keeps, the deformed counter reduces each mapped search basis again on the pivot columns of its block, with its labels.
If every search basis keeps its rank there, the family keeps its blocks, ranks, and coverage; otherwise the counter builds the family again.
A gate can change the parity of words, so the counter infers the weight divisor of [Rounding the certificate to a proven weight multiple](#sec-weight-multiples) again.
It can also change the distance and the weights of check words, so the counter searches for the distance again, and the MacWilliams method recomputes the weight distributions of the checks.

<a name="sec-references"></a>
## References

- [arXiv:1603.06757v2](https://arxiv.org/abs/1603.06757v2) (Algorithm 1): Section 2 and Algorithm 1 give the classical disjoint and rank-deficient bound.
- M. Grassl, "Searching for linear codes with large minimum distance," in *Discovering Mathematics with Magma*, Springer, 2006, pp. 287-313 ([doi:10.1007/978-3-540-37634-7_13](https://doi.org/10.1007/978-3-540-37634-7_13)): Algorithm 2.5 gives the certificate for overlapping information sets, footnote 2 on page 295 the certificate within a layer, page 299 the rounding to a weight divisor, and Algorithm 2.11 the choice between BZ and the weight distribution.
- P. Lisoněk and L. Trummer, "Algorithms for the minimum weight of linear codes," *Advances in Mathematics of Communications* 10(1), 195-207, 2016 ([doi:10.3934/amc.2016.10.195](https://doi.org/10.3934/amc.2016.10.195)): equation (5) gives the certificate within a layer, Proposition 5 the coverage-divided certificate, and Section 5.1 builds families of information sets with bounded coverage by matroid partitioning.
- G. White, *Enumeration-Based Algorithms in Linear Coding Theory*, PhD thesis, University of Sydney, 2006: Section 3.3 and Theorem 3.10 (pages 81-84) bound the weight of unexamined words of a code with known automorphisms by a sum over disjoint sets of columns: one term for each chosen cycle of an automorphism longer than `r`, from the images of a window of the cycle within an information set, rounded up on its own, and the usual terms on the other sets ([Images of an information set under automorphisms](#sec-automorphisms)).
- [arXiv:2408.10743v2](https://arxiv.org/abs/2408.10743v2): Section 2.4 describes the `Saved_isometry` symplectic-to-Hamming construction.
- A. M. Tillmann, "Computing the spark: mixed-integer programming for the (vector) matroid girth problem," *Computational Optimization and Applications* 74, 387-441, 2019 ([doi:10.1007/s10589-019-00114-9](https://doi.org/10.1007/s10589-019-00114-9)): Section 6 of its 2018 preprint ([Optimization Online 6525](https://optimization-online.org/2018/03/6525/)), with Lemma 13 and program (13), writes the girth of a matroid, the size of its smallest dependent set, as an integer program that asks a set to meet the complement of every basis, and program (14) applies it to the spark of a real matrix.
  For the columns of a binary parity-check matrix, with linear dependence over GF(2), the girth is the distance of the code, and the complements of bases are its information sets ([Overlapping search bases](#sec-overlapping-bases)).
- [gap-packages/guava](https://github.com/gap-packages/guava/blob/e3bce700b7dfd100b17241ebfd95b9cfe0757060/lib/codeops.gi#L568-L686), `lib/codeops.gi`, lines 568-686 at commit `e3bce70`: `MinimumDistance(C, w)` searches the coset `w + C` of a linear code `C` over disjoint information sets, each with an offset of the coset that is zero on that set ([Families for coset searches](#sec-coset-families)).
- [Quantinuum/Qubitserf](https://github.com/Quantinuum/Qubitserf/blob/04b9349a2d421815c35b2a3d53878167e7237f47/cpp/distfind/css.cpp#L150-L174), by S. Cercelescu, `cpp/distfind/css.cpp`, lines 150-174 at commit `04b9349`: `coset_problem` makes the reduction of [Families for coset searches](#sec-coset-families), and Qubitserf's BZ search runs the result by default; lines 9-36 of `cpp/distfind/op_weight.cpp` use `coset_problem` for the least weights of the X and Z parts of a Pauli operator modulo a CSS group.
- T. Junttila and P. Kaski, "Engineering an efficient canonical labeling tool for large and sparse graphs," in *Proceedings of the Ninth Workshop on Algorithm Engineering and Experiments (ALENEX)*, SIAM, 2007, pp. 135-149 ([doi:10.1137/1.9781611972870.13](https://doi.org/10.1137/1.9781611972870.13)), and "Conflict propagation and component recursion for canonical labeling," in *Theory and Practice of Algorithms in (Computer) Systems (TAPAS)*, Springer, 2011, pp. 151-162 ([doi:10.1007/978-3-642-19754-3_16](https://doi.org/10.1007/978-3-642-19754-3_16)): the bliss tool, which `pybliss` wraps ([Images of an information set under automorphisms](#sec-automorphisms)).
- G. Lee, "Bicycle race: small two-block codes on the Pareto frontier" ([arXiv:2610.13880v2](https://arxiv.org/abs/2610.13880v2)): Appendix C.4 takes the roots of a connected-cluster search from the automorphisms of the Tanner graph, which pynauty finds ([Images of an information set under automorphisms](#sec-automorphisms)).
- R. E. Korf and A. Felner, "Disjoint pattern database heuristics," *Artificial Intelligence* 134, 9-22, 2002 ([doi:10.1016/S0004-3702(01)00092-3](https://doi.org/10.1016/S0004-3702%2801%2900092-3)), and A. Felner, R. E. Korf, and S. Hanan, "Additive pattern database heuristics," *Journal of Artificial Intelligence Research* 22, 279-318, 2004 ([doi:10.1613/jair.1480](https://doi.org/10.1613/jair.1480)): disjoint and additive pattern databases ([Skipping completions by completion tables](#sec-completion-tables)).
- S. Bouyuklieva and I. Bouyukliev, "An extension of the Brouwer-Zimmermann algorithm for calculating the minimum weight of a linear code," *Mathematics* 9(19), 2354, 2021 ([doi:10.3390/math9192354](https://doi.org/10.3390/math9192354)): page 6 lists the codewords of a given weight without repetition ([Counting logical operators](#sec-counting)).
- The implementation and its tests: "[Where to look](README.md#sec-where-to-look)" in README.md says where to start for each part, and the tests of the [correctness requirements](brouwer_zimmermann_code.md#sec-requirements) cite that section in their docstrings.
