# Exact certification of bounded fault propagation in qLDPC

## 1. Purpose and scope

A noisy Clifford circuit that prepares a stabilizer state has bounded fault propagation if every set of faults leaves an output error equivalent to an error of weight at most the number of faults.
With post-selection, only the sets of faults that fire none of the post-selection detectors count.
This guide explains why the certifier can decide this exactly, and how it searches for a violation.
It is for contributors who need to understand or modify the certifier in this directory.
It is organized as follows:

- [Section 2](#2-definitions) defines faults, patterns, acceptance, syndromes, and weight.
- [Section 3](#3-facts-that-the-search-relies-on) proves the facts that the search relies on.
- [Section 4](#4-the-search) describes the search and why it finds a smallest violation.

## 2. Definitions

### 2.1 Faults, patterns, and acceptance

A **fault** is one error mechanism of the circuit's detector error model (DEM), and every fault costs one.
A **pattern** is a set of faults, and its **size** `|F|` is the number of faults in it.

Let `P` be the parity check matrix of the post-selection detectors for the faults: a binary matrix with one row for each post-selection detector and one column `p_j` for each fault `j`, whose entry is 1 if fault `j` flips that detector.
Flips add over GF(2), so a pattern `F` flips a detector exactly when an odd number of its faults do.
The **flags** of `F` are the sum of the columns `p_j` over the faults `j` in `F`, so the flags of a single fault `j` are its column `p_j`.
A pattern is **accepted** if its flags are zero, meaning that it flips none of the post-selection detectors.
Without post-selection, `P` has no rows and every pattern is accepted.

### 2.2 Syndromes and weight

Let `n` be the number of output qubits, and let `g_1, ..., g_n` be independent commuting Pauli operators that generate the stabilizer group of the prepared state, logical stabilizers included.
The **syndrome** of a Pauli error is the binary vector whose entry `i` is 1 if the error anticommutes with `g_i`.
Two errors are **equivalent** if they differ by an element of the stabilizer group, because then they leave the same state.

Pauli errors propagate linearly through a Clifford circuit.
Each fault therefore leaves a definite Pauli error on the output qubits, up to phase, with a syndrome `s_j`.
A pattern leaves the product of the errors of its faults, and the syndrome of this product is `S(F)`, the sum of the `s_j` over the faults `j` in `F`.

The **weight** of an error is its physical Pauli weight, the number of qubits that it acts on, or its CSS weight `max(|x|, |z|)`, where `x` and `z` are the X and Z parts of the error.
For a syndrome `s`, let `μ(s)` be the smallest weight of a Pauli error with syndrome `s`.
A pattern **violates** the bound if `μ(S(F)) > |F|`.
Fault propagation is bounded if no accepted nonempty pattern violates the bound, with the patterns limited to size at most `K` if there is a limit.

## 3. Facts that the search relies on

### 3.1 Syndromes identify equivalent errors

Two errors with the same syndrome differ by an operator that commutes with all `n` independent stabilizers of the `n`-qubit state.
Every such operator is a stabilizer up to a phase, so the two errors are equivalent.
Hence `μ(S(F))` is the weight of the lightest error that is equivalent to the error that the pattern causes.

### 3.2 Weight is subadditive

The weight of a product of two errors is at most the sum of their weights, so `μ(s + t) ≤ μ(s) + μ(t)`.
For physical weight, the product acts on no more qubits than the two errors together.
For CSS weight, `max(a + c, b + d) ≤ max(a, b) + max(c, d)`, where `(a, b)` and `(c, d)` are the X and Z weights of the two errors.
Since syndromes add, `μ(S(F))` is at most the sum of `μ(s_j)` over the faults `j` in `F`.

### 3.3 Without post-selection, single faults suffice

If every fault alone has `μ(s_j) ≤ 1`, then by subadditivity ([Section 3.2](#32-weight-is-subadditive)) `μ(S(F)) ≤ |F|` for every pattern `F`.
Without post-selection, the converse holds too, because a fault with `μ(s_j) ≥ 2` is itself an accepted violating pattern.
So single faults decide the bound exactly, for any limit `K ≥ 1`.
With post-selection, a fault that flips a post-selection detector is not accepted alone, so the converse can fail, and Sections [3.4](#34-irreducible-accepted-patterns-suffice) to [3.7](#37-with-independent-columns-pairs-suffice) are needed to decide the bound.

### 3.4 Irreducible accepted patterns suffice

An **irreducible accepted pattern** is a nonempty accepted pattern that has no accepted nonempty proper subset.
If `G` is an accepted pattern inside an accepted pattern `F`, the flags of `F \ G` are the sum of the flags of `F` and `G`, so `F \ G` is accepted too.
Repeating this shows that every accepted nonempty pattern is a disjoint union of irreducible accepted patterns.
If every irreducible accepted pattern `F_i` meets the bound, then [Section 3.2](#32-weight-is-subadditive) gives `μ(S(F)) ≤ Σ μ(S(F_i)) ≤ Σ |F_i| = |F|` for their union `F`.
Each `F_i` is no larger than `F`, so checking the irreducible accepted patterns of size at most `K` covers every accepted pattern of size at most `K`.

A violating pattern with the fewest faults is irreducible.
Otherwise it is a disjoint union of at least two irreducible accepted patterns, and one of them violates the bound with fewer faults, because the union meets the bound if all of them do.

### 3.5 Irreducible accepted patterns have at most `rank(P) + 1` faults

Let `P` be the parity check matrix of the post-selection detectors for the faults (as in [Section 2.1](#21-faults-patterns-and-acceptance)).
For a pattern `F`, let `P[:, F]` be the columns of `P` for the faults in `F`.
The nonzero null vectors of `P[:, F]` correspond one-to-one, through their supports, to the nonempty accepted patterns inside `F`.
A nonzero null vector other than the all-ones vector has a support that is a proper subset of `F`.
So `F` is an irreducible accepted pattern exactly when the null space of `P[:, F]` is spanned by the all-ones vector, which has dimension one.
The number of columns of `P[:, F]` is its rank plus the dimension of its null space, so `|F| = rank(P[:, F]) + 1 ≤ rank(P) + 1`.
So patterns of more than `rank(P) + 1` faults never need checking, and neither do patterns above the size limit `K`, if there is one.

### 3.6 Duplicate and zero signatures can be dropped

The **signature** of fault `j` is the pair `(p_j, s_j)`.
The flags and syndrome of a pattern depend only on the signatures of its faults, so faults with the same signature are interchangeable.
Dropping a fault with zero signature, or two faults with equal signatures, which cancel, leaves the flags and syndrome of a pattern unchanged and lowers its size.
So a violating pattern with the fewest faults has no such faults, and it is enough to keep one fault for each distinct nonzero signature.
Keeping the first one in DEM order makes a counterexample name the earliest faults.

### 3.7 With linearly independent columns, pairs of faults suffice

Suppose that the distinct nonzero columns of `P` are linearly independent, meaning that no nonempty set of them sums to zero.
This holds, for example, when every fault flips at most one post-selection detector.
Two faults may have equal columns, since the statement about linear independence is only about the distinct columns.
Then the irreducible accepted patterns of more than one fault are exactly the pairs of faults with the same nonzero column.

We prove this in two parts.

1. Such a pair is accepted and irreducible.
   - It is accepted, because its two equal columns cancel.
   - It is irreducible, because a single fault with a nonzero column is not accepted.
2. Every irreducible accepted pattern `F` with `|F| > 1` is such a pair.
   First, no fault of `F` has a zero column, since that fault alone would be an accepted proper subset of `F`.
   Then there are two cases:
   - No two faults of `F` have the same column.
     Then the columns of `F` are distinct and nonzero, so by linear independence they cannot sum to zero, which means that `F` cannot be accepted.
     This is a contradiction, so this case cannot occur.
   - Two faults of `F` have the same column.
     They form an accepted pattern inside `F`, so by irreducibility `F` consists of just these two faults, whose common column is nonzero.

So when the distinct nonzero columns are linearly independent, the only irreducible accepted patterns are the faults with a zero column and the pairs of faults with the same nonzero column, and no pattern of three or more faults needs checking.

## 4. The search

### 4.1 Enumerating irreducible accepted patterns

A fault with zero flags is an irreducible accepted pattern by itself, and it is in no larger one, because it would be an accepted proper subset of it.
The larger irreducible accepted patterns consist of faults with nonzero flags.
Order them by their position in the DEM, and write an irreducible accepted pattern `F` of size `m ≥ 2` in this order as `j_1 < ... < j_m`.
By [Section 3.5](#35-irreducible-accepted-patterns-have-at-most-rankp--1-faults), `rank(P[:, F]) = m - 1`, and the sum of the columns is zero, so any `m - 1` of the columns are independent and span the same space as all `m`.
So the **prefix** `j_1, ..., j_{m-1}` is independent, and the **closing** fault `j_m` has flags equal to the sum of the prefix flags.
Conversely, an independent prefix and a later fault with these flags form an accepted pattern of rank `m - 1`, which is irreducible by [Section 3.5](#35-irreducible-accepted-patterns-have-at-most-rankp--1-faults).
Requiring the closing fault to have the largest position lists each irreducible accepted pattern exactly once.

The search therefore extends independent prefixes one fault at a time, in increasing order.
It keeps an echelon basis of the prefix flags, with descending leading bits, to reject a new fault whose flags depend on the prefix.
No irreducible accepted pattern extends a dependent prefix, because every prefix of an irreducible accepted pattern is independent.
A table of the faults by their flags gives the closing faults that come after a prefix of length `m - 1`.

### 4.2 Ascending sizes find a smallest violation

The search checks the faults with zero flags first, and then the irreducible accepted patterns of size 2, 3, and so on, up to `rank(P) + 1`, limited by `K` if there is a limit, and by 2 if the distinct nonzero columns of `P` are linearly independent, as in [Section 3.7](#37-with-independent-columns-pairs-suffice).
It skips the larger sizes if every fault with nonzero flags has `μ(s_j) ≤ 1`, by [Section 3.3](#33-without-post-selection-single-faults-suffice).
It reports the first violation that it finds.

Let `m*` be the fewest faults in any violating accepted pattern of size at most `K`.
By [Section 3.4](#34-irreducible-accepted-patterns-suffice), a violating irreducible accepted pattern of size `m*` exists, and by [Section 3.6](#36-duplicate-and-zero-signatures-can-be-dropped) it remains one when each of its faults is replaced by the kept fault with the same signature.
An irreducible accepted pattern has at most `rank(P) + 1` faults by [Section 3.5](#35-irreducible-accepted-patterns-have-at-most-rankp--1-faults), and at most 2 if the distinct nonzero columns of `P` are linearly independent, by [Section 3.7](#37-with-independent-columns-pairs-suffice), so the search covers size `m*`.
No irreducible accepted pattern of a smaller size violates the bound, so the search does not stop before size `m*`, and it stops with a violation at that size.
The reported pattern therefore has the fewest faults of any violating pattern.

### 4.3 Deciding a pattern

For each pattern that it checks, the search decides whether `μ(S(F)) ≤ |F|`, with the size of the pattern as the weight threshold.
The decision comes from `PauliCosetSearch` in `src/qldpc/codes/distance/_bounded_weight_search.py`, which computes minimum weights of syndrome classes exactly.
For a violation, it also returns an error with the syndrome and a lower bound on `μ(S(F))`, which exceeds `|F|`.
