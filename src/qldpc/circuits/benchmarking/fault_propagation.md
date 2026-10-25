# Exact certification of bounded fault propagation in qLDPC

## 1. Purpose and scope

A noisy Clifford circuit that prepares a stabilizer state has bounded fault propagation if every set of faults leaves an output error that is equivalent to an error whose weight is at most the number of faults.
This guide explains why qLDPC can decide this exactly when detectors are post-selected, so that only sets of faults that fire none of them count, and how it searches for a violation.
It is written for contributors who need to understand, debug, or modify the certifier in this directory.
It is organized as follows:

- [Section 2](#2-definitions) defines faults, patterns, acceptance, syndromes, and weight.
- [Section 3](#3-facts-that-the-search-relies-on) proves the facts that the search relies on.
- [Section 4](#4-the-search) describes the search and why it finds a smallest violation.

## 2. Definitions

### 2.1 Faults, patterns, and acceptance

A **fault** is one error mechanism of the circuit's detector error model (DEM), and every fault costs one.
A **pattern** is a set of faults, and its **size** `|F|` is the number of faults that it contains.
Every set of faults is a pattern, and a fault occurs at most once in a pattern.

Let `P` be the binary matrix with one row for each post-selected detector and one column `p_j` for each fault `j`, whose entry is 1 if fault `j` flips that detector.
Flips add over GF(2), so a pattern `F` flips a detector exactly when an odd number of its faults do, and the **flags** of `F` are the sum of the columns `p_j` over the faults `j` in `F`.
A pattern is **accepted** if its flags are zero, meaning that it flips none of the post-selected detectors.
Without post-selection, `P` has no rows and every pattern is accepted.

### 2.2 Syndromes and weight

Let `n` be the number of output qubits, and let `g_1, ..., g_n` be independent commuting Pauli operators that generate the stabilizer group of the prepared state, logical stabilizers included.
The **syndrome** of a Pauli error is the binary vector whose entry `i` is 1 if the error anticommutes with `g_i`.
Two errors are **equivalent** if they differ by an element of the stabilizer group, because then they leave the same state.

Pauli errors propagate linearly through a Clifford circuit.
Each fault therefore leaves a definite Pauli error on the output qubits, up to phase, with a syndrome `s_j`.
A pattern leaves the product of the errors of its faults, and this product has syndrome `S(F)`, the sum of the `s_j` over GF(2) for the faults `j` in `F`.

The **weight** of an error is its physical Pauli weight, the number of qubits that it acts on, or its CSS weight `max(|x|, |z|)`, where `x` and `z` are the X and Z parts of the error.
For a syndrome `s`, let `μ(s)` be the smallest weight of a Pauli error with syndrome `s`.
A pattern **violates** the bound if `μ(S(F)) > |F|`.
Fault propagation is bounded if no accepted nonempty pattern violates the bound, with the patterns restricted to size at most a limit `K` if one is given.

## 3. Facts that the search relies on

### 3.1 Syndromes identify equivalent errors

Two errors with the same syndrome differ by an operator that commutes with all `n` independent stabilizers of the `n`-qubit state.
Every such operator is itself a stabilizer, up to a phase, so the two errors are equivalent.
Hence `μ(S(F))` is the weight of the lightest error that is equivalent to the error that the pattern causes.

### 3.2 Weight is subadditive

The weight of a product of two errors is at most the sum of their weights, so `μ(s + t) ≤ μ(s) + μ(t)`.
For physical weight, the product acts on no more qubits than the two errors together.
For CSS weight, `max(a + c, b + d) ≤ max(a, b) + max(c, d)`, where `(a, b)` and `(c, d)` are the X and Z weights of the two errors.
Since syndromes add, `μ(S(F))` is at most the sum of `μ(s_j)` over the faults `j` in `F`.

### 3.3 Without post-selection, single faults suffice

If every fault alone has `μ(s_j) ≤ 1`, then Section 3.2 gives `μ(S(F)) ≤ |F|` for every pattern, and no pattern needs checking.
Without post-selection, a fault with `μ(s_j) ≥ 2` is itself an accepted violating pattern.
The bound then holds exactly when `μ(s_j) ≤ 1` for every fault `j`, and a limit `K ≥ 1` cannot change this.
With post-selection, a fault that flips a post-selected detector is not accepted alone, so the converse fails and Sections 3.4 to 3.7 are needed.

### 3.4 Minimal patterns suffice

A pattern is **minimal** if it is accepted, nonempty, and contains no smaller accepted nonempty pattern.
If `G` is an accepted pattern inside an accepted pattern `F`, the flags of `F \ G` are the sum of the flags of `F` and `G`, so `F \ G` is accepted too.
Repeating this shows that every accepted nonempty pattern is a disjoint union of minimal patterns.
If every minimal pattern `F_i` meets the bound, then Section 3.2 gives `μ(S(F)) ≤ Σ μ(S(F_i)) ≤ Σ |F_i| = |F|` for their union `F`.
Each `F_i` has at most as many faults as `F`, so checking the minimal patterns of size at most `K` covers every accepted pattern of size at most `K`.

A violating pattern with the fewest faults is minimal.
Otherwise it is a disjoint union of at least two minimal patterns, and one of them would violate the bound with fewer faults, because the union meets the bound if all of them do.

### 3.5 Minimal patterns have at most rank + 1 faults

Let `J` be the set of faults of a pattern, and let `P[:, J]` be the columns of `P` for `J`.
The nonzero null vectors of `P[:, J]` are in one-to-one correspondence with the nonempty accepted patterns inside `J`, through their supports.
A nonzero null vector other than the all-ones vector has a support that is a proper subset of `J`.
So `J` is minimal exactly when the null space of `P[:, J]` is spanned by the all-ones vector, which has dimension one.
Then `rank(P[:, J]) = |J| - 1 ≤ rank(P)`, so `|J| ≤ rank(P) + 1`.
Patterns of more than `min(K, rank(P) + 1)` faults never need checking.

### 3.6 Duplicate and zero columns can be dropped

Call the pair `(p_j, s_j)` the **column** of fault `j`.
A fault with a zero column, meaning zero flags and zero syndrome, does nothing.
Two faults with equal columns cancel when they appear in the same pattern.
Dropping a zero-column fault, or both faults of an equal pair, from a violating pattern leaves its flags and syndrome unchanged and lowers its size, so the smaller pattern violates the bound too.
A violating pattern with the fewest faults therefore has no zero column and no repeated column.
Replacing each of its faults by another fault with the same column leaves the flags, syndrome, and size unchanged.
It is thus enough to keep one fault for each distinct nonzero column, which lowers the cost of the search without changing the result or the smallest size of a violation.
Keeping the first such fault in DEM order makes a counterexample name the earliest faults.

### 3.7 Independent flags give pairs

Suppose that the distinct nonzero flag columns `p_j` are linearly independent.
In an accepted pattern, group the faults by their flags `p_j`.
The flags of the pattern are the sum of the distinct nonzero columns that occur an odd number of times, so by independence every nonzero column occurs an even number of times.
Hence an accepted pattern that contains a fault with nonzero flags contains an accepted pair of faults with equal flags.
The minimal patterns with nonzero flags are therefore exactly the pairs of faults with equal flags, for any rank and any limit `K`.

## 4. The search

### 4.1 Enumerating minimal patterns

A fault with zero flags is a minimal pattern by itself, and it lies in no larger minimal pattern, because it is an accepted proper subset of it.
The larger minimal patterns consist of the remaining faults, which have nonzero flags.
Order them by their position in the DEM, and write a minimal pattern of size `m ≥ 2` in this order as `j_1 < ... < j_m`.
Section 3.5 gives `rank(P[:, J]) = m - 1`, and the sum of the columns is zero, so any `m - 1` of the columns are independent and span the same space as all `m`.
So the **prefix** `j_1, ..., j_{m-1}` is independent, and the **closing** fault `j_m` has flags equal to the sum of the prefix flags.
Conversely, an independent prefix and a later closing fault with these flags form an accepted pattern of rank `m - 1`, which is minimal by Section 3.5.
Requiring the closing fault to have the largest position lists each minimal pattern exactly once.

The search therefore extends independent prefixes by one fault at a time, in increasing order.
It keeps an echelon basis of the prefix flags, with descending leading bits, to reject a new fault whose flags depend on the prefix.
A dependent prefix cannot be extended to a minimal pattern, because every prefix of a minimal pattern is independent.
For a prefix of length `m - 1`, a table of the faults by their flags lists the closing faults that come after the prefix.

### 4.2 Ascending sizes find a smallest violation

The search checks the faults with zero flags first, and then the minimal patterns of size 2, 3, and so on, up to `min(K, rank(P) + 1)`, and up to 2 if the flags are independent as in Section 3.7.
It skips the larger sizes entirely if every fault with nonzero flags has `μ(s_j) ≤ 1`, by Section 3.3.
The first violation that it finds is reported.

Let `m*` be the fewest faults in any violating accepted pattern of size at most `K`.
Section 3.4 shows that a minimal violating pattern of size `m*` exists, and Section 3.6 shows that it remains one when each of its faults is replaced by the kept fault with the same column.
No minimal pattern of a smaller size violates the bound, so the search does not stop before size `m*`, and it stops with a violation at that size.
The reported pattern therefore has the fewest faults of any violating pattern.

### 4.3 Deciding a pattern

For each pattern that it checks, the search decides whether `μ(S(F)) ≤ |F|`, using the pattern's own size as the weight threshold and not the largest size that is searched.
The decision comes from `PauliCosetSearch` in `src/qldpc/codes/distance/_bounded_weight_search.py`, which computes minimum weights of syndrome classes exactly.
For a violation, it also returns an error with the syndrome and a proved lower bound on `μ(S(F))`, which exceeds `|F|`.
