# Exact certification of bounded fault propagation in qLDPC

## 1. Purpose and scope

A noisy Clifford circuit that prepares a stabilizer state has bounded fault propagation if every fault leaves an output error equivalent to an error whose Pauli weight is at most its fault weight.
With post-selection, only the faults that flip none of the post-selection detectors count.
This guide explains why the certifier can decide this exactly, and how it searches for a violation.
It is for contributors who need to understand or modify the certifier in this directory.
It is organized as follows:

- [Section 2](#2-definitions) defines error mechanisms, faults, primitive faults, flags and acceptance, syndromes, Pauli weight, and violation of the bound.
- [Section 3](#3-facts-that-the-search-relies-on) proves the facts that the search relies on.
- [Section 4](#4-the-search) describes the search and why it finds a smallest violation.

## 2. Definitions

### 2.1 Faults, primitive faults, and acceptance

An **error mechanism** is one error instruction of the circuit's detector error model (DEM).
With `N` error mechanisms, numbered in DEM order, a **fault** is a vector `F` in `GF(2)^N`, whose entry `j` is 1 if error mechanism `j` occurs.
Its **fault weight** `|F|` is its number of ones.
A **primitive fault** is a fault of fault weight one, that is, the unit vector of a single error mechanism `j`.
A fault **contains** the primitive fault `j` if its entry `j` is 1, and it contains another fault if it contains every primitive fault of that fault.

Let `P` be the parity check matrix of the post-selection detectors for the primitive faults: a binary matrix with one row for each post-selection detector and one column `p_j` for each primitive fault `j`, whose entry is 1 if primitive fault `j` flips that detector.
The **flags** of a fault `F` are `P F`, which record the post-selection detectors that it flips, so the flags of a primitive fault `j` are its column `p_j`.
A fault `F` is **accepted**, or passes post-selection, if its flags are zero, `P F = 0`.
So the accepted faults are exactly the codewords of the binary linear code with parity check matrix `P`.
Without post-selection, `P` has no rows and every fault is accepted.

### 2.2 Syndromes and Pauli weight

Let `n` be the number of output qubits, and let `g_1, ..., g_n` be `n`-qubit Pauli strings that generate the stabilizer group of the prepared state.
The **syndrome** of a Pauli error is the binary vector whose entry `i` is 1 if the error anticommutes with `g_i`.
Two errors are **equivalent** if they differ by an element of the stabilizer group, because then they leave the same state.

Let `S` be the **fault syndrome matrix**: an `n × N` binary matrix whose column `j` is the `n`-bit syndrome `s_j` of the output Pauli error induced by primitive fault `j`.
The syndrome of the output error of a fault `F` is then `S F`.

The **Pauli weight** of an error is either its **physical weight**, the number of qubits that it acts on, or its **CSS weight** `max(|x|, |z|)`, where `x` and `z` are the X and Z support vectors of the error.
For a syndrome `s`, let `μ(s)` be the smallest Pauli weight of an error with syndrome `s`.
A fault `F` **violates** the fault propagation bound if `μ(S F) > |F|`.
Fault propagation is bounded if no accepted nonzero fault violates the bound, with the faults limited to fault weight at most `K` if a positive integer limit `K` is given.

## 3. Facts that the search relies on

### 3.1 Syndromes identify equivalent errors

Two errors with the same syndrome differ by an operator that commutes with all `n` independent stabilizers of the `n`-qubit state.
Every such operator is a stabilizer up to a phase, so the two errors are equivalent.
Hence `μ(S F)` is the Pauli weight of the lightest error that is equivalent to the output error induced by fault `F`.

### 3.2 Pauli weight is subadditive

The Pauli weight of a product of two errors is at most the sum of their Pauli weights, so `μ(s + t) ≤ μ(s) + μ(t)`.
For physical weight, the product acts on no more qubits than the two errors together.
For CSS weight, let `(a, b)` and `(c, d)` be the values of `(|x|, |z|)` of the two errors, so that the X and Z parts of the product have weights at most `a + c` and `b + d`, and `max(a + c, b + d) ≤ max(a, b) + max(c, d)`.
So `μ(S F) ≤ Σ μ(s_j)` over the primitive faults `j` of `F`.

### 3.3 A violating fault contains a heavy primitive fault

Call a primitive fault **heavy** if `μ(s_j) ≥ 2`, and **light** otherwise.
By [Section 3.2](#32-pauli-weight-is-subadditive), `μ(S F) ≤ |F|` if every primitive fault that `F` contains is light.
So every fault `F` that violates the bound contains a heavy primitive fault.

Without post-selection, a heavy primitive fault is itself an accepted violating fault, so, for any limit `K`, the bound is violated exactly when some primitive fault is heavy.
With post-selection, a heavy primitive fault that flips a post-selection detector is not accepted alone, so Sections [3.4](#34-irreducible-accepted-faults-suffice) and [3.5](#35-irreducible-accepted-faults-have-fault-weight-at-most-rankp--1) are needed to decide the bound.

### 3.4 Irreducible accepted faults suffice

An **irreducible accepted fault** is a nonzero accepted fault that contains no other nonzero accepted fault (a minimal codeword).
A violating accepted fault `F` with the smallest fault weight is irreducible.
Otherwise `F` contains another nonzero accepted fault `G`, and `F + G` is also a nonzero accepted fault, which contains exactly the primitive faults of `F` that `G` does not.
So `|G| + |F + G| = |F|`, and by the choice of `F`, neither `G` nor `F + G` violates the bound.
Then [Section 3.2](#32-pauli-weight-is-subadditive) gives `μ(S F) ≤ μ(S G) + μ(S (F + G)) ≤ |G| + |F + G| = |F|`, which contradicts that `F` violates the bound.
So if the bound is violated at all, within the limit `K` if there is one, it is violated by an irreducible accepted fault, and it is enough to check those.

### 3.5 Irreducible accepted faults have fault weight at most `rank(P) + 1`

For a fault `F`, let `P[:, F]` be the columns of `P` for the primitive faults that `F` contains.
The null vectors of `P[:, F]`, padded with zeros at the other error mechanisms, are the accepted faults that `F` contains, and the all-ones vector gives `F`.
So a nonzero fault `F` is an irreducible accepted fault exactly when the all-ones vector spans the null space of `P[:, F]`.
By rank-nullity, an irreducible accepted fault `F` has `|F| = rank(P[:, F]) + 1 ≤ rank(P) + 1`.
So by [Section 3.4](#34-irreducible-accepted-faults-suffice), no fault of fault weight above `rank(P) + 1`, or above the limit `K` if there is one, needs checking.

### 3.6 Duplicate and zero signatures can be dropped

The **signature** of primitive fault `j` is the pair `(p_j, s_j)` of the columns of `P` and `S` at `j`.
The flags and syndrome of a fault depend only on the signatures of its primitive faults, so primitive faults with the same signature are interchangeable.
Dropping a primitive fault with zero signature, or two primitive faults with equal signatures, leaves the flags and syndrome of the fault unchanged and lowers its fault weight, so a violation stays a violation.
So a violating accepted fault with the smallest fault weight contains no such primitive faults, and it is enough to keep one primitive fault for each distinct nonzero signature.
Keeping the first one in DEM order makes a counterexample name the earliest primitive faults.
We call the primitive faults that remain the **kept primitive faults**.

## 4. The search

### 4.1 Checking one fault

The search decides exactly whether `μ(S F) ≤ |F|` for a fault `F` with `PauliCosetSearch`, a minimum-weight oracle for the Pauli errors with a given syndrome.
It uses the same oracle, with the threshold one, to decide whether a primitive fault is heavy.

### 4.2 Listing each irreducible accepted fault once

In this section, every fault contains only kept primitive faults of [Section 3.6](#36-duplicate-and-zero-signatures-can-be-dropped).
A primitive fault with zero flags is an irreducible accepted fault by itself, and no larger irreducible accepted fault contains one.
Every other irreducible accepted fault has fault weight `m ≥ 2` and contains only primitive faults with nonzero flags.

Order the kept primitive faults that have nonzero flags, with the heavy ones first, and write each irreducible accepted fault `F` of fault weight `m ≥ 2` as `j_1 < ... < j_m` in this order.
Call a fault **independent** if the columns of `P` for its primitive faults are linearly independent, that is, if it contains no nonzero accepted fault.
The **prefix** of `F` is the fault that contains `j_1, ..., j_{m-1}`, and its **closing** primitive fault is `j_m`.
By irreducibility, every fault that `F` contains other than `F` itself is independent, so the prefix is independent, and the closing primitive fault has flags equal to the sum of the prefix flags, because `F` is accepted.
Conversely, if `j_1 < ... < j_{m-1}` are independent and a later primitive fault `j_m` has flags equal to their sum, then the fault that contains these `m` primitive faults is accepted and its columns of `P` have rank `m - 1`, so it is irreducible by [Section 3.5](#35-irreducible-accepted-faults-have-fault-weight-at-most-rankp--1).
Since the closing primitive fault is the last one in the order, each irreducible accepted fault has exactly one prefix and closing primitive fault.

By [Section 3.3](#33-a-violating-fault-contains-a-heavy-primitive-fault), only a fault that contains a heavy primitive fault can violate the bound.
So the search builds only the prefixes that start with a heavy primitive fault.
Because the heavy primitive faults come first, this lists each irreducible accepted fault that contains a heavy primitive fault exactly once, and no other fault of fault weight at least 2.

### 4.3 Why the first violation has the smallest fault weight

The search checks the kept primitive faults with zero flags first, and then the faults of [Section 4.2](#42-listing-each-irreducible-accepted-fault-once) in order of fault weight 2, 3, and so on, up to `rank(P) + 1`, or up to `K` if there is a smaller limit.
Checking a fault means deciding whether it violates the fault propagation bound, as in [Section 4.1](#41-checking-one-fault).
The search reports the first violation that it finds, which is a violating accepted fault because every fault that it checks is accepted.

Suppose that some violating accepted fault exists, with fault weight at most `K` if there is a limit, and take one with the smallest fault weight.

1. It is irreducible, by [Section 3.4](#34-irreducible-accepted-faults-suffice).
2. By [Section 3.6](#36-duplicate-and-zero-signatures-can-be-dropped), it contains no primitive fault with zero signature and no two primitive faults with equal signatures, so replacing each of its primitive faults by the kept primitive fault with the same signature gives a violating accepted fault of the same fault weight, again irreducible by step 1.
3. If it has fault weight one, it is a primitive fault with zero flags, and the search checks it first.
   Otherwise, it contains a heavy primitive fault by [Section 3.3](#33-a-violating-fault-contains-a-heavy-primitive-fault) and has fault weight at most `rank(P) + 1` by [Section 3.5](#35-irreducible-accepted-faults-have-fault-weight-at-most-rankp--1) and at most `K` by assumption, so the search lists it.
4. No accepted fault with a smaller fault weight violates the bound, so the search finds no violation at a smaller fault weight, and it finds one at this fault weight.

So the reported fault has the smallest fault weight of any violating accepted fault.
If the search finds no violation, steps 1 to 3 show that no violating accepted fault exists, with fault weight at most `K` if there is a limit, so fault propagation is bounded.
