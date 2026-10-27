# Exact certification of bounded fault propagation in qLDPC

## Purpose and scope

A noisy Clifford circuit that prepares a stabilizer state has bounded fault propagation if every fault leaves an output error equivalent to an error whose Pauli weight is at most its fault weight.
With post-selection, only the faults that fire none of the post-selection detectors count.
This guide explains why the certifier can decide this exactly, and how it searches for a violation.
It is for contributors who need to understand or modify the certifier in this directory.

## Definitions

### Faults, primitive faults, and acceptance

An **error mechanism** is one error instruction of the circuit's detector error model (DEM).
With `N` error mechanisms, numbered in DEM order, a **fault** is a vector `F` in `GF(2)^N`, whose entry `j` is 1 if error mechanism `j` occurs.
Its **fault weight** `|F|` is its number of ones.
A **primitive fault** is a fault of fault weight one, that is, the unit vector of a single error mechanism `j`.
A fault **contains** the primitive fault `j` if its entry `j` is 1, and it contains another fault if it contains every primitive fault of that fault.

Let `P` be the binary matrix with one row for each post-selection detector and one column for each primitive fault, whose entry is 1 if that primitive fault flips that detector.
The **flags** of a fault `F` are `P F`, the post-selection detectors that it fires.
A fault `F` is **accepted**, or passes post-selection, if its flags are zero, `P F = 0`.
So the accepted faults are exactly the codewords of the binary linear code with parity check matrix `P`.
Without post-selection, `P` has no rows and every fault is accepted.

### Syndromes and Pauli weight

Let `n` be the number of output qubits, and let `g_1, ..., g_n` be `n`-qubit Pauli strings that generate the stabilizer group of the prepared state.
The **syndrome** of a Pauli error is the binary vector whose entry `i` is 1 if the error anticommutes with `g_i`.
Two errors are **equivalent** if they differ by an element of the stabilizer group, up to a phase, because then they leave the same state.
Errors are equivalent exactly when they have the same syndrome, because an operator that commutes with `g_1, ..., g_n` is a stabilizer up to a phase.

Let `S` be the **fault syndrome matrix**: an `n × N` binary matrix whose column `j` is the `n`-bit syndrome `s_j` of the output Pauli error induced by primitive fault `j`.
The syndrome of the output error of a fault `F` is then `S F`.

The **Pauli weight** of an error is either its **physical weight**, the number of qubits that it acts on, or its **CSS weight** `max(|x|, |z|)`, where `x` and `z` are the X and Z support vectors of the error.
For a syndrome `s`, let `μ(s)` be the smallest Pauli weight of an error with syndrome `s`, that is, of an error in that equivalence class.
A fault `F` **violates** the fault propagation bound if `μ(S F) > |F|`.
Fault propagation is bounded up to fault weight `K`, the largest fault weight to certify (a positive integer, or ∞ for no limit), if no nonzero accepted fault of fault weight at most `K` violates the bound.

## Facts that the search relies on

### Pauli weight is subadditive

The Pauli weight of a product of two errors is at most the sum of their Pauli weights, so `μ(s + t) ≤ μ(s) + μ(t)`.
For physical weight, the product acts on no more qubits than the two errors together.
For CSS weight, let `(a, b)` and `(c, d)` be the values of `(|x|, |z|)` of the two errors, so that the X and Z parts of the product have weights at most `a + c` and `b + d`, and `max(a + c, b + d) ≤ max(a, b) + max(c, d)`.
So `μ(S F) ≤ Σ μ(s_j)` over the primitive faults `j` of `F`.

### A violating fault contains a heavy primitive fault

Call a primitive fault **heavy** if `μ(s_j) ≥ 2`.
By [subadditivity](#pauli-weight-is-subadditive), `μ(S F) ≤ |F|` if `F` contains no heavy primitive fault, since every term of the sum is then at most one.
So every fault `F` that violates the bound contains a heavy primitive fault.

### Irreducible accepted faults suffice

An **irreducible accepted fault** is a nonzero accepted fault that contains no other nonzero accepted fault (a minimal codeword).
A violating accepted fault `F` with the smallest fault weight is irreducible.
Otherwise `F` contains another nonzero accepted fault `G`, and `F + G` is also a nonzero accepted fault, which contains exactly the primitive faults of `F` that `G` does not.
So `|G| + |F + G| = |F|`, and by the choice of `F`, neither `G` nor `F + G` violates the bound.
Then [subadditivity](#pauli-weight-is-subadditive) gives `μ(S F) ≤ μ(S G) + μ(S (F + G)) ≤ |G| + |F + G| = |F|`, which contradicts that `F` violates the bound.
So if some accepted fault of fault weight at most `K` violates the bound, so does an irreducible one, and it is enough to check those.

### Irreducible accepted faults have fault weight at most `rank(P) + 1`

For a fault `F`, let `P[:, F]` be the columns of `P` for the primitive faults that `F` contains.
The null vectors of `P[:, F]`, padded with zeros at the other error mechanisms, are the accepted faults that `F` contains, and the all-ones vector gives `F`.
So a nonzero fault `F` is an irreducible accepted fault exactly when the all-ones vector spans the null space of `P[:, F]`.
By rank-nullity, an irreducible accepted fault `F` has `|F| = rank(P[:, F]) + 1 ≤ rank(P) + 1`.
So, since [irreducible accepted faults suffice](#irreducible-accepted-faults-suffice), no fault of fault weight above `min(K, rank(P) + 1)` needs checking.

### Duplicate and zero signatures can be dropped

The **signature** of a primitive fault is the pair of its flags and its syndrome.
The flags and syndrome of a fault depend only on the signatures of its primitive faults, so primitive faults with the same signature are interchangeable.
Dropping a primitive fault with zero signature, or a pair with equal signatures, changes neither the flags nor the syndrome and lowers the fault weight, so a violating accepted fault stays one.
So a violating accepted fault with the smallest fault weight contains no such primitive faults, and it is enough to keep one primitive fault for each distinct nonzero signature.
We call the primitive faults that remain the **kept primitive faults**.

## The search

### Checking one fault

The search decides exactly whether `μ(S F) ≤ |F|` for a fault `F` with `PauliCosetSearch`, a minimum-weight oracle for the Pauli errors with a given syndrome.
It skips the oracle when a product of low-weight errors with syndromes `s_j`, one for each primitive fault `j` of `F`, has Pauli weight at most `|F|`, since that product has syndrome `S F`.
It uses the same oracle, with the threshold one, to decide whether a primitive fault is heavy.

### Listing each irreducible accepted fault once

In this section, every fault contains only [kept primitive faults](#duplicate-and-zero-signatures-can-be-dropped).
A primitive fault with zero flags is an irreducible accepted fault by itself, and no larger irreducible accepted fault contains one.
Every other irreducible accepted fault has fault weight `m ≥ 2` and contains only primitive faults with nonzero flags.

Order the kept primitive faults that have nonzero flags, with the heavy ones first, and write each irreducible accepted fault `F` of fault weight `m ≥ 2` as `j_1 < ... < j_m` in this order.
Call a fault **independent** if the flags of its primitive faults are linearly independent, that is, if it contains no nonzero accepted fault.
The **prefix** of `F` is the fault that contains `j_1, ..., j_{m-1}`, and its **closing** primitive fault is `j_m`.
The prefix is independent because `F` is irreducible, and the closing primitive fault has flags equal to the sum of the prefix flags because `F` is accepted.
Conversely, if `j_1 < ... < j_{m-1}` are independent and a later primitive fault `j_m` has flags equal to their sum, then the fault that contains these `m` primitive faults is accepted and its flags have rank `m - 1`, so it is [irreducible](#irreducible-accepted-faults-have-fault-weight-at-most-rankp--1).
So the irreducible accepted faults of fault weight `m ≥ 2` correspond one to one to the independent prefixes with a later primitive fault of matching flags, since the closing primitive fault is the last one in the order.

Only a fault that [contains a heavy primitive fault](#a-violating-fault-contains-a-heavy-primitive-fault) can violate the bound.
So the search builds only the prefixes that start with a heavy primitive fault.
Since the heavy primitive faults come first, such a fault has a heavy `j_1`, so this lists each of them once and lists nothing else.

Most choices of `j_{m-1}` have no closing primitive fault, so for fault weight `m ≥ 4` the search saves time by choosing `j_{m-1}` and `j_m` together.
Choosing them together removes about one factor of `N` (the number of primitive faults) from the search time, at the cost of a table whose size is quadratic in `N`.
The search keeps a table of all pairs of these primitive faults indexed by the sum of their flags, if the table fits within a user-specified memory limit.
For each choice of `j_1, ..., j_{m-2}`, the search looks up the pairs `j_{m-1} < j_m` that come after `j_{m-2}` and whose flags sum to the flags of `j_1, ..., j_{m-2}`, and keeps the pairs for which the prefix is independent.
The table lookup lists the same faults in the same order as a loop over `j_{m-1}`.

### Why the first violation has the smallest fault weight

The search checks the kept primitive faults with zero flags first, and then the [listed faults](#listing-each-irreducible-accepted-fault-once) in order of fault weight 2, 3, and so on, up to `min(K, rank(P) + 1)`.
The search reports the first violation that it finds, which is a violating accepted fault because every fault that it checks is accepted.

Suppose that some violating accepted fault of fault weight at most `K` exists, and take one with the smallest fault weight.

1. It is [irreducible](#irreducible-accepted-faults-suffice).
2. Its primitive faults have [distinct nonzero signatures](#duplicate-and-zero-signatures-can-be-dropped), so replacing each by the kept primitive fault with the same signature gives a violating accepted fault of the same fault weight, which is irreducible by step 1 as well.
3. If it has fault weight one, it is a primitive fault with zero flags, and the search checks it first.
   Otherwise, it [contains a heavy primitive fault](#a-violating-fault-contains-a-heavy-primitive-fault), and it has [fault weight at most `rank(P) + 1`](#irreducible-accepted-faults-have-fault-weight-at-most-rankp--1) and at most `K` by assumption, so the search lists it.
4. No accepted fault with a smaller fault weight violates the bound, so the search finds no violation at a smaller fault weight, and it finds one at this fault weight.

So the reported fault has the smallest fault weight of any violating accepted fault.
If the search finds no violation, steps 1 to 3 show that no violating accepted fault of fault weight at most `K` exists, so fault propagation is bounded up to fault weight `K`.
