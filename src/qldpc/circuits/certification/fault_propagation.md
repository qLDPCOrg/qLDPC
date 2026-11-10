# Exact certification of bounded fault propagation in qLDPC

## Purpose and scope

A noisy Clifford circuit that prepares a stabilizer state has bounded fault propagation if every fault has [residual weight](#syndromes-and-pauli-weight) at most its fault weight.
Bounded fault propagation is a sufficient criterion for fault-tolerant state preparation (Definition 3 in Section 4.2 of [arXiv:0904.2557](https://arxiv.org/abs/0904.2557)): a few faults can leave only correspondingly few errors on the prepared state.
The residual weight of a fault is the smallest Pauli weight of the Pauli error that the fault leaves on the output qubits, after multiplying that error by any stabilizer of the prepared state.
With post-selection, only the faults that fire none of the post-selection detectors count.
This guide explains why the certifier can decide this exactly, and how it searches for a violation.
It is for contributors who need to understand or modify the certifier in this directory.

## Definitions

### Faults, primitive faults, and acceptance

An **error mechanism** is one error instruction of the detector error model (DEM) that the certifier returns, which is the DEM of the circuit after adding any flag detectors and one noiseless detector for each stabilizer of the prepared state.
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

The **output error** of a fault is the Pauli error that it leaves on the output qubits.
Let `S` be the **fault syndrome matrix**: an `n × N` binary matrix whose column `j` is the `n`-bit syndrome `s_j` of the output error of primitive fault `j`.
The syndrome of the output error of a fault `F` is then `S F`.

The **Pauli weight** of a Pauli error is either its **physical weight**, the number of qubits that it acts on, or its **CSS weight** `max(|x|, |z|)`, where `x` and `z` are the X and Z support vectors of the error.
For a syndrome `s`, let `μ(s)` be the smallest Pauli weight of a Pauli error with syndrome `s`, that is, of a Pauli error in that equivalence class.
The **residual weight** of a fault `F` is `μ(S F)`, the smallest Pauli weight of an error equivalent to its output error.
A fault `F` **violates** the fault propagation bound if its residual weight exceeds its fault weight, `μ(S F) > |F|`.
Fault propagation is bounded up to fault weight `K`, the largest fault weight to certify (a positive integer, or ∞ for no limit), if no nonzero accepted fault of fault weight at most `K` violates the bound.

### Physical errors

A **physical error** is one case of one noise instruction of the circuit, such as one Pauli error of a `DEPOLARIZE1` instruction on one qubit.
Each error mechanism is the effect of a single physical error, and Stim merges physical errors that flip the same detectors and observables into one error mechanism, which does not change the result.
A set of `t` physical errors therefore has the same flags and syndrome as the sum of their error mechanisms, which is a fault of fault weight at most `t`, and conversely every fault of fault weight `t` arises from `t` physical errors.
So the smallest fault weight of a violating accepted fault is also the smallest number of physical errors that together pass post-selection and violate the bound.

## Facts that the search relies on

### Pauli weight is subadditive

The Pauli weight of a product of two errors is at most the sum of their Pauli weights, so `μ(s + t) ≤ μ(s) + μ(t)`.
For physical weight, the product acts on no more qubits than the two errors together.
For CSS weight, let `(a, b)` and `(c, d)` be the values of `(|x|, |z|)` of the two errors, so that the X and Z parts of the product have weights at most `a + c` and `b + d`, and `max(a + c, b + d) ≤ max(a, b) + max(c, d)`.
So `μ(S F) ≤ Σ μ(s_j)` over the primitive faults `j` of `F`.

### A violating fault contains a heavy primitive fault

Call a primitive fault **heavy** if its residual weight is at least two, `μ(s_j) ≥ 2`, and **light** otherwise.
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
Each column of `P` for a dropped primitive fault is zero or equals the column of a kept primitive fault, so the flags of the kept primitive faults have rank `rank(P)`, and the search computes `rank(P)` from them.

## The search

### Checking one fault

The search decides exactly whether `μ(S F) ≤ |F|` for a fault `F` with a minimum-weight oracle for the Pauli errors with a given syndrome.
It uses the same oracle, with the threshold one, to decide whether a primitive fault is heavy.
The oracle is `CosetLeaderSearch`, named for coset leaders, the lightest errors with a given syndrome.

A stabilizer state is **CSS** if its stabilizer group is generated by **pure-X stabilizers**, which have no Z part, and **pure-Z stabilizers**, which have no X part.
For the CSS weight on a CSS state, the oracle runs two smaller classical searches; otherwise it searches the Pauli errors with the syndrome directly.
The X parts of the pure-X stabilizers span a binary code `C_X`, and the Z parts of the pure-Z stabilizers a binary code `C_Z`.
Multiplying an error with X part `x` and Z part `z` by a stabilizer adds a word of `C_X` to `x` and a word of `C_Z` to `z`, and every such pair of words comes from a stabilizer.
The two words can be chosen independently, so the smallest CSS weight of an error equivalent to this one is `max(min |x + a|, min |z + b|)`, over the words `a` of `C_X` and `b` of `C_Z`.
Each of the two minima is the smallest Hamming weight in a coset of a classical code, and `CosetLeaderSearch` finds it with a search on `n` columns rather than `2n`.
The coset of `x` is determined by the syndrome bits of the pure-Z stabilizers, and that of `z` by those of the pure-X stabilizers.
`CosetLeaderSearch` computes these bits from the syndrome for `g_1, ..., g_n`.
For the physical weight, or for a state that is not CSS, the X and Z parts do not separate in this way.

By default, each search of the oracle is a Brouwer–Zimmermann search of the cosets of a binary code, which [brouwer_zimmermann.md](../../codes/distance/brouwer_zimmermann.md) describes, with the improvements that it uses for cosets.
When a high threshold makes a search costly, the oracle may search with another family of information sets that it expects to take less work.
Section 4.4 of [brouwer_zimmermann.md](../../codes/distance/brouwer_zimmermann.md#44-heuristics-and-cost-models) says when, from the expected work that the field `overlapping_minimum_work` of the search tuning sets.
The oracle builds these families once for each cached search (see `CSS_SEARCH_CACHE_SIZE` and `PAULI_SEARCH_CACHE_SIZE` in `qldpc.codes.distance._tuning`).
Another family can change the error that the oracle returns, and the lower bound that it gives, but never whether it finds an error of weight at most the threshold.
The thresholds of certifications are usually small, so their searches rarely cost that much, but those of the residual-weight estimates of `qldpc.experimental` often do.

The search skips the oracle for a fault `F` in two cases, each of which cannot hide a violation.

1. For each primitive fault `j`, the search keeps one Pauli error with syndrome `s_j`, its **representative**.
   If the product of the representatives of the primitive faults of `F` has Pauli weight at most `|F|`, then `F` satisfies the bound, because that product has syndrome `S F`.
2. If a fault `G` that the search listed earlier has the syndrome `S F`, then `F` cannot be the first violation.
   The search lists faults by increasing fault weight, so `|G| ≤ |F|`.
   If `G` violates the bound, the search stops there.
   Otherwise `μ(S F) = μ(S G) ≤ |G| ≤ |F|`, so `F` satisfies the bound.
   The search remembers the syndromes of the faults that it lists in a table, which grows within a user-specified memory limit.
   A full table takes no more syndromes, so the search may check some syndromes more than once.

### Listing each irreducible accepted fault once

In this section, every fault contains only [kept primitive faults](#duplicate-and-zero-signatures-can-be-dropped).
A primitive fault with zero flags is an irreducible accepted fault by itself, and no larger irreducible accepted fault contains one.
Every other irreducible accepted fault has fault weight `m ≥ 2` and contains only primitive faults with nonzero flags.

A **flag class**, or class, is the set of all the kept primitive faults that have the same nonzero flags, and its **members** are these primitive faults.
Two primitive faults of the same class form an accepted fault of fault weight two.
So an irreducible accepted fault of fault weight two is two primitive faults of one class, and an irreducible accepted fault of fault weight `m ≥ 3` contains at most one primitive fault of each class, since otherwise it would contain a smaller accepted fault.

Call a set of classes, or a fault, **independent** if their flags are linearly independent, that is, if no nonempty subset has flags that sum to zero.
For `m ≥ 3`, call `m` distinct classes a **class codeword** if their flags sum to zero and every proper nonempty subset of them is independent.
In other words, a class codeword is an irreducible accepted fault of `m` classes, as if each class were a single primitive fault.
Then choosing one primitive fault from each class of a class codeword gives an irreducible accepted fault: its flags are those of the classes, so it is accepted and contains no smaller accepted fault.
Conversely, the classes of the primitive faults of an irreducible accepted fault of fault weight `m ≥ 3` form a class codeword.
So choosing a class codeword of `m ≥ 3` classes and one primitive fault from each of its classes gives each irreducible accepted fault of fault weight `m` exactly once.

Fix an order of the classes (the search's order is given below), and write a class codeword as `c_1 < ... < c_m` in this order.
The **prefix** of the codeword is `c_1, ..., c_{m-1}`, which is independent, and its **closing** class is `c_m`, whose flags equal the sum of the prefix flags.
Conversely, if the classes `c_1 < ... < c_{m-1}` are independent and a later class `c_m` has flags equal to their sum, then these `m` classes form a class codeword, since their flags have rank `m - 1` and sum to zero, so the only dependency among them involves all `m`.
So the class codewords correspond one to one to the independent prefixes with a later closing class, since the closing class is the last one in the order.

Only a fault that [contains a heavy primitive fault](#a-violating-fault-contains-a-heavy-primitive-fault) can violate the bound.
The search orders the kept primitive faults that have nonzero flags with the heavy ones first, lists the members of each class in this order, and orders the classes by their first members.
Then a class that contains a heavy primitive fault, a **start class**, comes before every class that does not, and its heavy members come before its light ones.
The code calls the heavy primitive faults the **start members**.
So a fault of fault weight `m ≥ 3` contains a heavy primitive fault only if its class codeword starts with a start class, and the search builds only the prefixes that start with a start class.
For each class codeword, it chooses one primitive fault from each class in turn.
While all the primitive faults that it has chosen are light, it skips the rest of the current class as soon as every primitive fault that it could still choose, in this class or in a later class of the codeword, is light, since then no fault that completes the current choice contains a heavy primitive fault.
For fault weight two, it chooses two primitive faults from a start class in order, the first of them heavy.
This lists each irreducible accepted fault that contains a heavy primitive fault once, and lists nothing else.

The search looks up the closing class in a hash map from a 64-bit linear hash of the flags of each class to the classes with that hash, where linear means that the hash of a sum of flags is the sum of their hashes.
Different flags can share a hash, so the search compares the flags of every class that it finds with the sum of the prefix flags.
Most choices of `c_{m-1}` have no closing class, so for fault weight `m ≥ 4` the search saves time by choosing `c_{m-1}` and `c_m` together, as a **closing pair**.
Choosing them together removes about one factor of the number of classes from the search time, at the cost of a table whose size is quadratic in the number of classes.
If it fits within the memory limit, the search keeps a table of all pairs of classes, sorted by the hash of the sum of their flags, with a hash map from each hash to its pairs; this table has first claim on the memory limit.
For each choice of independent classes `c_1, ..., c_{m-2}`, the search looks up the pairs `c_{m-1} < c_m` with the hash of the sum of the flags of `c_1, ..., c_{m-2}`, and keeps those that come after `c_{m-2}` and whose flags sum to exactly that sum.
Of these, it keeps only the pairs whose first class `c_{m-1}` keeps the prefix `c_1, ..., c_{m-1}` independent.
That `c_1, ..., c_{m-2}` are independent is not enough, since the flags of `c_{m-1}` could be the sum of some of theirs, and then the `m` classes would contain a smaller set of classes whose flags sum to zero.
With this condition, `c_m` has flags equal to the sum of the independent prefix `c_1, ..., c_{m-1}`, so the `m` classes form a class codeword, as for a closing class found by a lookup of single classes.
The table keeps the pairs with each hash in lexicographic order, so the table lookup lists the same class codewords in the same order as a loop over `c_{m-1}`, and the faults are listed in the same order whether or not the table fits within the memory limit.

Many primitive faults can share their flags, which is why the search works with classes.
Its tests of independence, its lookups of closing classes, and its table of pairs work with classes, of which there can be far fewer than primitive faults, and it handles individual primitive faults only when it expands each class codeword into faults.

### Why the first violation has the smallest fault weight

The search checks the kept primitive faults with zero flags first, and then the [listed faults](#listing-each-irreducible-accepted-fault-once) in order of fault weight 2, 3, and so on, up to `min(K, rank(P) + 1)`.
The search reports the first violation that it finds, which is a violating accepted fault because every fault that it checks is accepted.

Suppose that some violating accepted fault of fault weight at most `K` exists, and take one with the smallest fault weight.

1. It is [irreducible](#irreducible-accepted-faults-suffice).
2. Its primitive faults have [distinct nonzero signatures](#duplicate-and-zero-signatures-can-be-dropped), so replacing each by the kept primitive fault with the same signature gives a violating accepted fault of the same fault weight that contains only kept primitive faults.
   This fault also has the smallest fault weight, so it is irreducible by step 1.
3. If this fault has fault weight one, it is a primitive fault with zero flags, and the search checks it first.
   Otherwise, it [contains a heavy primitive fault](#a-violating-fault-contains-a-heavy-primitive-fault), and it has [fault weight at most `rank(P) + 1`](#irreducible-accepted-faults-have-fault-weight-at-most-rankp--1) and at most `K` by assumption, so the search lists it.
4. No accepted fault with a smaller fault weight violates the bound, so the search finds no violation at a smaller fault weight, and it finds one at this fault weight.

So the reported fault has the smallest fault weight of any violating accepted fault.
If the search finds no violation, steps 1 to 3 show that no violating accepted fault of fault weight at most `K` exists, so fault propagation is bounded up to fault weight `K`.

## A speed-up that the search does not use

### Skipping faults whose heavy part is light enough

Split a fault `F` into three faults, `F = G + L0 + L1`.
The **heavy part** `G` contains the heavy primitive faults of `F`.
`L0` contains its light primitive faults with zero syndrome, and `L1` its light primitive faults with nonzero syndrome, each of which has `μ(s_j) = 1`.
By [subadditivity](#pauli-weight-is-subadditive), `μ(S F) ≤ μ(S G) + |L1|`.
Since `|F| = |G| + |L0| + |L1|`, the fault `F` satisfies the bound whenever `μ(S G) ≤ |G| + |L0|`.

In a class codeword, the start classes come first, and only they contain heavy primitive faults.
So once the search has chosen a primitive fault from each start class of a class codeword, every fault that completes the choice has the same heavy part as the chosen primitive faults, and contains every zero-syndrome light primitive fault among them.
The search could therefore skip all these faults if it finds a Pauli error with syndrome `S G` whose Pauli weight is at most `|G|` plus the number of zero-syndrome light primitive faults chosen so far.
Every skipped fault satisfies the bound, so skipping them would change neither the result nor the reported counterexample.

The search does not use this rule, because the time that it saves does not justify the added complexity.
A trial implementation, in an earlier version of the search that worked with primitive faults rather than classes, applied the rule to each partial choice of primitive faults that contained a light primitive fault.
It looked for such a Pauli error in the first two layers of the oracle's search of Pauli errors, at each light primitive fault added to a partial choice, and reused the answer for each heavy part.
On color-code state preparations, certifications up to fault weight four took about as long as without the rule.
In benchmarks that treated every listed fault as satisfying the bound, so that the search listed every fault, the rule made the search 1.1 to 1.2 times faster at fault weight five, and 1.7 times faster at fault weight six.
Two variants were slower: deciding exactly whether `μ(S G)` is small enough, and also applying the rule when adding the last primitive fault of a fault.
The speed-up grows with the fault weight, so the rule may pay off for searches at larger fault weights.
