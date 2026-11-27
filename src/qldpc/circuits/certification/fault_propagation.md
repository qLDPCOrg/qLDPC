# Exact certification of bounded fault propagation in qLDPC

<a name="sec-scope"></a>
## Purpose and scope

A noisy Clifford circuit that prepares a stabilizer state has bounded fault propagation if every fault has [residual weight](#sec-syndromes) at most its fault weight.
Bounded fault propagation is a sufficient criterion for fault-tolerant state preparation (Definition 3 in Section 4.2 of [arXiv:0904.2557](https://arxiv.org/abs/0904.2557)): a few faults can leave only correspondingly few errors on the prepared state.
The residual weight of a fault is the smallest Pauli weight of the Pauli error that the fault leaves on the output qubits, after multiplying that error by any stabilizer of the prepared state.
With post-selection, only the faults that fire none of the post-selection detectors count.
This guide explains why the certifier can decide this exactly, and how it searches for a violation.
It is for contributors who need to understand or modify the certifier in this directory.

<a name="sec-definitions"></a>
## Definitions

<a name="sec-faults"></a>
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

<a name="sec-syndromes"></a>
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

<a name="sec-physical-errors"></a>
### Physical errors

A **physical error** is a case of the circuit's noise whose probability is of first order in the noise's error parameters, such as the X error of an `X_ERROR(p)`.
For example, one Pauli error of a `DEPOLARIZE1` instruction on one qubit is a physical error.
A Pauli error that needs `k` independent physical errors counts as `k` physical errors: for example, an `X_ERROR(p)` and a `Z_ERROR(q)` on one qubit make a Y error, of probability `pq`, from two physical errors.
If independent errors are written as separate noise instructions, rather than as one channel such as `PAULI_CHANNEL_1(p(1-q), pq, q(1-p))`, each error mechanism is the effect of a single physical error, and Stim merges physical errors that flip the same detectors and observables into one error mechanism, which does not change the result.
A set of `t` physical errors therefore has the same flags and syndrome as the sum of their error mechanisms, which is a fault of fault weight at most `t`, and conversely every fault of fault weight `t` arises from `t` physical errors.
So the smallest fault weight of a violating accepted fault is also the smallest number of physical errors that together pass post-selection and violate the bound.

<a name="sec-facts"></a>
## Facts that the search relies on

<a name="sec-subadditive-weight"></a>
### Pauli weight is subadditive

The Pauli weight of a product of two errors is at most the sum of their Pauli weights, so `μ(s + t) ≤ μ(s) + μ(t)`.
For physical weight, the product acts on no more qubits than the two errors together.
For CSS weight, let `(a, b)` and `(c, d)` be the values of `(|x|, |z|)` of the two errors, so that the X and Z parts of the product have weights at most `a + c` and `b + d`, and `max(a + c, b + d) ≤ max(a, b) + max(c, d)`.
So `μ(S F) ≤ Σ μ(s_j)` over the primitive faults `j` of `F`.

<a name="sec-heavy-primitive-fault"></a>
### A violating fault contains a heavy primitive fault

Call a primitive fault **heavy** if its residual weight is at least two, `μ(s_j) ≥ 2`, and **light** otherwise.
By [subadditivity](#sec-subadditive-weight), `μ(S F) ≤ |F|` if `F` contains no heavy primitive fault, since every term of the sum is then at most one.
So every fault `F` that violates the bound contains a heavy primitive fault.

<a name="sec-irreducible-faults"></a>
### Irreducible accepted faults suffice

An **irreducible accepted fault** is a nonzero accepted fault that contains no other nonzero accepted fault (a minimal codeword).
A violating accepted fault `F` with the smallest fault weight is irreducible.
Otherwise `F` contains another nonzero accepted fault `G`, and `F + G` is also a nonzero accepted fault, which contains exactly the primitive faults of `F` that `G` does not.
So `|G| + |F + G| = |F|`, and by the choice of `F`, neither `G` nor `F + G` violates the bound.
Then [subadditivity](#sec-subadditive-weight) gives `μ(S F) ≤ μ(S G) + μ(S (F + G)) ≤ |G| + |F + G| = |F|`, which contradicts that `F` violates the bound.
So if some accepted fault of fault weight at most `K` violates the bound, so does an irreducible one, and it is enough to check those.

<a name="sec-fault-weight-bound"></a>
### Irreducible accepted faults have fault weight at most `rank(P) + 1`

For a fault `F`, let `P[:, F]` be the columns of `P` for the primitive faults that `F` contains.
The null vectors of `P[:, F]`, padded with zeros at the other error mechanisms, are the accepted faults that `F` contains, and the all-ones vector gives `F`.
So a nonzero fault `F` is an irreducible accepted fault exactly when the all-ones vector spans the null space of `P[:, F]`.
By rank-nullity, an irreducible accepted fault `F` has `|F| = rank(P[:, F]) + 1 ≤ rank(P) + 1`.
So, since [irreducible accepted faults suffice](#sec-irreducible-faults), no fault of fault weight above `min(K, rank(P) + 1)` needs checking.
For a finite `K`, the search computes the rank only up to `K - 1`, since every rank of at least `K - 1` gives `min(K, rank(P) + 1) = K`.

<a name="sec-dropped-signatures"></a>
### Duplicate and zero signatures can be dropped

The **signature** of a primitive fault is the pair of its flags and its syndrome.
The flags and syndrome of a fault depend only on the signatures of its primitive faults, so primitive faults with the same signature are interchangeable.
Dropping a primitive fault with zero signature, or a pair with equal signatures, changes neither the flags nor the syndrome and lowers the fault weight, so a violating accepted fault stays one.
So a violating accepted fault with the smallest fault weight contains no such primitive faults, and it is enough to keep one primitive fault for each distinct nonzero signature.
We call the primitive faults that remain the **kept primitive faults**.
Each column of `P` for a dropped primitive fault is zero or equals the column of a kept primitive fault, so the flags of the kept primitive faults have rank `rank(P)`, and the search computes `rank(P)` from them.

<a name="sec-search"></a>
## The search

<a name="sec-checking-one-fault"></a>
### Checking one fault

The search decides exactly whether `μ(S F) ≤ |F|` for a fault `F` with a minimum-weight oracle for the Pauli errors with a given syndrome.
It uses the same oracle, with the threshold one, to decide whether a primitive fault is heavy.
The oracle is `qldpc.codes.distance.CosetLeaderSearch`, named for coset leaders, the lightest errors with a given syndrome.

A stabilizer state is **CSS** if its stabilizer group is generated by **pure-X stabilizers**, which have no Z part, and **pure-Z stabilizers**, which have no X part.
For the CSS weight on a CSS state, the oracle runs two smaller classical searches; otherwise it searches the Pauli errors with the syndrome directly.
The X parts of the pure-X stabilizers span a binary code `C_X`, and the Z parts of the pure-Z stabilizers a binary code `C_Z`.
Multiplying an error with X part `x` and Z part `z` by a stabilizer adds a word of `C_X` to `x` and a word of `C_Z` to `z`, and every such pair of words comes from a stabilizer.
The two words can be chosen independently, so the smallest CSS weight of an error equivalent to this one is `max(min |x + a|, min |z + b|)`, over the words `a` of `C_X` and `b` of `C_Z`.
Each of the two minima is the smallest Hamming weight in a coset of a classical code, and `CosetLeaderSearch` finds it with a search on `n` columns rather than `2n`.
The coset of `x` is determined by the syndrome bits of the pure-Z stabilizers, and that of `z` by those of the pure-X stabilizers.
`CosetLeaderSearch` computes these bits from the syndrome for `g_1, ..., g_n`.
For the physical weight, or for a state that is not CSS, the X and Z parts do not separate in this way.
Let `W_X` and `W_Z` be the binary matrices of the X and Z parts of `g_1, ..., g_n`.
To choose the default Pauli weight, the certifier tests whether the prepared state is CSS by their ranks.
Up to phases, the pure-X stabilizers form a space of dimension `n - rank(W_Z)`, the pure-Z stabilizers one of dimension `n - rank(W_X)`, and the two meet only in the identity, so together they generate a space of dimension `2n - rank(W_X) - rank(W_Z)`.
This space is the whole stabilizer group, of dimension `n`, exactly when `rank(W_X) + rank(W_Z) = n`.

By default, each search of the oracle is a Brouwer-Zimmermann search of the cosets of a binary code, which [brouwer_zimmermann.md](../../codes/distance/brouwer_zimmermann.md) describes, with the improvements that it uses for cosets.
When a high threshold makes a search costly, the oracle may search with another family of information sets that it expects to take less work.
The section [Heuristics and cost models](../../codes/distance/brouwer_zimmermann_code.md#sec-heuristics) of `brouwer_zimmermann_code.md` says when, from the expected work that the field `coset_route_minimum_work` of the search tuning sets.
The oracle builds these families once for each search that it caches, and `qldpc/codes/distance/_tuning.py` sets how many searches of each kind it caches.
Another family can change the error that the oracle returns, and the lower bound that it gives, but never whether it finds an error of weight at most the threshold.
The costliest of these searches also skip errors that tables of bounds prove to weigh more than the threshold ([Skipping completions by completion tables](../../codes/distance/brouwer_zimmermann.md#sec-completion-tables)), which can change the error and the lower bound in the same way.
So the `output_pauli` and `residual_weight_lower_bound` of a counterexample can change between versions of qLDPC when the queries that find it are costly, while the verdict and the smallest violating fault do not.
The thresholds of certifications are usually small, so their searches rarely cost that much, but the maximum weights of the residual-weight estimates of `qldpc.experimental` often are not.

The search skips the oracle for a fault `F` in two cases, each of which cannot hide a violation.

1. For each primitive fault `j`, the search keeps one Pauli error with syndrome `s_j`, its **representative**.
   If the product of the representatives of the primitive faults of `F` has Pauli weight at most `|F|`, then `F` satisfies the bound, because that product has syndrome `S F`.
2. If a fault `G` that the search listed earlier has the syndrome `S F`, then `F` cannot be the first violation.
   The search lists faults by increasing fault weight, so `|G| ≤ |F|`.
   If `G` violates the bound, the search stops there.
   Otherwise `μ(S F) = μ(S G) ≤ |G| ≤ |F|`, so `F` satisfies the bound.
   The search remembers the syndrome of each fault that it lists.
   It keeps them in a table, which grows within a user-specified memory limit, up to `2^31` syndromes.
   A full table takes no more syndromes, so the search may check some syndromes more than once.

<a name="sec-listing-once"></a>
### Listing each irreducible accepted fault once

In this section, every fault contains only [kept primitive faults](#sec-dropped-signatures).
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

Only a fault that [contains a heavy primitive fault](#sec-heavy-primitive-fault) can violate the bound.
The search orders the kept primitive faults that have nonzero flags with the heavy ones first, lists the members of each class in this order, and orders the classes by their first members.
Then a class that contains a heavy primitive fault, a **start class**, comes before every class that does not, and its heavy members come before its light ones.
The code calls the heavy primitive faults the **start members**.
So a fault of fault weight `m ≥ 3` contains a heavy primitive fault only if its class codeword starts with a start class, and the search lists only the class codewords that start with a start class.
For each class codeword, it chooses one primitive fault from each class in turn.
While all the primitive faults that it has chosen are light, it skips the rest of the current class as soon as every primitive fault that it could still choose, in this class or in a later class of the codeword, is light, since then no fault that completes the current choice contains a heavy primitive fault.
For fault weight two, it chooses two primitive faults from a start class in order, the first of them heavy.
This lists each irreducible accepted fault that contains a heavy primitive fault once, and lists nothing else.

The search lists the class codewords that start with a start class by the [walk by shared flags](#sec-shared-flags).

Many primitive faults can share their flags, so there can be far fewer classes than primitive faults.
That is why the search works with classes, and handles individual primitive faults only when it expands each class codeword into faults.

<a name="sec-shared-flags"></a>
### Listing class codewords by shared flags

Order the classes as the search does, with the start classes first, and write a class codeword as `c_1 < ... < c_m` in this order.
Its classes `c_1, ..., c_{m-1}` are independent, and the flags of `c_m` are their sum.
So one could list the class codewords by choosing every independent set of `m - 1` classes that starts with a start class, and looking for a later class with the sum of their flags.
For `q` classes, there are about `q^{m-2} / (m-2)!` such sets for each start class.
When each flag bit is set by few classes, nearly none of them has such a later class, and choosing them all is out of reach for fault weights of four or five on circuits with tens of thousands of classes.
The **walk by shared flags** visits far fewer sets of classes.
It adds to the classes chosen so far only a class that shares a flag bit with the sum of their flags, as the depth-first search of [connected_cluster.md](../../codes/distance/connected_cluster.md#sec-depth-first-search) does for the codewords of a sparse code.
In its terms, the classes are the variables, and the flag bits are the checks.

For a set `A` of classes, let `σ(A)` be the sum of their flags, and `f(c)` the flags of a class `c`.
Each class holds all the kept primitive faults with its nonzero flags, so the flags of the classes are nonzero and pairwise distinct.
So a single class is independent, and at most one class has given flags.
A class `z` **closes** a set `B` of classes if `z` is not in `B` and `f(z) = σ(B)`; `z` need not come after the classes of `B`.
The **root** of a class codeword is its first class, and the walk lists the class codewords of fault weight `m ≥ 3` whose root is a given start class `r`.
It visits **states**, each an independent set `A` of classes that contains `r`, with a set `Φ(A)` of **forbidden** classes.
It starts at `A = {r}` with no class forbidden, and adds classes only to states with `|A| ≤ m - 3`, so every state has at most `m - 2` classes.
A class is **available** at a state if it comes after `r` and is neither in `A` nor in `Φ(A)`.
At each state, the walk chooses a bit `b` of `σ(A)`, which is nonzero since `A` is independent.
The **candidates** of the state are the available classes whose flags have bit `b`, in their order, and the walk tries each of them, `x`, in turn.

- If `|A| ≤ m - 3` and `A ∪ {x}` is independent, the walk visits the state `A ∪ {x}`, a **child** of `A`, with `Φ(A ∪ {x})` the union of `Φ(A)` and the candidates of `A` before `x`.
  A state is **below** the states from which the walk reaches it through children.
  So forbidden classes stay forbidden below a state, and once the walk has tried a candidate of `A`, it is forbidden in the later children of `A` and below them.
- If `|A| = m - 2` and `A ∪ {x}` is independent, the walk **finds** the class codeword `A ∪ {x, z}` if an available class `z` closes `A ∪ {x}`, that is, has flags `σ(A) + f(x)`.
  The classes have distinct flags, so at most one class can.

The bit `b` can be any bit of `σ(A)`, and the arguments below hold for any choice.
The walk chooses the bit set by the fewest classes after `r`, and the lowest of these, since that gives the fewest candidates as long as few classes are forbidden.
An index from each bit to the classes that have it, in their order, gives the classes after `r` by a binary search.

**The walk finds only class codewords with root `r`.**
When it finds `A ∪ {x, z}`, the `m - 1` classes of `A ∪ {x}` are independent and `z` has the sum of their flags.
`z` is not in `A`, since it is available, and it is not `x`, since `σ(A)` is not zero.
So these `m` classes have flags of rank `m - 1` that sum to zero.
By rank-nullity, the subsets of them whose flags sum to zero form a space of dimension one, which holds the whole set.
So no proper nonempty subset has flags that sum to zero, and they form a class codeword.
Every class of it other than `r` was available at the state where the walk added it or found the class codeword, so it comes after `r`, and `r` is its root.

**The walk finds every class codeword with root `r`.**
Let `C` be such a class codeword, so that `r` is its least class.
Call a state **on the way to** `C` if `A` is contained in `C` and no class of `C \ A` is in `Φ(A)`.
The first state, `A = {r}`, is on the way to `C`.
At a state on the way to `C`, every class of `C \ A` is available, since it comes after `r` and is neither in `A` nor forbidden.
`σ(C) = 0`, so `σ(C \ A) = σ(A)`, and every bit of `σ(A)`, the chosen bit `b` among them, is set in an odd number of classes of `C \ A`, and so in at least one.
So some class of `C \ A` is a candidate; let `x` be the first one.
If `|A| ≤ m - 3`, then `A ∪ {x}` is a proper subset of `C`, so it is independent, and the walk visits the child `A ∪ {x}`.
The classes of `Φ(A ∪ {x})` are those of `Φ(A)` and the candidates before `x`, none of which is in `C \ A`, so the state `A ∪ {x}` is on the way to `C` too.
If `|A| = m - 2`, then `C \ A = {x, z}` for a class `z` with flags `σ(A) + f(x)`, which is available, and `A ∪ {x}` is a proper subset of `C`, so it is independent, and the walk finds `C`.
By induction on `|A|`, the walk finds `C`.

**The walk finds each class codeword once.**
The walk never reaches the same set of classes twice.
If one visit is below another, the lower one has more classes.
Otherwise the two visits are at or below different children `A ∪ {u}` and `A ∪ {u'}` of a visit `A`, with `u` before `u'`.
The sets at or below `A ∪ {u}` all contain `u`, while those at or below `A ∪ {u'}` do not, since `u` is in `Φ(A ∪ {u'})`, forbidden classes stay forbidden below a state, and the walk adds only available classes.
So the walk visits each state once, and `Φ(A)` depends only on `A`.
That alone does not keep a class codeword from being found twice, since the walk could find it at different states, from different sets of `m - 1` of its classes.
The requirement that `z` be available, and so not forbidden, prevents this.
For example, take the classes 0, 1, 2, and 3 with flags `100`, `101`, `110`, and `111`, fault weight four, and the root 0.
The walk branches on bit 2, with the candidates 1, 2, and 3.
At the state `{0, 1}`, with sum `001`, the candidate 3 has bit 0, and class 2 closes `{0, 1, 3}`, so the walk finds `{0, 1, 2, 3}`.
At the state `{0, 2}`, with sum `010`, the candidate 3 has bit 1, and class 1 would close `{0, 2, 3}`, but class 1 is forbidden there, since it was tried before class 2.

In general, suppose that the walk finds `C` at two states `A` and `A'`, with candidates `x` and `x'` and classes `z` and `z'` that close `A ∪ {x}` and `A' ∪ {x'}`.
If `A = A'`, then `C \ A = {x, z} = {x', z'}`, and since `σ(A) = f(x) + f(z)` has bit `b`, exactly one of `x` and `z` has bit `b`.
Both `x` and `x'` are candidates, and so have bit `b`, so `x = x'` and `z = z'`, and the two findings are one.
Otherwise neither state is below the other, since both have `m - 2` classes.
Swapping `A` and `A'` if needed, let `B` be the last state that both are below, where their paths first diverge, and `B ∪ {u}` and `B ∪ {u'}` the children of `B` that `A` and `A'` are at or below, with `u` before `u'`.
Then `u` is in `Φ(B ∪ {u'})`, and so in `Φ(A')`.
Then `u` is in `A`, and so in `C`.
But `u` is not in `A'`: it is not in `B` or `u'`, since it was a candidate of `B`, and the walk adds below `B ∪ {u'}` only classes that are available, and so not forbidden.
It is not `x'`, which is available at `A'`, and it is not `z'`, since the walk finds `C` at `A'` only if `z'` is neither forbidden nor in `A'`.
So `u` is not in `C`, a contradiction, and the walk finds no class codeword twice.

**Order.**
The walk lists a root in one or more **passes**, each a walk of the states from the root as above.
A pass **keeps** class codewords that it finds in an array of a fixed number `D ≥ 2` of rows, outside the memory limit, each as a row of its classes in increasing order.
Once the pass is done, the walk sorts the rows in lexicographic order, and only then does the search **release** them, passing them on to be expanded into faults.
A root with more class codewords than fit in the array takes several passes, each releasing the next of them in lexicographic order, as the next paragraph shows.
The search lists the roots in order, and the root is the first class of each of its class codewords, so the search releases all its class codewords in lexicographic order.
The result depends on the listing only through this sequence of class codewords (see [threads](#sec-listing-on-threads)).
The search reports the first violating fault of the smallest fault weight in this sequence, so a fixed order keeps the reported fault the same between versions of qLDPC.

**Roots with many class codewords.**
At a state with `|A| = m - 2`, each class after `r` with bit `b` that the walk examines, candidate or not, adds at most one class codeword, so the walk makes room before it examines one.
If the array is full, the walk **halves** it: it sorts its rows and keeps the first `⌊D/2⌋ ≥ 1` of them.
From then on, it keeps only class codewords that come before the first one dropped, its **ceiling**, which it lowers at each halving.
A later pass over a root also has a **floor**, the last class codeword released by the pass before, and keeps only class codewords after it.
A missing floor or ceiling imposes no bound.
The floor and the ceiling decide only which class codewords a pass keeps, never which states it visits, so each pass finds every class codeword of the root once, as shown above.

By induction over the steps of a pass, at each moment the array holds exactly the class codewords that the pass has found so far and that come after the floor and before the current ceiling.
Each such class codeword entered when it was found, since the walk had made room for it, and it came before the ceiling of that moment, which is no earlier than the current one.
No halving dropped it, since a halving drops exactly the rows at or after its new ceiling, the rows being distinct and sorted.
Conversely, a class codeword enters only when it is found between the floor and the ceiling of that moment.
Suppose that a class codeword that entered comes at or after the current ceiling.
The ceiling when it entered came after it, so some halving after it entered has a new ceiling no later than it.
The halvings before the first of these had later ceilings and kept it, and the first of these dropped it.
The pass finds each class codeword once, so it does not enter again.

When the pass is done, the array therefore holds every class codeword of the root between the floor and the ceiling, and the search releases them in order.
If the pass halved the array, it releases at least the `⌊D/2⌋ ≥ 1` rows that the last halving kept; the last one released becomes the new floor, and the walk makes another pass.
Otherwise these are all the class codewords of the root after the floor, and the root is done.
So, by induction over the passes, the class codewords of a root released so far are exactly those up to its floor, and each pass releases the next of them.
A root with `k` class codewords takes at most `1 + k/⌊D/2⌋` passes.
Each pass ends, since it visits each of finitely many states once, and each halving is followed by the examination of a class.
Sorting a full array takes about as much memory again, which is also outside the memory limit.
`_fault_search.py` sets `D` and says how much memory the array takes.

**The probes.**
To find the class `z` with flags `σ(A) + f(x)`, the walk probes the **map of classes by hash**, a hash map from a 64-bit linear hash of the flags of each class to the classes with that hash, where linear means that the hash of a sum of flags is the sum of their hashes.
So the walk gets the hash of `σ(A) + f(x)` by adding that of `f(x)` to that of `σ(A)`, which it keeps for each state.
A class with these flags has this hash, so a probe that finds no class with this hash rules out a class codeword.
Otherwise, the walk checks every class with this hash, comparing its flags in full, since different flags can share a hash.
For the class with these flags, if there is one, it checks whether that class is available and whether `A ∪ {x}` is independent.
These are the checks of the walk's definition above, applied only when a probe finds a class, so the walk finds exactly the class codewords of that definition, while most candidates cost only a probe.

<a name="sec-skipping-listed-syndromes"></a>
### Skipping faults whose syndromes are listed

Many faults of a fault weight share their syndromes, so the table of syndromes leaves out most of the faults that the search lists (see [Checking one fault](#sec-checking-one-fault), case 2).
The search skips many of them without looking them up.
It expands each class codeword by choosing one member of each of its classes in turn, as [above](#sec-listing-once) describes.
Take two choices `U` and `U'` of members of the first `ℓ < m` classes of a class codeword of `m` classes, where the search chooses `U` first, and either both or neither contain a heavy primitive fault.
The rule above that skips the rest of a class depends only on the member, the later classes, and whether a heavy primitive fault was chosen before it, so the search completes `U` and `U'` by the same choices `R` of members of the other classes.
If `S U = S U'`, each fault `U' + R` has the syndrome of `U + R`, which comes earlier in the listing order, and the search can skip all the faults `U' + R` without looking them up.
It does so when it has remembered the syndrome of `U`, which it does for a limited number of choices of each class codeword.
At fault weight two, the search chooses two members of one start class, so the only choices to compare are single members of that class.
These have the same flags, and so distinct syndromes, since kept primitive faults have distinct signatures, so the search skips no faults of fault weight two.

Suppose that the table of syndromes has had room for every syndrome so far.
Then, when the search reaches a fault, the table holds the syndrome of every fault that comes earlier in the listing order, whether the search looked it up or skipped it.
This holds by induction on the listing order: a fault that the search looked up had its syndrome added to the table, or found it there, and a skipped fault `U' + R` has the syndrome of the earlier fault `U + R`, which by induction is in the table.
So the table holds the syndrome of each skipped fault, and it holds only syndromes of faults that the search looked up earlier, so by case 2 of [Checking one fault](#sec-checking-one-fault), no skipped fault is the first violation.
The table would also have left out each skipped fault, so skipping changes neither the faults that the search checks nor the table.
The search skips faults only while the table has had room for every syndrome.
This keeps the faults that it checks the same.
It is not needed for exactness.
Even when the table is full, each skipped fault has the syndrome of a fault listed earlier, so the first fault in the listing order with a given syndrome is never skipped, and the table cannot hold its syndrome when the search reaches it.
The first violation is such a fault, since an earlier fault with its syndrome would violate the bound too, so the search checks it.

<a name="sec-first-violation"></a>
### Why the first violation has the smallest fault weight

The search checks the kept primitive faults with zero flags first, and then the [listed faults](#sec-listing-once) in order of fault weight 2, 3, and so on, up to `min(K, rank(P) + 1)`.
The search reports the first violation that it finds, which is a violating accepted fault because every fault that it checks is accepted.

Suppose that some violating accepted fault of fault weight at most `K` exists, and take one with the smallest fault weight.

1. It is [irreducible](#sec-irreducible-faults).
2. Its primitive faults have [distinct nonzero signatures](#sec-dropped-signatures), so replacing each by the kept primitive fault with the same signature gives a violating accepted fault of the same fault weight that contains only kept primitive faults.
   This fault also has the smallest fault weight, so it is irreducible by step 1.
3. If this fault has fault weight one, it is a primitive fault with zero flags, and the search checks it first.
   Otherwise, it [contains a heavy primitive fault](#sec-heavy-primitive-fault), and it has [fault weight at most `rank(P) + 1`](#sec-fault-weight-bound) and at most `K` by assumption, so the search lists it.
4. No accepted fault with a smaller fault weight violates the bound, so the search finds no violation at a smaller fault weight, and it finds one at this fault weight.

So the reported fault has the smallest fault weight of any violating accepted fault.
If the search finds no violation, steps 1 to 3 show that no violating accepted fault of fault weight at most `K` exists, so fault propagation is bounded up to fault weight `K`.

<a name="sec-threads"></a>
## Threads

<a name="sec-listing-on-threads"></a>
### Listing class codewords on threads

A listing of class codewords counts its **work** in units of a probe of the [map of classes by hash](#sec-shared-flags).
The walk counts one unit for each bit of `σ(A)` that it examines to choose a bit, and for each class after `r` with bit `b` that it examines, including those that it skips as chosen or forbidden.
It counts `64` units for each reduction of flags: that of the root at the start of each pass, and that of a candidate, whether the candidate gives a child, is found dependent, or completes a class codeword.
This count estimates the time of the walk; it decides only when a listing moves to threads and when a thread pauses, never what the walk lists.
A listing is **costly** if, once the calling thread, listing alone, has done `2^18` probes of work for each thread, at least as much work for each thread is estimated to be left, as described below.
A costly listing runs on `SearchTuning.num_threads` threads, as the oracle's searches do.
[Threads](../../codes/distance/README.md#threads) in the README of the distance package gives the default of that number.
Only listing class codewords moves to threads, since it takes most of the lister's time: in trials of an earlier version of the search, which listed class codewords by another walk, four fifths or more whenever the lister took over half a second.
The calling thread expands the class codewords into faults in the order in which a listing on a single thread lists them, which we call the **serial order**, while the threads list later ones.
So the search looks up and skips the same faults, keeps the same table of syndromes, and sends the oracle the same syndromes in the same batches as on one thread, and its result does not depend on the number of threads.

The search lists the class codewords [root by root](#sec-shared-flags), so its **work items** are the roots.
Each pass over a root starts with only the root chosen and no class forbidden, and a pass after the first depends only on the class codewords of the root that earlier passes released, so the listing of a root, its work, and its class codewords depend only on the root.
So the listings of consecutive ranges of roots list the class codewords in the serial order, and the work of all the ranges adds up to that of a single thread.
The search **weighs** each root `r` by `C(q - 1 - r, m - 2)`, for `q` classes and fault weight `m`: the number of choices of the classes of a class codeword with root `r` other than the root and the one that the walk finds by a probe.
The candidates of every state of the walk come after `r`, so the work of a root falls with `r` about as fast as its weight does if the classes that share each flag bit are spread over the order of the classes.
A root is never split between threads, so a listing with fewer start classes than threads, or with one root much costlier than the others, runs on fewer threads for part of its time.

The calling thread first lists the first roots alone, however costly the first one is, in ranges of 1, 2, 4, and more roots, until it has done `2^18` probes of work for each thread.
If the work left, estimated as its work times the ratio of the weights of the other roots to those of its own, is less than `2^18` probes for each thread, it lists the other roots alone too.
Otherwise the listing is costly, and the other roots move to threads.
The **thread runner** is the code of `qldpc/codes/distance/_threads.py` that runs the threads of a listing.
On `n` threads, it splits the roots that move to threads into ranges, each of the fewest roots whose weights reach `1 / (256 t)` of those of all these roots, for `t = min(n - 1, number of these roots)` threads.
The threads claim the ranges in order, and the calling thread takes the class codewords of each range in turn, as its thread lists them.
The weights can estimate the work poorly, so that a range of a given share of the weights can hold much more of the work than its share.
Small ranges keep such a range from holding up the calling thread while the other threads, far ahead, reach the cap of held batches below and wait.
A thread hands the class codewords to the calling thread in batches of at most `2^10` class codewords, so that the batches that the thread runner holds, at most 64 for each thread, take bounded memory.
Each call of the walk on a thread returns after about `2^20` probes of work, also within a pass over a root, whose state the walk keeps for its next call.
The thread then checks a stop event that the thread runner gives the threads, and that closing the listing sets.
A thread hands on no empty batch, since empty batches would fill the cap of held batches and stall the threads that list ahead.

The expansion is that of a single thread, since it gets the same class codewords in the serial order, and how they are split into batches of class codewords does not change it.
Batches of class codewords split only between class codewords; the expansion starts each class codeword with no remembered choices, and keeps its table of syndromes and its batch of faults from one class codeword to the next, within a batch of class codewords and across batches.
So it pauses at the same faults, where its batch of faults is full or its table of syndromes should grow.
The threads can list class codewords after the first violation, which the calling thread never expands.
An exception on a thread, such as a `MemoryError`, can be raised before the calling thread reaches a violation that one thread would have found first.

While the threads list, the calling thread can run a costly batch of the oracle, which moves to threads of its own, so that up to `2 n - 1` threads can be busy for a time.
Listing on threads also takes more CPU time than on one thread, since each thread lists more slowly when more threads list.
The docstring of `certify_bounded_fault_propagation` says how to keep a certification on one thread.

<a name="sec-thread-requirements"></a>
### Requirements on threads

Changes to the listing must keep these requirements.
The tests that cite this section check them on small instances, and review checks what these tests miss.

- **Expand class codewords only on the calling thread, in the serial order.**
  The table of syndromes, the choices of members that the expansion remembers, and the batches of faults depend on the order of the faults, and [skipping faults](#sec-skipping-listed-syndromes) relies on the table holding the syndrome of each earlier fault.
  Expanding class codewords on threads would need a table of syndromes for each thread, and its own argument that the search still checks the faults of one thread.
  On any number of threads, the search must check the faults of one thread.
  This includes listings in which threads pause within passes over roots, a root takes several passes, the table of syndromes fills up, a violation stops the listing while threads list, the calling thread lists roots alone first, or a later range of roots is listed before the first.
- **Split the listing only between roots, and start each pass over a root with only the root chosen and no class forbidden.**
  The passes over a root must stay a function of the root, also when a thread pauses within a root.
  So the ranges of roots list the class codewords in the serial order, and their work adds up to that of one thread, as [Listing class codewords on threads](#sec-listing-on-threads) shows.
- **Keep the walk by shared flags writing only to the arrays of its own walk, and the expansion into faults only to the arrays of the calling thread,** since both run in compiled code without the global interpreter lock while threads list.
  The threads read the rows of the classes, the index of their flag bits, and the map of classes by hash, which the lister builds once and never changes.
  The docstring of each compiled function of `_fault_search.py` that runs without the lock says which arrays it writes.
  A test runs the threads with the arrays that the lister builds once marked read-only, so that numba rejects a write to them in any branch of compiled code, and checks that they do not change.
  Review checks the writes that the read-only marking misses, such as those through `.flat`, `np.ascontiguousarray`, `np.asfortranarray`, `np.nditer`, or `np.fill_diagonal`, which numba compiles anyway, and those on paths that the tests do not run.
- **Stop the threads whenever a listing ends, also by an exception.**
  The search closes the listing of class codewords of a fault weight whenever it stops taking class codewords from it: at a violation, at the end of the listing, or at an exception.
  The thread runner then stops the threads and waits for them.
  The thread runner hands on the class codewords in the order of the ranges, and holds at most `64 t` batches of class codewords on `t` threads that list, and one more for the range that the calling thread reads.
  Each batch that a thread hands to the calling thread holds at most `2^10` class codewords, so that the held batches take bounded memory.
  A thread checks the stop event after each call of the walk rather than hand on empty batches, and closing the listing sets that event.
- **Call no numba-parallel code on threads, and read the search tuning only on the calling thread.**
  The search reads its number of threads from the search tuning on the calling thread, before it lists, so that a certification lists on the number of threads of the tuning that it runs within.
  A fixture of `qldpc/codes/distance/conftest.py`, which `conftest.py` here imports, fails each test of this package whose threads call numba-parallel code from Python or read the search tuning through the function of `qldpc/codes/distance/_tuning.py` that returns it.
  A test checks that this fixture is active here, on the threads that the thread runner starts.
  qLDPC compiles none of its functions numba-parallel, which `qldpc/_compiled_code_rules_test.py` checks.
  The docstring of each function of this package compiled with `nogil=True` cites this section, which a test checks.

<a name="sec-unused-speed-up"></a>
## A speed-up that the search does not use

<a name="sec-light-heavy-part"></a>
### Skipping faults whose heavy part is light enough

Split a fault `F` into three faults, `F = G + L0 + L1`.
The **heavy part** `G` contains the heavy primitive faults of `F`.
`L0` contains its light primitive faults with zero syndrome, and `L1` its light primitive faults with nonzero syndrome, each of which has `μ(s_j) = 1`.
By [subadditivity](#sec-subadditive-weight), `μ(S F) ≤ μ(S G) + |L1|`.
Since `|F| = |G| + |L0| + |L1|`, the fault `F` satisfies the bound whenever `μ(S G) ≤ |G| + |L0|`.

In a class codeword, the start classes come first, and only they contain heavy primitive faults.
So once the search has chosen a primitive fault from each start class of a class codeword, every fault that completes the choice has the same heavy part as the chosen primitive faults, and contains every zero-syndrome light primitive fault among them.
The search could therefore skip all these faults if it finds a Pauli error with syndrome `S G` whose Pauli weight is at most `|G|` plus the number of zero-syndrome light primitive faults chosen so far.
Every skipped fault satisfies the bound, so skipping them would change neither the result nor the reported counterexample.
But their syndromes would not enter the table of syndromes, so the search could check later faults that the table would have left out.

The search does not use this rule, because in trials the time that it saved did not justify the added complexity.
A trial implementation, in an earlier version of the search that worked with primitive faults rather than classes, applied the rule to each partial choice of primitive faults that contained a light primitive fault.
It looked for such a Pauli error in the first two layers of the oracle's search of Pauli errors, at each light primitive fault added to a partial choice, and reused the answer for each heavy part.
It did not look when adding the last primitive fault of a fault, since only one fault then completes the choice.
The number of faults that complete a choice grows with the fault weight.
Up to fault weight four, looking for such a Pauli error cost about as much time as it saved.
In searches that listed every fault, it saved little time at fault weight five but made the search about twice as fast at fault weight six.
Later trials reached fault weight ten, where the search skips most faults because [their syndromes are listed](#sec-skipping-listed-syndromes).
There, expanding class codewords into faults took under a fifth of the listing's time, and the rule, with the smallest Pauli weight of each heavy part from the oracle, would have removed under a third of the choices of members that remain.
