# Connected-cluster distance search in qLDPC

## 1. Purpose and scope

The connected-cluster search computes the exact distance of a binary code from the Tanner graph of its parity checks.
The code can be classical, CSS, or a qubit code that is not CSS, and a quantum code can be a subsystem code.
The search is `method="connected_cluster"` of `get_distance_exact` on the code classes, and the default `method="auto"` runs it for the distances of quantum codes and the fault distances of circuits, switching to Brouwer-Zimmermann (BZ, see [brouwer_zimmermann.md](brouwer_zimmermann.md)) when that is predicted to cost less (section "Choosing a method" of [README.md](README.md)).
It also finds light vectors with a given syndrome for `CosetLeaderSearch`, which `certify_bounded_fault_propagation` uses.
The search is the connected-cluster algorithm of Dumer, Kovalev, and Pryadko, and later works cite their 2017 paper for it (see the [references](#references)).
Its earliest known application to quantum codes is the linked-cluster technique of Kovalev, Dumer, and Pryadko (2013), for the distances of quantum LDPC codes.
Dumer, Kovalev, and Pryadko (2014) then grew each cluster through the checks that it leaves unsatisfied, as qLDPC does.
qLDPC follows the branch and bound of Okada and Kasai (Algorithm 2 in Section V) and of Lee et al. (Appendix B), who also extend it to stabilizer codes that are not CSS (Appendix B.2).
They branch on the unsatisfied check with the fewest choices, forbid the variables of earlier siblings, prune by lower bounds, and search from one root per orbit of cyclic shifts.
G. Lee (Appendix C.4) takes the roots from general groups of automorphisms of the Tanner graph, and [Section 3.5](#35-symmetries) says which automorphisms qLDPC uses.
Related searches of classical LDPC codes look for small stopping sets and low-weight codewords: those of Richter, of Rosnes and Ytrehus, and of Khatami et al.
The method's name, `"connected_cluster"`, follows `dist-m4ri`, the program of Pryadko and Zeng, which calls the search the "connected-cluster (CC) algorithm" of Dumer, Kovalev, and Pryadko.
Webster, Jacob, and Higgott (Section 4.3.2) note that it is "also referred to as the irreducible cluster or linked cluster algorithm".
This guide says what the cost of the search depends on ([Section 2](#2-what-the-cost-of-the-search-depends-on)), explains the algorithm ([Section 3](#3-the-algorithm)), maps it to the source code ([Section 4](#4-guide-to-the-source-code)), and lists what every change must preserve ([Section 5](#5-correctness-requirements)).

## 2. What the cost of the search depends on

The search grows supports, sets of variables (the columns of the check matrix `H`), from a root variable, and adds variables only through unsatisfied checks, those that hold an odd number of the variables of the support ([Section 3.3](#33-the-depth-first-search)).
Such a check offers at most `w - 1` further variables, where `w` is the largest weight of a check, so a search for words of weight at most `W` visits at most of the order of `(w - 1)^(W - 1)` supports from each root, fewer once pruning ([Section 3.4](#34-pruning)) cuts subtrees.
The cost thus grows with the weight of the checks, with `W`, and with the number of roots, which symmetries reduce ([Section 3.5](#35-symmetries)).
The search raises `W` until it finds a nontrivial word ([Section 3.6](#36-from-a-search-to-the-distance)), each limit costing roughly a constant factor more than the last, so dense checks, whose factor is large, make the search slow unless the distance is small.
Checks much lighter than the distance, as in surface codes, also slow the search: a support can leave few checks unsatisfied yet need many more variables, so the bounds of Section 3.4 prune little.

The stabilizers that `get_distance_css` and `get_distance_pauli` derive for a subsystem code can be much heavier than its lightest stabilizer generators ([Section 3.9](#39-sparser-stabilizers-of-subsystem-codes)).
A caller who knows sparse stabilizer generators can search them as given.
For a CSS subsystem code with Z-type stabilizer generators `S_Z` and X-type gauge generators `G_X`, `get_distance_from_checks(S_Z, G_X, method="connected_cluster")` gives the dressed X-distance; passing X-type stabilizers in place of `G_X` would count gauge operators that are not stabilizers as logical operators.

A code that is not CSS costs more to search than an equivalent CSS code: each qubit of a check offers two Paulis that flip the check, and a `Y` can flip more checks, which weakens the bounds ([Section 3.7](#37-qubit-codes-that-are-not-css)).
So `get_distance_pauli` searches checks as a CSS code whenever X-type and Z-type operators generate their group, even if the checks mix X and Z, but searches a code that is CSS only up to local Cliffords, such as a CSS code with Hadamards on some qubits, as a code that is not CSS.

Sparse checks alone do not make the search faster than BZ, whose work depends not on sparsity but on the distance and on the dimension of the space that it searches, stabilizers included for a quantum code ([Section 3.6 of brouwer_zimmermann.md](brouwer_zimmermann.md#36-work-estimates-and-exhaustive-enumeration)).
The connected-cluster search is often much faster on quantum LDPC codes, but can be much slower on some of them, and on classical LDPC codes whose checks are only moderately sparse.
The script [experiments/distance_tuning/benchmark.py](../../../../experiments/distance_tuning/benchmark.py) compares the methods on known codes.

## 3. The algorithm

### 3.1 The problem

Let `H` and `G` be binary matrices with `rowspan(G) ⊆ ker(H)`.
The vectors of `ker(H)` are **words**, a word outside `rowspan(G)` is a **witness**, and the distance of the pair is

$$
d(H, G) = \min \{ \operatorname{wt}(x) : Hx = 0,\; x \notin \operatorname{rowspan}(G) \}.
$$

Coordinates are **variables**, and the rows of `H` are **checks**.
The **support** of a vector is the set of its nonzero variables, and its **syndrome** is the set of its **unsatisfied** checks, those with an odd number of variables in the support.

A CSS code with X-type checks `H_X` and Z-type checks `H_Z` has two **sectors**, each an instance of this problem: its X-type logical operators, of distance `d(H_Z, H_X)`, and its Z-type ones, of distance `d(H_X, H_Z)`.
Its distance is the smaller of the two.
For a CSS subsystem code with gauge generators `G_X` and `G_Z` and stabilizers `S_X` and `S_Z`, the sectors are `(S_Z, G_X)` and `(S_X, G_Z)`, and a witness is a dressed logical operator.
For a classical code, `H` is its parity-check matrix and `G` is empty, and the search has one sector.
[Section 3.7](#37-qubit-codes-that-are-not-css) treats qubit codes that are not CSS, whose search also has one sector.

### 3.2 Light words are connected

**Lemma.**
Let `x` be a witness of minimum weight.
Its support is connected in the Tanner graph of `H`, which joins each check to its variables, and no restriction of `x` to a proper nonempty subset of its support, equal to `x` there and zero elsewhere, is in `ker(H)`.

*Proof.*
No check holds variables of two components of the support, so the restrictions of `x` to the components are in `ker(H)`; they sum to `x`, so not all are in `rowspan(G)`, and one of them is a witness, which by minimality is `x` itself.
If a restriction `y` to a proper nonempty subset were in `ker(H)`, then `y` and `x + y` would be lighter words that sum to `x`, so one of them would be a lighter witness. ∎

The lemma is due to Dumer, Kovalev, and Pryadko (see the [references](#references)), though none of their papers states it in one place for `ker(H)` outside `rowspan(G)`.
Theorem 1 of their 2013 paper states its first part for the minimum-weight words of a linear code.
The second part of the lemma says that `x` is irreducible: it is not the sum of two nonzero words of `ker(H)` with disjoint supports.
Their 2015 paper defines irreducible Pauli operators, and its Lemma 4 writes every undetectable operator as a product of irreducible ones with disjoint supports.
That paper also builds every irreducible operator outside the stabilizer group by a search through unsatisfied stabilizer generators.
Lemma 3 of their 2017 paper, which cites the 2015 paper, states that a minimum-weight word of a linear code is irreducible.
Lemma 2 of Okada and Kasai states both parts for one witness in `ker(H)` outside `rowspan(G)`, with the proof above.
The proof holds for every witness of minimum weight, which [Section 3.5](#35-symmetries) uses.

### 3.3 The depth-first search

Each variable has a **rank**, its position in an order that [Section 3.5](#35-symmetries) chooses.
A **state** is a support `S` with a set `F` of **forbidden** variables, and the search from a **root** `r` starts at `S = {r}`, with every variable of lower rank forbidden.
At a state whose syndrome is empty, the search reports `S` if it is the support of a witness, and goes no deeper.
At any other state, it picks an unsatisfied check `c` with the fewest **available** variables, those in neither `S` nor `F`, and searches the **children** `S ∪ {v}`, for the available `v` of `c`, depth first.
Once it leaves the subtree of a child, it forbids the variable of that child in the later children, which are its later **siblings**.
Since an unsatisfied check holds a variable of `S`, every support that the search visits is connected, and since the variables of earlier siblings are forbidden, none recurs.

The children come in decreasing order of repair ([Section 3.4](#34-pruning)).
The order does not affect completeness, and is a heuristic: the most repairing child satisfies the most weight, and its variable is then forbidden in all its later siblings, which can help prune them.

**Completeness.**
Let `x` be a witness of minimum weight at most `W`, and `r` its variable of lowest rank.
Take a state of the search from `r` with `S ⊊ supp(x)` and `F ∩ supp(x) = ∅`, such as `S = {r}`.
By the lemma, the syndrome of `S` is not empty, and every check holds an even number of the variables of `supp(x)`, so each unsatisfied check, such as the check `c` on which the search branches, holds an available variable of `supp(x) \ S`.
The child of the first such variable of `c` is another state of this kind, since the variables of its earlier siblings lie outside `supp(x)`.
So the search reaches `supp(x)`, unless it reports another witness first.
The pruning of [Section 3.4](#34-pruning) keeps these states, since its bounds hold for any state that a witness of weight at most `W` extends, and it forbids only variables that no such witness holds.

### 3.4 Pruning

At a state with `s` variables, a **completion** is a set `T` of available variables such that `S ∪ T` is the support of a witness, and the search prunes the state if it proves that every completion has more than `W - s` variables.
Every unsatisfied check holds an odd number of the variables of `S` and an even number of those of `S ∪ T`, so it holds a variable of `T`.
So `T` is a **hitting set** of the available variables of the unsatisfied checks, and `|T| ≥ Y` for any feasible solution of the dual of the hitting-set problem: weights `y_c ≥ 0` on the unsatisfied checks, with sum `Y`, such that the checks that hold any one available variable weigh at most one in total.

- The bound `b1` of Okada and Kasai takes `y_c = 1/Δ`, where `Δ` is the largest number of checks of a variable.
- Their bound `b2` takes `y_c = 1` on some unsatisfied checks whose available variables are pairwise disjoint, and `y_c = 0` on the others.
- qLDPC takes `y_c = 1/m_c`, where the **overlap** of a variable is the number of unsatisfied checks that hold it, and `m_c` is the largest overlap of an available variable of `c`.
  An available variable of overlap `k` lies in `k` checks with `m_c ≥ k`, whose weights total at most one.
  These weights are at least those of `b1`, since `m_c ≤ Δ`, and if no two unsatisfied checks share an available variable, `Y` equals `b2` with every unsatisfied check chosen.
  Where unsatisfied checks share available variables, `b2` can be larger, but it needs a choice of disjoint checks at each state, and qLDPC does not compute it.

The weights are integers in units of `1 / lcm(1, ..., Δ)`, so that comparisons are exact; if `lcm(1, ..., Δ)` exceeds `2**40`, the unit is `2**-40`, and the weights round down, which keeps them feasible.

**Forbidding variables for a subtree.**
The **repair** `R_v` of an available variable `v` is the total weight of the unsatisfied checks that hold `v`.
A completion `T` that holds `v` must cover the total weight `Y`, of which `v` covers `R_v` and each other variable at most one, so `|T| ≥ 1 + Y - R_v`.
If this exceeds `W - s`, no completion within the limit holds `v`, and otherwise `v` is **viable**.
A completion `T` of a descendant `S ∪ A` gives the completion `A ∪ T` of `S`, so the search forbids each variable that is not viable at `S` throughout the subtree of `S`.
A variable outside the unsatisfied checks has a repair of zero, so if `1 + Y > W - s`, the subtree uses only viable variables of unsatisfied checks.
The search prunes the state if `Y > W - s` or an unsatisfied check has no viable variable, and otherwise branches on the check with the fewest.
Forbidding variables can lower the `m_c` in the subtree, which raises the weights there, and so prunes more.
Setting `restrict_viable_symbols` of `SearchTuning` in `_tuning.py` to `False` turns this forbidding off.
The search then prunes a state if `Y > W - s` or an unsatisfied check has no available variable, and otherwise branches on the unsatisfied check with the fewest available variables.
It stays exact, since the completeness argument of [Section 3.3](#33-the-depth-first-search) holds with fewer forbidden variables.

Before it enters a child, the search applies the cheaper `b1`: it skips the child if the unsatisfied checks of the child outnumber `Δ(W - s - 1)`.

qLDPC does not use dual ascent, which raises each `y_c` as far as its constraints allow, at the cost of further passes over the unsatisfied checks of every state.

### 3.5 Symmetries

Let `Γ` be a group of column permutations that map `ker(H)` and `rowspan(G)` onto themselves, and so map witnesses to witnesses of the same weight.
For a ranking of the variables, the **root** of an orbit of `Γ` is its variable of lowest rank.
Among the witnesses of minimum weight, take one, `x`, whose lowest-ranked variable `m` ranks lowest.
If a variable `u` of the orbit of `m` ranked lower, an element of `Γ` that maps `m` to `u` would map `x` to a witness of minimum weight that holds `u`, against the choice of `x`, so `m` is a root.
By the lemma, the completeness argument of [Section 3.3](#33-the-depth-first-search) applies to `x` from `m`, so it suffices to search from the roots.
Since the lemma holds for every witness of minimum weight, the images of `x` included, `Γ` need not permute the rows of `H`: qLDPC checks only that each generator maps `ker(H)` and `rowspan(G)` onto themselves, with `verify_automorphisms`, as BZ does.

qLDPC ranks the variables orbit by orbit, with the orbits in the order of their lowest indices, so that the search from each root forbids every earlier orbit.
A root forbids only variables of orbits whose roots rank lower, so for a given order of the orbits, no ranking forbids more.
Without symmetries, every variable is a root, ranked by its index.
Okada and Kasai write the variables of a pair-partition code as `(ℓ, a)`, for a block column `ℓ` and a position `a` modulo the lift size `P`, search from the roots `(ℓ, 0)` of the orbits under cyclic translations of `a`, and rank `(ℓ, a)` by `ℓP + a`; their implementation uses these roots only if `P` is prime and `W < P`.
qLDPC takes the orbits of any group of automorphisms that it verifies.

The generators of `Γ` are automorphisms that the caller of a distance function supplies on request, such as the `_found_qudit_automorphisms` of quantum codes; classical codes supply none.
For a large code, finding the automorphisms, verifying them, and ordering their orbits can cost as much as visiting many states.
So the search of each sector asks for them at most once, before the first weight limit at which it expects to have visited `automorphism_minimum_states` states over all its limits.
That field of `SearchTuning` in `_tuning.py` defaults to a round number of states assumed to cost about as much to visit as asking.
The search predicts the number of states of the next limit from the ratio between the numbers of the last two.
The search predicts rather than counts because it can ask only between limits, each a single compiled call, and the limit in which it would reach `automorphism_minimum_states` can be a large part of the search.
A first limit that a `cutoff` sets above the weight divisor ([Section 3.6](#36-from-a-search-to-the-distance)) has no prediction, so the search asks before it.
A first limit at the divisor, the cheapest ([Section 2](#2-what-the-cost-of-the-search-depends-on)), is treated as visiting no states, so the search asks before it only if `automorphism_minimum_states` is zero.
Setting `automorphism_minimum_states` to `MAX_VALUE` turns asking off.

With symmetries, the search from the first root does most of the work: it forbids nothing, and every witness that meets the first orbit has an image that holds this root.
A witness with several variables in the first orbit can have several such images, one for each such variable if `Γ` acts regularly on the orbit, so the search can visit several images of one state; rejecting such duplicates soundly for general groups is left open.

### 3.6 From a search to the distance

**Logical test.**
A word is in `rowspan(G)` exactly if it is orthogonal to `ker(G)`, which holds the rows of `H`, so a word is a witness exactly if it has an odd product with a row of a basis of `ker(G)` modulo `rowspan(H)`.
The search packs the column of each variable in these rows into 64-bit blocks, and tests a state with an empty syndrome by XORing the blocks of its variables, which gives zero exactly for a word of `rowspan(G)`.

**Weight divisor.**
If the weight of every word is a multiple of a **weight divisor** `d`, the search tries only weight limits that are multiples of `d`.
For example, each block row of the checks of a pair-partition code sums to the all-ones vector, so its words have even weight.
`infer_weight_divisor` of `_information_sets/basis_families.py` finds `d` from a basis of `ker(H)`, as BZ does.

**Exact distance.**
For each sector, the search keeps a lower bound `L` on the weights of its witnesses, starting at its divisor, and searches the sector with the lowest `L` at the weight limit `W = L`.
A witness found then is a lightest one over all sectors; otherwise every witness of the sector weighs more than `W`, and its `L` rises to the next multiple of its divisor.
The search ends once the lightest witness found weighs at most every `L`, and so needs no initial upper bound.
The sectors share the lightest witness found, as they do in BZ, and an upper bound that the caller passes, such as a known distance of another sector, acts as a witness of that weight.
A `cutoff` above the `L` of a sector makes its first limit the largest multiple of its divisor at most `cutoff`, and a witness within that limit is returned at once, though it can be heavier than the distance.

**Termination.**
Each limit ends, since the search from a root is a finite tree of depth at most `W`.
A sector without witnesses is not searched.
The lightest witness of any other sector, with `n` variables and divisor `d`, weighs a multiple of `d` that is at most `n`, so the sector takes at most `n // d` limits.
A search that needs more limits than the sum of these over its sectors has a bug, and raises a `RuntimeError` rather than looping forever.

**Interrupts.**
Each weight limit runs in one compiled call, which a keyboard interrupt cannot stop.
An interrupt takes effect after that call, as a `KeyboardInterrupt` or a `SystemError` that it caused, so the wait can be as long as the last limit of a large search.

### 3.7 Qubit codes that are not CSS

A Pauli operator on `n` qubits, up to a phase, is a vector `(a | b)` of `2n` bits, which acts on qubit `j` as `X` if `(a_j, b_j) = (1, 0)`, as `Z` if it is `(0, 1)`, and as `Y` if it is `(1, 1)`.
Two operators commute if their symplectic product vanishes, and the weight of an operator is the number of qubits on which it is not the identity.
For a qubit code, `H` holds the stabilizers, the words are the operators that commute with them, and `G` holds the gauge generators, which are the stabilizers for a stabilizer code.

Each qubit is a variable whose nonzero **values** are `X`, `Z`, and `Y`, and a **symbol** is a variable with one of its values.
A set of symbols on distinct variables stands for the operator with those values there, and a symbol **flips** a check that anticommutes with it on its qubit.
The Tanner graph joins each check to the symbols that flip it.
In the binary problems of Section 3.1, symbols are variables, each with one value.
Sections 3.2 to 3.6 carry over with symbols for variables:

- **Lemma.**
  The components of a support are those of the graph that joins each check to the qubits on which it acts.
  A check meets at most one component, so it commutes with the restriction of `x` to each, and the rest of the proof only sums restrictions.
  A symbol flips only checks that act on its qubit, so the search grows supports that are connected in this graph.
- **Search.**
  A state is a set `S` of symbols on distinct variables, with a set `F` of forbidden symbols, and the available symbols of an unsatisfied check `c` flip `c`, are not in `F`, and lie on variables outside `S`.
  While the search is in the subtree of a child, it blocks the other values of the variable of the child, since an operator has one value on each qubit.
  The symbol of a child joins `F` for its later siblings, but the other values of its variable do not, since a later sibling can lead to a witness with another value there.
  The search from a root starts from the empty state, whose children are the three symbols of the root.
- **Completeness.**
  For a state whose `S` holds only symbols of `x` and whose `F` holds none, each unsatisfied check `c` anticommutes with the restriction of `x` to the variables of `S` and commutes with `x`, so a symbol of `x` on another variable flips `c`.
  The first such symbol among the children of `c` is visited before it joins `F`, and earlier siblings block their other values only within their own subtrees.
- **Pruning.**
  A completion has one symbol on each new variable, and the overlaps, weights, and repairs of [Section 3.4](#34-pruning) are those of symbols.
  A `Y` flips the checks that act on its qubit as `X` or `Z`, an `X` those that act as `Z` or `Y`, and a `Z` those that act as `X` or `Y`, so in a CSS code with Hadamards on some qubits, where no check acts as `Y`, a `Y` flips every check on its qubit, which raises overlaps and lowers weights.
- **Symmetries.**
  A permutation of the qubits moves both halves of each vector, and qLDPC verifies it on all `2n` columns; the roots and ranks are those of variables, and the search from a root forbids every value of the variables of lower rank.
- **Logical test and divisor.**
  The test block of a `Y` is the sum of those of `X` and `Z`, and qLDPC gives these problems no weight divisor, so their limits step by one.

### 3.8 Vectors with a given syndrome

A **syndrome query** asks for a vector `x` of weight at most `W` with `Hx = σ`, for a nonzero syndrome `σ`; its solutions form a coset of `ker(H)`.
If `H` holds the stabilizers of a stabilizer state, a lightest solution is a coset leader, a lightest Pauli error with syndrome `σ`.
For a code with logical qubits, the query ranges over every logical class and ignores `G`.
The query searches as [Section 3.3](#33-the-depth-first-search) does, but from the empty support and without roots or ranks: the syndrome of a state is `σ + Hx_S`, where `x_S` is the vector with support `S`, and the search reports `S` once that syndrome is empty.

**Completeness.**
Let `x` be a lightest solution, and take a state with `S ⊊ supp(x)` and `F ∩ supp(x) = ∅`, such as the empty state.
The syndrome of the state is that of the restriction of `x` to `supp(x) \ S`, which is not empty, or else `x_S` would be a lighter solution, so the argument of [Section 3.3](#33-the-depth-first-search) carries over.
So do the pruning of [Section 3.4](#34-pruning), with completions that empty the syndrome, and the values of [Section 3.7](#37-qubit-codes-that-are-not-css).
The support of `x` need not be connected, but the restriction of `x` to each component flips a check of `σ`, through which the search can enter that component.

The roots and ranks, symmetries, weight divisor, and logical test of a distance search do not carry over: the nonzero syndrome gives the empty state a check to branch on, a syndrome is not invariant under the symmetries of the code, the weights of the solutions need not share a divisor, and every solution counts.

`CosetLeaderSearch` of `coset_leaders.py`, which `certify_bounded_fault_propagation` and the experimental `estimate_residual_weight_histogram` use, answers its queries by syndrome queries when built with `method="connected_cluster"`.
It searches Pauli errors for the physical weight, but the CSS weight `max(|a|, |b|)` of an error `(a | b)` is not a sum over qubits, so it supports that weight only on a CSS state, where it searches the X and Z parts of errors separately, and raises a `ValueError` on any other state.
Its one or two coset searches, each a `ConnectedClusterCosetSearch` in `_connected_cluster/coset.py`, take checks that can be dense ([Section 2](#2-what-the-cost-of-the-search-depends-on)), such as the row-reduced stabilizers that `certify_bounded_fault_propagation` passes, or the parts of pure-X and pure-Z stabilizers that `CosetLeaderSearch` row-reduces for the CSS weight.
So each replaces its checks with sparser checks of the same span, and maps the syndrome of each query to them.

The sparsest such checks are a lightest basis of the span, which the greedy algorithm for matroids finds by taking, again and again, the lightest word outside the span of those taken; but each step would be an exact search, whose cost grows quickly with the weight of the word.
So `get_sparser_rows` of `_connected_cluster/sparse_checks.py` instead takes candidates from row reductions in random orders of the columns, and keeps a lightest basis of the candidates and the current checks, so that their total weight never rises.
`max_sparsifying_rounds` and `sparsifying_rounds_without_improvement` limit the rounds, each a row reduction, and so trade the time to build a coset search against the speed of its queries.
Before and after the rounds, `get_sparser_rows` replaces a check by its sum with another while that makes it lighter.
Checks with the same span give the same answers and bounds, so the random orders, which a fixed seed makes reproducible, change only the speed and which errors the queries return.

The rounds need not find a lightest basis: a word of the span is a row of a reduction exactly if its support holds just one pivot, since each row is zero on the other pivots, and a heavier word is less likely to hold just one.
Even a lightest basis can be heavier than the checks of a code: for the physical weight on the state `|0⟩` of an `[[n, k]]` code, the span is the stabilizer group of the state, and at most `n - k` of its independent elements are stabilizers of the code.
So every basis holds at least `k` logical operators, each at least as heavy as the distance, which the greedy algorithm would have to find by exact searches.

A coset search first looks up the syndrome of a query in a table of the syndromes of the vectors of weight at most one, and searches only the syndromes that the table lacks, which have no solution of weight below two.
A search at limit `W` that finds no solution proves that every solution is heavier than `W`.
These two facts give the lower bounds of the answers, which the class docstring of `ConnectedClusterCosetSearch` states for each kind of query.

A query at limit `W` can wander through many heavy supports before it reaches a much lighter solution.
`certify_bounded_fault_propagation` queries batches of syndromes most of which have a solution within `W`, so for such batches a coset search tries the limits 2, 3, …, `W` in turn, and stops at the first within which it finds a solution, a lightest one, at the cost of repeating the lower limits for syndromes without one.
For `estimate_residual_weight_histogram`, `CosetLeaderSearch` finds minimum weights with the same increasing limits; [README.md](README.md#the-coset-leader-search) gives the floor of its searches by CSS weight.
A single query, `CosetLeaderSearch.get_min_weight_bound`, searches at `W` directly, since a caller may already raise `W` one step at a time.

Without reuse, a state computes the overlap and repair ([Section 3.4](#34-pruning)) of a symbol once for each unsatisfied check that it examines and the symbol flips.
The checks of a syndrome query can stay dense after the rounds, so a symbol can flip many unsatisfied checks, and the query computes both at most once per symbol and state.
A distance search, whose checks are usually sparse, recomputes both, since the bookkeeping would cost more than it saves.

### 3.9 Sparser stabilizers of subsystem codes

The distance search of a subsystem code needs checks whose row space is its stabilizer group ([Section 3.1](#31-the-problem)).
`get_distance_css` and `get_distance_pauli` derive a basis of that group, the **derived basis**, from the gauge generators by row reduction; it can be much heavier than the lightest generators of the group.
For example, in the rotated toric code on 256 qubits with one logical qubit made a gauge qubit, the plaquettes, of weight 4, generate the stabilizer group, but the derived basis has a mean weight of about 15 and rows of weight up to 46, so its search branches on many more variables at each state ([Section 2](#2-what-the-cost-of-the-search-depends-on)).
So with `method="connected_cluster"`, these functions give the search two sets of checks:

- the gauge generators that commute with every gauge generator, which are stabilizers, as given, with any redundant rows, but without those heavier than every row of the second set;
- the derived basis, made sparser by `get_sparser_rows` ([Section 3.8](#38-vectors-with-a-given-syndrome)) in up to `max_distance_sparsifying_rounds` rounds, which `sparsifying_rounds_without_improvement` can end sooner.
  With `max_distance_sparsifying_rounds` set to 0, a distance search skips sparsifying entirely and keeps the derived basis as it is; a coset search with `max_sparsifying_rounds` set to 0 still makes these sums ([Section 3.8](#38-vectors-with-a-given-syndrome)).

Together, the two sets span exactly the stabilizer group, since the second spans it and the first lies in it; so they have the kernel of the derived basis, and the automorphisms of the code apply as before ([Section 3.5](#35-symmetries)).
A distance search needs only the span of its checks, and no map of syndromes, so it can keep redundant checks.
These can strengthen it, since each unsatisfied check can prune a state or offer fewer branches.
Since a distance search uses its checks once, while a `CosetLeaderSearch` sparsifies once for many queries, `max_distance_sparsifying_rounds` defaults to fewer rounds than `max_sparsifying_rounds`.

Sparsity is not all that matters: bases of equal weight can make the search visit very different numbers of states.
So sparsifying changes the checks only when their total weight drops, which leaves a basis that is already as sparse as possible, such as the derived basis of a Bacon-Shor code, as it is.

## 4. Guide to the source code

The names below are in `_connected_cluster/search.py`, unless the table says otherwise.

| Concept | Code |
| --- | --- |
| Pair `(H, G)`, its words and witnesses | `checks` and `excluded` of `SupportSearch` and `get_distance_connected_cluster` |
| Sectors of a CSS code | `_search_css` of `core.py`, which `get_distance_css` and `get_distance_pauli` call, and `get_css_stabilizers` of `_binary.py` |
| Classical codes, and qubit codes that are not CSS | `get_distance_from_checks` and `get_distance_pauli` of `core.py` |
| Problems of Section 3.7, values, and symbols | `symplectic` of `SupportSearch`, its `_get_symbol_columns`, and `_shift_other_values` |
| Distance over several sectors | `get_distance_connected_cluster`, which advances `ConnectedClusterSearch` one weight limit at a time |
| Distance of a vector `v` from a classical code with generator matrix `G_C` | `ClassicalCode.get_distance_exact(vector=v)` in `src/qldpc/codes/common.py`, which, by the connected-cluster method, searches `([H \| Hv], [G_C \| 0])` and subtracts one |
| Search through a weight limit `W` | `SupportSearch.find_witness` |
| Syndrome query | `SupportSearch.find_vector_with_syndrome`, which coset searches skip for `SupportSearch._find_vector_with_packed_syndrome`, and the sentinel root `-1` of `_search_supports` |
| Overlaps and repairs that a syndrome query reuses within a state | `memoize` of `_search_supports`, and `memos` and `stamp` of `_get_branches` |
| Coset leaders of Pauli errors | `ConnectedClusterCosetSearch` in `_connected_cluster/coset.py`, and `CosetLeaderSearch` in `coset_leaders.py` |
| Sparser checks with the same span | `get_sparser_rows` and `_sparsify` in `_connected_cluster/sparse_checks.py`, and `_get_sparser_stabilizers` of `core.py` |
| Rounds of sparsifying | `max_sparsifying_rounds`, `max_distance_sparsifying_rounds`, and `sparsifying_rounds_without_improvement` of `SearchTuning` in `_tuning.py` |
| Table of light vectors, and increasing limits for batches and minimum weights | `ConnectedClusterCosetSearch` (its `_singles` and `get_min_weight_bound`), and `_get_batch_cluster_bound` and `_get_min_cluster_weight` of `CosetLeaderSearch` |
| Logical test | `SupportSearch._get_tests`, and `_is_outside_excluded` |
| Roots, ranks, and symmetries | `SupportSearch._use_automorphisms`, and `verify_automorphisms` and `get_orbits` of `_information_sets/automorphism_families.py` |
| States, branching, and undoing what a state forbade when the search leaves it | `_search_supports`, whose `levels` mark the available symbols |
| Weights, pruning, and the symbols that a state forbids for its subtree | `_get_branches`, and `restrict_viable_symbols` of `SearchTuning` in `_tuning.py`, whose `False` turns that forbidding off |
| When a search asks for symmetries | `automorphism_minimum_states` of `SearchTuning` in `_tuning.py` |
| Weight divisor | `infer_weight_divisor` of `_information_sets/basis_families.py` |

`get_distance_from_checks`, `get_distance_css`, and `get_distance_pauli` of `core.py` build the sectors of the search and pass it the active tuning, the last two after they derive the stabilizers of subsystem codes and make them sparser ([Section 3.9](#39-sparser-stabilizers-of-subsystem-codes)).
`SupportSearch` validates every sector, and the `get_distance_exact` methods of the code classes in `src/qldpc/codes/common.py` call these functions.
Nonbinary codes, `logical_counts.py`, and `get_distance_from_generators` reject `method="connected_cluster"`: counting needs every word of a weight rather than one witness, and generators are not checks.
Hypergraph product codes get their closed-form distances by searching the classical seed codes with the given method.
`_search_supports`, `_get_branches`, and `_is_outside_excluded` are compiled with `always_compiled`, and their small helpers with `always_compiled(inline=True)`, as [README.md](README.md) explains.

## 5. Correctness requirements

Changes to the search can use other bounds and orders, but must keep these requirements.

- **Prune only by bounds that hold for every witness that extends the state** ([Section 3.4](#34-pruning)).
  A bound must be a lower bound on the number of variables that such a witness adds, all of which are available ([Section 3.3](#33-the-depth-first-search)).
  A bound that is too strong can still give the right distance, so the tests need cases where the bound is tight at the distance, such as the tests of known distances.
- **Forbid a variable for a subtree only if no completion of the state within the weight limit holds it**, and make it available again when the search leaves the state.
- **Give the same exact distances, and the same answers and lower bounds of coset queries, with `restrict_viable_symbols` on and off** ([Section 3.4](#34-pruning)), so the tests compare both settings with enumeration.
- **Give each variable at most one value**: block the other values of the variable of a symbol while the symbol is in the support, unblock them when the search leaves its branch, and let a sibling forbid only its own symbol.
  Without the blocking, distances would not change, but a witness could hold two symbols on one variable and overstate its weight, so the tests check witnesses at every weight limit.
- **Forbid each earlier sibling's symbol, and every value of the variables of lower rank than the root.**
  Without this, the results would not change, and the search would only be slower, so the tests compare `num_states` with a count by hand.
- **Verify the automorphisms, and make each root the lowest-ranked variable of its orbit under all generators**, since roots from unverified permutations are unsound.
- **Preserve the logical test**: a state with an empty syndrome holds a witness exactly if its product with some test row is odd, and no word of `rowspan(G)` may be reported.
- **Use a proven weight divisor**, since a divisor that some witness lacks skips its weight.
- **Start a syndrome query from its syndrome alone**, without ranks, symmetries, divisor, or logical test ([Section 3.8](#38-vectors-with-a-given-syndrome)).
  The search keeps only the distinct nonzero checks, so the syndrome of the given checks must be mapped onto them, and a syndrome with a nonzero entry for a zero check, or with different entries for two equal checks, has no solution.
- **Reuse an overlap or a repair only within the state that computed it**, since another state can have another syndrome or other weights.
  Compute a repair only after weighing all the unsatisfied checks, since it sums their weights.
- **Give coset queries only proven lower bounds**: a bound of 2 needs the syndrome to be missing from the table of the vectors of weight at most 1, and a bound above a limit needs a complete search at that limit ([Section 3.8](#38-vectors-with-a-given-syndrome)).
  A caller that raises its limit to each bound would silently get wrong weights from a bound above the minimum, so the tests compare the exact bounds with brute force.
- **Keep the span of the checks of a coset query**, with an invertible map of the syndromes to the sparser checks, or the queries would search other cosets.
- **Keep the span of the stabilizers of a subsystem code exactly** ([Section 3.9](#39-sparser-stabilizers-of-subsystem-codes)): a smaller span would admit words that anticommute with a stabilizer, and a larger one would miss dressed logical operators; redundant rows need no map.
- **Return the exact minimum unless a cutoff permits an early exit**: the limits rise one divisor at a time, so that the first witness is a lightest one, and an early exit returns the weight of a known witness, an upper bound.

## References

- I. Dumer, A. A. Kovalev, and L. P. Pryadko, "Distance verification for classical and quantum LDPC codes", IEEE Transactions on Information Theory 63, 4675 (2017), [arXiv:1611.07164](https://arxiv.org/abs/1611.07164): the irreducible-cluster technique, and Lemma 3.
- A. A. Kovalev, I. Dumer, and L. P. Pryadko, "Linked-cluster technique for finding the distance of a quantum LDPC code", Information Theory and Applications Workshop, 2013, pp. 1–6, [arXiv:1302.1845](https://arxiv.org/abs/1302.1845): the linked-cluster technique, and Theorem 1.
- I. Dumer, A. A. Kovalev, and L. P. Pryadko, "Numerical techniques for finding the distances of quantum codes", IEEE International Symposium on Information Theory, 2014, pp. 1086–1090, [arXiv:1405.0348](https://arxiv.org/abs/1405.0348): the technique driven by unsatisfied checks.
- I. Dumer, A. A. Kovalev, and L. P. Pryadko, "Thresholds for correcting errors, erasures, and faulty syndrome measurements in degenerate quantum codes", Physical Review Letters 115, 050502 (2015), [arXiv:1412.6172](https://arxiv.org/abs/1412.6172): irreducible Pauli operators.
- K. Okada and K. Kasai, "Pair-partition constructions for CPM-based quantum LDPC codes", Section V, [arXiv:2607.14091](https://arxiv.org/abs/2607.14091): the branch and bound, and Lemma 2, with the implementation in [kasaikenta/pair-partition-cpm-css-codes](https://github.com/kasaikenta/pair-partition-cpm-css-codes).
- J. Y. Lee, K. Okada, N. Maskara, K. Kasai, and H. Zhou, "Design principles for ultra-high-rate quantum codes", [arXiv:2609.30069](https://arxiv.org/abs/2609.30069): the pair-partition codes, with the instances of [ultra-high-rate-qec/design-principles-data](https://github.com/ultra-high-rate-qec/design-principles-data), and the branch and bound in Appendix B.
- G. Lee, "Bicycle race: small two-block codes on the Pareto frontier", Appendix C.4, [arXiv:2610.13880](https://arxiv.org/abs/2610.13880): roots from the automorphisms of the Tanner graph.
- G. Richter, "Finding small stopping sets in the Tanner graphs of LDPC codes", 4th International Symposium on Turbo Codes and Related Topics and 6th International ITG-Conference on Source and Channel Coding, 2006, pp. 1–5.
- E. Rosnes and Ø. Ytrehus, "An efficient algorithm to find all small-size stopping sets of low-density parity-check matrices", IEEE Transactions on Information Theory 55, 4167 (2009).
- S. M. Khatami, L. Danjean, D. V. Nguyen, and B. Vasić, "An efficient exhaustive low-weight codeword search for structured LDPC codes", Information Theory and Applications Workshop, 2013, pp. 401–410.
- [QEC-pages/dist-m4ri](https://github.com/QEC-pages/dist-m4ri), by L. P. Pryadko and W. Zeng: a program whose connected-cluster (CC) method computes exact distances of classical codes, of quantum CSS codes, and of detector error models.
- M. Webster, A. Jacob, and O. Higgott, "Distance-finding algorithms for quantum codes and circuits", Section 4.3.2, [arXiv:2603.22532](https://arxiv.org/abs/2603.22532): other names of the algorithm.
