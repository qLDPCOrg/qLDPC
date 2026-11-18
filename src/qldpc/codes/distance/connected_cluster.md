# Connected-cluster distance search in qLDPC

## 1. Purpose and scope

The connected-cluster search computes the exact distance of a binary code from the Tanner graph of its parity checks.
The code can be classical, CSS, or a qubit code that is not CSS, and a quantum code can be a subsystem code.
The search is `method="connected_cluster"` of `get_distance_exact` on the code classes.
The default, `method="auto"`, runs it for the distances of quantum codes and the fault distances of circuits, unless enumerating every word is cheap.
`"auto"` switches to Brouwer-Zimmermann (BZ, see [brouwer_zimmermann.md](brouwer_zimmermann.md)) when BZ is predicted to cost less (section "Choosing a method" of [README.md](README.md)).
The connected-cluster search also finds light vectors with a given syndrome for `CosetLeaderSearch`, which `certify_bounded_fault_propagation` uses.
The search is the connected-cluster algorithm of Dumer, Kovalev, and Pryadko, and later works cite their 2017 paper for it (see the [references](#references)).
Its earliest known application to quantum codes is the linked-cluster technique of Kovalev, Dumer, and Pryadko (2013), for the distances of quantum LDPC codes.
Dumer, Kovalev, and Pryadko (2014) then grew each cluster through the checks that it leaves unsatisfied, as qLDPC does.
qLDPC follows the branch and bound of Okada and Kasai (Algorithm 2 in Section V) and of Lee et al. (Appendix B), who also extend it to stabilizer codes that are not CSS (Appendix B.2).
They branch on the unsatisfied check with the fewest choices, forbid the variables of earlier siblings, prune by lower bounds, and search from one root per orbit of cyclic shifts.
G. Lee (Appendix C.4) takes the roots from general groups of automorphisms of the Tanner graph, and [Section 3.5](#35-symmetries) says which automorphisms qLDPC uses.
Related searches of classical LDPC codes look for small stopping sets and low-weight codewords: those of Richter, of Rosnes and Ytrehus, and of Khatami et al.
The method's name, `"connected_cluster"`, follows `dist-m4ri`, the program of Pryadko and Zeng, which calls the search the "connected-cluster (CC) algorithm" of Dumer, Kovalev, and Pryadko.
Webster, Jacob, and Higgott (Section 4.3.2) note that it is "also referred to as the irreducible cluster or linked cluster algorithm".
This guide says what the cost of the search depends on ([Section 2](#2-what-the-cost-of-the-search-depends-on)), explains the algorithm and the techniques that it does not use ([Section 3](#3-the-algorithm)), maps it to the source code ([Section 4](#4-guide-to-the-source-code)), and lists what every change must preserve ([Section 5](#5-correctness-requirements)).

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
If `_search_css` finds a qubit permutation that maps each sector onto the other, the search takes the X sector alone.
It finds only the identity and, with pybliss, permutations that map the nonzero rows of `G_X`, as listed, one to one onto those of `G_Z` and back, where `G_X = H_X` and `G_Z = H_Z` for a stabilizer code (section "Searching one sector of a CSS code" of [README.md](README.md)).
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
The pruning of [Section 3.4](#34-pruning) keeps these states, since its bounds hold for any state that a witness of weight at most `W` extends, and it forbids only variables that no such witness holds, except for the overlap cuts, which [Section 3.4](#34-pruning) shows keep the states of one witness of minimum weight.

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

**Overlaps with known words.**
The bounds above hold for every witness that extends a state.
The search also forbids variables by a rule that holds only for one witness, chosen as follows.
Order the witnesses of minimum weight by their supports, each listed in increasing rank, compared lexicographically, and call the least one the **canonical** witness `x*`.
Let `g` be a **known word**: a row of `G`, which is in `rowspan(G)`, so that `x + g` is a witness whenever `x` is.
Then `wt(x + g) = wt(x) + |g| - 2|x ∩ g|`, so a witness `x` with `|x ∩ g| > |g|/2` is not of minimum weight.
If `|x* ∩ g| = |g|/2`, then `x* + g` is another witness of minimum weight, which differs from `x*` only on the support of `g`; if `x*` lacked the lowest-ranked variable `u` of `g`, then `x* + g` would hold it and agree with `x*` on every variable of lower rank, so `x* + g` would come first.
So `x*` holds `u`, and in either case `2|x* ∩ (g \ {u})| ≤ |g| - 1`.
If `|g|` is odd, a stronger rule needs no tie to break: `2|x ∩ g| ≤ |g|` gives `2|x ∩ g| ≤ |g| - 1` for every witness `x` of minimum weight, `u` included, as in Qubitserf ([references](#references)).
The **load** of a support `S` on a known word `g` is `2|S ∩ g|` if `|g|` is odd, and `2|S ∩ (g \ {u})|` otherwise, and `S` **passes** the cut of `g` if its load is at most the **capacity** `|g| - 1`.
So `x*` passes every cut.
The search forbids each variable that would raise the load of the support on `g` above the capacity, until the load drops.
The load of a state is at most that of any witness that extends it, so no variable of `x*` is forbidden at a state `S ⊊ supp(x*)`, and the completeness argument of [Section 3.3](#33-the-depth-first-search), applied to `x*`, still holds; since `x*` has the lowest-ranked least variable of all witnesses of minimum weight, the argument of [Section 3.5](#35-symmetries) applies to it too.
Since `x*` is never cut along its path, its completion from each such state uses only available variables, so the bounds and the forbidding of variables that are not viable, above, still keep its states.
The same argument applies to any lightest witness that passes every cut, and the search visits only supports that pass every cut.
So at any limit `W` at least the distance, the search finds a witness, though not necessarily a lightest one if `W` exceeds the distance.
At `W` equal to the distance, a root finds a witness exactly if it is the lowest-ranked variable of a lightest witness that passes every cut.
Below the distance, there is nothing to find.
A support within the limit has a load of at most `2W` on any word, so the search skips words with a capacity of at least `2W`.
Setting `stabilizer_overlap_cuts` of `SearchTuning` in `_tuning.py` to `False` turns these cuts off, and the search stays exact.

For a CSS code, the known words of the search of X-type logical operators are its X-type stabilizer generators, and those of a subsystem code its gauge generators.
The distance search of a classical code has no known words, since its `G` is empty, so the cuts change nothing there.
The search for the distance of a vector from a classical code (see [Section 4](#4-guide-to-the-source-code)) excludes the rows of the generator matrix of the code, which are rows of `G`, so it cuts by them, and the argument above applies.
The half-overlap rule is a property of minimum-weight vectors of a coset; the search programs Qubitserf (`cpp/distfind/cc.cpp`, which also leaves out the first variable of each even word), QEDist (`cssdist/native_css/exact.cpp`), public-regular-logical-frames (`src/lp_exclusion.cpp`) of J. Y. Lee, and tropidist (`cdist/gmitm.c`) prune by it ([references](#references)).
The cuts made the searches of surface, toric, and color codes, whose stabilizers are much lighter than their distances, visit about 4 to over 1000 times fewer states, and those of bivariate bicycle codes up to 3 times fewer.
For pair-partition and quantum Tanner codes, the cuts changed the number of states visited by at most about 5 percent, and for lifted-product codes by up to about a quarter either way.
At its last limit, the search stops at its first witness, and the cuts can forbid that witness, so that the search goes on to a later one.
The searches of these three families took up to several percent longer.

**Sums of pairs of checks.**
A distance search also adds sums of two of its checks to its checks, and sums of two of its known words to its known words, as Qubitserf and DistQLDPC do ([references](#references)).
A sum of two checks is in `rowspan(H)`, so every word satisfies it: the words, the witnesses, and the supports with an empty syndrome stay the same.
The arguments above use an unsatisfied check only through this: it holds an odd number of the variables of the support and an even number of those of any word, and so a variable of every completion.
In the problems of [Section 3.7](#37-qubit-codes-that-are-not-css), a sum of two checks likewise commutes with every word, so a symbol of every completion flips each unsatisfied sum.
So the completeness argument of [Section 3.3](#33-the-depth-first-search), the bounds, and the forbidding of variables that are not viable hold with the sums among the checks, if `Δ` in `b1` counts the sums too.
A sum of two rows of `G` is in `rowspan(G)`, so the argument of the overlap cuts holds for it as a known word.
The symmetries, the logical test, and the weight divisor depend only on `ker(H)` and `rowspan(G)`, which do not change.

A sum of two checks helps the search if an unsatisfied sum has fewer available variables than the unsatisfied checks, or adds weight to the dual, but every added check costs time at each state that it is unsatisfied in, and raises the overlaps that weigh the other checks.
So the search adds a sum only if it is not zero, differs from every check, and is no heavier than the heavier of its two checks, as both programs do.
A sum is no heavier than the heavier of its two checks only if the two share at least half of the lighter one.
The search weighs the sum of each pair of checks that share a variable by counting the variables that they share and those on which they agree, without forming the sum.
It takes these sums lightest first, until their total weight would exceed that of the checks, so that each state examines at most about twice as many entries of checks.
It chooses sums of known words in the same way, but sums only the known words no heavier than the heaviest check.
In a CSS code, the known words of each sector are the checks or gauge generators of the other sector.
But the rows of the generator matrix that the distance of a vector from a classical code excludes can be dense.
Finding their sums can cost far more than the search, and the cut of a word cuts only within limits above about half its weight.
Qubitserf adds sums only if every check weighs at most 16 and there are at most 512 checks on at most 2048 variables, and stops at four times as many rows as checks, or at 2048 rows.
The connected-cluster engine of DistQLDPC adds them by default, up to four times as many sums as checks.
The sums made the searches of quantum Tanner codes visit 5 to 8 times fewer states and take 2 to 4 times less time, the most for one with Hadamards on half of its qubits, which is not CSS.
They made the searches of some random codes with dense checks, CSS and not, take about a sixth to two fifths less time.
They made the search of the `[[47, 1, 11]]` quantum quadratic-residue code visit about a third fewer states, but take about a third longer.
They made the searches of color codes visit up to about 40 percent fewer states, but take up to a few milliseconds longer.
Without the bound on their total weight, the searches of quantum Tanner codes were faster still, but those of codes with dense checks took up to about 7 times longer.
Two checks of a bivariate bicycle or pair-partition code, or of a toric code, share too few variables for a light sum, so their searches do not change.
Surface codes have a few light sums at their boundaries, which changed their searches by about one percent in states and a few milliseconds in time.
Setting `pair_sum_checks` of `SearchTuning` in `_tuning.py` to `False` turns the sums off, and the search stays exact.

Every pruning rule must also keep the requirements on threads of [Section 5](#5-correctness-requirements), since threads search a costly weight limit by rebuilding each state from its path, and answer several syndrome queries of one search at once ([Section 3.11](#311-threads)).
[Section 3.10](#310-techniques-not-used) explains why qLDPC does not use some stronger bounds, forbidding rules, and lookups of other tools.

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

The generators of `Γ` are automorphisms that the caller of a distance function supplies on request, such as the `_known_permutation_automorphisms` of binary classical and quantum codes.
For a large code, finding the automorphisms, verifying them, and ordering their orbits can cost as much as visiting many states.
So the search of each sector asks for them at most once, before the first weight limit at which it expects to have visited `automorphism_minimum_states` states over all its limits.
That field of `SearchTuning` in `_tuning.py` defaults to a round number of states assumed to cost about as much to visit as asking.
The search predicts the number of states of the next limit from the ratio between the numbers of the last two.
These numbers are those of one thread on any number of threads, and a prediction must read nothing that depends on the threads ([Sections 3.11](#311-threads) and [5](#5-correctness-requirements)).
The search predicts rather than counts because it can ask only between limits, and the limit in which it would reach `automorphism_minimum_states` can be a large part of the search.
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
[Section 3.10](#310-techniques-not-used) explains why qLDPC does not search best first instead.

**Termination.**
Each limit ends, since the search from a root is a finite tree of depth at most `W`.
A sector without witnesses is not searched.
The lightest witness of any other sector, with `n` variables and divisor `d`, weighs a multiple of `d` that is at most `n`, so the sector takes at most `n // d` limits.
A search that needs more limits than the sum of these over its sectors has a bug, and raises a `RuntimeError` rather than looping forever.

**Interrupts.**
On one thread, each weight limit runs in one compiled call, which a keyboard interrupt cannot stop.
An interrupt takes effect after that call, as a `KeyboardInterrupt` or a `SystemError` that it caused, so the wait can be as long as the last limit of a large search.
A limit that runs on threads ([Section 3.11](#311-threads)) is one compiled call for each work item.
An interrupt takes effect once the items that the threads are searching end, which can take long, since an item can hold much of the work of a limit.

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
- **Overlaps with known words.**
  The known words are the gauge generators, the stabilizers of a stabilizer code.
  If `x` and a known word `g` have equal values on `a` qubits and different nonzero values on `b` qubits, then `wt(x + g) = wt(x) + |g| - 2a - b`, so the argument of [Section 3.4](#34-pruning) gives `2a + b ≤ |g|` for a witness of minimum weight, and `2a + b ≤ |g| - 1` without the lowest-ranked qubit `u` of `g` for the canonical one, which compares supports as before.
  So a symbol on a qubit of `g` other than `u` adds 2 to the load of `g` if it has the value of `g` there and 1 otherwise, and the capacity is again `|g| - 1`.
  Several witnesses of minimum weight can share the least support with different values; any of them serves as the canonical one, since the argument compares only supports.
  The rule for words of odd weight does not carry over, since `2a + b` can be odd: it applies only in binary problems, in which every symbol adds 2.
- **Symmetries.**
  A permutation of the qubits moves both halves of each vector, and qLDPC verifies it on all `2n` columns; the roots and ranks are those of variables, and the search from a root forbids every value of the variables of lower rank.
- **Logical test and divisor.**
  The test block of a `Y` is the sum of those of `X` and `Z`, and qLDPC gives these problems no weight divisor, so their limits step by one.

### 3.8 Vectors with a given syndrome

A **syndrome query** asks for a vector `x` of weight at most `W` with `Hx = σ`, for a nonzero syndrome `σ`; its solutions form a coset of `ker(H)`.
If `H` holds the stabilizers of a stabilizer state, a lightest solution is a coset leader, a lightest Pauli error with syndrome `σ`.
For a code with logical qubits, the query ranges over every logical class, and does not use `G` to decide which vectors count.
The query searches as [Section 3.3](#33-the-depth-first-search) does, but from the empty support, without roots, and with ranks only in its overlap cuts (below): the syndrome of a state is `σ + Hx_S`, where `x_S` is the vector with support `S`, and the search reports `S` once that syndrome is empty.

**Completeness.**
Let `x` be a lightest solution, and take a state with `S ⊊ supp(x)` and `F ∩ supp(x) = ∅`, such as the empty state.
The syndrome of the state is that of the restriction of `x` to `supp(x) \ S`, which is not empty, or else `x_S` would be a lighter solution, so the argument of [Section 3.3](#33-the-depth-first-search) carries over.
So do the pruning of [Section 3.4](#34-pruning), with completions that empty the syndrome, and the values of [Section 3.7](#37-qubit-codes-that-are-not-css).
The support of `x` need not be connected, but the restriction of `x` to each component flips a check of `σ`, through which the search can enter that component.

**Overlaps with known words.**
For any word `g` of `ker(H)`, `x + g` has the same syndrome as `x`, so the cuts of [Section 3.4](#34-pruning) apply with known words of `ker(H)`, and a lightest solution whose support comes first, the canonical one, passes every cut.
The order of the supports is that of the current ranks of the search, by index unless symmetries of an earlier distance search set them; any fixed order of the variables keeps the argument.
This is the classical property that a coset leader meets each word of the code in at most half of that word.
A query uses the rows of `G` and the words that `SupportSearch.use_query_overlap_words` adds as known words.
Distance searches do not use the added words: for a word `g` outside `rowspan(G)`, `x + g` need not be a witness, so the cut of `g` can forbid every lightest witness.
The stabilizers of a stabilizer state commute with each other, so they lie in `ker(H)`, which they span if the state is pure.
So `CosetLeaderSearch` gives the search of Pauli errors its own sparser checks as known words, and, for the CSS weight, gives the search of X parts the X parts of the pure-X stabilizers, the sparser checks of the search of Z parts, and vice versa.
The cuts can change which solution a query returns, but not whether the query finds one within a limit, so the answers and lower bounds of the queries do not change.

**Sums of pairs of checks.**
The sums of checks of [Section 3.4](#34-pruning) carry over to queries: every vector of `ker(H)` satisfies a sum of two checks, so the completeness argument above holds with it among the checks, and its syndrome bit is the sum of those of its two checks, which a query adds to its syndrome.
So a `SupportSearch` with `pair_sum_checks` adds them to its queries too, but not the sums of known words, which only its distance searches use.
The coset searches of `CosetLeaderSearch` add no sums: they made batches of queries on states of the gross code, whose sparser stabilizers have light sums, visit 2 to 3.5 times more states and take 3 to 4.5 times as long, and saved no time on the other codes that we timed.

The roots, the forbidding of variables of lower rank than the root, the symmetries, the weight divisor, and the logical test of a distance search do not carry over: the nonzero syndrome gives the empty state a check to branch on, a syndrome is not invariant under the symmetries of the code, the weights of the solutions need not share a divisor, and every solution counts.

`CosetLeaderSearch` of `coset_leaders.py`, which `certify_bounded_fault_propagation` and the experimental `estimate_residual_weight_histogram` use, answers its queries by syndrome queries when built with `method="connected_cluster"`.
It searches Pauli errors for the physical weight, but the CSS weight `max(|a|, |b|)` of an error `(a | b)` is not a sum over qubits, so it supports that weight only on a CSS state, where it searches the X and Z parts of errors separately, and raises a `ValueError` on any other state.
Its one or two coset searches, each a `ConnectedClusterCosetSearch` in `_connected_cluster/coset.py`, take checks that can be dense ([Section 2](#2-what-the-cost-of-the-search-depends-on)), such as the row-reduced stabilizers that `certify_bounded_fault_propagation` passes, or the parts of pure-X and pure-Z stabilizers that `CosetLeaderSearch` row-reduces for the CSS weight.
So each replaces its checks with sparser checks of the same span, and maps the syndrome of each query to them.
Batches run the queries of one coset search on several threads at once ([Section 3.11](#311-threads)), and so can callers that share a `CosetLeaderSearch` (section "Threads" of [README.md](README.md#threads)).
So a query only reads its coset search and the tables of its `SupportSearch` ([Section 5](#5-correctness-requirements)).

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
  With `max_distance_sparsifying_rounds` set to 0, a distance search skips sparsifying entirely and keeps the derived basis as it is; a coset search with `max_sparsifying_rounds` set to 0 still replaces checks by lighter sums with other checks, before and after the rounds ([Section 3.8](#38-vectors-with-a-given-syndrome)).

Together, the two sets span exactly the stabilizer group, since the second spans it and the first lies in it; so they have the kernel of the derived basis, and the automorphisms of the code apply as before ([Section 3.5](#35-symmetries)).
A distance search needs only the span of its checks, and no map of syndromes, so it can keep redundant checks.
These can strengthen it, since each unsatisfied check can prune a state or offer fewer branches.
Since a distance search uses its checks once, while a `CosetLeaderSearch` sparsifies once for many queries, `max_distance_sparsifying_rounds` defaults to fewer rounds than `max_sparsifying_rounds`.

Sparsity is not all that matters: bases of equal weight can make the search visit very different numbers of states.
So sparsifying changes the checks only when their total weight drops, which leaves a basis that is already as sparse as possible, such as the derived basis of a Bacon-Shor code, as it is.

### 3.10 Techniques not used

qLDPC tried or considered the techniques below, from public tools and papers, and does not use them.
This section records why, so that later contributors do not repeat them without new evidence.
The paragraphs use the terms of [Section 3.4](#34-pruning): the number `s` of variables of a state, the weight limit `W`, completions, the weights `y_c = 1/m_c` of the unsatisfied checks and their sum `Y`, and viable variables.
Some also use symbols ([Section 3.7](#37-qubit-codes-that-are-not-css)), which are the variables in binary problems.

**Dual ascent.**
Dual ascent raises each weight `y_c` of the bound `Y` as far as its constraints allow.
It costs further passes over the unsatisfied checks of every state, and qLDPC does not use it.

**Repairs through satisfied checks.**
QEDist prunes by repairs through satisfied checks, the checks that the support satisfies ([references](#references)); these repairs move dual weight onto satisfied checks, and are not the repairs `R_v` of [Section 3.4](#34-pruning).
Such a check holds an even number of the variables of a completion.
So a completion that holds an available variable `v` of the check holds another available variable of it.
A dual solution may therefore move weight from `v`, if `v` carries more than one, onto the other available variables of the check.
The bound can then exceed every bound of the hitting-set problem of [Section 3.4](#34-pruning).
QEDist adds weight greedily to up to 16 unsatisfied checks.
Each time, it moves the excess of each overloaded variable onto the first satisfied check with room.
It also searches a bounded number of steps for more disjoint checks.
A trial in qLDPC started from the weights `y_c = 1/m_c` and moved the excess weight in the same way.
It visited about 3 times fewer states on the [[144,12,12]] and [[288,12,18]] bivariate bicycle codes, about 1.3 to 1.9 times fewer on quantum Tanner and pair-partition codes, and about 1.1 to 1.3 times fewer on surface, toric, and color codes.
The search for disjoint checks alone would have removed under a tenth of the states.
But a run of the bound visits the satisfied checks of every overloaded variable, which costs several states on codes with heavy checks.
So the searches took about 1.2 to 1.4 times less time on the two bivariate bicycle codes.
But they took about 1.8 times longer on a quantum Tanner code, 1.5 times longer on a pair-partition code, 1.2 times longer on a larger one, and about 1.1 times longer on a color code.
Other variants of the trial pruned more on some codes, but none was faster than qLDPC's search on the quantum Tanner or color code.

**Erasure bound.**
Qubitserf and DistQLDPC prune by an erasure bound ([references](#references)).
Only the available variables of the unsatisfied checks carry weight in the dual of [Section 3.4](#34-pruning).
So a completion of at most `W - s` variables holds at least `⌈Y⌉` of them, and at most `e = W - s - ⌈Y⌉` other available variables.
For `e` up to 2, the erasure bound prunes the state unless some available variables of the unsatisfied checks, in any number, and at most `e` other available variables complete the support to a witness.
In a syndrome query, they must complete it to a vector with the syndrome.
Peeling and elimination over GF(2) decide this.
A hash of the checks lets them try every other variable, and every pair of them, at once.
Unlike the bounds of Section 3.4, the erasure bound also uses the checks that the support satisfies.
A trial in qLDPC visited up to about 15 times fewer states with it.
But each run of the bound cost about as much as 10 states.
So the searches took about a quarter longer on a quantum Tanner code and on the [[288,12,18]] bivariate bicycle code, and up to about 2.5 times longer on pair-partition codes and on syndrome queries.
They took about a third less time only on the [[144,12,12]] bivariate bicycle (gross) code and a color code, which took a few hundredths of a second.

**Forbidding the only available symbol of a satisfied check.**
A completion holds an even number of the symbols that flip a check that the support satisfies.
In a syndrome query, a check is satisfied if it is not in the syndrome `σ + Hx_S` of the state ([Section 3.8](#38-vectors-with-a-given-syndrome)).
So a completion cannot hold a symbol that is the only available one of such a check, and the search could forbid that symbol for the subtree.
Forbidding it can leave another satisfied check with one available symbol, or an unsatisfied check with none, which prunes the state.
Nearly all such checks arise at states that allow only viable symbols, where a pass over the checks of the allowed symbols finds them.
A trial in qLDPC ran this pass, and its closure, at those states.
It visited about 1.2 to 1.9 times fewer states on bivariate bicycle, quantum Tanner, pair-partition, and color codes, and on searches of pair-partition codes that are not CSS.
But the states that it removed were mostly ones that the search would have left at once, since adding the forbidden symbol leaves the check unsatisfied with no available symbol.
So the searches took about the same time on the bivariate bicycle and color codes and on a small pair-partition code.
They took about 1.05 to 1.15 times longer on a quantum Tanner code, on a larger pair-partition code, and on searches of pair-partition codes that are not CSS.

**Pairs that one variable could replace.**
tropidist forbids a variable that, with a variable of the support, a single variable could replace ([references](#references)).
It gives each variable `v` a logical label, the column of `v` in the rows of its logical observables, which play the role of the test rows of [Section 3.6](#36-from-a-search-to-the-distance).
With `h_v` the column of `v` in `H`, it forbids `b` and `c` in the subtree of `a` if `h_a + h_b = h_c` and the labels of `a` and `b` sum to that of `c`.
Then `{a, b, c}` is the support of a word of `rowspan(G)` of weight 3.
The sum of this word with a witness that holds two of `a`, `b`, and `c` is a lighter witness, so no witness of minimum weight holds two of them.
A syndrome query could use any word of `ker(H)` of weight 3.
A search with the values of [Section 3.7](#37-qubit-codes-that-are-not-css) could also use a word of weight 2, whose qubits replace a symbol on each by one symbol on one of them.
In binary problems, the rule is the overlap cut of [Section 3.4](#34-pruning) by a known word of weight 3 that is not a row of `G`.
The distance search of a classical code has no such words, since its `G` is empty.
A count found none of weight 3 in the surface, toric, color, bivariate bicycle, generalized bicycle, pair-partition, quantum Tanner, and quantum Golay codes that we timed, in their sectors, their searches with Hadamards on some qubits, and their queries on the encoded state `|0…0⟩`.
A CSS code whose stabilizers all have even weights, as these do, has none in `rowspan(G)` in any case.
Only the surface codes have words of weight 2, their boundary stabilizers, which are rows of `G` and so already cut in binary problems.
So the rule would change none of these searches except the searches and queries of surface codes with the values of Section 3.7, and qLDPC does not use it.

**Lookups of the last variables.**
dist-m4ri and Qubitserf look up the last one or two variables of a support in a table of syndromes ([references](#references)).
With the bounds and the overlap cuts of [Section 3.4](#34-pruning), states with room for at most two more variables were observed to be few of the states that a search visits.
Lookups of one or two variables were tried on the codes of `experiments/distance_tuning/workloads.py` and on the surface, toric, color, bivariate bicycle, pair-partition, and quantum Tanner codes that we timed.
They saved no measurable time.

**Best-first search.**
qLDPC raises the weight limit one divisor at a time, and searches each limit depth first ([Section 3.6](#36-from-a-search-to-the-distance)).
This is iterative deepening, and with the bound `⌈Y⌉`, it is the iterative-deepening form of A*.
The decoder Tesseract and the search of Banfi et al. instead expand states in the order of the cost of `S` plus such a bound ([references](#references)).
The cost of `S` is `|S|` for unit costs.
Any such search must still expand every state that the last limit without a witness visits.
So it can save at most the states of the other limits: the lower ones, and the one that finds a witness.
On the slower searches that we measured, these were about half of the states of a quantum Tanner code, a quarter of those of the [[288,12,18]] bivariate bicycle code and of a search of a pair-partition code that is not CSS, and a tenth of those of the search of a larger pair-partition code that is CSS.
Estimates for qLDPC's search found that best-first variants would visit about 0.6 to 2.5 times as many states as it does on these slower searches.
The reason is that they lose part of the forbidding of variables that are not viable, which depends on the limit.
They would also keep millions of states in memory.
On surface, toric, and color codes, whose searches take hundredths of a second, they would visit about 0.1 to 0.5 times as many states.
Best-first variants would also replace the depth-first search of each limit, on which the predictions of [Section 3.5](#35-symmetries), `method="auto"`, interrupts ([Section 3.6](#36-from-a-search-to-the-distance)), and the split of a limit across threads ([Section 3.11](#311-threads)) rely, so qLDPC does not use them.

### 3.11 Threads

A costly weight limit of a distance search runs on `num_threads` threads of `SearchTuning`, which share the read-only tables of the search; the section "Threads" of [README.md](README.md#threads) says how to set their number.
The threads give the same witness and count of states as a search on one thread, so that the decisions that depend on them, such as when to ask for symmetries ([Section 3.5](#35-symmetries)), the growth of the states, and the switch of `method="auto"` (section "Choosing a method" of [README.md](README.md)), do not depend on the number of threads.

**Work items.**
The **frontier** at depth `F` is the set of the states with `F` symbols that the search enters, and the frontier at depth 0 is the set of roots.
A **work item** is a state of the frontier with its subtree, and is named by its root and by the branch, counting from zero, that each state on its path takes.
In DFS order, the items partition the states of at least `F` symbols, and the other states are **above** the frontier.

**Replay.**
A thread searches an item by replaying its path from the root: at each state of the path, it marks the earlier siblings of the next state as tried, and takes the next state.
What the search has forbidden, blocked, and loaded at a state depends only on the state's path, its earlier siblings, and the forbidden symbols of the root, since the search undoes what a state forbade, blocked, and loaded on the overlap cuts ([Sections 3.4](#34-pruning) and [3.7](#37-qubit-codes-that-are-not-css)) when it leaves the state.
Once the search leaves the subtree of an earlier sibling, the sibling's net effect is only that its symbol stays tried, even if the bound `b1` skipped its state.
So the replay rebuilds the forbidden and blocked symbols, the restrictions, and the loads exactly as one thread has them, and the thread then searches the subtree of the item as one thread does, until it goes back above the frontier.

**Choosing the frontier.**
The search starts with the roots as items, and deepens the frontier one depth at a time while it has fewer than 32 items for each thread, so that the claims below can share the work among the threads.
Each pass replays the current items, enters their states, which are then above the frontier, and lists their children rather than entering them.
The deepening ends at the weight limit, or at a witness that a pass enters, after which no listed item comes in DFS order.
It also ends once no state is deep enough, and then the passes have searched the whole limit.
The search keeps the frontier with the most items, unless a pass enters a witness, when it keeps that pass's frontier.
Each state above the final frontier is entered by one pass, so the passes cost about as much as those states and the paths of the items, and, if the search keeps a wider frontier than the last, also the states that the later passes entered.

**Claims.**
The threads claim ranges of items in DFS order, each about `1 / (2 num_threads)` of the weight of the items that no thread has claimed, so that the ranges shrink as the work ends (`run_work_items` of `_threads.py`).
The subtrees of the roots shrink quickly in the order of their ranks, so the work of an item can differ from that of the next by orders of magnitude.
So each item weighs one more than an even share, over the items of its root, of the states from that root in the last limit of the search without a witness, if that limit had the same roots.
Without such a limit, as at the first limit after the search starts or takes symmetries, every item weighs the same, and the first claims can hold most of the work.
The weights change only which thread searches an item, not the result.

**Ordered commit.**
A thread searches the items of a range one at a time.
Once a thread finds a witness in an item, the threads skip the later items that have not started, and finish the earlier ones, whose witnesses would come first; an item that is already running ends on its own.
The search returns the witness of the first item that has one, or else the witness that a pass entered, if any.
It counts the states above the frontier before that witness in DFS order, the states of the earlier items, and the states of the item of the witness up to the witness, which is the count of one thread; without a witness, it counts every state.

**When to split.**
A limit runs on threads only if `num_threads` exceeds one and the search predicts ([Section 3.5](#35-symmetries)) that the limit visits more than `num_threads * 2**information_set_batch_exponent` states, as many as the supports of a chunk of a BZ scan for each thread ([Section 4.5 of brouwer_zimmermann.md](brouwer_zimmermann.md#45-behavior-spanning-files)).
A split costs far fewer states than the gate, which reuses the chunk size of BZ scans, so the gate is conservative: lowering `information_set_batch_exponent` lets smaller limits split.
By the predictions of [Section 3.5](#35-symmetries), a first limit that a `cutoff` sets above the weight divisor runs on threads, and a first limit at the divisor does not.

**Syndrome queries.**
A syndrome query ([Section 3.8](#38-vectors-with-a-given-syndrome)) runs on one thread.
The batch queries of `CosetLeaderSearch` instead move the rest of a batch to threads, by syndromes, once their queries on the calling thread have visited `num_threads * 2**information_set_batch_exponent` states, and give the answers of one thread (section "The coset-leader search" of [README.md](README.md#the-coset-leader-search)).

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
| Weight limits on threads: the frontier, work items, replay, weighted claims, ordered commit, and when to split | `_search_on_threads`, `paths` and `frontier_depth` of `_search_supports`, `num_threads` and `split_minimum_states` of `SupportSearch`, and `run_work_items` of `_threads.py`; `num_threads` of `SearchTuning` in `_tuning.py` |
| Batches of syndrome queries on threads | `_find_first_cluster_bound` and `_get_min_cluster_weights` of `CosetLeaderSearch` in `coset_leaders.py`, `ConnectedClusterCosetSearch._get_min_weight_bound_and_states` in `_connected_cluster/coset.py`, which counts the work of a query, and `run_blocks` of `_threads.py` |
| Weights, pruning, and the symbols that a state forbids for its subtree | `_get_branches`, and `restrict_viable_symbols` of `SearchTuning` in `_tuning.py`, whose `False` turns that forbidding off |
| Overlaps with known words: loads, capacities, and the symbols that they forbid | `_get_overlap_cuts`, `_shift_cut_loads`, `SupportSearch.use_query_overlap_words`, and `stabilizer_overlap_cuts` of `SearchTuning` in `_tuning.py`, whose `False` turns the cuts off; `ConnectedClusterCosetSearch.use_query_overlap_words` and the constructor of `CosetLeaderSearch` for queries |
| Sums of pairs of checks and of known words | `get_light_pair_sums` in `_connected_cluster/sparse_checks.py`, the `summed_pairs` of `SupportSearch` and its `_find_vector_with_packed_syndrome`, which maps syndromes to them, its `witness_words`, the known words of `find_witness`, and `pair_sum_checks` of `SearchTuning` in `_tuning.py`, whose `False` turns them off |
| When a search asks for symmetries | `automorphism_minimum_states` of `SearchTuning` in `_tuning.py` |
| Weight divisor | `infer_weight_divisor` of `_information_sets/basis_families.py` |

`get_distance_from_checks`, `get_distance_css`, and `get_distance_pauli` of `core.py` build the sectors of the search and pass it the active tuning, the last two after they derive the stabilizers of subsystem codes and make them sparser ([Section 3.9](#39-sparser-stabilizers-of-subsystem-codes)).
`SupportSearch` validates every sector, and the `get_distance_exact` methods of the code classes in `src/qldpc/codes/common.py` call these functions.
Nonbinary codes, `logical_counts.py`, and `get_distance_from_generators` reject `method="connected_cluster"`: counting needs every word of a weight rather than one witness, and generators are not checks.
Hypergraph product codes get their closed-form distances by searching the classical seed codes with the given method.
`_search_supports`, `_get_branches`, and `_is_outside_excluded` are compiled with `always_compiled`, and their small helpers with `always_compiled(inline=True)`, as [README.md](README.md) explains.
Threads run `_search_supports` at once, so it is compiled with `nogil=True`, and writes only to arrays that it allocates.

## 5. Correctness requirements

Changes to the search can use other bounds and orders, but must keep these requirements.

- **Prune only by bounds that hold for every witness that extends the state** ([Section 3.4](#34-pruning)).
  A bound must be a lower bound on the number of variables that such a witness adds, all of which are available ([Section 3.3](#33-the-depth-first-search)).
  A bound that is too strong can still give the right distance, so the tests need cases where the bound is tight at the distance, such as the tests of known distances.
- **Forbid a variable for a subtree only if no completion of the state within the weight limit holds it**, and make it available again when the search leaves the state.
- **Forbid a symbol by an overlap cut at a state only if the canonical witness, or canonical solution, does not extend the state or lacks the symbol** ([Section 3.4](#34-pruning)), with the lowest-ranked variable of each word under the current ranks of the search, which a syndrome query uses too, and loads that are undone when a symbol leaves the support.
  The canonical witness has a support that comes first in the order of these ranks, with ties in values broken arbitrarily.
  A distance search may use only rows of `G` as known words, while a syndrome query may use any word of `ker(H)` ([Section 3.8](#38-vectors-with-a-given-syndrome)).
  A cut that ignored the ranks that symmetries set could forbid the only witness that the roots reach, so the tests check that each root finds exactly the lightest witnesses that pass every cut, with and without symmetries.
- **Add only checks in the span of the checks, and only known words in `rowspan(G)` for a distance search or in `ker(H)` for a syndrome query** (Sections [3.4](#34-pruning) and [3.8](#38-vectors-with-a-given-syndrome)), map the syndrome of a query to the added checks, and count the added checks in `Δ`.
  A sum of a check and a vector outside the span would cut witnesses, and too small a `Δ` would make `b1` prune states that a witness extends.
- **Give the same exact distances, and the same answers and lower bounds of coset queries, with `restrict_viable_symbols` and `stabilizer_overlap_cuts` on and off, and the same exact distances and answers of the syndrome queries of `SupportSearch` with `pair_sum_checks` on and off** ([Section 3.4](#34-pruning)), so the tests compare these settings with enumeration.
- **Give each variable at most one value**: block the other values of the variable of a symbol while the symbol is in the support, unblock them when the search leaves its branch, and let a sibling forbid only its own symbol.
  Without the blocking, distances would not change, but a witness could hold two symbols on one variable and overstate its weight, so the tests check witnesses at every weight limit.
- **Forbid each earlier sibling's symbol, and every value of the variables of lower rank than the root.**
  Without this, the results would not change, and the search would only be slower, so the tests compare `num_states` with a count by hand.
- **Verify the automorphisms, and make each root the lowest-ranked variable of its orbit under all generators**, since roots from unverified permutations are unsound.
- **Preserve the logical test**: a state with an empty syndrome holds a witness exactly if its product with some test row is odd, and no word of `rowspan(G)` may be reported.
- **Use a proven weight divisor**, since a divisor that some witness lacks skips its weight.
- **Start a syndrome query from its syndrome alone**, without roots, symmetries, divisor, or logical test, and with ranks only in its overlap cuts ([Section 3.8](#38-vectors-with-a-given-syndrome)).
  The search keeps only the distinct nonzero checks, so the syndrome of the given checks must be mapped onto them, and a syndrome with a nonzero entry for a zero check, or with different entries for two equal checks, has no solution.
- **Reuse an overlap or a repair only within the state that computed it**, since another state can have another syndrome or other weights.
  Compute a repair only after weighing all the unsatisfied checks, since it sums their weights.
- **Give coset queries only proven lower bounds**: a bound of 2 needs the syndrome to be missing from the table of the vectors of weight at most 1, and a bound above a limit needs a complete search at that limit ([Section 3.8](#38-vectors-with-a-given-syndrome)).
  A caller that raises its limit to each bound would silently get wrong weights from a bound above the minimum, so the tests compare the exact bounds with brute force.
- **Keep the span of the checks of a coset query**, with an invertible map of the syndromes to the sparser checks, or the queries would search other cosets.
- **Keep the span of the stabilizers of a subsystem code exactly** ([Section 3.9](#39-sparser-stabilizers-of-subsystem-codes)): a smaller span would admit words that anticommute with a stabilizer, and a larger one would miss dressed logical operators; redundant rows need no map.
- **On threads, give the same witness and count of states as a search on one thread, and give the batches of `CosetLeaderSearch` the answers of one thread, including the first index of a syndrome without an error within `max_weight`** ([Section 3.11](#311-threads)): replay the path of each work item exactly, run every item before the first with a witness in full, and count the states above the frontier once.
  The decisions that depend on these counts would otherwise depend on the number of threads.
  So of what a limit leaves, decisions must read only its witness and count of states, and the states of each root only after a limit without a witness, since on threads the items after the first witness may or may not have run.
  In `_connected_cluster/search_test.py`, `test_limits_on_threads_give_the_witnesses_and_states_of_one_thread`, `test_distance_searches_on_threads_take_the_steps_of_one_thread`, and `test_each_support_is_visited_once_above_its_least_variable` compare limits on threads with one thread, including a witness above the frontier and in the first and last items.
  `test_limits_on_threads_keep_the_first_witness_in_dfs_order` checks the first witness in either order of the threads, and `test_a_range_on_threads_stops_after_an_earlier_item_ends_the_limit` checks that a thread searches no later item of its range once an earlier item has a witness or an error.
  In `coset_leaders_test.py`, `test_connected_cluster_batches_on_threads_answer_as_on_one_thread` and `test_connected_cluster_batches_keep_the_least_index_whichever_block_ends_first` compare the batches on threads with one thread.
  - **Keep what the search has forbidden, blocked, and loaded at a state a function of the state's path and earlier siblings, so that replay rebuilds it** ([Section 3.11](#311-threads)).
    A pruning rule may keep per-state data only if leaving the state undoes it, as `levels` and the loads of the overlap cuts do, and may add checks and known words only as tables built before the search, as the light sums of pairs of checks are ([Section 3.4](#34-pruning)).
    It must not carry memos, learned constraints, or sets of states already seen from one state or root to another, and must keep the depth-first order of the search.
    A change that breaks this, such as a best-first search ([Section 3.10](#310-techniques-not-used)), needs its own way to split a limit across threads.
  - **Keep `_search_supports` writing only to arrays that it allocates**, and never to the tables of the search, since threads run it without the global interpreter lock.
    The other code that threads run writes only its own entries of shared results, such as the counts of states of its items and the weights of its rows, or writes under a lock, as when it records a witness or a syndrome without an error in reach.
    Nothing that threads run calls numba code compiled with `parallel=True` or `prange` (`run_work_items` of `_threads.py`).
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
- [QEC-pages/dist-m4ri](https://github.com/QEC-pages/dist-m4ri), by L. P. Pryadko and W. Zeng: a program whose connected-cluster (CC) method computes exact distances of classical codes, of quantum CSS codes, and of detector error models; its `src/dist_m4ri.c`, lines 827–877 at v0.16.1 (commit `4cc9c12`), looks up the last one or two columns of a support by a hash of the syndrome.
- M. Webster, A. Jacob, and O. Higgott, "Distance-finding algorithms for quantum codes and circuits", Section 4.3.2, [arXiv:2603.22532](https://arxiv.org/abs/2603.22532): other names of the algorithm.
- [Quantinuum/Qubitserf](https://github.com/Quantinuum/Qubitserf/blob/04b9349a2d421815c35b2a3d53878167e7237f47/cpp/distfind/cc.cpp), `cpp/distfind/cc.cpp`, lines 251–288 and 421–431 at commit `04b9349`: cuts by the overlaps with stabilizers, leaving out the first qubit of each stabilizer of even weight; and lines 445–475 and 551–554: a hash lookup of the last two qubits of a support, and a scan of one check for the last qubit.
  Lines 182–213 and 217–219: sums of pairs of checks, and of pairs of stabilizers, no heavier than the heavier of each pair, added as checks and known words until there are four times as many rows, or 2048, and only if every row weighs at most 16 and there are at most 512 rows and 2048 columns (lines 183, 187, and 196).
  Its [`cpp/distfind/erasure_bound.hpp`](https://github.com/Quantinuum/Qubitserf/blob/04b9349a2d421815c35b2a3d53878167e7237f47/cpp/distfind/erasure_bound.hpp), lines 114–260, with lines 527–540 of `cc.cpp`, prunes by the erasure bound.
- [guluchen/DistQLDPC](https://github.com/guluchen/DistQLDPC/blob/fae495bce77d6fff2cbb308aaca0676b45c8e98e/src/cc/cc_half.h), `src/cc/cc_half.h`, lines 170–205 at commit `fae495b`: the erasure bound, behind a gate that times its tests; and lines 139–160: up to four times as many sums of pairs of checks, and of pairs of stabilizers, no heavier than the heavier of each pair, as redundant checks and known words, which its connected-cluster engine adds by default (its rules `sparse` and `sparseov`, on by default at lines 229 and 237–241, and in its `README.md`, line 160).
- [yeon-lee/QEDist](https://github.com/yeon-lee/QEDist/blob/71a0f0c1020ffa48f6839e15c1a24b6642a8480e/cssdist/native_css/exact.cpp), `cssdist/native_css/exact.cpp`, lines 136–138 and 222 at commit `71a0f0c`, and its repairs through satisfied checks at lines 267–308, with lines 57–102 of `cssdist/native_css/even_methods.inc`, and its search for disjoint checks at lines 189–201 and 311–320, and J. Y. Lee, "Designing group-valued codes with full regular low-weight bases", [arXiv:2610.06820](https://arxiv.org/abs/2610.06820), with [yeon-lee/public-regular-logical-frames](https://github.com/yeon-lee/public-regular-logical-frames/blob/04ad4795ad15cb89ee47b7868ff925c30b2fc731/src/lp_exclusion.cpp), `src/lp_exclusion.cpp`, lines 68–69 at commit `04ad479`: pruning of supports that meet a stabilizer in more than half of it.
- [Fanerst/tropidist](https://github.com/Fanerst/tropidist/blob/b03ec2d4077f287f89f97b5c2c16ce5ce17c2c58/cdist/gmitm.c), `cdist/gmitm.c`, lines 388–416 at commit `b03ec2d`, the program of Cheng, Guo, and Pan, [arXiv:2611.08247](https://arxiv.org/abs/2611.08247): the same rule, as "kernel dominance", by sets of variables with zero syndrome and zero logical value (see lines 207–210 for the name); and lines 364–385, with lines 207–231 of `cdist/search.py`, which forbid a variable that, with a variable of the support, one variable could replace.
- L. Aghababaie Beni, O. Higgott, and N. Shutty, "Tesseract: a search-based decoder for quantum error correction", Section 3, Algorithms 1–3, [arXiv:2503.10988](https://arxiv.org/abs/2503.10988): best-first search by the cost of the chosen errors plus a bound of the same kind as `⌈Y⌉`.
- M. Banfi, M. Ferrari, A. Favano, A. Tarable, and L. Barletta, "Certified non-degenerate maximum-likelihood benchmarking of quantum LDPC decoders", Sections V-A to V-C, [arXiv:2610.28008](https://arxiv.org/abs/2610.28008): best-first search by the cost of the chosen errors plus a packing bound.
