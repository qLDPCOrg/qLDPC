# Connected-cluster distance search in qLDPC

## 1. Purpose and scope

The connected-cluster search computes the exact distance of a binary code from the Tanner graph of its parity checks.
The code can be classical, CSS, or a qubit stabilizer code that is not CSS, and a quantum code can be a subsystem code.
The search is the method `"connected_cluster"` of the `get_distance_exact` methods of `ClassicalCode`, `QuditCode`, and `CSSCode`.
Those methods use Brouwer-Zimmermann (BZ, see [brouwer_zimmermann.md](brouwer_zimmermann.md)) by default.
The search also finds light vectors with a given syndrome, such as light Pauli errors on a stabilizer state.
`CosetLeaderSearch`, which `certify_bounded_fault_propagation` uses, can answer its queries with this search.
The search is the connected-cluster algorithm of Dumer, Kovalev, and Pryadko, and later works cite their 2017 paper for it (see the [references](#references)).
Its earliest known application to quantum codes is the linked-cluster technique of Kovalev, Dumer, and Pryadko (2013), for the distances of quantum LDPC codes.
Dumer, Kovalev, and Pryadko (2014) then grew each cluster through the checks that it leaves unsatisfied, as qLDPC does.
qLDPC follows the branch and bound of Okada and Kasai (Algorithm 2 in Section V) and of Lee et al. (Appendix B), who also extend it to stabilizer codes that are not CSS (Appendix B.2).
They branch on the unsatisfied check with the fewest choices, forbid the variables of earlier siblings, prune by lower bounds, and search from one root per orbit of cyclic shifts.
G. Lee (Appendix C.4) takes the roots from general groups of automorphisms of the Tanner graph, and [Section 3.5](#35-symmetries) says which automorphisms qLDPC uses.
Related searches of classical LDPC codes look for small stopping sets and low-weight codewords: those of Richter, of Rosnes and Ytrehus, and of Khatami et al.
The method's name, `"connected_cluster"`, follows `dist-m4ri`, the program of Pryadko and Zeng, which calls the search the "connected-cluster (CC) algorithm" of Dumer, Kovalev, and Pryadko.
Webster, Jacob, and Higgott (Section 4.3.2) note that it is "also referred to as the irreducible cluster or linked cluster algorithm".
This guide explains the search, maps it to the source code, and states what a change to the search must preserve.
It is organized as follows:

- [Section 2](#2-what-the-cost-of-the-search-depends-on) says what the cost of the search depends on.
- [Section 3](#3-the-algorithm) explains the algorithm, and where qLDPC differs from Okada and Kasai.
  [Section 3.7](#37-qubit-codes-that-are-not-css) extends the algorithm to qubit codes that are not CSS, and [Section 3.8](#38-vectors-with-a-given-syndrome) to vectors with a given syndrome.
  [Section 3.9](#39-sparser-stabilizers-of-subsystem-codes) says how the search of a subsystem code makes its checks sparser.
- [Section 4](#4-guide-to-the-source-code) maps the algorithm to the source code.
- [Section 5](#5-correctness-requirements) lists the requirements that every change to the search must keep.

## 2. What the cost of the search depends on

The search looks for light words of a code: vectors `x` that satisfy its parity checks `H`, so that `Hx = 0`, and that are not trivial ([Section 3.1](#31-the-problem) states the problem).
The columns of `H` are the variables, and the support of `x` is the set of the variables on which `x` is nonzero.
The search grows supports from one variable, a root.
It adds variables to a support only through the unsatisfied checks of the support, the rows of `H` that hold an odd number of its variables.
Such a check holds at least one variable of the support, so it offers at most `w - 1` further variables, where `w` is the largest number of variables of a check.
A search with weight limit `W` looks only for words of weight at most `W`, so it adds at most `W - 1` variables to a root.
So from each root, it visits at most of the order of `(w - 1)^(W - 1)` supports, and fewer once pruning ([Section 3.4](#34-pruning)) cuts subtrees.
The cost of the search therefore depends on the sparsity of the checks, on the weight limit, and on the number of roots, which symmetries reduce ([Section 3.5](#35-symmetries)).

The search of a code raises its weight limit step by step until it finds a nontrivial word ([Section 3.6](#36-from-a-search-to-the-distance)).
For a given code, each weight limit costs roughly a constant factor more than the limit before, so the last limits cost the most.
That factor is large if the checks are dense, since every unsatisfied check then offers many variables.
So dense checks make the search slow unless the distance is small.
If the checks weigh much less than the distance, as those of surface codes do, a support can leave few checks unsatisfied while it still needs many further variables.
The bounds of [Section 3.4](#34-pruning), which count unsatisfied checks, then prune little, and the search visits many supports lighter than the distance.

The checks of a subsystem code are its stabilizers ([Section 3.1](#31-the-problem)), which can be much heavier than its gauge generators.
`get_distance_css` and `get_distance_pauli` derive a basis of the stabilizer group by row reduction, and this basis can be much heavier than the lightest generators of the group.
So these functions search sparser generators of the group instead ([Section 3.9](#39-sparser-stabilizers-of-subsystem-codes)), though not necessarily the sparsest.
A caller who knows sparse generators `S` of the stabilizer group, as Pauli strings, and gauge generators `G`, can search `S` as given: `get_distance_from_checks(S, G, symplectic=True, method="connected_cluster")` returns the dressed distance.
For a CSS code, let the rows of `S_Z` be the supports of sparse Z-type stabilizer generators, and the rows of `G_X` the supports of X-type gauge generators.
Then `get_distance_from_checks(S_Z, G_X, method="connected_cluster")` returns the dressed X-distance.

Applying Hadamards to some qubits of a CSS code gives a code that is not CSS, with the same distance.
The search of that code costs more than the searches of the two sectors of the CSS code ([Section 3.1](#31-the-problem)), for two reasons that [Section 3.7](#37-qubit-codes-that-are-not-css) explains.
Each qubit of a check offers two Paulis that flip the check, the two that anticommute with it on that qubit, so a check offers twice as many branches.
And a `Y` can flip both a check that acts on its qubit as `X` and a check that acts there as `Z`, so it lies in more unsatisfied checks, which weakens the bounds of [Section 3.4](#34-pruning).
So `get_distance_pauli` searches Pauli checks as a CSS code if X-type and Z-type operators generate the group of the checks, even if the checks mix X and Z.
A code that is CSS only up to local Cliffords, such as the code with Hadamards above, is still searched as a code that is not CSS.

Sparse checks alone do not make the search faster than the BZ search.
A BZ search examines combinations of the rows of bases of the words that it searches ([Section 3.6 of brouwer_zimmermann.md](brouwer_zimmermann.md#36-work-estimates-and-exhaustive-enumeration)).
So its work depends on the dimension of the space of these words and on the distance, but not on the sparsity of the checks.
For a quantum code, that space holds the stabilizers as well as the logical operators, so its dimension is large even if the code encodes few qubits.
On quantum LDPC codes, the connected-cluster search is often much faster than the BZ search.
But it can be much slower on some quantum LDPC codes, and on classical LDPC codes whose checks are only moderately sparse.

The script [experiments/distance_tuning/benchmark.py](../../../../experiments/distance_tuning/benchmark.py) compares the exact distance searches of qLDPC on known codes: the pair-partition codes of Okada and Kasai and of Lee et al. (see the [references](#references)), and a few others.

## 3. The algorithm

### 3.1 The problem

Let `H` and `G` be binary matrices with `rowspan(G) ⊆ ker(H)`.
The vectors of `ker(H)` are the **words**, and a word outside `rowspan(G)` is a **witness**.
The distance of the pair is the least weight of a witness:

$$
d(H, G) = \min \{ \operatorname{wt}(x) : Hx = 0,\; x \notin \operatorname{rowspan}(G) \}.
$$

Coordinates are called **variables**, and the rows of `H` are the **checks**.
The **support** of a vector `x` is the set of its nonzero variables, and its **syndrome** is the set of the checks that have an odd number of variables in that support.
A check is **unsatisfied** if it is in the syndrome.

A CSS code has X-type checks `H_X` and Z-type checks `H_Z`.
Its X-type logical operators have the distance `d(H_Z, H_X)`, and its Z-type logical operators the distance `d(H_X, H_Z)`.
A search treats these two problems, the **sectors** of the code, separately, and the distance of the code is the smaller of the two.
A CSS subsystem code has X-type and Z-type gauge generators, `G_X` and `G_Z`, and X-type and Z-type stabilizers, `S_X` and `S_Z`.
Its X sector is the pair `(S_Z, G_X)`: the kernel of `S_Z` holds the X-type operators that commute with every stabilizer, and `G_X` spans the trivial ones.
Its Z sector is the pair `(S_X, G_Z)`.
A witness is then a dressed logical operator, as in BZ.
For a classical code, `H` is its parity-check matrix, and `G` is empty.

The distance of a vector `v` from a classical code `C` with parity-check matrix `H` and generator matrix `G_C` is one less than `d([H | Hv], [G_C | 0])`.
The vectors of the kernel of `[H | Hv]` whose last bit is 0 are the words of `C` with a 0 appended, which `[G_C | 0]` excludes.
The others are the words of the coset `v + C`, each with a 1 appended.
[Section 3.7](#37-qubit-codes-that-are-not-css) treats qubit codes that are not CSS.

### 3.2 Light words are connected

The **Tanner graph** of `H` joins each check to its variables.
The **restriction** of a vector `x` to a set of variables agrees with `x` on that set and is zero elsewhere.

**Lemma.**
Let `x` be a witness of minimum weight.
Its support is connected in the Tanner graph of `H`, and no restriction of `x` to a proper nonempty subset of its support is in `ker(H)`.

*Proof.*
Split the support of `x` into its connected components.
No check holds variables of two components, so the restriction of `x` to each component has an empty syndrome.
If each of these restrictions were in `rowspan(G)`, their sum `x` would be too.
So one of them is a witness, and since `x` has minimum weight, that restriction is all of `x`: the support of `x` is a single component.
Now let `y` be the restriction of `x` to a proper nonempty subset of its support, and suppose that `y` is in `ker(H)`.
Then `y` and `x + y`, the restriction of `x` to the rest of its support, are in `ker(H)`, and both are lighter than `x`.
They cannot both be in `rowspan(G)`, since their sum is `x`.
So one of them would be a lighter witness, which is a contradiction. ∎

The lemma is due to Dumer, Kovalev, and Pryadko (see the [references](#references)), though none of their papers states it in one place for `ker(H)` outside `rowspan(G)`.
Theorem 1 of their 2013 paper states its first part for the minimum-weight words of a linear code.
The second part of the lemma says that `x` is irreducible: it is not the sum of two nonzero words of `ker(H)` with disjoint supports.
Their 2015 paper defines irreducible Pauli operators, and its Lemma 4 writes every undetectable operator as a product of irreducible ones with disjoint supports.
That paper also builds every irreducible operator outside the stabilizer group by a search through unsatisfied stabilizer generators.
Lemma 3 of their 2017 paper, which cites the 2015 paper, states that a minimum-weight word of a linear code is irreducible.
Lemma 2 of Okada and Kasai states both parts for one witness in `ker(H)` outside `rowspan(G)`, with the proof above.
The proof above holds for every witness of minimum weight, and [Section 3.5](#35-symmetries) relies on this.

### 3.3 The depth-first search

Each variable has a **rank**, its position in a total order of the variables that [Section 3.5](#35-symmetries) chooses.
A **state** is a support `S`, with a set `F` of **forbidden** variables.
The search from a **root** `r` starts from the state with `S = {r}`, in which every variable of lower rank than `r` is forbidden.
A state whose support has an empty syndrome has no children: the search reports `S` if `S` is the support of a witness, and otherwise leaves the state.
At every other state, the search chooses an unsatisfied check `c` with the fewest **available** variables, those of `c` that are in neither `S` nor `F`.
The **children** of the state are the states `S ∪ {v}`, one for each available variable `v` of `c`.
The search is depth first: it visits the whole subtree of one child before it visits the next child.
The **stack** holds the states on the path from the first state of the search to the state that it visits.
Once the search leaves the subtree of the child `S ∪ {v}`, it forbids `v` in the later children, the later **siblings** of that child.

The search visits the children of a state in decreasing order of their **repairs**.
The repair of the child `S ∪ {v}` is the total weight of the unsatisfied checks that contain `v`, for the weights of checks that [Section 3.4](#34-pruning) defines.
The order does not change whether the search finds a witness within its weight limit.
It changes how soon the search finds one, which ends the search, and which variables are forbidden in the subtree of each child.
Taking the most repairing child first is a heuristic.
The variable of that child satisfies unsatisfied checks of the most weight.
Forbidding that variable in the subtrees of all later siblings can raise the weights of the checks in those subtrees, and so strengthen the bounds there ([Section 3.4](#34-pruning)).

The search adds a variable only through an unsatisfied check, which holds a variable of the support, so every support that it visits is connected.
Forbidding the variables of earlier siblings makes the search visit each support at most once.

**Completeness.**
Let `x` be a witness of minimum weight, of weight at most the weight limit `W`, and let `r` be the variable of lowest rank in its support.
Consider the search from the root `r`, and a state of that search with `S ⊊ supp(x)` and `F ∩ supp(x) = ∅`, such as its first state, `S = {r}`.
By the lemma, the restriction of `x` to `S` is not in `ker(H)`, so the syndrome of `S` is not empty.
Every check holds an even number of the variables of `supp(x)`, so an unsatisfied check `c` holds a variable of `supp(x) \ S`.
Every such variable is available, since `F` misses `supp(x)`.
Let `u` be the first of these variables in the order in which the search visits the children.
The variables of the earlier siblings of the child `S ∪ {u}` lie outside `supp(x)`, so that child is again a state of this kind, unless its support is `supp(x)`.
So the search reaches `supp(x)`, unless it reports another witness first.
The pruning of [Section 3.4](#34-pruning) never cuts a state of this kind.
Its bounds hold for every state that a witness of weight at most `W` extends, as `x` extends every state of this kind, and the variables that it forbids for a subtree lie in no such witness.

### 3.4 Pruning

Let `W` be the weight limit, and consider a state whose support `S` has `s` variables, so that `W - s` variables are left to spare.
A **completion** of the state is a set `T` of available variables, those in neither `S` nor `F`, such that `S ∪ T` is the support of a witness of weight at most `W`.
The search prunes the state, and skips its subtree, if it can show that every completion has more than `W - s` variables.
Each unsatisfied check holds an odd number of the variables of `S`, and an even number of those of `S ∪ T`, so it holds a variable of `T`.
So `T` is a **hitting set**: it meets the available variables of every unsatisfied check.
The size of a hitting set is at least the value of any feasible solution of the dual of the hitting-set problem.
Such a solution gives each unsatisfied check `c` a weight `y_c ≥ 0`, such that for each available variable, the weights of the unsatisfied checks that contain it total at most one.
Its value is the sum `Y` of the weights `y_c`, so `|T| ≥ Y`.

- The bound `b1` of Okada and Kasai takes `y_c = 1/Δ`, where `Δ` is the largest number of checks of a variable.
- Their bound `b2` chooses unsatisfied checks whose available variables are pairwise disjoint, gives each of them `y_c = 1`, and gives the other unsatisfied checks `y_c = 0`.
- qLDPC defines the **overlap** of an available variable as the number of unsatisfied checks that contain it.
  It sets `y_c = 1/m_c`, where `m_c` is the largest overlap of the available variables of `c`.
  An available variable with overlap `k` lies in `k` unsatisfied checks, each with `m_c ≥ k`, so their weights total at most one.
  Since `m_c ≤ Δ`, these weights are at least those of `b1`.
  If no two unsatisfied checks share an available variable, every overlap is one, every weight is one, and `Y` equals `b2` with every unsatisfied check chosen.
  Where unsatisfied checks share available variables, `b2` can be larger, but it needs a choice of checks with pairwise disjoint available variables at each state, and qLDPC does not compute it.

The search stores the weights as integers in units of `1 / lcm(1, ..., Δ)`, so that every comparison is exact.
If `lcm(1, ..., Δ)` exceeds `2**40`, the unit is `2**-40` instead, and the weights are rounded down, which keeps them feasible.

**Forbidding variables for a subtree.**
The **repair** `R_v` of an available variable `v` is the sum of the weights `y_c` of the unsatisfied checks that contain `v`.
Suppose that a completion `T` contains `v`.
Every unsatisfied check holds a variable of `T`.
The weights of the unsatisfied checks that hold `v` total `R_v`, and, for each other variable of `T`, the weights of the unsatisfied checks that hold that variable total at most one.
So `Y ≤ R_v + |T| - 1`, and if `1 + Y - R_v > W - s`, then no completion contains `v`.
Otherwise `v` is **viable**.
If a state in the subtree has the support `S'`, every completion of that state, together with `S' \ S`, is a completion of the state with the support `S`.
So the search forbids every variable that is not viable in the whole subtree of the state, not only in its children.
A variable outside the unsatisfied checks has a repair of zero, so if `1 + Y > W - s`, the subtree can use only viable variables of unsatisfied checks.
The search prunes a state if `Y > W - s`, or if an unsatisfied check has no viable variable.
Otherwise the state branches on the unsatisfied check with the fewest viable variables.
Forbidding variables can raise the weights `y_c` in the subtree, since its checks then have fewer available variables, and so lets the bounds prune more of the subtree.

Before the search enters a child, it applies the cheaper bound `b1`: it skips the child if the unsatisfied checks of the child outnumber `Δ` times the variables that the child has to spare.

Other bounds hold, and qLDPC does not use them:

- Dual ascent raises each weight `y_c` as far as the constraints of the available variables of `c` allow.
  It strengthens the bound `Y`, at the cost of further passes over the unsatisfied checks of every state.
- A girth bound says that a variable with `k` unsatisfied checks of its `Δ` needs `⌈(Δ - k)/λ⌉` further variables, where `λ` is the largest number of checks that two variables share.
- Rounding the bound `s + Y` on the weight of a witness up to a multiple of the weight divisor of [Section 3.6](#36-from-a-search-to-the-distance) prunes nothing, since every weight limit is such a multiple.

### 3.5 Symmetries

Let `Γ` be a group of column permutations that map both `ker(H)` and `rowspan(G)` onto themselves.
`Γ` maps every witness to a witness of the same weight.
For a ranking of the variables, call the variable of lowest rank in each orbit of `Γ` the root of that orbit.
Among the witnesses of minimum weight, take one, `x`, whose variable of lowest rank, `m`, ranks as low as possible.
Then `m` is the root of its orbit.
Indeed, if a variable `u` of that orbit ranked lower than `m`, an element of `Γ` that maps `m` to `u` would map `x` to a witness of minimum weight that holds `u`, against the choice of `x`.
So it suffices to search from the roots of the orbits.
By the lemma of [Section 3.2](#32-light-words-are-connected), the support of `x` is connected, so the completeness argument of [Section 3.3](#33-the-depth-first-search) applies to `x` and the root `m`.
The lemma holds for every witness of minimum weight, so `Γ` need not map the rows of `H` to each other: it only has to map `ker(H)` and `rowspan(G)` onto themselves.
qLDPC verifies that each generator of `Γ` does so, with `verify_automorphisms` of `_information_sets/automorphism_families.py`, as BZ does.

The search from a root forbids the variables of lower rank, so the ranking decides how much each search is cut.
qLDPC ranks the variables orbit by orbit, with the orbits in the order of their smallest variables, so that the search from each root forbids every variable of the earlier orbits.
The search from a root `r` can forbid only variables that rank lower than `r`, and each of them lies in an orbit whose root ranks lower than `r`.
So, for a given order of the orbits, no ranking forbids more.
Without symmetries, every variable is a root, and the rank of each variable is its index.
Okada and Kasai instead write the variables of a pair-partition code as pairs `(ℓ, a)`, for a block column `ℓ` and a position `a` modulo the lift size `P`.
They search from the roots `(ℓ, 0)` of the orbits of the cyclic translations, which add the same constant to every `a`, and rank `(ℓ, a)` by `ℓP + a`.
Their implementation does so only if `P` is prime and `W < P`.
qLDPC takes the orbits of any group of automorphisms that it verifies.

The generators of `Γ` are the automorphisms that the caller of a distance function passes.
The `get_distance_exact` methods of quantum codes pass the automorphisms that `_found_qudit_automorphisms` returns.
Without automorphisms, as for the classical codes of `get_distance_exact`, the search starts from every variable.
For a large code, finding the automorphisms, verifying them, and ordering their orbits can cost as much as visiting many states.
So the search of each problem, such as a sector of a CSS code, asks for them at most once, and only once it expects its own work to outweigh that cost.
The field `automorphism_minimum_states` of `SearchTuning` in `_tuning.py` sets the threshold: the search asks before the first weight limit at which it expects to have visited that many states over all of its limits.
The default of that field is a round number of states, assumed to cost about as much to visit as asking does.
To predict the number of states that the next weight limit visits, the search multiplies the number of states of the last limit by the ratio between the numbers of states of the last two limits.
A first weight limit above the weight divisor, which a cutoff can set ([Section 3.6](#36-from-a-search-to-the-distance)), has no prediction.
So the search asks before that limit, unless `automorphism_minimum_states` is `MAX_VALUE`.
The search can ask only between weight limits, since each limit runs in one compiled call ([Section 3.6](#36-from-a-search-to-the-distance)).
If it waited until it had visited `automorphism_minimum_states` states, it would run the limit in which it reaches that number without symmetries, and that limit can be a large part of the search ([Section 2](#2-what-the-cost-of-the-search-depends-on)).

With symmetries, the search from the root of the first orbit does most of the work.
That search forbids no variable, and every witness that meets the first orbit has an image under `Γ` that holds this root.
A witness with several variables in the first orbit can have several such images: one for each of these variables if `Γ` acts regularly on the orbit.
So the search can visit several images of one state.
Rejecting such duplicates soundly for general groups is left open.

### 3.6 From a search to the distance

**Logical test.**
A vector of `ker(H)` is in `rowspan(G)` exactly if it is orthogonal to `ker(G)`.
Every vector of `ker(H)` is also orthogonal to the rows of `H`, which lie in `ker(G)`, since `rowspan(G) ⊆ ker(H)`.
So a word is a witness exactly if it has an odd product with a row of a basis of `ker(G)` modulo `rowspan(H)`.
The search keeps the columns of these **test rows**, packed into one 64-bit block per 64 rows.
It tests a state whose support has an empty syndrome by XORing the columns of the variables of its support: the support is that of a witness exactly if the result is not zero.

**Weight divisor.**
If the weight of every vector of `ker(H)` is a multiple of an integer, the search can take that integer as its **weight divisor**, and try only weight limits that are multiples of it.
For example, each block row of the checks of a pair-partition code sums to the all-ones vector, so every vector of `ker(H)` has even weight, and the divisor can be two.
`infer_weight_divisor` of `_information_sets/basis_families.py` finds a divisor from a basis of `ker(H)`, as BZ does.

**Exact distance.**
`get_distance_connected_cluster` searches several problems, such as the two sectors of a CSS code, and keeps, for each problem, a lower bound `L` on the weights of its witnesses, at first its weight divisor.
It searches the problem with the lowest `L`, with the weight limit `W = L`.
If that search finds a witness, the witness is a lightest one over all problems, since every witness of a problem weighs at least the `L` of that problem.
Otherwise every witness of the problem is heavier than `W`, and its `L` rises to the next multiple of its divisor.
The searches end once the lightest witness found weighs at most the lowest `L`, and the weight of that witness is the distance.
The search needs no initial upper bound.
The searches of the problems share the lightest witness found, as the BZ searches of the sectors of a CSS code do.
A caller can pass an upper bound, such as a known distance of another sector, which acts as a witness of that weight.
A caller can also pass a `cutoff`, to stop at the first witness of weight at most `cutoff`.
If the cutoff is above the `L` of a problem, the first weight limit of that problem is the largest multiple of its divisor at most the cutoff.
The search then returns the weight of the first witness that it finds within that limit, which can exceed the distance.

**Termination.**
Each weight limit ends, since the search from a root is a tree of depth at most `W`, with finitely many children at each state.
Take a problem with `n` variables and weight divisor `d`.
Every limit at which its search finds no witness is a multiple of `d` below the weight of its lightest witness.
That weight is itself a multiple of `d`, and at most `n`, so it is at most `d · (n // d)`, and the problem takes at most `n // d` weight limits.
A search that needs more limits than the sum of `n // d` over its problems has a bug, and raises a `RuntimeError` rather than looping forever.

**Interrupts.**
Each weight limit of a problem runs in one compiled call, which a keyboard interrupt cannot stop, so an interrupt takes effect only once that call ends.
Python then raises a `KeyboardInterrupt`, or a `SystemError` that the interrupt caused.
In a large search, the last weight limits, and so the wait for an interrupt, can take a long time.

### 3.7 Qubit codes that are not CSS

A Pauli operator on `n` qubits, up to a phase, is a symplectic vector `x = (a | b)` of `2n` bits.
The operator acts on qubit `j` as `X` if `a_j = 1` and `b_j = 0`, as `Z` if `a_j = 0` and `b_j = 1`, and as `Y` if both are 1.
Two operators commute if their symplectic product vanishes.
The weight of an operator is the number of qubits on which it is not the identity.
For a qubit code, the rows of `H` are its stabilizers, and the words are the operators that commute with every row.
The rows of `G` are its gauge generators, which are its stabilizers if it is not a subsystem code.
The distance is the least weight of a word outside `rowspan(G)`, as in [Section 3.1](#31-the-problem), but with weight counting qubits.

The search makes each qubit a variable, whose nonzero **values** are `X`, `Z`, and `Y`.
A **symbol** is a variable with one of its values.
A set of symbols on distinct variables stands for the operator that has those values there and is the identity elsewhere.
A symbol **flips** a check if the symbol and the check anticommute on the qubit of the symbol.
The syndrome of an operator is the set of the checks that it anticommutes with.
The Tanner graph joins each check to the symbols that flip it.
A problem that is not symplectic, such as a classical code or a sector of a CSS code, has one value per variable, so its symbols are its variables.

Each argument of Sections 3.2 to 3.6 carries over, with symbols where it uses variables:

- **Lemma.**
  The components of the support of `x` are now those of the graph that joins each check to the qubits on which the check acts.
  A check acts on qubits of at most one component, so it commutes with the restriction of `x` to each component, as it commutes with `x`.
  The rest of the proof uses only that the restrictions of `x` to the parts of a partition of its support sum to `x`.
  A symbol flips a check only if the check acts on its qubit, so the search grows supports that are connected in this graph.
- **Search.**
  A state is a set `S` of symbols on distinct variables, with a set `F` of forbidden symbols.
  The available symbols of an unsatisfied check `c` are those that flip `c`, are not in `F`, and lie on a variable outside `S`.
  A state that branches on `c` has one child for each of these symbols.
  While the search is in the subtree of a child, it blocks the other values of the variable of that child, since an operator has one value on each qubit.
  It unblocks them when it leaves that subtree.
  The symbol of a child joins `F` for the later siblings of the child, but the other values of its variable do not, since a later sibling can lead to a witness that has another value there.
  The search from a root starts from the empty state, whose children are the three symbols of the root.
- **Completeness.**
  The argument of [Section 3.3](#33-the-depth-first-search) holds with `S` a set of symbols of `x`, and with `F` missing every symbol of `x`.
  An unsatisfied check `c` anticommutes with the restriction of `x` to the variables of `S`, and commutes with `x`, so a symbol of `x` on a variable outside `S` flips `c`.
  The search visits the first such symbol, in the order of the children, before that symbol joins `F`.
  Each earlier sibling blocked the other values of its variable only while the search was in its subtree.
- **Pruning.**
  A completion `T` has one symbol on each new variable, so its size is the number of variables that it adds.
  For each available symbol, the weights `y_c` of the unsatisfied checks that it flips must total at most one, and the overlaps and repairs of [Section 3.4](#34-pruning) are those of symbols.
  A `Y` flips the checks that act on its qubit as `X` or as `Z`, while an `X` flips those that act there as `Z` or `Y`, and a `Z` those that act there as `X` or `Y`.
  So a `Y` can flip more checks than an `X` or a `Z`.
  In a CSS code with Hadamards applied to some of its qubits, no check acts as `Y`, so a `Y` flips every check on its qubit, which raises overlaps and so lowers the weights `y_c`.
- **Symmetries.**
  A permutation of the qubits moves both halves `a` and `b` of each vector, and qLDPC verifies it on all `2n` columns.
  The roots and ranks are those of variables, and the search from a root blocks every value of the variables of lower rank.
- **Logical test and divisor.**
  In the logical test of [Section 3.6](#36-from-a-search-to-the-distance), the column of a `Y` is the sum of the columns of the `X` and the `Z` on its qubit.
  qLDPC infers no weight divisor for these problems, and steps their weight limits by one.

### 3.8 Vectors with a given syndrome

A **syndrome query** asks for a vector `x` of weight at most `W` with `Hx = σ`, for a given nonzero syndrome `σ`.
The vectors with syndrome `σ` form a coset of `ker(H)`.
If the rows of `H` are the stabilizers of a stabilizer state, `ker(H)` is the stabilizer group, so a lightest vector with syndrome `σ` is a coset leader: a lightest Pauli error with that syndrome.
If the rows of `H` are the checks of a code that encodes logical qubits, the query ranges over every logical class, and ignores `G`.

The query searches as [Section 3.3](#33-the-depth-first-search) does, but from the empty support, without roots or ranks.
The syndrome of a state is now `σ + Hx_S`, where `x_S` is the vector with support `S`, and the search reports `S` once that syndrome is empty.

**Completeness.**
Suppose that `x` is a lightest vector with syndrome `σ`, and take a state with `S ⊊ supp(x)` and `F ∩ supp(x) = ∅`, as the empty state is.
The syndrome `σ + Hx_S` of the state is that of the restriction of `x` to `supp(x) \ S`.
It is not empty, since otherwise `x_S` would be a lighter vector with syndrome `σ`.
So an unsatisfied check holds a variable of `supp(x) \ S`, and the argument of [Section 3.3](#33-the-depth-first-search) carries over.
The pruning of [Section 3.4](#34-pruning) holds with completions that empty the syndrome of the state, and the values of [Section 3.7](#37-qubit-codes-that-are-not-css) carry over.
The support of `x` need not be connected.
But the restriction of `x` to each component of its support flips a check of `σ`, so the search can enter each component through such a check.

Four parts of a distance search do not carry over:

- The nonzero syndrome gives the empty state an unsatisfied check to branch on, so the query needs no roots or ranks.
- A syndrome is not invariant under the symmetries of the code, so the query uses none.
- The weights of the vectors with a syndrome need not share a divisor, so the query has none.
- Every vector with the syndrome counts, so the query has no logical test.

`certify_bounded_fault_propagation` in `qldpc.circuits.certification`, and the experimental `estimate_residual_weight_histogram`, use `CosetLeaderSearch` of `coset_leaders.py` to find light Pauli errors with given syndromes on a stabilizer state.
A `CosetLeaderSearch` built with `method="connected_cluster"` answers each of its queries by a table lookup, or by syndrome queries at one or more weight limits.
For the physical weight, the number of qubits on which an error acts, it searches the Pauli errors with a given syndrome.
The CSS weight `max(|x|, |z|)` of an error with X part `x` and Z part `z` is not a sum over qubits.
So for the CSS weight, a `CosetLeaderSearch` searches the X and Z parts separately on a CSS state, and raises an error on any other state.

Each coset search of a `CosetLeaderSearch` by this method is a `ConnectedClusterCosetSearch`.
Its checks can be dense, such as the row-reduced stabilizers that `certify_bounded_fault_propagation` passes, and dense checks make the search slow ([Section 2](#2-what-the-cost-of-the-search-depends-on)).
So the coset search first replaces its checks with sparser checks that span the same space, and maps the syndrome of each query to them.
The sparsest checks with that span form a lightest basis of the span.
The greedy algorithm for matroids finds such a basis by taking, again and again, the lightest word of the span outside the span of the words taken so far.
But each of its steps would be an exact search for a light word of the span, whose cost grows quickly with the weight of that word.
So the coset search instead takes candidates from row reductions of its checks, with the columns in random orders.
It keeps a lightest basis among these candidates and its current checks, so the total weight of its checks never rises.
`get_sparser_rows` in `_connected_cluster/sparse_checks.py` states the procedure.
The fields `max_sparsifying_rounds` and `sparsifying_rounds_without_improvement` of `SearchTuning` limit the rounds of `get_sparser_rows`.
Each round costs one row reduction, so these fields trade the time to build the coset search against the speed of its queries.
Checks with the same span give the same answers and bounds, so the random orders change only the speed of the queries and which errors the queries return.
A fixed seed makes the checks reproducible.

The checks that the rounds find need not be a lightest basis of the span.
A word of the span is a row of a reduction exactly if its support holds just one pivot column of that reduction, since each row of the reduction is zero on the pivots of the other rows.
The heavier a word, the less likely a random order of the columns gives it just one pivot.
So the heavier the words of a lightest basis, the less likely the rounds are to find them.
Even a lightest basis can be heavier than the checks of a code.
Take the physical weight on the logical state `|0⟩` of an `[[n, k, d]]` stabilizer code.
The span of the checks is then the stabilizer group of that state, which the stabilizers of the code and its logical Z operators generate.
At most `n - k` independent words of that group are stabilizers of the code.
So every basis of the group holds at least `k` words that are nontrivial logical operators of the code, and weigh at least `d`.

An exact second stage, which qLDPC does not implement, could make the checks lighter still with the greedy algorithm for matroids.
Each of its steps would find a lightest word of the span outside the span of the words taken so far.
Each step is a witness search ([Section 3.1](#31-the-problem)): its excluded rows are the words taken so far, and its checks are sparse checks whose kernel is the span.
For Pauli errors on a stabilizer state, the sparse checks from the rounds serve as these checks, since the span is the set of the Paulis that commute with all of them.
For the X parts, the sparse checks of the Z parts serve as these checks, and for the Z parts, those of the X parts.
But for the state `|0⟩` of a code, such a stage would have to find the `k` logical operators that every basis holds, and each weighs at least `d`, so each is costly to find.
So such a stage would pay only where the queries far outnumber its searches.

A `ConnectedClusterCosetSearch` first looks up the syndrome of a query in a table of the syndromes of the vectors of weight at most one.
It searches only the syndromes that the table lacks, which have no vector of weight below two.
A syndrome query at a weight limit `W` is complete, so if it finds no vector, every vector with the syndrome weighs more than `W`.
These two facts give the lower bounds of the answers, which the class docstring of `ConnectedClusterCosetSearch`, in `_connected_cluster/coset.py`, states for each kind of query.

A syndrome query with weight limit `W` can visit many heavy supports before it reaches a much lighter vector.
`certify_bounded_fault_propagation` queries batches of syndromes, most of which have a vector within `W`.
For these batches, the coset search tries the limits 2, 3, …, `W` in turn, and stops at the first limit within which it finds a vector.
That vector is a lightest one, and its weight is the lower bound.
The search of a syndrome without a vector within `W` repeats the work of the lower limits.
For `estimate_residual_weight_histogram`, `CosetLeaderSearch` finds the minimum weight of each coset with the same increasing limits.
For the CSS weight, it starts the search of the Z parts of a class at the minimum weight of the X parts of that class ([Section 3.10 of brouwer_zimmermann.md](brouwer_zimmermann.md#310-minimum-weights-of-cosets)).
A single query, of `CosetLeaderSearch.get_min_weight_bound`, searches at `W` directly, so that it repeats no work for a caller that already raises `W` one step at a time.

Without reuse, a state computes the overlap and the repair ([Section 3.4](#34-pruning)) of a symbol once for each unsatisfied check that the symbol flips and that the state examines.
The checks of a syndrome query can stay dense even after the rounds above, and a symbol can then flip many unsatisfied checks.
So a syndrome query computes the overlap and the repair of each symbol at most once per state, and reuses them.
A distance search, whose checks are usually sparse, computes them again each time, since a symbol then flips few unsatisfied checks, and the bookkeeping of reuse would cost more than it saves.

### 3.9 Sparser stabilizers of subsystem codes

The distance search of a subsystem code needs checks whose row space is its stabilizer group ([Section 3.1](#31-the-problem)).
`get_distance_css` and `get_distance_pauli` derive a basis of that group from the gauge generators by row reduction, the **derived basis**, which can be much heavier than the lightest generators of the group.
For example, take the rotated toric code on 256 qubits, and make one of its logical qubits a gauge qubit.
The plaquettes of the toric code, which weigh 4, generate the stabilizer group of the resulting subsystem code, but its derived basis has a mean weight of about 15 and rows of weight up to 46.
A search of the derived basis can then branch on many more variables at each state than a search of the plaquettes ([Section 2](#2-what-the-cost-of-the-search-depends-on)).
So with `method="connected_cluster"`, `get_distance_css` and `get_distance_pauli` give the distance search of a subsystem code two sets of checks:

- The gauge generators that commute with every gauge generator are stabilizers.
  The first set holds them as given, with any redundant rows, except those heavier than every row of the second set.
- The second set is the derived basis, made sparser by `get_sparser_rows` ([Section 3.8](#38-vectors-with-a-given-syndrome)) in up to `max_distance_sparsifying_rounds` rounds, which `sparsifying_rounds_without_improvement` can end sooner.

The second set spans the stabilizer group, and the first set lies in that group, so together they span exactly the stabilizer group.
The two sets together have the kernel of the derived basis, so the automorphisms of the code apply as before ([Section 3.5](#35-symmetries)).
Unlike a coset search, a distance search needs only the span of its checks, and no map of syndromes to them, so it can keep redundant rows.
Redundant checks can strengthen the search, since each unsatisfied check can prune a state or offer fewer branches.
A `CosetLeaderSearch` makes its checks sparser once, and then answers many queries with them.
A distance search uses its checks for one search, so `max_distance_sparsifying_rounds` defaults to fewer rounds than `max_sparsifying_rounds`.

Sparsity is not the only property of the checks that matters.
The derived basis of a Bacon-Shor code is already as sparse as possible: each of its rows acts on two lines of the lattice, one of which is the last line.
Another basis of the same weight, whose rows act on neighboring lines, makes the search visit far more states.
So the rounds change the checks only when they lower the total weight of the checks, and they leave the derived basis of a Bacon-Shor code as it is.

## 4. Guide to the source code

The names below are in `_connected_cluster/search.py`, unless the table says otherwise.

| Concept | Code |
| --- | --- |
| Pair `(H, G)`, its words and witnesses | `checks` and `excluded` of `SupportSearch` and `get_distance_connected_cluster` |
| Sectors of a CSS code | `_search_css` of `core.py`, which `get_distance_css` and `get_distance_pauli` call, and `get_css_stabilizers` of `_binary.py` |
| Classical codes, and qubit codes that are not CSS | `get_distance_from_checks` and `get_distance_pauli` of `core.py`, which the `get_distance_exact` methods of `ClassicalCode` and `QuditCode` call |
| Symplectic problems, values, and symbols | `symplectic` of `SupportSearch`, its `_get_symbol_columns`, and `_shift_other_values` |
| Distance of several sectors | `get_distance_connected_cluster`, which `_search_css` of `core.py` calls |
| Search through a weight limit `W` | `SupportSearch.find_witness` |
| Syndrome query | `SupportSearch.find_vector_with_syndrome`, which coset searches skip for `SupportSearch._find_vector_with_packed_syndrome`, and the root `-1` of `_search_supports` |
| Overlaps and repairs that a syndrome query reuses within a state | `memoize` of `_search_supports`, and `memos` and `stamp` of `_get_branches` |
| Coset leaders of Pauli errors | `ConnectedClusterCosetSearch` in `_connected_cluster/coset.py`, and `CosetLeaderSearch` in `coset_leaders.py` |
| Sparser checks with the same span, for coset searches and for the distance searches of subsystem codes | `get_sparser_rows` and `_sparsify` in `_connected_cluster/sparse_checks.py`, and `_get_sparser_stabilizers` of `core.py` |
| How many rounds make the checks of a coset search, or of a distance search of a subsystem code, sparser | `max_sparsifying_rounds`, `max_distance_sparsifying_rounds`, and `sparsifying_rounds_without_improvement` of `SearchTuning` in `_tuning.py` |
| Table of light vectors, and increasing limits for batches and minimum weights | `ConnectedClusterCosetSearch` (its `_singles` and `get_min_weight_bound`) in `_connected_cluster/coset.py`, and `_get_batch_cluster_bound` and `_get_min_cluster_weight` of `CosetLeaderSearch` in `coset_leaders.py` |
| Logical test | `SupportSearch._get_tests`, and `_is_outside_excluded` |
| Roots, ranks, and symmetries | `SupportSearch._use_automorphisms`, and `verify_automorphisms` and `get_orbits` of `_information_sets/automorphism_families.py` |
| States, branching, and undoing what a state forbade when the search leaves it | `_search_supports`, whose `levels` mark the available symbols |
| Weights, pruning, and the symbols that a state forbids for its subtree | `_get_branches` |
| When a search asks for symmetries | `automorphism_minimum_states` of `SearchTuning` in `_tuning.py` |
| Weight divisor | `infer_weight_divisor` of `_information_sets/basis_families.py` |

`get_distance_from_checks`, `get_distance_css`, and `get_distance_pauli` of `core.py` build the problems of the search.
`get_distance_css` and `get_distance_pauli` also derive the stabilizers of subsystem codes and make them sparser ([Section 3.9](#39-sparser-stabilizers-of-subsystem-codes)).
These three functions pass the active tuning to the search, and `SupportSearch` validates every problem.
The `get_distance_exact` methods of the code classes in `src/qldpc/codes/common.py` call these functions.
Nonbinary codes reject `method="connected_cluster"`.
`logical_counts.py` does not accept this method, since counting needs every word of a weight, and not one witness.
`get_distance_from_generators` rejects it, since its inputs are generators and not parity checks.
The closed forms of hypergraph product codes search their classical seed codes with the method that they are given.
`_search_supports`, `_get_branches`, and `_is_outside_excluded` are compiled with `always_compiled`, and their small helpers with `always_compiled(inline=True)`, as [README.md](README.md) explains.

## 5. Correctness requirements

Changes to the search can use other bounds and orders, while keeping these requirements.

- **Every pruning of a state must hold for each witness that extends the state.**
  A bound may count only available variables, those outside `S` and `F`, since a witness that the search must reach from the state adds only such variables ([Section 3.3](#33-the-depth-first-search)).
  `F` holds the variables of the earlier siblings of the states in the stack, the variables of lower rank than the root, and the variables that a state in the stack forbade for its subtree.
  A bound must be a lower bound on the number of variables that such a witness adds.
  A search with a bound that is too strong still finds the distance unless the bound prunes the lightest witnesses.
  So the tests need cases in which the bound is tight at the distance, as in the tests of the distances of known codes.
- **Forbid a variable for a subtree only if no completion of the state contains it.**
  The variables that a state forbids must become available again when the search leaves the state.
- **Give each variable at most one value.**
  While a symbol is in the support, the other values of its variable must be blocked, and they must become available again once the search leaves the branch of that symbol.
  A sibling forbids only its own symbol, not the other values of its variable.
  Without this blocking, distances would not change, but a witness could have two symbols on one variable, and its number of symbols would overstate its weight.
  So the tests check the witnesses at every weight limit.
- **Block siblings and variables of lower rank than the root.**
  A search from a root must use only variables of higher rank, with any of their values, and the symbol of a child must be forbidden for the later siblings of that child.
  Without this blocking, the results would not change, and the search would only take longer, so the tests compare `num_states` with a count by hand on a small instance.
- **Verify the automorphisms, and make each root the lowest-ranked variable of its orbit.**
  Roots from unverified permutations are unsound.
  The orbits come from all generators.
- **Preserve the logical test.**
  The support of a state with an empty syndrome is that of a witness exactly if it fails a test, and the search must never report a word of `rowspan(G)`.
- **Use a proven weight divisor.**
  A divisor that does not divide the weight of some witness makes the search skip that weight.
- **Start a syndrome query from its syndrome alone.**
  The query must block no variable by rank, and use no symmetry, divisor, or logical test, as [Section 3.8](#38-vectors-with-a-given-syndrome) explains.
  The search keeps only the distinct nonzero checks, so a syndrome of the given checks must be mapped onto those.
  A syndrome with a nonzero entry for a zero check, or with different entries for two equal checks, has no vector.
- **Reuse an overlap or a repair only within the state that computed it.**
  A state can prune unsoundly if it reuses values from another state whose syndrome or check weights differ.
  A state may compute a repair only once it has weighed all of its unsatisfied checks, since a repair sums their weights.
- **Give coset queries only proven lower bounds.**
  A lower bound of 2 needs the syndrome to be missing from a table of the syndromes of every vector of weight at most 1, and a lower bound above a limit needs a complete search at that limit ([Section 3.8](#38-vectors-with-a-given-syndrome)).
  A caller that raises its weight limit to each lower bound would silently get wrong weights from a bound above the minimum weight.
  So the tests compare the bounds that should be exact with brute force.
- **Keep the span of the checks of a coset query.**
  Sparser checks must span the same space as the given checks, with an invertible transform that maps the syndrome of each query to a syndrome for the sparser checks, or the query would search other cosets ([Section 3.8](#38-vectors-with-a-given-syndrome)).
  How sparse the checks are changes only the speed of the queries and which errors the queries return.
- **Keep the span of the stabilizers of a subsystem code.**
  The checks of its distance search must span exactly its stabilizer group ([Section 3.9](#39-sparser-stabilizers-of-subsystem-codes)).
  A smaller span would admit words that anticommute with a stabilizer, and a larger one would miss dressed logical operators.
  The checks need no transform, and may have redundant rows.
- **Return the exact minimum unless a cutoff permits an early exit.**
  The weight limits rise one divisor at a time, so that the first witness found is a lightest one.
  An early exit returns the weight of a known witness, which is an upper bound.

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
