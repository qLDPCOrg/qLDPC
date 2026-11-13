# Connected-support distance search in qLDPC

## 1. Purpose and scope

The connected-support search computes the exact distance of a binary code from the Tanner graph of its parity checks.
The code can be classical, CSS, or a qubit stabilizer code that is not CSS, and the quantum codes can be subsystem codes.
It is the method `"connected_support"` of the `get_distance_exact` methods of `ClassicalCode`, `QuditCode`, and `CSSCode`.
Brouwer-Zimmermann (BZ, see [brouwer_zimmermann.md](brouwer_zimmermann.md)) is the default.
The same search also finds light vectors with a given syndrome, such as Pauli errors, which `CosetLeaderSearch` and the certifier of bounded fault propagation can use.
It follows Algorithm 2 in Section V of Okada and Kasai (see the [references](#references)), which searches CSS codes by excluding the row space of their stabilizers.
The idea is older: Richter, Rosnes and Ytrehus, and Khatami et al. search the Tanner graphs of classical LDPC codes for small stopping sets and low-weight codewords.
Dumer, Kovalev, and Pryadko use the sparsity of the checks in the same way, with irreducible clusters, to verify the distances of classical and quantum LDPC codes, and the program `dist-m4ri` implements a connected-cluster search for exact distances.
The method is named after what it searches, as `"brute_force"` is, rather than after one of these works.
This guide explains the search, maps it to the code, and states what a change to it must preserve.
It is organized as follows:

- [Section 2](#2-what-the-cost-of-the-search-depends-on) says what the cost of the search depends on.
- [Section 3](#3-the-algorithm) explains the algorithm, and where qLDPC differs from the paper.
  [Section 3.7](#37-qubit-codes-that-are-not-css) extends the algorithm to qubit codes that are not CSS, and [Section 3.8](#38-vectors-with-a-given-syndrome) to vectors with a given syndrome.
  [Section 3.9](#39-sparser-stabilizers-of-subsystem-codes) says how the search of a subsystem code gets sparser checks.
- [Section 4](#4-guide-to-the-source-code) maps the algorithm to the source code.
- [Section 5](#5-correctness-requirements) lists the requirements that every change to the search must keep.

## 2. What the cost of the search depends on

The search enumerates supports, the sets of variables on which a word is nonzero, that grow through unsatisfied checks ([Section 3.1](#31-the-problem) defines these terms).
So its cost depends on the sparsity of the checks, on the weight limit, and on the number of variables from which it starts, which symmetries reduce ([Section 3.5](#35-symmetries)).
For a given code, its cost grows by a roughly constant factor with each step of the weight limit.
That factor is large if the checks are dense, because every unsatisfied check offers many variables to branch on.
The search of a subsystem code uses the Tanner graph of its stabilizers, which can be dense even if its gauge generators are sparse.
`get_distance_css` and `get_distance_pauli` derive these stabilizers as a row-reduced basis, which can be dense even if the stabilizer group has sparse generators.
So they search sparser generators instead ([Section 3.9](#39-sparser-stabilizers-of-subsystem-codes)).
These need not be the sparsest.
A caller who knows sparser generators `S` of the stabilizer group, and gauge generators `G`, can get the same dressed distance by searching them as given, with `get_distance_from_checks(S, G, symplectic=True, method="connected_support")`.
For a CSS code with sparse X-type and Z-type stabilizer generators, the X-distance comes from `get_distance_from_checks(S_Z, G_X, method="connected_support")`, where `S_Z` holds the supports of the Z-type stabilizer generators and `G_X` those of the X-type gauge generators.
If the checks weigh much less than the distance, as those of surface codes do, many supports grow without being pruned ([Section 3.4](#34-pruning)), so the search visits many states below the distance.
The search of a qubit code that is not CSS costs more than the searches of the sectors of an equivalent CSS code, such as the same code without Hadamards on some of its qubits.
A `Y` can flip checks that act as `X` and checks that act as `Z` on its qubit at once, which weakens the bounds of [Section 3.4](#34-pruning), and each qubit of a check offers two symbols that flip it, the two Paulis that anticommute with the check there ([Section 3.7](#37-qubit-codes-that-are-not-css)).
So `get_distance_pauli` searches Pauli checks as a CSS code if X-type and Z-type operators generate their group, even if the checks mix X and Z.
A code that is CSS only up to local Cliffords, such as those Hadamards, is still searched as a code that is not CSS.

The script [experiments/distance_tuning/benchmark.py](../../../../experiments/distance_tuning/benchmark.py) times the exact distance searches of qLDPC on the pair-partition codes of Okada and Kasai and of Lee et al. (see the [references](#references)), and on a few other codes.

## 3. The algorithm

### 3.1 The problem

Let `H` and `G` be binary matrices with `rowspan(G) ⊆ ker(H)`.
The distance of the pair is

$$
d(H, G) = \min \{ \operatorname{wt}(x) : Hx = 0,\; x \notin \operatorname{rowspan}(G) \}.
$$

A vector `x` in `ker(H)` outside `rowspan(G)` is a **witness**.
Coordinates are called **variables**.
The **support** of `x` is the set of its nonzero variables, and its **syndrome** is the set of the rows of `H`, the **checks**, that have an odd number of variables in the support.
A check is **unsatisfied** if it is in the syndrome.

A **sector** of a CSS code is its X-type or its Z-type logical operators, which a search treats as separate problems.
For a CSS code, the distance of the X-type logical operators is `d(H_Z, H_X)`, and the distance of the Z-type logical operators is `d(H_X, H_Z)`.
The distance of the code is the smaller of the two.
For a subsystem code, `H` holds the stabilizers of the opposite type, so that its kernel holds the X-type or Z-type operators that commute with every stabilizer, and `G` holds the gauge generators of the same type.
A witness is then a dressed logical operator, as in BZ.
For a classical code, `H` is its parity-check matrix, and `G` is empty.
The distance of a vector `v` from a classical code `C` with generator matrix `G_C` is one less than `d([H | Hv], [G_C | 0])`.
The words of the kernel of `[H | Hv]` whose last bit is 0 are the words of `C` with a 0 appended, which `[G_C | 0]` excludes.
The others are the words of the coset `v + C`, each with a 1 appended.
[Section 3.7](#37-qubit-codes-that-are-not-css) treats qubit codes that are not CSS.

### 3.2 Light words are connected

**Lemma** (after Lemma 2 of [Okada and Kasai](#references)).
Every witness `x` of minimum weight has a support that is connected in the Tanner graph of `H`.
No restriction of `x` to a proper nonempty subset of its support, which agrees with `x` there and is zero elsewhere, is in `ker(H)`.

*Proof.*
Split the support of a minimum-weight witness `x` into its connected components.
No check is adjacent to variables of two components, so each component has an empty syndrome.
If every component were in `rowspan(G)`, their sum `x` would be too.
So one component is a witness, and by minimality it is all of `x`.
Now let `y` be a restriction of `x` to a proper nonempty subset of its support, and suppose that `y` is in `ker(H)`.
Then `y` and `x + y`, the restriction of `x` to the rest of its support, are in `ker(H)`, and are lighter than `x`.
They cannot both be in `rowspan(G)`, since their sum is `x`.
One of them would be a lighter witness, which is a contradiction. ∎

The paper states the lemma for one witness.
The proof holds for every witness of minimum weight, which matters for the symmetries of [Section 3.5](#35-symmetries).

### 3.3 The depth-first search

The variables have a **rank**, a total order that [Section 3.5](#35-symmetries) chooses.
A **state** is a support `S`, with a set `F` of **forbidden** variables.
The search starts from a **root** `r`, with `S = {r}`, and every variable of lower rank than `r` forbidden.
At a state with an empty syndrome, the search stops branching.
It reports `S` if it is a witness, and otherwise leaves the state.
At any other state, it chooses an unsatisfied check `c` with the fewest **available** variables, those of `c` that are neither in `S` nor in `F`.
It visits the states `S ∪ {v}` for each available variable `v` of `c`, and adds `v` to `F` for its later siblings once its state is done.
Its children come in the order of their repairs ([Section 3.4](#34-pruning)), most repairing first.
On the [[276, 98, 14]] pair-partition code of Okada and Kasai, this order visits a quarter as many states as the order of the variables, up to the weight limit at which the search finds a witness.
Every state grows only through unsatisfied checks, so every support that it visits is connected.

**Completeness.**
Suppose that `x` is a witness of minimum weight, whose variable of lowest rank is the root `r`.
By the lemma, the restriction of `x` to a proper nonempty subset of its support has a syndrome that is not empty.
Take a state with `S ⊊ supp(x)` and `F ∩ supp(x) = ∅`.
The syndrome of `S` is not empty, and the parity of `supp(x)` at every check is even, so an unsatisfied check `c` has a variable of `supp(x) \ S`.
That variable is available, since `F` misses `supp(x)`.
The first variable of `supp(x) \ S` in the order of the children of `c` is visited before it joins `F`, and gives a state of the same kind.
So the search reaches `supp(x)` itself.
Every bound of [Section 3.4](#34-pruning) holds for a state of this kind, since `x` is a witness that extends it, and no variable that [Section 3.4](#34-pruning) forbids for a subtree is in `supp(x)`.

Forbidding the variables of earlier siblings makes the search visit each support at most once.

### 3.4 Pruning

A search through a weight limit `W` has `W - s` variables to spare at a state with `s` variables.
A **completion** of the state is a set `T` of further variables whose union with `S` is a witness of weight at most `W`.
A state prunes itself if its unsatisfied checks need more than `W - s` further variables.
Every unsatisfied check needs a variable of its available set, since a witness that extends the state has no variable of `F`.
So the number of further variables is at least the value of any feasible solution of the dual of this hitting-set problem.
That solution gives each unsatisfied check `c` a weight `y_c ≥ 0`, so that the weights of the unsatisfied checks that contain any available variable total at most one.

- The paper's bound `b1` takes `y_c = 1/Δ`, where `Δ` is the largest number of checks of a variable.
- The paper's bound `b2` takes `y_c` in `{0, 1}`, for unsatisfied checks whose available sets are pairwise disjoint.
- qLDPC defines the **overlap** of an available variable as the number of unsatisfied checks that contain it.
  It sets `y_c` to the inverse of the largest overlap `m_c` of the available variables of `c`.
  An available variable with overlap `k` is available to each of its `k` unsatisfied checks, whose `m_c` are at least `k`, so their weights total at most one.
  Since `m_c ≤ Δ`, these weights are at least those of `b1`.
  Adding `b2` changed the number of states by at most 7% in tests on the [[276, 98, 14]] code, so qLDPC does not use it.

The weights are integers in units of `1 / lcm(1, ..., Δ)`, so that every comparison is exact.
If that scale exceeds `2**40`, the scale is `2**40`, and the weights are rounded down, which keeps them feasible.

**Forbidding variables for a subtree.**
Let `Y` be the sum of the weights `y_c` of the unsatisfied checks of a state, and let the **repair** `R_v` of an available variable `v` be the weight of the unsatisfied checks that contain it.
Suppose that a completion `T` of the state contains `v`.
Every unsatisfied check meets `T`, the variables of `T` other than `v` repair at most one each, and `v` repairs `R_v`.
So `|T| ≥ 1 + Y - R_v`, and if this exceeds `W - s`, then `v` is in no completion of the state.
Otherwise `v` is **viable**.
A variable that is not viable is forbidden in the whole subtree of the state, not only for its children.
This holds for every variable outside the unsatisfied checks, whose repair is zero, as soon as `1 + Y > W - s`, so such a state restricts its subtree to the viable variables of its unsatisfied checks.
A state is pruned if an unsatisfied check has no viable variable, and otherwise it branches on the check with the fewest.
Forbidding variables for a subtree can raise the weights of its states, since their checks have fewer variables, and on the [[276, 98, 14]] code it removes more than half of the states.

A cheaper test runs first: a child is not visited if its unsatisfied checks outnumber `Δ` times its remaining weight limit, which is `b1`.

Other bounds that hold, and that qLDPC tested and does not use, added little on the pair-partition codes:

- Raising each weight `y_c` to the room that its available variables leave (dual ascent) removed 8% to 14% of the states, which did not pay for its cost.
- A girth bound, that a variable with `k` unsatisfied checks of its `Δ` needs `⌈(Δ - k)/λ⌉` further variables, where `λ` is the largest number of checks that two variables share, removed under 1%.
- Rounding the bound up to the weight divisor prunes nothing, since every weight limit `W` is a multiple of the divisor.

### 3.5 Symmetries

Let `Γ` be a group of column permutations that map both `ker(H)` and `rowspan(G)` onto themselves.
It maps every witness to a witness of the same weight, so it maps the set of minimum-weight witnesses to itself.
Rank the variables, and call the variable of lowest rank of each orbit of `Γ` its root.
Among the minimum-weight witnesses, take one whose variable `m` of lowest rank is as low as possible.
Then `m` is the root of its orbit, since a variable of the orbit of lower rank than `m` would give an image with a variable of lower rank.
So it suffices to search from the roots.
By the lemma, this witness is connected, so the completeness argument of [Section 3.3](#33-the-depth-first-search) applies to it.
The lemma holds for every minimum-weight witness, so `Γ` need not map the rows of `H` to each other, and only has to map `ker(H)` and `rowspan(G)` to themselves.
qLDPC verifies this with `verify_automorphisms` of `_information_sets/automorphism_families.py`, as BZ does.

The search from a root forbids the variables of lower rank, so the ranks decide how much each search is cut.
qLDPC ranks the variables orbit by orbit, in the order of their least variables, so that the search from each root forbids every earlier orbit.
A root forbids only variables of orbits whose roots rank lower, so for a given order of the orbits no ranking forbids more.
On the pair-partition codes, this removes a quarter of the states of the natural order of the variables.
Without symmetries, every variable is a root, and the ranks are the variables themselves.
Okada and Kasai instead take the roots `(ℓ, 0)` of the cyclic translations of a pair-partition code, with variables `(ℓ, a)` ranked by `ℓP + a`, and their implementation does so only if `P` is prime and `W < P`.
qLDPC takes the orbits of any group of automorphisms that it verifies.

Generators of `Γ` are the automorphisms that the caller of a distance function passes, such as the `_found_qudit_automorphisms` that the `get_distance_exact` methods of quantum codes pass.
Without automorphisms, as for the classical codes of `get_distance_exact`, a search starts from every variable.
Finding the automorphisms of a large code can take as long as a search of many states.
So the search of each problem, such as a sector of a CSS code, asks for them at most once, before the first weight limit at which it expects to reach `automorphism_minimum_states` states, a field of the tuning of `_tuning.py`.
It expects the number of states of each weight limit to grow by the ratio between those of the last two.
A first limit above the divisor, which `cutoff` can set, has no such estimate, so the search asks for them before it, unless `automorphism_minimum_states` is `MAX_VALUE`.
Waiting until it has visited that many states instead would let a search that prunes well run a whole weight limit without them, which can be a large part of the search.

With symmetries, the search from the first root does most of the work, since every witness that meets the first orbit is found through it.
A witness that meets the first orbit several times can have several images that contain the first root, as many as its variables in that orbit if the group acts regularly on the orbit, and the search can visit images of a state more than once.
In the [[276, 98, 14]] code, two thirds of the states of that search hold two or more variables of the first orbit.
Rejecting such duplicates soundly for general groups is left open.

### 3.6 From a search to the distance

**Logical test.**
A vector `x` of `ker(H)` is in `rowspan(G)` exactly if it is orthogonal to `ker(G)`, and it is orthogonal to the rows of `H`, which lie in `ker(G)`.
So `x` is a witness exactly if it has an odd product with a row of a basis of `ker(G)` modulo `rowspan(H)`.
The search keeps these rows as packed columns, one 64-bit block per 64 rows, and tests a state with an empty syndrome by XORing the columns of its variables.

**Weight divisor.**
If every vector of `ker(H)` has even weight, as for the pair-partition codes, whose block rows of checks sum to the all-ones vector, weights are multiples of two.
`infer_weight_divisor` of `_information_sets/basis_families.py` finds this from a basis of `ker(H)`, as BZ does.

**Exact distance.**
Let `L` be the smallest weight that an unexamined witness can have, which is the divisor at first.
`get_distance_connected_support` searches the problem with the lowest `L`, such as one of the sectors of a CSS code, through the weight limit `W = L`.
A witness is then a lightest one if the earlier limits found none, since weights are multiples of the divisor.
Otherwise every witness is heavier than `W`, and `L` rises to the next multiple of the divisor.
This ends when `L` is at least the weight of the best known witness, the distance.
A search never needs a good upper bound, and the searches of the sectors of a CSS code share their bounds, as in BZ.
If `cutoff` is above `L`, the first limit is the largest multiple of the divisor at most `cutoff`, and a witness within that limit is returned at once.
An upper bound from a known sector distance acts as a witness of that weight.

**Termination.**
Each weight limit ends, since the search from a root is a tree of depth at most `W` with finitely many branches at each state.
A limit without a witness is a multiple of the divisor `d` below the weight of every witness.
So is the weight of a lightest witness, which is therefore at most `d · (n // d)` for a problem with `n` variables.
A witness ends the search, so a correct search takes at most the sum of `n // d` weight limits over its problems.
A search that needs more has a bug, and raises a `RuntimeError` rather than looping forever.

**Interrupts.**
Each weight limit of a problem runs in one compiled call, which a keyboard interrupt cannot stop, so an interrupt takes effect only once that weight limit ends.
Python then raises a `KeyboardInterrupt`, or a `SystemError` that the interrupt caused.
In a large search, the last weight limits, and so the wait for an interrupt, can take a long time.

### 3.7 Qubit codes that are not CSS

A Pauli operator on `n` qubits, up to a phase, is a symplectic vector `x = (a | b)` of `2n` bits, which has `X` on qubit `j` if `a_j = 1` and `b_j = 0`, `Z` if `a_j = 0` and `b_j = 1`, and `Y` if both are 1.
Two operators commute if their symplectic product vanishes.
The weight of an operator is the number of qubits on which it is not the identity.
For a qubit code, `H` holds the stabilizers, and the words are the operators that commute with them; `G` holds the gauge generators, which are the stabilizers if the code is not a subsystem code.
The distance is the least weight of a word outside `rowspan(G)`, as in [Section 3.1](#31-the-problem), but with weight counting qubits.

The search makes each qubit a variable, whose nonzero **values** are `X`, `Z`, and `Y`.
A **symbol** is a variable with one of its values, and the word of a set of symbols on distinct variables has those values there and is the identity elsewhere.
A symbol **flips** a check if the two anticommute on its qubit, and the syndrome of a word is the set of checks that it anticommutes with.
The Tanner graph joins each check to the symbols that flip it.
A homogeneous problem, such as a classical code or a sector of a CSS code, is the case of one value per variable, so its symbols are its variables.

Each argument of Sections 3.2 to 3.6 carries over, with symbols where it uses variables:

- **Lemma.**
  The components of `supp(x)` are those of the graph that joins each check to the qubits on which it acts.
  A check acts on qubits of at most one component, so it commutes with the restriction of `x` to every component, as it commutes with `x`.
  The rest of the proof uses only that the restrictions of `x` to the parts of a partition of its support sum to `x`.
  A symbol flips a check only if the check acts on its qubit, so the search grows supports that are connected in this graph.
- **Search.**
  A state is a set `S` of symbols on distinct variables, with a set `F` of forbidden symbols.
  The children of an unsatisfied check `c` are its available symbols: those that flip `c`, are not in `F`, and whose variable is not in `S`.
  While a child is in the stack, the other values of its variable are blocked, since a word has one value on each qubit, and they are available again once the search leaves that child.
  A child joins `F` for its later siblings, but the other values of its variable do not, since a later sibling can lead to a witness that has another value there.
  The search from a root starts from the empty state, whose children are the three symbols of the root.
- **Completeness.**
  The argument of [Section 3.3](#33-the-depth-first-search) holds with `S` a set of symbols of `x`, and with `F` missing every symbol of `x`.
  An unsatisfied check `c` anticommutes with the restriction of `x` to `S`, and commutes with `x`, so it is flipped by a symbol of `x` on a variable outside `S`.
  The first such symbol among the children of `c` is visited before it joins `F`.
  The earlier siblings block the other values of their variables only while their own branches are in the stack.
- **Pruning.**
  A completion `T` has one symbol on each new variable, so its size is the number of variables that it adds.
  The dual weights `y_c` must make the weights of the unsatisfied checks that each available symbol flips total at most one, and the overlap and repair of [Section 3.4](#34-pruning) are those of symbols.
  A `Y` flips the checks that act on its qubit as `X` or as `Z`, while an `X` flips those that act as `Z` or `Y`, and a `Z` those that act as `X` or `Y`.
  So a `Y` can flip more checks than either: in a CSS code with Hadamards applied to some of its qubits, no check acts as `Y`, and a `Y` flips every check on its qubit, which raises overlaps and lowers the weights.
- **Symmetries.**
  A permutation of the qubits moves both halves of each word, and qLDPC verifies it on the `2n` columns.
  The roots and ranks are those of variables, and the search from a root blocks every value of the variables of lower rank.
- **Logical test and divisor.**
  The test columns of a symbol are those of its word: for `Y`, the sum of those of `X` and `Z`.
  qLDPC infers no weight divisor for these problems, and steps their weight limits by one.

### 3.8 Vectors with a given syndrome

A **syndrome query** asks for a vector `x` with `H x = σ`, for a given nonzero syndrome `σ`, of weight at most `W`.
The vectors with a syndrome are a coset of the kernel of `H`.
For the stabilizers of a pure stabilizer state, that kernel is the stabilizer group, so a lightest vector is a coset leader of the Pauli errors with that syndrome.
For a code with logical operators, the query ranges over every logical class, and ignores `G`.

The query searches as [Section 3.3](#33-the-depth-first-search) does, but from the empty support with syndrome `σ`, without roots or ranks.
At a state, the unsatisfied checks are those of `σ + H x_S`, where `x_S` is the vector of the state, and the search reports `S` once that syndrome is empty.

**Completeness.**
Suppose that `x` is a lightest vector with syndrome `σ`, and take a state with `S ⊊ supp(x)` and `F ∩ supp(x) = ∅`, as the empty state is.
The syndrome of the state is that of the rest of `x`, which is not empty, since the restriction of `x` to `S` would otherwise be a lighter vector with syndrome `σ`.
So an unsatisfied check has a variable of `supp(x) \ S`, and the argument of [Section 3.3](#33-the-depth-first-search) carries over.
The pruning of [Section 3.4](#34-pruning) holds with completions that empty the syndrome of the state, and the values of [Section 3.7](#37-qubit-codes-that-are-not-css) carry over.
`x` need not be connected: each component of its support flips a check of `σ`, so the empty state reaches it through one of them.

Four parts of a distance search do not carry over:

- The nonzero syndrome gives the empty state an unsatisfied check to branch on, so the query needs no roots or ranks.
- A syndrome is not invariant under the symmetries of the code, so the query uses none.
- The weights of the vectors with a syndrome need not share a divisor, so the query has none.
- Every vector with the syndrome counts, so the query has no logical test.

The fault-propagation certifier in `qldpc.circuits.certification`, and the experimental residual-weight estimator, use `CosetLeaderSearch` in `coset_leaders.py`.
When it is built with `method="connected_support"`, `CosetLeaderSearch` answers each of its queries by a table lookup, or by such queries at one or more weight limits.
It searches the Pauli errors of a state by their physical weight.
The CSS weight `max(|x|, |z|)` is not a sum over qubits, so for it, on a CSS state, it searches the X and Z parts separately, and on any other state it raises an error.
For the residual-weight estimator, it finds the minimum weight of each coset by trying increasing weight limits, and starts the search of a Z part at the weight of its X part ([Section 3.10 of brouwer_zimmermann.md](brouwer_zimmermann.md#310-minimum-weights-of-cosets)).

The search is fast only if its checks are sparse.
But the stabilizers that certification passes are row-reduced, and so are the X and Z parts that `CosetLeaderSearch` searches for the CSS weight.
So a search first replaces its checks with sparser ones that span the same space, and maps the syndrome of each query to them.
The sparsest such checks are a lightest basis.
The greedy algorithm for matroids would find one by taking, again and again, the lightest word of the span outside the span of those taken so far.
But each step would be an exact search for a light word of the span, whose cost grows quickly with its weight, so a search takes its candidates from row reductions instead.
It first adds its checks to each other while that makes them lighter.
Then each round row-reduces the given checks with pivots in a random order of the columns, which gives the words of the span that have exactly one pivot column in their support.
Those that are lighter than the heaviest check join the checks as candidates.
Each round then keeps, by the greedy algorithm for matroids, the lightest independent candidates, so the total weight of the checks never rises.
The rounds stop after `max_sparsifying_rounds`, or once `sparsifying_rounds_without_improvement` rounds in a row have left the total weight unchanged.
Both limits are fields of `SearchTuning`.
A last pass adds the checks to each other again where that helps.
Checks with the same span give the same answers and bounds, so the random orders change only the speed of the queries and which errors they return.
A fixed seed makes the checks reproducible.
For the physical weight on the state `|0⟩` of the [[144, 12, 12]] bivariate bicycle code, this lowers the mean weight of the 144 generators of its stabilizer group from 26 to 6.5, the least possible.
No stabilizer of the code weighs less than 6, the weight of its checks, and only 132 of the generators can lie in its stabilizer group.
The other 12 are logical operators, so they weigh at least the distance, 12.
On the state `|0⟩` of the [[232, 62, 12]] pair-partition code, this lowers the mean weight from 54 to 9.5, while the checks of the code weigh 8.
On longer codes of higher rate, a round rarely makes a light word one of its rows, and the checks stay much heavier than those of the code.
On the [[944, 478, 20]] pair-partition code, whose checks weigh 16, it lowers the mean weight from 177 only to about 63.
An exact second stage, which qLDPC does not implement, could lower it further.
The greedy algorithm for matroids would take, again and again, the lightest word of the span outside the span of the words taken so far.
The connected-support search finds such a word as a witness, with the words taken so far as its excluded rows ([Section 3.1](#31-the-problem)), and it is fast only if its checks are sparse, as the rounds make them.
For Pauli errors, the words of the span are the Paulis that commute with every sparse check, since the state is pure.
For X or Z parts, they are the vectors that satisfy the sparse checks of the other part.
But the cost of a search grows quickly with the weight of the word that it finds, and the words outside the stabilizer group of the code, such as the logical operators among the generators of `|0⟩`, weigh at least its distance.
So such a stage would pay only where the queries far outnumber its searches.

A search first looks up the syndrome in a table of the syndromes of the vectors of weight at most 1, and searches only syndromes that the table lacks, which have no vector of weight below 2.
A syndrome in the table gets the weight of its vector as the lower bound.
A single query of any other syndrome with limit `W` has the lower bound `max(2, W + 1)` if no vector lies within `W`, and 2 if the search finds one.
These bounds are often weaker than those of BZ.
A search with limit `W` can wander through many heavy supports before it reaches a much lighter vector.
So for the batches of certification, where most syndromes have a vector within `W`, a search tries the limits 2, 3, …, `W` in turn, and stops at the first limit within which it finds a vector.
That vector is a lightest one, and its weight is the lower bound.
The search of a syndrome without a vector within `W` repeats the work of the lower limits.
Single queries search at `W` directly, so that the search repeats no work for a caller that already raises `W` one step at a time.

Without reuse, a state computes the overlap and the repair ([Section 3.4](#34-pruning)) of a symbol once for each unsatisfied check that the symbol flips and that the state examines.
The checks of a syndrome query can stay dense even after the sparsifying rounds above.
Then a symbol can flip many unsatisfied checks.
So a syndrome query computes the overlap and the repair of each symbol at most once per state, and reuses them.
A distance search, whose checks are usually sparse, computes them again each time, since keeping them would cost more than it saves.

### 3.9 Sparser stabilizers of subsystem codes

The search of a subsystem code needs checks whose row space is its stabilizer group `S` ([Section 3.1](#31-the-problem)).
`get_distance_css` and `get_distance_pauli` derive `S` from the gauge generators as a row-reduced basis, which can be several times heavier than sparse generators of `S`.
For example, take the toric code on 256 qubits, and make one of its logical qubits a gauge qubit.
Its stabilizers are generated by the plaquettes of the toric code, which weigh 4, but the derived basis has a mean weight of 15 and rows of weight up to 46.
A search of the derived basis takes more than three minutes, and a search of the plaquettes about a second.
So with `method="connected_support"`, the search of a subsystem code gets two sets of checks:

- The gauge generators that commute with every gauge generator are stabilizers.
  The search keeps them as given, with any redundant rows.
  It drops those heavier than every row of the second set.
- The rounds of row reductions in random orders of [Section 3.8](#38-vectors-with-a-given-syndrome) find sparser rows with the span of the derived basis.
  `max_distance_sparsifying_rounds`, a field of `SearchTuning`, limits the rounds.

The second set spans `S`, and the first lies in `S`, so together they span `S`.
The kernel of the checks is that of the derived basis, so the automorphisms of the code apply as before ([Section 3.5](#35-symmetries)).
A coset search maps the syndrome of each query to its sparser checks, but a distance search needs only their span.
Redundant checks can strengthen the search, since each unsatisfied check can prune a state or offer fewer branches.
On the [[288, 12, 18]] bivariate bicycle code with 6 of its logical qubits made gauge qubits, the gauge generators that are stabilizers roughly halve the number of states that the search visits.
A `CosetLeaderSearch` makes its checks sparser once for many queries, but a distance search uses its checks for one search, so it takes fewer rounds.

Sparsity is not the only property of the checks that matters.
The derived basis of a Bacon-Shor code is already as sparse as possible: each of its stabilizers acts on two parallel lines of the lattice, one of which is the last line.
Another basis of the same weight, whose stabilizers act on neighboring lines, makes the search of the 9×9 Bacon-Shor code visit more than 100 times as many states.
So the rounds change a row only when they find a strictly lighter one, which leaves the derived basis of a Bacon-Shor code as it is.

## 4. Guide to the source code

The names below are in `_connected_support/search.py`, unless the table says otherwise.

| Concept | Code |
| --- | --- |
| Pair `(H, G)`, its words and witnesses | `checks` and `excluded` of `SupportSearch` and `get_distance_connected_support` |
| Sectors of a CSS code | `_search_css` of `core.py`, which `get_distance_css` and `get_distance_pauli` call, and `get_css_stabilizers` of `_binary.py` |
| Classical codes, and qubit codes that are not CSS | `get_distance_from_checks` and `get_distance_pauli` of `core.py`, which the `get_distance_exact` methods of `ClassicalCode` and `QuditCode` call |
| Symplectic problems, values, and symbols | `homogeneous` of `SupportSearch`, its `_get_symbol_columns`, and `_shift_other_values` |
| Distance of several sectors | `get_distance_connected_support`, which `_search_css` of `core.py` calls |
| Search through a weight limit `W` | `SupportSearch.find_witness` |
| Syndrome query | `SupportSearch.find_vector_with_syndrome`, which coset searches skip for `SupportSearch._find_vector_with_packed_syndrome`, and the root `-1` of `_search_supports` |
| Overlaps and repairs that a syndrome query reuses within a state | `memoize` of `_search_supports`, and `memos` and `stamp` of `_get_branches` |
| Coset leaders of Pauli errors | `ConnectedSupportCosetSearch` in `_connected_support/coset.py`, and `CosetLeaderSearch` in `coset_leaders.py` |
| Sparser checks with the same span, for coset searches and for the distance searches of subsystem codes | `get_sparser_rows` and `_sparsify` in `_connected_support/sparse_checks.py`, and `_get_sparser_stabilizers` of `core.py` |
| How many rounds make the checks of a coset search, or of a distance search of a subsystem code, sparser | `max_sparsifying_rounds`, `max_distance_sparsifying_rounds`, and `sparsifying_rounds_without_improvement` of `SearchTuning` in `_tuning.py` |
| Table of light vectors, and increasing limits for batches and minimum weights | `ConnectedSupportCosetSearch` (its `_singles` and `get_min_weight_bound`) in `_connected_support/coset.py`, and `_get_batch_support_bound` and `_get_min_support_weight` of `CosetLeaderSearch` in `coset_leaders.py` |
| Logical test | `SupportSearch._get_tests`, and `_is_outside_excluded` |
| Roots, ranks, and symmetries | `SupportSearch._use_automorphisms`, and `verify_automorphisms` and `get_orbits` of `_information_sets/automorphism_families.py` |
| States, branching, and undoing what a state forbade when the search leaves it | `_search_supports`, whose `levels` mark the available symbols |
| Weights, pruning, and the symbols that a state forbids for its subtree | `_get_branches` |
| When a search asks for symmetries | `automorphism_minimum_states` of `SearchTuning` in `_tuning.py` |
| Weight divisor | `infer_weight_divisor` of `_information_sets/basis_families.py` |

`get_distance_from_checks`, `get_distance_css`, and `get_distance_pauli` of `core.py` build the problems of the search, and the last two find the stabilizers of subsystem codes and make them sparser ([Section 3.9](#39-sparser-stabilizers-of-subsystem-codes)).
They pass the search the active tuning, and `SupportSearch` validates every problem.
The `get_distance_exact` methods of the code classes in `src/qldpc/codes/common.py` call these functions.
Nonbinary codes reject the method.
`logical_counts.py` does not accept the method, since counting needs every word of a weight, and not one witness.
`get_distance_from_generators` rejects it, since its inputs are generators and not parity checks.
The closed forms of hypergraph product codes search their classical seed codes with the method that they are given.
`_search_supports`, `_get_branches`, and `_is_outside_excluded` are compiled with `always_compiled`, and their small helpers with `always_compiled(inline=True)`, as [README.md](README.md) explains.

## 5. Correctness requirements

Changes to the search can use other bounds and orders, while keeping these requirements.

- **Every pruning must hold for a witness that extends the state.**
  A bound may use the available variables, which exclude `S` and `F`, since a minimum-weight witness extending the state has no other variables.
  `F` includes the earlier siblings of the states in the stack, the variables of lower rank than the root, and the variables that a state forbade for its subtree.
  A bound must be a lower bound on the number of further variables that such a witness needs.
  A bound that is too strong can still find the distance, unless it prunes a lightest witness, so a test needs a case where the bound is tight at the distance, as the tests of the distances of known codes are.
- **Forbid a variable for a subtree only if no completion of the state contains it.**
  The variables that a state forbids must become available again when the search leaves the state.
- **Give each variable at most one value.**
  While a symbol is in the support, the other values of its variable must be blocked, and they must be available again once the search leaves its branch.
  A sibling forbids only its own symbol, not the other values of its variable.
  Omitting the blocking would not change distances, but would return witnesses with two symbols on a variable, whose number overstates their weight, so the tests check witnesses at every weight limit.
- **Block siblings and variables of lower rank than the root.**
  A search from a root must use only variables of higher rank, with any of their values, and a symbol must be forbidden for its later siblings.
  Omitting this blocking would not change the results, and would only lose time, so `num_states` is tested on a hand-counted instance.
- **Verify the automorphisms, and make each root the lowest-ranked variable of its orbit.**
  Roots from unverified permutations are unsound.
  The orbits come from all generators.
- **Preserve the logical test.**
  A state with an empty syndrome is a witness exactly if it fails a test, and a stabilizer must not be reported.
- **Use a proven weight divisor.**
  A divisor that some witness lacks skips its weight.
- **Start a syndrome query from its syndrome alone.**
  It must block no variable by rank, and use no symmetry, divisor, or logical test, as [Section 3.8](#38-vectors-with-a-given-syndrome) explains.
  The syndrome of the given checks must be mapped onto the distinct nonzero checks.
  A syndrome with a nonzero entry for a zero check, or with different entries for two equal checks, has no vector.
- **Reuse an overlap or a repair only within the state that computed it.**
  A state can prune unsoundly if it reuses values from another state whose syndrome or check weights differ.
  A state may compute a repair only once it has weighed all of its unsatisfied checks, since a repair sums their weights.
- **Give coset queries only proven lower bounds.**
  A lower bound of 2 needs the syndrome to be missing from a table of the syndromes of every vector of weight at most 1, and a lower bound above a limit needs a complete search at that limit ([Section 3.8](#38-vectors-with-a-given-syndrome)).
  A consumer that raises its limit to each bound would silently get wrong weights from a bound above the minimum, so the tests compare the exact bounds with brute force.
- **Keep the span of the checks of a coset query.**
  Sparser checks must span the same space as the given ones, with an invertible transform that maps the syndrome of each query to a syndrome for them, or the query would search other cosets ([Section 3.8](#38-vectors-with-a-given-syndrome)).
  How sparse they are changes only the speed of the queries, and which errors they return.
- **Keep the span of the stabilizers of a subsystem code.**
  The checks of its distance search must span exactly its stabilizer group ([Section 3.9](#39-sparser-stabilizers-of-subsystem-codes)).
  A smaller span would admit words that anticommute with a stabilizer, and a larger one would miss dressed logical operators.
  The checks need no transform, and may have redundant rows.
- **Return the exact minimum unless a cutoff permits an early exit.**
  The weights of the limits rise one divisor at a time, so that the first witness is the lightest.
  An early exit returns the weight of a known witness, which is an upper bound.

## References

- K. Okada and K. Kasai, "Pair-partition constructions for CPM-based quantum LDPC codes", Section V, [arXiv:2607.14091](https://arxiv.org/abs/2607.14091): the connected-support search for the pair-partition codes, with the implementation in [kasaikenta/pair-partition-cpm-css-codes](https://github.com/kasaikenta/pair-partition-cpm-css-codes).
- J. Y. Lee, K. Okada, N. Maskara, K. Kasai, and H. Zhou, "Design principles for ultra-high-rate quantum codes", [arXiv:2609.30069](https://arxiv.org/abs/2609.30069): the pair-partition codes, with the instances of [ultra-high-rate-qec/design-principles-data](https://github.com/ultra-high-rate-qec/design-principles-data).
- G. Richter, "Finding small stopping sets in the Tanner graphs of LDPC codes", 4th International Symposium on Turbo Codes and Related Topics and 6th International ITG-Conference on Source and Channel Coding, 2006, pp. 1–5.
- E. Rosnes and Ø. Ytrehus, "An efficient algorithm to find all small-size stopping sets of low-density parity-check matrices", IEEE Transactions on Information Theory 55, 4167 (2009).
- S. M. Khatami, L. Danjean, D. V. Nguyen, and B. Vasić, "An efficient exhaustive low-weight codeword search for structured LDPC codes", Information Theory and Applications Workshop, 2013, pp. 401–410.
- I. Dumer, A. A. Kovalev, and L. P. Pryadko, "Distance verification for classical and quantum LDPC codes", IEEE Transactions on Information Theory 63, 4675 (2017), [arXiv:1611.07164](https://arxiv.org/abs/1611.07164): the irreducible-cluster technique.
- [QEC-pages/dist-m4ri](https://github.com/QEC-pages/dist-m4ri): a program whose connected-cluster method computes exact distances of classical codes and of quantum CSS codes.
