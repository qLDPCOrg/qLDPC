# Connected-support distance search in qLDPC

## 1. Purpose and scope

The connected-support search computes the exact distance of a binary code from the Tanner graph of its parity checks.
The code can be classical, CSS, or a qubit stabilizer code that is not CSS, and the quantum codes can be subsystem codes.
It is the method `"connected_support"` of the `get_distance_exact` methods of `ClassicalCode`, `QuditCode`, and `CSSCode`.
Brouwer–Zimmermann (BZ, see [brouwer_zimmermann.md](brouwer_zimmermann.md)) is the default.
It follows Algorithm 2 in Section V of Okada and Kasai (see the [references](#references)).
That algorithm extends the searches for small stopping sets and low-weight codewords of classical LDPC codes to CSS codes, by excluding the row space of their stabilizers.
The classical searches are those of Richter, Rosnes and Ytrehus, and Khatami et al., also in the references.
This guide explains the search, maps it to the code, and states what a change to it must preserve.
It is organized as follows:

- [Section 2](#2-when-to-use-which-method) says when to use this method rather than BZ.
- [Section 3](#3-the-algorithm) explains the algorithm, and where qLDPC differs from the paper.
  [Section 3.7](#37-qubit-codes-that-are-not-css) extends the algorithm to qubit codes that are not CSS.
- [Section 4](#4-guide-to-the-source-code) maps the algorithm to the source code.
- [Section 5](#5-correctness-requirements) lists the requirements that every change to the search must keep.

## 2. When to use which method

A **sector** of a CSS code is its X-type or its Z-type logical operators, which a search treats as separate problems ([Section 3.1](#31-the-problem)).
BZ examines combinations of the rows of bases of the searched space in layers of increasing coefficient weight, and its cost grows like `binom(r, d/m)`.
Here `r` is the dimension of the searched space, `d` is the distance, and `m` is the number of disjoint information sets.
The sectors of CSS codes usually have `r > n/2`, hence `m = 1`, and a high rate pushes `r` towards `n`, so that BZ becomes infeasible at modest distances.
The connected-support search instead enumerates supports that grow through unsatisfied checks, so its cost depends on the sparsity of the checks and on `d`, and not on `r`.
Its cost grows by a roughly constant factor for each additional unit of weight, which is about 5 for the pair-partition codes below, with row weight 8 to 16, and about 3 for surface codes.

The table lists times on one core of an Apple M3 Max laptop, run at low priority, with the benchmark script [experiments/connected_support_distance/benchmark.py](../../../../experiments/connected_support_distance/benchmark.py).
The codes are the pair-partition codes of Okada and Kasai and of Lee et al., which are quasi-cyclic CSS codes, mostly of high rate, with parity checks of column weight 3 to 5 (see the [references](#references)).
The instances, and their distances, come from the data of Lee et al., and the repository of Okada and Kasai also publishes the [[276, 98, 14]], [[518, 228, 16]], and [[944, 478, 20]] codes.
The [[372, 130, 18]] code is the instance `qc_372_130_18` of that data, which is not the [[372, 130, 16]] code of the table of Okada and Kasai.
Times are CPU times, for the whole code unless they say per sector.
The BZ times marked "estimated" scale the measured time of the [[232, 62, 12]] code by the ratio of the work that BZ predicts for each code, with the cyclic automorphisms that qLDPC finds for these codes (Sections 3.6 and 3.7 of [brouwer_zimmermann.md](brouwer_zimmermann.md)).
That prediction ignores the completion tables of Section 3.9 of that guide, whose savings differ between codes, so these estimates are rough.

| Code | BZ | Connected support | Faster |
| --- | --- | --- | --- |
| [[110, 8, 12]] | 0.4 s | 0.3 s | connected support |
| [[232, 62, 12]] | 49 s | 0.2 s | connected support |
| [[276, 98, 14]] | about a day and a half (estimated) | 6.4 s | connected support |
| [[518, 228, 16]] | about 5 months (estimated) | 252 s | connected support |
| [[372, 130, 18]] | about 4 months (estimated) | about 10 minutes per sector (extrapolated) | connected support |
| [[170, 8, 20]] | about 45 minutes (estimated) | about an hour per sector (extrapolated) | BZ |
| [[944, 478, 20]] | about a year (estimated) | about half a day per sector (extrapolated) | connected support |

Choose the connected-support search for sparse codes, and especially for high-rate ones, and BZ for codes with dense checks, as the quantum Reed–Muller code below shows.
For sparse codes of low rate and high distance, BZ can be faster, as the [[170, 8, 20]] row shows.
The connected-support times of the [[372, 130, 18]], [[170, 8, 20]], and [[944, 478, 20]] codes are not measured.
They are extrapolated from timed searches through lower weights, whose costs grow by a factor of 20 to 33 per two units of weight for these codes, and assume that this growth continues.
The script measured the other connected-support times of the table, and those of the surface codes below.

The cost of the connected-support search grows quickly if the checks are dense, because every unsatisfied check offers many variables to branch on.
The [[128, 70, 8]] quantum Reed–Muller code, which has checks of weight 32, takes 0.9 s, and BZ takes 0.1 s.
These times are for a plain `CSSCode` built from the matrices of `QuantumReedMullerCode(2, 7)`, since that class knows its distance without a search.
The stabilizers of surface codes weigh much less than their distances, so that many supports grow without being pruned, but the search is still much faster than BZ.
Searching rotated surface codes takes 0.03 s for distance 11, 0.8 s for distance 15, 6.1 s for distance 17, and 47 s for distance 19.
BZ takes 0.5 s for distance 11, and more than 200 s for distance 15.
A subsystem code searches the Tanner graph of its stabilizers, which can be dense.

A classical code is searched like a sector with nothing excluded.
A qubit code that is not CSS has no sectors, and its search branches on the Paulis `X`, `Z`, and `Y` of each qubit ([Section 3.7](#37-qubit-codes-that-are-not-css)).
This search is slower than that of the sectors of a CSS code: a `Y` can flip checks that act as `X` and checks that act as `Z` on its qubit at once, which weakens the bounds of [Section 3.4](#34-pruning).
The table times the codes above, and others, with Hadamards on the second half of their qubits.
These codes are not CSS, have the same distances, and are searched as `QuditCode` objects (the option `--hadamards` of the script), with the automorphisms that `pybliss` finds.

| Code, with Hadamards | BZ | Connected support |
| --- | --- | --- |
| Rotated surface code, distance 9 | 2.7 s | 0.2 s |
| Rotated surface code, distance 11 | more than 300 s | 0.6 s |
| [[144, 12, 12]] bivariate bicycle code | more than 300 s | 0.4 s |
| [[110, 8, 12]] | more than 300 s | 9.1 s |
| [[232, 62, 12]] | not run | 1.6 s |
| [[276, 98, 14]] | not run | 389 s |

These searches take 8 to 60 times as long as those of the same codes without Hadamards.

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

For a CSS code, the distance of the X-type logical operators is `d(H_Z, H_X)`, and the distance of the Z-type logical operators is `d(H_X, H_Z)`.
The distance of the code is the smaller of the two.
For a subsystem code, `H` holds the stabilizers of the opposite type, so that its kernel holds the X-type or Z-type operators that commute with every stabilizer, and `G` holds the gauge generators of the same type.
A witness is then a dressed logical operator, as in BZ.
For a classical code, `H` is its parity-check matrix, and `G` is empty.
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
On the [[276, 98, 14]] code, this order visits a quarter as many states as the order of the variables, up to the weight limit at which the search finds a witness.
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
qLDPC verifies this with `verify_automorphisms` of `_search/automorphism_families.py`, as BZ does.

The search from a root forbids the variables of lower rank, so the ranks decide how much each search is cut.
qLDPC ranks the variables orbit by orbit, in the order of their least variables, so that the search from each root forbids every earlier orbit.
A root forbids only variables of orbits whose roots rank lower, so for a given order of the orbits no ranking forbids more.
On the pair-partition codes, this removes a quarter of the states of the natural order of the variables.
Without symmetries, every variable is a root, and the ranks are the variables themselves.
Okada and Kasai instead take the roots `(ℓ, 0)` of the cyclic translations of a pair-partition code, with variables `(ℓ, a)` ranked by `ℓP + a`, and their implementation does so only if `P` is prime and `W < P`.
qLDPC takes the orbits of any group of automorphisms that it verifies.

The `_found_qudit_automorphisms` of a quantum code provide generators of `Γ`.
Classical codes are searched without symmetries, from every variable.
Finding them takes milliseconds for codes of hundreds of qubits, but about half a second with `pybliss` for a code of 5,200 qubits, as long as a search of more than a million states on one core.
So the search of each problem, such as a sector of a CSS code, asks for them at most once, before the first weight limit at which it expects to reach `automorphism_minimum_states` states, a field of the tuning of `_tuning.py`.
It expects the number of states of each weight limit to grow by the ratio between those of the last two.
A first limit above the divisor, which `cutoff` can set, has no such estimate, so the search asks for them before it.
Waiting until it has visited that many states instead would let a search that prunes well run a whole weight limit without them: on the [[276, 98, 14]] code, that limit took nearly half of the time of the search.

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
`infer_weight_divisor` of `_search/information_sets.py` finds this from a basis of `ker(H)`, as BZ does.

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
A limit without a witness is a multiple of the divisor `d` below the weight of every witness, which is at most the number of variables `n`, and a witness ends the search.
A correct search therefore takes at most the sum of `n // d` over its problems, so one that takes more has a bug, and raises a `RuntimeError` rather than loop forever.

**Interrupts.**
Each weight limit of a problem runs in one compiled call, which a keyboard interrupt cannot stop, so an interrupt takes effect only once that weight limit ends.
Python then raises a `KeyboardInterrupt`, or a `SystemError` that it caused.
The last weight limits of the largest searches of [Section 2](#2-when-to-use-which-method) can take hours.

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
  A search starts from the empty state, whose children are the three symbols of the root.
- **Completeness.**
  The argument of [Section 3.3](#33-the-depth-first-search) holds with `S` a set of symbols of `x`, and with `F` missing every symbol of `x`.
  An unsatisfied check `c` anticommutes with the restriction of `x` to `S`, and commutes with `x`, so it is flipped by a symbol of `x` on a variable outside `S`.
  The first such symbol among the children of `c` is visited before it joins `F`.
  The earlier siblings block the other values of their variables only while their own branches are in the stack.
- **Pruning.**
  A completion `T` has one symbol on each new variable, so its size is the number of variables that it adds.
  The dual weights `y_c` must make the weights of the unsatisfied checks that each available symbol flips total at most one, and the overlap and repair of [Section 3.4](#34-pruning) are those of symbols.
  A `Y` flips the checks that act on its qubit as `X` or as `Z`, while an `X` flips those that act as `Z` or `Y`, and a `Z` those that act as `X` or `Y`.
  So a `Y` can flip more checks than either: in a CSS code with Hadamards, no check acts as `Y`, and a `Y` flips every check on its qubit, which raises overlaps and lowers the weights.
- **Symmetries.**
  A permutation of the qubits moves both halves of each word, and qLDPC verifies it on the `2n` columns.
  The roots and ranks are those of variables, and the search from a root blocks every value of the variables of lower rank.
- **Logical test and divisor.**
  The test columns of a symbol are those of its word: for `Y`, the sum of those of `X` and `Z`.
  qLDPC infers no weight divisor for these problems, and steps their weight limits by one.

## 4. Guide to the source code

The names below are in `_search/connected_support.py`, unless the table says otherwise.

| Concept | Code |
| --- | --- |
| Pair `(H, G)`, its words and witnesses | `checks` and `excluded` of `SupportSearch` and `get_distance_connected_support` |
| Sectors of a CSS code | `CSSCode._get_parity_check_sector` in `src/qldpc/codes/common.py` |
| Classical codes, and qubit codes that are not CSS | `ClassicalCode.get_distance_exact` and `QuditCode.get_distance_exact` in `src/qldpc/codes/common.py` |
| Symplectic problems, values, and symbols | `homogeneous` of `SupportSearch`, its `_get_symbol_columns`, and `_shift_other_values` |
| Distance of several sectors | `get_distance_connected_support`, which `CSSCode._get_distance_of_sectors` in `src/qldpc/codes/common.py` reaches through `core.py` |
| Search through a weight limit `W` | `SupportSearch.find_witness` |
| Logical test | `SupportSearch._get_tests`, and `_is_outside_excluded` |
| Roots, ranks, and symmetries | `SupportSearch._use_automorphisms`, and `verify_automorphisms` and `get_orbits` of `_search/automorphism_families.py` |
| States, branching, and undoing what a state forbade when the search leaves it | `_search_supports`, whose `levels` mark the available symbols |
| Weights, pruning, and the symbols that a state forbids for its subtree | `_get_branches` |
| When a search asks for symmetries | `automorphism_minimum_states` of `SearchTuning` in `_tuning.py` |
| Weight divisor | `infer_weight_divisor` of `_search/information_sets.py` |

`CSSCode._get_distance_of_sectors` in `src/qldpc/codes/common.py` builds the sectors and calls `_get_distance_connected_support` of `core.py`, which passes the active tuning to the search.
`ClassicalCode.get_distance_exact` and `QuditCode.get_distance_exact` call it with their problems, and `SupportSearch` validates every problem.
Nonbinary codes reject the method.
`logical_counts.py` does not accept the method, since counting needs every word of a weight, and not one witness.
`get_distance_classical` and `get_distance_quantum` reject it, since their inputs are generators and not parity checks.
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
- **Return the exact minimum unless a cutoff permits an early exit.**
  The weights of the limits rise one divisor at a time, so that the first witness is the lightest.
  An early exit returns the weight of a known witness, which is an upper bound.

## References

- K. Okada and K. Kasai, "Pair-partition constructions for CPM-based quantum LDPC codes", Section V, [arXiv:2607.14091](https://arxiv.org/abs/2607.14091): the connected-support search for the pair-partition codes, with the implementation in [kasaikenta/pair-partition-cpm-css-codes](https://github.com/kasaikenta/pair-partition-cpm-css-codes).
- J. Y. Lee, K. Okada, N. Maskara, K. Kasai, and H. Zhou, "Design principles for ultra-high-rate quantum codes", [arXiv:2609.30069](https://arxiv.org/abs/2609.30069): the pair-partition codes, with the instances of [ultra-high-rate-qec/design-principles-data](https://github.com/ultra-high-rate-qec/design-principles-data).
- G. Richter, "Finding small stopping sets in the Tanner graphs of LDPC codes", 4th International Symposium on Turbo Codes and Related Topics and 6th International ITG-Conference on Source and Channel Coding, 2006, pp. 1–5.
- E. Rosnes and Ø. Ytrehus, "An efficient algorithm to find all small-size stopping sets of low-density parity-check matrices", IEEE Transactions on Information Theory 55, 4167 (2009).
- S. M. Khatami, L. Danjean, D. V. Nguyen, and B. Vasić, "An efficient exhaustive low-weight codeword search for structured LDPC codes", Information Theory and Applications Workshop, 2013, pp. 401–410.
