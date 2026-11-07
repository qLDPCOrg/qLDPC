# Connected-support distance search in qLDPC

## 1. Purpose and scope

The connected-support search computes the exact distance of a binary CSS code, including a subsystem code, from the Tanner graphs of its parity checks.
It is the method `"connected_support"` of `CSSCode.get_distance_exact`.
Brouwer–Zimmermann (BZ, see [brouwer_zimmermann.md](brouwer_zimmermann.md)) is the default.
The connected-support search is a quantum version of the searches for small stopping sets and low-weight codewords of classical LDPC codes (Richter, Rosnes and Ytrehus, and Khatami et al. in the [references](#references)), and follows Section V of Okada and Kasai (also in the references).
This guide explains the search, maps it to the code, and states what a change to it must preserve.
It is organized as follows:

- [Section 2](#2-when-to-use-which-method) says when to use this method rather than BZ.
- [Section 3](#3-the-algorithm) explains the algorithm, and where qLDPC differs from the paper.
- [Section 4](#4-guide-to-the-source-code) maps the algorithm to the source code.
- [Section 5](#5-correctness-requirements) lists the requirements that every change to the search must keep.

## 2. When to use which method

BZ examines the words of a basis of the searched space in layers of increasing coefficient weight, and its cost grows like `binom(r, d/m)`.
Here `r` is the dimension of the searched space, `d` is the distance, and `m` is the number of disjoint information sets.
The sectors of CSS codes usually have `r > n/2`, hence `m = 1`, and a high rate pushes `r` towards `n`, so that BZ becomes infeasible at modest distances.
The connected-support search instead enumerates supports that grow through unsatisfied checks, so its cost depends on the sparsity of the checks and on `d`, and not on `r`.
Its cost grows by a roughly constant factor for each additional unit of weight, which is about 5 for the pair-partition codes below, with row weight 8 to 16, and about 3 for surface codes.

A **sector** of a CSS code is its X-type or its Z-type logical operators, which a search treats as separate problems ([Section 3.1](#31-the-problem)).
The table lists times on one core of an Apple M3 Max laptop, run at low priority, with the benchmark script [experiments/connected_support_distance/benchmark.py](../../../../experiments/connected_support_distance/benchmark.py).
The codes are the pair-partition codes of Okada and Kasai and of Lee et al., which are quasi-cyclic CSS codes, mostly of high rate, with parity checks of column weight 3 to 5 (see the [references](#references)).
The instances, and their distances, come from the data of Lee et al., and Okada and Kasai also publish the [[276, 98, 14]] code.
Times are CPU times, for the whole code unless they say per sector.
The BZ times of the larger codes are estimated from the work that BZ predicts for them (Sections 3.6 and 3.7 of [brouwer_zimmermann.md](brouwer_zimmermann.md)), with the cyclic automorphisms that qLDPC finds for these codes.

| Code | BZ | Connected support | Faster |
| --- | --- | --- | --- |
| [[110, 8, 12]] | 0.5 s | 0.3 s | connected support |
| [[232, 62, 12]] | 212 s | 0.2 s | connected support |
| [[276, 98, 14]] | about 7 days (estimated) | 6.4 s | connected support |
| [[518, 228, 16]] | infeasible | 252 s | connected support |
| [[372, 130, 18]] | about 1.5 years (estimated) | about 10 minutes per sector (extrapolated) | connected support |
| [[170, 8, 20]] | about 4 hours (estimated) | about an hour per sector (extrapolated) | comparable |
| [[944, 478, 20]] | infeasible | about half a day per sector (extrapolated) | connected support |

Choose the connected-support search for sparse codes, and especially for high-rate ones, and BZ for codes with dense checks, as the quantum Reed–Muller code below shows.
For sparse codes of low rate and high distance, the two can be comparable, as the [[170, 8, 20]] row shows.
The script measures the other connected-support times of the table, and those of the surface codes below, and the extrapolations come from timed searches through lower weights, whose costs grow by a factor of 20 to 33 per two units of weight for these codes.
They assume that this growth continues, and the connected-support times of the [[372, 130, 18]], [[170, 8, 20]], and [[944, 478, 20]] codes are not measured.

The cost of the connected-support search grows quickly if the checks are dense, because every unsatisfied check offers many variables to branch on.
The [[128, 70, 8]] quantum Reed–Muller code, which has checks of weight 32, takes 0.9 s, and BZ takes 0.1 s.
These times are for a plain `CSSCode` built from the matrices of `QuantumReedMullerCode(2, 7)`, since that class knows its distance without a search.
The stabilizers of surface codes weigh much less than their distances, so that many supports grow without being pruned, but the search is still much faster than BZ.
Searching rotated surface codes takes 0.03 s for distance 11, 0.8 s for distance 15, 6.1 s for distance 17, and 47 s for distance 19.
BZ takes 0.5 s for distance 11, and more than 200 s for distance 15.
A subsystem code searches the Tanner graph of its stabilizers, which can be dense.

## 3. The algorithm

### 3.1 The problem

Let `H` and `G` be binary matrices with `rowspan(G) ⊆ ker(H)`.
The distance of the pair is

$$
d(H, G) = \min \{ \operatorname{wt}(x) : Hx = 0,\; x \notin \operatorname{rowspan}(G) \}.
$$

A vector `x` in `ker(H)` outside `rowspan(G)` is a **witness**.
Its **support** is the set of its nonzero coordinates, which are called **variables**, and its **syndrome** is the set of the rows of `H`, the **checks**, that have an odd number of variables in the support.
A check is **unsatisfied** if it is in the syndrome.

For a CSS code, the distance of the X-type logical operators is `d(H_Z, H_X)`, and the distance of the Z-type logical operators is `d(H_X, H_Z)`.
The distance of the code is the smaller of the two.
For a subsystem code, `H` holds the stabilizers of the opposite type, so that its kernel holds the X-type or Z-type operators that commute with every stabilizer, and `G` holds the gauge generators of the same type.
A witness is then a dressed logical operator, as in BZ.

### 3.2 Light words are connected

**Lemma** (after Lemma 2 of [Okada and Kasai](#references)).
Every witness of minimum weight has a support that is connected in the Tanner graph of `H`, and that contains no support of a nonzero vector of `ker(H)` but its own.

*Proof.*
Split the support of a minimum-weight witness `x` into its connected components.
No check is adjacent to variables of two components, so each component has an empty syndrome.
If every component were in `rowspan(G)`, their sum `x` would be too.
So one component is a witness, and by minimality it is all of `x`.
Now let `y` be a nonzero vector of `ker(H)` whose support is a proper subset of that of `x`.
Then `y` and `x + y` are lighter than `x`, and they cannot both be in `rowspan(G)`, since their sum is `x`.
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
Its children come in the order of their repairs ([Section 3.4](#34-pruning)), most repairing first, which on the [[276, 98, 14]] code visits a quarter of the states of the order of the variables through the weight limit at which the search finds a witness.
Every state grows only through unsatisfied checks, so every support that it visits is connected.

**Completeness.**
Suppose that `x` is a witness of minimum weight, whose variable of lowest rank is the root `r`.
By the lemma, no proper nonempty subset of its support has an empty syndrome.
Take a state with `S ⊊ supp(x)` and `F ∩ supp(x) = ∅`.
The syndrome of `S` is not empty, and the parity of `supp(x)` at every check is even, so an unsatisfied check `c` has a variable of `supp(x) \ S`.
That variable is available, since `F` misses `supp(x)`.
The first variable of `supp(x) \ S` in the order of the children of `c` is visited before it joins `F`, and gives a state of the same kind.
So the search reaches `supp(x)` itself.
Every bound of [Section 3.4](#34-pruning) holds for a state of this kind, since `x` is a witness that extends it, and so does every variable that [Section 3.4](#34-pruning) forbids for a subtree.

Forbidding the variables of earlier siblings makes the search visit each support at most once.

### 3.4 Pruning

A search through a weight limit `W` has `W - s` variables to spare at a state with `s` variables.
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
Let `Y` be the total weight of a state, and let the **repair** `R_v` of an available variable `v` be the weight of the unsatisfied checks that contain it.
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
- Rounding the bound up to the weight divisor prunes nothing, since every weight limit `W` is a multiple of the divisor, except a first limit that `cutoff` sets.

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

The `_found_qudit_automorphisms` of the code provide generators of `Γ`.
Finding them takes milliseconds for codes of hundreds of qubits, but about half a second with `pybliss` for a code of 5,200 qubits, as long as a search of more than a million states on one core.
So the search of each sector asks for them once, before the first weight limit at which it expects to reach `automorphism_minimum_states` states, a field of the tuning of `_tuning.py`.
It expects each weight limit to grow by the ratio between its last two.
Waiting until it has visited that many states instead would let a search that prunes well run a whole weight limit without them: on the [[276, 98, 14]] code, that limit took nearly half of the time of the search.

With symmetries, the search from the first root does most of the work, since every witness that meets the first orbit is found through it.
A witness that meets the first orbit several times can have several images that contain the first root, as many as its variables in that orbit if the group acts regularly on the orbit, and the search can visit images of a state more than once.
In the [[276, 98, 14]] code, two thirds of the states of that search hold two or more variables of the first orbit.
Rejecting such duplicates soundly for general groups is left open.

### 3.6 From a search to the distance

**Logical test.**
A vector `x` of `ker(H)` is in `rowspan(G)` exactly if it is orthogonal to `ker(G)`, and it is orthogonal to the rows of `H`, which lie in `ker(G)`.
So `x` is a witness exactly if it has an odd product with a row of a basis of `ker(G)` modulo `rowspan(H)`.
The search keeps these rows as packed columns, one word per 64 rows, and tests a state with an empty syndrome by XORing the columns of its variables.

**Weight divisor.**
If every vector of `ker(H)` has even weight, as for the pair-partition codes, whose block rows of checks sum to the all-ones vector, weights are multiples of two.
`infer_weight_divisor` of `_search/information_sets.py` finds this from a basis of `ker(H)`, as BZ does.

**Exact distance.**
Let `L` be the smallest weight that an unexamined witness can have, which is the divisor at first.
`get_distance_connected_support` searches the sector with the lowest `L` through the weight limit `W = L`.
A witness is then a lightest one if the earlier limits found none, since weights are multiples of the divisor.
Otherwise every witness is heavier than `W`, and `L` rises to the next multiple of the divisor.
This ends when `L` is at least the weight of the best known witness, the distance.
A search never needs a good upper bound, and a search of one sector shares the bound of the other, as in BZ.
If `cutoff` is above `L`, the first limit is `cutoff`, and a witness of weight at most `cutoff` is returned at once.
An upper bound from a known sector distance acts as a witness of that weight.

## 4. Guide to the source code

The names below are in `_search/connected_support.py`, unless the table says otherwise.

| Concept | Code |
| --- | --- |
| Pair `(H, G)`, its words and witnesses | `checks` and `excluded` of `SupportSearch` and `get_distance_connected_support` |
| Sectors of a code | `CSSCode._get_parity_check_sector` in `src/qldpc/codes/common.py` |
| Distance of several sectors | `get_distance_connected_support` |
| Search through a weight limit `W` | `SupportSearch.find_witness` |
| Logical test | `SupportSearch._get_tests`, and `_is_outside_excluded` |
| Roots, ranks, and symmetries | `SupportSearch._use_automorphisms`, and `verify_automorphisms` of `_search/automorphism_families.py` |
| States, branching, and undoing what a state forbade when the search leaves it | `_search_supports`, whose `levels` mark the available variables |
| Weights, pruning, and the variables that a state forbids for its subtree | `_get_branches` |
| When a search asks for symmetries | `automorphism_minimum_states` of `SearchTuning` in `_tuning.py` |
| Weight divisor | `infer_weight_divisor` of `_search/information_sets.py` |

`CSSCode.get_distance_exact` in `src/qldpc/codes/common.py` builds the sectors and calls `_get_distance_css_connected_support` of `core.py`, which validates them.
`logical_counts.py` does not accept the method, since counting needs every word of a weight, and not one witness.
`get_distance_classical` and `get_distance_quantum` reject it, since their inputs are generators and not parity checks.
Classical codes reject it too, so the closed forms of hypergraph product codes, which search their classical seed codes, search them with Brouwer–Zimmermann instead.
`_search_supports`, `_get_branches`, and `_is_outside_excluded` are compiled with `always_compiled`, and their small helpers with `always_compiled(inline=True)`, as [README.md](README.md) explains.

## 5. Correctness requirements

Changes to the search can use other bounds and orders, while keeping these requirements.

- **Every pruning must hold for a witness that extends the state.**
  A bound may use the available variables, which exclude `S` and `F`, since a minimum-weight witness extending the state has no other variables.
  `F` includes the variables of lower rank than the root, and the variables that a state forbade for its subtree.
  A bound must be a lower bound on the number of further variables that such a witness needs.
  A bound that is too strong can still find the distance, unless it prunes a lightest witness, so a test needs a case where the bound is tight at the distance, as the tests of the distances of known codes are.
- **Forbid a variable for a subtree only if no completion of the state contains it.**
  The variables that a state forbids must become available again when the search leaves the state.
- **Block siblings and variables of lower rank than the root.**
  A search from a root must use only variables of higher rank, and a variable must be forbidden for its later siblings.
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
- G. Richter, "Finding small stopping sets in the Tanner graphs of LDPC codes", 4th International Symposium on Turbo Codes and Related Topics, 2006.
- E. Rosnes and Ø. Ytrehus, "An efficient algorithm to find all small-size stopping sets of low-density parity-check matrices", IEEE Transactions on Information Theory 55, 4167 (2009).
- S. M. Khatami, L. Danjean, D. V. Nguyen, and B. Vasić, "An efficient exhaustive low-weight codeword search for structured LDPC codes", Information Theory and Applications Workshop, 2013, pp. 401–410.
