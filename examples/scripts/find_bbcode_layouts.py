#!/usr/bin/env python3
"""Find bivariate bicycle code qubit layouts with minimal real-space qubit communication distances.

The qubit placement strategy is described in arXiv:2404.18809.
"""

import functools
import itertools
import math
import sys
import time
from collections.abc import Callable, Iterator

import numpy as np
import numpy.typing as npt
import scipy.optimize
from sympy.abc import x, y

import qldpc

Basis2D = tuple[tuple[int, int], tuple[int, int]]
LayoutParams = tuple[bool, Basis2D, Basis2D, tuple[int, int]]

DEFAULT_PRECISION = 0.01


def get_best_known_layout_params(code: qldpc.codes.BBCode, folded_layout: bool) -> LayoutParams:
    """Retrieve the best known layout parameters for a bivariate bicycle code.

    This function can be used to identify optimized qubit layout parameters, the locations of qubits
    with that layout, and the maximum qubit communication distance for the layout with:
    ```
    layout_params = get_best_known_layout_params(code, folded_layout)
    get_qubit_pos = get_qubit_pos_func(code, layout_params)
    for node in code.graph.nodes:
        print(node, get_qubit_pos(node))
    max_distance = get_max_comm_distance(code, layout_params)
    print("max_distance:", max_distance)
    ```
    """
    basis_l = basis_r = shift_lr = None

    if folded_layout:
        if code == qldpc.codes.BBCode({x: 6, y: 6}, x**3 + y + y**2, y**3 + x + x**2):
            basis_l = ((1, 1), (1, 2))
            basis_r = ((-1, -1), (-1, -2))
            shift_lr = (-2, -1)
        elif code == qldpc.codes.BBCode({x: 15, y: 3}, x**9 + y + y**2, 1 + x**2 + x**7):
            basis_l = ((2, 0), (0, 1))
            basis_r = ((2, 0), (0, 1))
            shift_lr = (0, 0)
        elif code == qldpc.codes.BBCode({x: 9, y: 6}, x**3 + y + y**2, y**3 + x + x**2):
            basis_l = ((-2, 2), (-3, -1))
            basis_r = ((2, -2), (3, 1))
            shift_lr = (-4, 0)
        elif code == qldpc.codes.BBCode({x: 12, y: 6}, x**3 + y + y**2, y**3 + x + x**2):
            basis_l = ((1, 1), (0, 1))
            basis_r = ((-1, -1), (0, -1))
            shift_lr = (-1, 0)
        elif code == qldpc.codes.BBCode({x: 12, y: 12}, x**3 + y**2 + y**7, y**3 + x + x**2):
            basis_l = ((0, 5), (1, -3))
            basis_r = ((0, 5), (1, -3))
            shift_lr = (-1, 2)

    if basis_l is None or basis_r is None or shift_lr is None:
        raise ValueError(
            f"Layout parameters unknown for the following BBCode with"
            f" folded_layout={folded_layout}:\n{code}"
        )

    return folded_layout, basis_l, basis_r, shift_lr


def find_layout_params(
    code: qldpc.codes.BBCode,
    folded_layout: bool,
    *,
    precision: float = DEFAULT_PRECISION,
    restricted_search: bool = True,
    verbose: bool = True,
) -> LayoutParams:
    """Optimize BBCode layout parameters, as described in arXiv:2404.18809.

    This function can be used to identify optimized qubit layout parameters, the locations of qubits
    with that layout, and the maximum qubit communication distance for the layout with:
    ```
    layout_params = find_layout_params(code, folded_layout)
    get_qubit_pos = get_qubit_pos_func(code, layout_params)
    for node in code.graph.nodes:
        print(node, get_qubit_pos(node))
    max_distance = get_max_comm_distance(code, layout_params)
    print("max_distance:", max_distance)
    ```
    """
    # initialize the optimized (min-max) communication distance to an upper bound
    optimal_distance = get_max_qubit_distance(code)

    # precompute the support of parity checks
    check_supports = get_check_supports(code)

    # iterate over layout parameters of the BBCode
    for basis_l, basis_r, shift_lr in get_layout_search_space(code, restricted_search):
        # compute a matrix of squared communication distances for this layout
        layout_params = folded_layout, basis_l, basis_r, shift_lr
        placement_matrix = get_placement_matrix(
            code, layout_params, check_supports=check_supports, validate=False
        )
        # minimize the maximum communication distance for this layout
        min_max_distance = get_min_max_communication_distance(
            placement_matrix, optimal_distance, precision=precision
        )
        if min_max_distance < optimal_distance:
            optimal_basis_l = basis_l
            optimal_basis_r = basis_r
            optimal_shift_lr = shift_lr
            optimal_distance = min_max_distance
            if verbose:
                print()
                print("new best found:", min_max_distance)
                print("basis_l:", basis_l)
                print("basis_r:", basis_r)
                print("shift_lr:", shift_lr)
                sys.stdout.flush()

    return folded_layout, optimal_basis_l, optimal_basis_r, optimal_shift_lr


def get_max_qubit_distance(code: qldpc.codes.BBCode) -> float:
    """Get the maximum distance between two qubits in the given code."""
    return 2 * math.sqrt(sum(xx**2 for xx in code.orders))


def get_check_supports(code: qldpc.codes.BBCode) -> npt.NDArray[np.int_]:
    """Identify the support of the parity checks of the given code."""
    return np.vstack(
        [np.where(stabilizer) for stabilizer in itertools.chain(code.matrix_x, code.matrix_z)]
    )


def get_layout_search_space(
    code: qldpc.codes.BBCode, restricted_search: bool
) -> Iterator[tuple[Basis2D, Basis2D, tuple[int, int]]]:
    """Iterate over layout parameters of a BBCode.

    Each instance of layout parameters corresponds to:
    - a basis of lattice vectors for determining the placement of L-type qubits,
    - a basis of lattice vectors for determining the placement of R-type qubits,
    - a relative shift (between L and R) of the qubit plaquettes of the BBCode.
    """
    # identify the sets of lattice vectors that are used to relabel qubit plaquettes
    vector_pairs = itertools.combinations(np.ndindex(code.orders), 2)
    lattice_vectors = [
        (vec_a, vec_b) if code.get_order(vec_a) >= code.get_order(vec_b) else (vec_b, vec_a)
        for vec_a, vec_b in vector_pairs
        if code.is_valid_basis(vec_a, vec_b)
    ]

    # iterate over all lattice vector bases for L-type qubits
    for basis_l in lattice_vectors:
        order_0 = code.get_order(basis_l[0])

        # determine the search space of lattice vectors for R-type qubits
        if restricted_search:
            # only consider the same basis, or "minus" the same basis
            (aa, bb), (cc, dd) = basis_l
            bases_r = [((aa, bb), (cc, dd)), ((-aa, -bb), (-cc, -dd))]
        else:
            # consider all lattice vectors with the same orders
            bases_r = [
                (vec_a, vec_b)
                for vec_a, vec_b in lattice_vectors
                if code.get_order(vec_a) == order_0
            ] + [
                (vec_b, vec_a)
                for vec_a, vec_b in lattice_vectors
                if vec_a != vec_b and code.get_order(vec_b) == order_0
            ]

        for basis_r in bases_r:
            for shift_lr in np.ndindex(code.orders):
                yield basis_l, basis_r, shift_lr


def get_min_max_communication_distance(
    placement_matrix: npt.NDArray[np.int_],
    distance_cutoff: float,
    *,
    precision: float = DEFAULT_PRECISION,
) -> float:
    """Minimize the maximum communication distance for a given placement matrix.

    The optimization stategy is to initialize lower and upper bounds for the min-max communication
    distance, and repeatedly bisect these bounds until we reach a desired precision for the min-max.

    The distance_cutoff argument is used for early stopping: if the lower bound for the  min-max is
    greater than the distance_cutoff, then quit early and return a number greater than the
    distance_cutoff.

    We initialize the upper bound to twice the distance_cutoff plus a little bit, so that if the
    min-max for this placement_matrix is above the cutoff, the first bisection will fail and we will
    terminate right away.
    """
    low, high = 0.0, 2 * distance_cutoff + precision
    while True:
        mid = (low + high) / 2
        if has_perfect_matching(placement_matrix <= int(mid**2)):
            high = mid
            if high - low < precision:
                return mid
        else:
            low = mid
            if high - low < precision or low > distance_cutoff:
                return high


def get_placement_matrix(
    code: qldpc.codes.BBCode,
    layout_params: LayoutParams,
    *,
    check_supports: npt.NDArray[np.int_] | None = None,
    validate: bool = True,
) -> npt.NDArray[np.int_]:
    """Construct a placement matrix of squared maximum communication distances.

    Rows and columns of the placement matrix are indexed by candidate qubit locations (locs) and
    check qubits for placement, such that the value at placement_matrix[loc_index, node_index]
    is the answer to the question: when the given node is placed at the given location, what is that
    node's maximum squared distance to any of its neighbors in the Tanner graph of the code?
    """
    # identify how data qubits are permuted relative to the default data qubit layout
    layout_data_locs = get_data_qubit_locs(code, layout_params, validate=validate)
    layout_perm = np.lexsort(layout_data_locs.T)
    inverse_perm = np.empty_like(layout_perm)
    inverse_perm[layout_perm] = np.arange(layout_perm.size, dtype=int)

    # identify matrix of squared distances between all pairs of (check, data) qubit locations
    folded_layout, (vec_a, vec_b) = layout_params[:2]
    orders = code.get_order(vec_a), code.get_order(vec_b)
    squared_distance_matrix = get_squared_distance_matrix(folded_layout, orders)[:, inverse_perm]

    """
    Compute a tensor of squared distances, with shape (num_sites, num_checks, num_neighbors), for
    which squared_distance_tensor[loc_index, check_index, neighbor_index] is the squared distance
    between a check qubit and one of its neighbors when the check qubit in a given location.
    """
    check_supports = check_supports if check_supports is not None else get_check_supports(code)
    squared_distance_tensor = squared_distance_matrix[:, check_supports]

    # matrix of maximum squared communication distances
    return np.max(squared_distance_tensor, axis=2)


def get_data_qubit_locs(
    code: qldpc.codes.BBCode, layout_params: LayoutParams, *, validate: bool = True
) -> npt.NDArray[np.int_]:
    """Get the locations of data qubits in particular layout of a BBCode."""
    folded_layout, basis_l, basis_r, shift_lr = layout_params
    orders = (code.get_order(basis_l[0]), code.get_order(basis_l[1]))
    if validate:
        assert code.is_valid_basis(*basis_l)
        assert code.is_valid_basis(*basis_r)
        assert orders == (code.get_order(basis_r[0]), code.get_order(basis_r[1]))

    # get qubit locations in each sector
    qubit_locs_l = get_layout_sector_qubit_locs(basis_l, orders, code.orders, folded_layout, "L")
    qubit_locs_r = get_layout_sector_qubit_locs(basis_r, orders, code.orders, folded_layout, "R")

    # shift the plaquettes of R-type qubits
    qubit_locs_r = qubit_locs_r.reshape(*orders, 2)
    qubit_locs_r = np.roll(qubit_locs_r, shift=shift_lr, axis=(0, 1))
    qubit_locs_r = qubit_locs_r.reshape(-1, 2)

    return np.vstack([qubit_locs_l, qubit_locs_r])


@functools.cache
def get_layout_sector_qubit_locs(
    basis: Basis2D,
    basis_orders: tuple[int, int],
    torus_shape: tuple[int, int],
    folded_layout: bool,
    sector: str,
) -> npt.NDArray[np.int_]:
    """Get the locations of qubits in a particular sector of a particular layout of a BBCode.

    First, relabel qubit plaquettes, which naturally live on torus with shape torus_shape and are
    thereby labeled by their "plaquette coordinates" on a rectangular grid.  If the old plaquette
    label was (x, y) (mod torus_shape), the new label is the coefficients (a, b) (mod basis_orders)
    for which (x, y) = a * basis[0] + b * basis[1].

    After relabeling plaquettes by their coordinates in a new basis for a torus, assign qubit
    positions identically to qldpc.codes.BBCode.get_qubit_pos in a vectorized manner.
    """
    vec_a, vec_b = basis

    # invert the modular system of equations (x, y) = a * basis[0] + b * basis[1]
    plaquette_map = np.empty(torus_shape + (2,), dtype=int)
    for aa, bb in itertools.product(range(basis_orders[0]), range(basis_orders[1])):
        plaquette_map[
            (aa * vec_a[0] + bb * vec_b[0]) % torus_shape[0],
            (aa * vec_a[1] + bb * vec_b[1]) % torus_shape[1],
        ] = aa, bb

    # assign qubit locations identically to qldpc.codes.BBCode.get_qubit_pos
    qubit_locs = plaquette_map.reshape(-1, 2)
    qubit_locs = 2 * qubit_locs
    if sector == "R" or sector == "X":
        qubit_locs[:, 0] += 1
    if sector == "L" or sector == "X":
        qubit_locs[:, 1] += 1
    if folded_layout:
        for axis in range(2):
            low = qubit_locs[:, axis] < basis_orders[axis]
            qubit_locs[low, axis] = 2 * qubit_locs[low, axis]
            qubit_locs[~low, axis] = (2 * basis_orders[axis] - 1 - qubit_locs[~low, axis]) * 2 + 1

    return qubit_locs


@functools.cache
def get_squared_distance_matrix(
    folded_layout: bool, torus_shape: tuple[int, int]
) -> npt.NDArray[np.int_]:
    """Compute a matrix of squared distances between default check and data qubit locations.

    Here torus_shape is the dimensions of the grid of qubit plaquettes that the qubits live on.
    """
    # identify locations of data and check qubits
    data_locs = get_default_qubit_locs(folded_layout, torus_shape, data=True)
    check_locs = get_default_qubit_locs(folded_layout, torus_shape, data=False)

    # construct a tensor of all pair-wise displacements
    displacements = check_locs[:, None, :] - data_locs[None, :, :]

    # square displacements to get squared distances
    return np.einsum("...i,...i->...", displacements, displacements)


@functools.cache
def get_default_qubit_locs(
    folded_layout: bool, torus_shape: tuple[int, int], *, data: bool
) -> npt.NDArray[np.int_]:
    """Identify the default locations qubits when placed on a torus with the given dimensions."""
    num_checks = 2 * torus_shape[0] * torus_shape[1]
    nodes = [qldpc.objects.Node(index, is_data=data) for index in range(num_checks)]
    locs = [
        qldpc.codes.BBCode.get_qubit_pos_from_orders(node, folded_layout, torus_shape)
        for node in nodes
    ]

    # sort the locations and return
    locs_array = np.array(locs, dtype=int)
    return locs_array[np.lexsort(locs_array.T)]


def has_perfect_matching(biadjacency_matrix: npt.NDArray[np.bool_]) -> bool | np.bool_:
    """Does a bipartite graph with the given biadjacenty matrix have a perfect matching?"""
    # quit early if any vertex has no indicent edges <--> any row/column is all zeros
    if np.any(~np.any(biadjacency_matrix, axis=0)) or np.any(~np.any(biadjacency_matrix, axis=1)):
        return False
    rows, cols = scipy.optimize.linear_sum_assignment(biadjacency_matrix, maximize=True)
    return np.all(biadjacency_matrix[rows, cols])


def get_qubit_pos_func(
    code: qldpc.codes.BBCode,
    layout_params: LayoutParams,
    max_comm_distance: float | None = None,
    *,
    precision: float = DEFAULT_PRECISION,
    validate: bool = True,
) -> Callable[[qldpc.objects.Node], tuple[int, int]]:
    """Construct a function that gives positions of qubits in particular layout of a BBCode.

    Check qubits are required to satisfy a maximum communication distance constraint.  If no such
    constraint is provided, minimize the maximum communication distance.
    """
    # identify candidate check qubit locations
    folded_layout, (vec_a, vec_b) = layout_params[:2]
    orders = code.get_order(vec_a), code.get_order(vec_b)
    candidate_locs = get_default_qubit_locs(folded_layout, orders, data=False)

    # identify the maximum qubit communication distance that we will tolerate
    placement_matrix = get_placement_matrix(code, layout_params)
    if not max_comm_distance:
        distance_cutoff = get_max_qubit_distance(code)
        max_comm_distance = get_min_max_communication_distance(
            placement_matrix, distance_cutoff, precision=precision
        )

    # assign check qubits to locations
    biadjacency_matrix = placement_matrix <= (max_comm_distance + 1e-15) ** 2
    check_loc_indices, check_qubit_indices = scipy.optimize.linear_sum_assignment(
        biadjacency_matrix, maximize=True
    )
    if not np.all(biadjacency_matrix[check_loc_indices, check_qubit_indices]):
        raise ValueError(f"A maximum communication distance of {max_comm_distance} is unachievable")

    data_qubit_locs = get_data_qubit_locs(code, layout_params, validate=validate)
    check_qubit_locs = candidate_locs[np.argsort(check_qubit_indices)]

    def get_qubit_pos(node: qldpc.objects.Node) -> tuple[int, int]:
        """Get the position of a qubit in a BBCode (with a particular layout)."""
        loc = data_qubit_locs[node.index] if node.is_data else check_qubit_locs[node.index]
        return tuple(loc)

    return get_qubit_pos


def get_max_comm_distance(
    code: qldpc.codes.BBCode, layout_params: LayoutParams, *, precision: float = DEFAULT_PRECISION
) -> float:
    """Get the maximum qubit communication distance in particular BBCode layout."""
    get_qubit_pos = get_qubit_pos_func(code, layout_params, precision=precision)

    max_squared_distance = 0
    for check_qubit_index in range(len(code)):
        node = qldpc.objects.Node(check_qubit_index, is_data=False)
        loc_0 = get_qubit_pos(node)
        for neighbor in code.graph.successors(node):
            loc_1 = get_qubit_pos(neighbor)
            squared_distance = (loc_0[0] - loc_1[0]) ** 2 + (loc_0[1] - loc_1[1]) ** 2
            if squared_distance > max_squared_distance:
                max_squared_distance = squared_distance

    return math.sqrt(max_squared_distance)


def get_completed_qubit_pos_func(
    code: qldpc.codes.BBCode,
    data_qubit_locs: npt.NDArray[np.int_],
    *,
    lattice_shape: tuple[int, int] | None = None,
    check_supports: npt.NDArray[np.int_] | None = None,
    max_comm_distance: float | None = None,
    precision: float = DEFAULT_PRECISION,
) -> Callable[[qldpc.objects.Node], tuple[int, int]]:
    """Complete a qubit location assignment for a BBCode.

    Here "data_qubit_locs" is a 2-D array with shape (num_data_qubits, 2) for which
    data_qubit_locs[data_qubit_index] is the location of the data qubit with index data_qubit_index.
    From that data, this method returns a function that maps (any) qubit to a location.

    Other optional arguments that may speed things up:
    - lattice_shape: the dimensions of the rectangular grid that the qubits live on
    - check_supports: the supports of the parity checks; should be get_check_supports(code)
    - max_comm_distance: the maximum communication distance to enforce
    - precision: the precision to which we minimize max_comm_distance, if a value was not provided
    """
    if lattice_shape is None:
        # try to guess the shape of the qubit grid
        lattice_shape = (2 * code.orders[0], 2 * code.orders[1])
        if (
            np.max(data_qubit_locs[:, 0]) >= lattice_shape[0]
            or np.max(data_qubit_locs[:, 1]) >= lattice_shape[1]
        ):
            lattice_shape = lattice_shape[::-1]

    # identify indices of lattice sites occupied by data qubits
    data_qubit_loc_indices = np.ravel_multi_index(data_qubit_locs.T, dims=lattice_shape)

    # identify unoccupied lattice sites and the indices of those lattice sites
    all_loc_indices = np.arange(len(code), dtype=int)
    check_qubit_loc_indices = all_loc_indices[~np.isin(all_loc_indices, data_qubit_loc_indices)]
    candidate_locs = np.array(
        np.unravel_index(check_qubit_loc_indices, shape=lattice_shape), dtype=int
    ).T

    # construct a matrix of squared distances between (check_qubit_location, data_qubit) pairs
    squared_distance_matrix = get_full_squared_distance_matrix(lattice_shape)[
        np.ix_(check_qubit_loc_indices, data_qubit_loc_indices)
    ]

    # matrix of squared maximum comm distances for (check_qubit_loc, check_qubit) assignments
    check_supports = check_supports if check_supports is not None else get_check_supports(code)
    squared_distance_tensor = squared_distance_matrix[:, check_supports]
    placement_matrix = np.max(squared_distance_tensor, axis=2)

    if max_comm_distance is None:
        # minimize the maximum communication distance for all check qubit location assignments
        max_qubit_distance = get_max_qubit_distance(code)
        max_comm_distance = get_min_max_communication_distance(
            placement_matrix, max_qubit_distance, precision=precision
        )

    # assign check qubits to candidate locations
    biadjacency_matrix = placement_matrix <= (max_comm_distance + 1e-15) ** 2
    check_loc_indices, check_qubit_indices = scipy.optimize.linear_sum_assignment(
        biadjacency_matrix, maximize=True
    )

    check_qubit_locs = candidate_locs[np.argsort(check_qubit_indices)]

    def get_qubit_pos(node: qldpc.objects.Node) -> tuple[int, int]:
        """Get the position of a qubit in a BBCode (with a particular layout)."""
        loc = data_qubit_locs[node.index] if node.is_data else check_qubit_locs[node.index]
        return tuple(loc)

    return get_qubit_pos


@functools.cache
def get_full_squared_distance_matrix(lattice_shape: tuple[int, int]) -> npt.NDArray[np.int_]:
    """Construct a matrix of squared distances between all pair of lattice points."""
    locs = np.array(list(np.ndindex(lattice_shape)))
    displacements = locs[:, None, :] - locs[None, :, :]
    return np.einsum("...i,...i->...", displacements, displacements)


if __name__ == "__main__":
    # the codes in Table 1 of arXiv:2404.18809, whose layouts we want to optimize
    codes = [
        qldpc.codes.BBCode(
            {x: 6, y: 6},
            x**3 + y + y**2,
            y**3 + x + x**2,
        ),
        qldpc.codes.BBCode(
            {x: 15, y: 3},
            x**9 + y + y**2,
            1 + x**2 + x**7,
        ),
        qldpc.codes.BBCode(
            {x: 9, y: 6},
            x**3 + y + y**2,
            y**3 + x + x**2,
        ),
        qldpc.codes.BBCode(
            {x: 12, y: 6},
            x**3 + y + y**2,
            y**3 + x + x**2,
        ),
        qldpc.codes.BBCode(
            {x: 12, y: 12},
            x**3 + y**2 + y**7,
            y**3 + x + x**2,
        ),
    ]
    # consider "folded" qubit layouts, as in Figure 2 of arXiv:2404.18809
    folded_layout = True

    for code in codes:
        print()
        print("(n, k):", (len(code), code.dimension))
        print(code)

        start = time.time()
        layout_params = get_best_known_layout_params(code, folded_layout)
        # layout_params = find_layout_params(code, folded_layout)
        # print("optimization time:", time.time() - start)

        # print qubit positions
        get_qubit_pos = get_qubit_pos_func(code, layout_params)
        # for node in sorted(code.graph.nodes):
        #     print(node, get_qubit_pos(node))

        max_distance = get_max_comm_distance(code, layout_params)
        print("maximum communication distance:", max_distance)
