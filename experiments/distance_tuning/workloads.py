# SPDX-License-Identifier: Apache-2.0

"""Named codes and workloads for timing and retuning the exact distance searches of qLDPC.

A code name is one of:

- ``pp<n>``: an embedded pair-partition instance (see ``INSTANCES``), such as ``pp110``;
- ``gross``: the [[144, 12, 12]] bivariate bicycle code;
- ``bb288``: ``BBCode({x: 12, y: 12}, x**3 + y**2 + y**7, y**3 + x + x**2)``, the [[288, 12, 18]]
  code of the width data set of ``qldpc.codes.distance._tuning``;
- ``qt7``: ``QTCode.random(CyclicGroup(7), HammingCode(3), seed=1)``, also of that data set;
- ``lp<order>s<seed>``: the lifted product of a 3 x 4 matrix of monomials of
  ``CyclicGroup(order)``, whose exponents NumPy's default generator draws from ``seed``, such as
  ``lp13s5`` and ``lp13s8``;
- ``surface:<d>`` and ``toric:<d>``: rotated surface and toric codes of distance ``d``;
- ``classical:<bits>:<checks>:<seed>``: ``ClassicalCode.random(bits, checks, seed=seed)``;
- ``sparse:<bits>:<dimension>:<row_weight>:<seed>``: a classical code of low distance, whose
  generator matrix has ``dimension`` rows of ``row_weight`` ones each, in random columns that
  NumPy's default generator draws from ``seed``;
- a path to a ``.cpm`` file (see ``parse_cpm``).

A workload name is a kind, then a code name, then the arguments of the kind, separated by ``/``:

- ``distance/<code>/<sector>``: ``get_distance_exact`` of a quantum code, for the sector ``X``,
  ``Z``, or ``both``, or of a classical code, whose sector must be ``-``;
- ``hadamard/<code>``: ``get_distance_exact`` of the code that Hadamards on the second half of the
  qubits of a CSS code make, which is not CSS;
- ``count/<code>/<weight>``: ``CSSLogicalCounter.count_logical_ops(weight)`` by the BZ method;
- ``ccount/<code>/<weight>``: the same count, in a process whose warm-up first calls
  ``compile_search_setup`` (see ``trial.py``);
- ``coset/<code>/<weighing>/<error_weight>/<max_weight>/<num_syndromes>``: ``get_min_weights`` of
  a ``CosetLeaderSearch`` on the logical state ``|0...0>`` of a CSS code, by ``css`` or
  ``physical`` weight, for the syndromes of random Pauli errors of ``error_weight`` qubits.

Each workload returns a short string of its result, which every tuning must reproduce.
"""

from __future__ import annotations

import functools
import pathlib
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from sympy.abc import x, y

from qldpc import abstract, codes
from qldpc.codes.distance import CSSLogicalCounter
from qldpc.codes.distance._search.coset_leader_search import CosetLeaderSearch
from qldpc.objects import Pauli

####################################################################################################
# embedded pair-partition instances

# The parameters [[n, k, d]] and the ``.cpm`` text of each embedded instance.  These quasi-cyclic
# CSS codes come from https://github.com/ultra-high-rate-qec/design-principles-data
# (``codes/css/cpm_pp/instances``); https://github.com/kasaikenta/pair-partition-cpm-css-codes also
# publishes the ``276``, ``518``, and ``944`` instances.
INSTANCES: dict[str, tuple[tuple[int, int, int], str]] = {
    "110": (
        (110, 8, 12),
        """
        CPM_CSS_V1 j5l10_p11_c0002 11 10 5 5 0 0 0 0 0 0 0 0 0 0 0 1 6 3 10 9 5 7 8 2 0 8 4 2 3 1
        6 9 5 7 0 5 8 4 6 7 2 10 3 1 0 10 5 8 1 6 9 2 4 3 0 6 3 7 5 6 5 0 3 7 0 3 7 9 8 7 0 9 8 3
        0 9 10 5 2 0 9 10 5 2 0 4 2 1 7 1 2 7 4 0 0 2 1 6 9 9 6 2 0 1
        """,
    ),
    "170": (
        (170, 8, 20),
        """
        CPM_CSS_V1 j5l10_p17_c0000 17 10 5 5 0 0 0 0 0 0 0 0 0 0 0 15 4 3 8 7 6 5 10 2 0 16 12 5 3
        1 10 14 13 15 0 10 11 12 4 9 3 16 8 1 0 6 10 16 15 11 1 2 9 7 0 11 8 15 6 11 6 0 8 15 0 7
        9 14 10 9 0 14 10 7 0 1 16 13 2 0 1 16 13 2 0 9 3 1 5 1 3 5 9 0 0 2 1 10 7 7 10 2 0 1
        """,
    ),
    "232": (
        (232, 62, 12),
        """
        CPM_CSS_V1 j3l8_p29_g8_orbit_05025 29 8 3 3 0 0 0 0 0 0 0 0 0 5 4 18 2 1 8 16 0 10 24 15 3
        23 11 12 0 16 3 19 3 0 19 16 0 1 2 12 0 1 2 12 0 27 23 2 2 23 27 0
        """,
    ),
    "276": (
        (276, 98, 14),
        """
        CPM_CSS_V1 qc_276_98_14 23 12 4 4 9 3 11 13 14 19 5 22 10 15 2 21 16 20 11 9 18 12 18 17
        19 4 7 4 19 20 18 11 4 18 17 8 11 21 2 14 5 18 1 13 12 13 13 5 1 14 9 3 20 2 17 7 15 8 16
        21 16 9 3 10 2 8 11 5 3 22 5 2 15 4 18 13 22 21 7 4 8 20 19 12 4 10 3 17 12 14 13 14 6 19
        20 1 13 15 3 9
        """,
    ),
    "372": (
        (372, 130, 18),
        """
        CPM_CSS_V1 qc_372_130_18 31 12 4 4 21 3 24 1 7 23 17 30 24 4 19 23 5 7 27 23 9 28 24 5 2 1
        2 10 10 6 7 2 0 13 28 24 8 16 4 27 12 11 23 6 3 5 17 17 18 14 5 27 20 24 26 15 2 9 16 20
        26 18 14 9 2 5 3 16 11 0 27 7 26 8 0 7 13 8 0 9 29 10 25 22 15 9 6 30 19 27 27 29 14 25 24
        2 22 6 16 16
        """,
    ),
    "518": (
        (518, 228, 16),
        """
        CPM_CSS_V1 qc_518_228_16 37 14 4 4 3 3 4 5 6 7 8 9 10 11 12 13 14 15 12 13 16 31 29 24 13
        3 32 1 32 6 5 33 10 18 29 33 8 21 31 10 6 21 10 16 9 15 1 25 25 15 31 15 20 11 30 35 15 20
        4 28 13 8 27 13 10 36 23 13 33 19 27 18 24 7 2 21 15 6 9 9 29 20 31 13 11 14 32 18 2 24 12
        1 7 34 31 30 6 34 2 21 15 14 9 20 11 3 18 9 34 35 16 23 29 15 12 22
        """,
    ),
    "944": (
        (944, 478, 20),
        """
        CPM_CSS_V1 qc_944_478_20 59 16 4 4 13 53 38 20 16 56 9 15 54 11 15 32 4 20 58 27 17 2 2 11
        17 49 55 51 54 1 22 38 49 29 4 8 18 30 51 22 37 31 58 22 12 33 56 15 44 26 3 22 21 38 30
        48 15 35 19 58 19 14 54 58 29 8 26 57 34 17 25 46 14 37 37 42 18 57 13 53 31 46 39 55 10
        11 15 38 15 2 12 24 4 14 20 31 22 56 16 24 11 50 52 19 31 45 50 17 9 26 17 29 45 18 57 16
        10 13 40 20 14 55 15 40 1 10 15 57 1 18 1 46
        """,
    ),
}


def parse_cpm(text: str) -> codes.PairPartitionCode:
    """Build the pair-partition code of a ``.cpm`` token stream.

    The stream is whitespace-separated: ``CPM_CSS_V1``, a name, the lift size ``P``, the number
    ``L`` of block columns, the numbers ``J_X`` and ``J_Z`` of block rows, and then the X-type
    exponents and the Z-type exponents in row-major order.
    """
    magic, _name, *numbers = text.split()
    if magic != "CPM_CSS_V1":
        raise ValueError(f"Expected a CPM_CSS_V1 file, not {magic!r}")
    lift_size, block_length, num_rows_x, num_rows_z, *exponents = map(int, numbers)
    values = np.array(exponents)
    split = num_rows_x * block_length
    exponents_x = values[:split].reshape(num_rows_x, block_length)
    exponents_z = values[split:].reshape(num_rows_z, block_length)
    return codes.PairPartitionCode(exponents_x, exponents_z, lift_size)


####################################################################################################
# codes


@functools.cache
def get_code(name: str) -> codes.ClassicalCode | codes.CSSCode:
    """Build a named code (see the module docstring)."""
    kind, _, args = name.partition(":")
    if kind.startswith("pp") and kind[2:] in INSTANCES:
        return parse_cpm(INSTANCES[kind[2:]][1])
    if kind == "gross":
        return codes.GrossCode()
    if kind == "bb288":
        return codes.BBCode({x: 12, y: 12}, x**3 + y**2 + y**7, y**3 + x + x**2)
    if kind == "qt7":
        return codes.QTCode.random(abstract.CyclicGroup(7), codes.HammingCode(3), seed=1)
    if kind.startswith("lp") and "s" in kind:
        order, seed = map(int, kind[2:].split("s"))
        return get_lifted_product_code(order, seed)
    if kind == "surface":
        return codes.SurfaceCode(int(args))
    if kind == "toric":
        return codes.ToricCode(int(args), rotated=True)
    if kind == "classical":
        bits, checks, seed = map(int, args.split(":"))
        return codes.ClassicalCode.random(bits, checks, seed=seed)
    if kind == "sparse":
        bits, dimension, row_weight, seed = map(int, args.split(":"))
        rng = np.random.default_rng(seed)
        generator = np.zeros((dimension, bits), dtype=int)
        for row in generator:
            row[rng.choice(bits, size=row_weight, replace=False)] = 1
        return codes.ClassicalCode.from_generator(generator)
    if pathlib.Path(name).is_file():
        return parse_cpm(pathlib.Path(name).read_text(encoding="utf-8"))
    raise ValueError(f"Unknown code {name!r}")


def get_published_parameters(name: str) -> tuple[int, int, int] | None:
    """The published parameters [[n, k, d]] of an embedded instance, or None for other codes."""
    if name.startswith("pp") and name[2:] in INSTANCES:
        return INSTANCES[name[2:]][0]
    return None


def get_lifted_product_code(order: int, seed: int) -> codes.LPCode:
    """The lifted product of a 3 x 4 matrix of random monomials of a cyclic group."""
    group = abstract.CyclicGroup(order)
    generator = group.generators[0]
    exponents = np.random.default_rng(seed).integers(order, size=(3, 4))
    matrix = abstract.RingArray.build(
        [[generator ** int(power) for power in row] for row in exponents], group
    )
    return codes.LPCode(matrix)


def get_css_code(name: str) -> codes.CSSCode:
    """Build a named CSS code."""
    code = get_code(name)
    if not isinstance(code, codes.CSSCode):
        raise TypeError(f"{name!r} is not a CSS code")
    return code


####################################################################################################
# workloads


def run_workload(name: str) -> str:
    """Run a named workload (see the module docstring), and return a string of its result."""
    kind, *args = name.split("/")
    if kind not in WORKLOADS:
        raise ValueError(f"Unknown workload kind {kind!r}; choose from {sorted(WORKLOADS)}")
    return WORKLOADS[kind](*args)


def run_distance(code_name: str, sector: str) -> str:
    """The exact distance of a code, for one sector of a CSS code."""
    code = get_code(code_name)
    code.forget_distance()
    if isinstance(code, codes.CSSCode):
        if sector == "both":
            return str(code.get_distance_exact())
        if sector == "X":
            return str(code.get_distance_exact(Pauli.X))
        if sector == "Z":
            return str(code.get_distance_exact(Pauli.Z))
        raise ValueError(f"Unknown sector {sector!r}; choose 'X', 'Z', or 'both'")
    if sector != "-":
        raise ValueError(f"A classical code has no sector {sector!r}; use '-'")
    return str(code.get_distance_exact())


def run_hadamard_distance(code_name: str) -> str:
    """The exact distance of a CSS code after Hadamards on the second half of its qubits."""
    code = get_hadamard_code(code_name)
    code.forget_distance()
    return str(code.get_distance_exact())


@functools.cache
def get_hadamard_code(code_name: str) -> codes.QuditCode:
    """The code that Hadamards on the second half of the qubits of a CSS code make."""
    css_code = get_css_code(code_name)
    num_qubits = len(css_code)
    return codes.QuditCode(css_code.conjugated(range(num_qubits // 2, num_qubits)).matrix)


def run_count(code_name: str, weight: str) -> str:
    """Count the logical operators of a CSS code of a given weight, by the BZ method."""
    counts = CSSLogicalCounter(get_css_code(code_name)).count_logical_ops(int(weight))
    return f"{counts.weight}:{counts.count_x}:{counts.count_z}"


def run_cosets(
    code_name: str, weighing: str, error_weight: str, max_weight: str, num_syndromes: str
) -> str:
    """The minimum weights of the cosets of random errors on the logical state |0...0>."""
    if weighing not in ("css", "physical"):
        raise ValueError(f"Unknown weighing {weighing!r}; choose 'css' or 'physical'")
    stabilizers = get_zero_state_stabilizers(code_name)
    syndromes = get_random_syndromes(stabilizers, int(error_weight), int(num_syndromes))
    search = CosetLeaderSearch(stabilizers, css_weight=weighing == "css")
    weights = search._get_min_weights(syndromes, int(max_weight))
    return ",".join(
        f"{weight}:{count}" for weight, count in zip(*np.unique(weights, return_counts=True))
    )


def run_first_heavy_coset(
    code_name: str, weighing: str, error_weight: str, max_weight: str, num_syndromes: str
) -> str:
    """Find the first random error on |0...0> whose coset has no error within a maximum weight."""
    if weighing not in ("css", "physical"):
        raise ValueError(f"Unknown weighing {weighing!r}; choose 'css' or 'physical'")
    stabilizers = get_zero_state_stabilizers(code_name)
    syndromes = get_random_syndromes(stabilizers, int(error_weight), int(num_syndromes))
    search = CosetLeaderSearch(stabilizers, css_weight=weighing == "css")
    index, bound = search.find_first_not_within_max_weight(syndromes, int(max_weight))
    return f"{index}:{None if bound is None else bound.within_max_weight}"


@functools.cache
def get_zero_state_stabilizers(code_name: str) -> npt.NDArray[np.uint8]:
    """The stabilizers, in ``[X|Z]`` order, of the logical state |0...0> of a CSS code."""
    code = get_css_code(code_name)
    stabilizers = code.get_stabilizer_ops(canonicalized=True, symplectic=True)
    logicals = code.get_logical_ops(Pauli.Z, symplectic=True)
    return np.vstack([stabilizers, logicals]).view(np.ndarray).astype(np.uint8)


def get_random_syndromes(
    stabilizers: npt.NDArray[np.uint8], error_weight: int, num_syndromes: int, seed: int = 0
) -> npt.NDArray[np.uint64]:
    """The packed syndromes of random Pauli errors on ``error_weight`` qubits."""
    num_qubits = stabilizers.shape[1] // 2
    rng = np.random.default_rng(seed)
    errors = np.zeros((num_syndromes, 2 * num_qubits), dtype=np.uint8)
    for error in errors:
        support = rng.choice(num_qubits, size=error_weight, replace=False)
        pauli = rng.integers(1, 4, size=error_weight)  # 1, 2, 3 for X, Z, and Y
        error[support] = pauli & 1
        error[num_qubits + support] = pauli >> 1
    # the symplectic product of each error with each stabilizer
    bits = (
        errors[:, :num_qubits] @ stabilizers[:, num_qubits:].T
        + errors[:, num_qubits:] @ stabilizers[:, :num_qubits].T
    ) % 2
    num_words = (num_qubits + 63) // 64
    padded = np.zeros((num_syndromes, 64 * num_words), dtype=np.uint64)
    padded[:, :num_qubits] = bits
    weights = np.uint64(1) << np.arange(64, dtype=np.uint64)
    return (padded.reshape(num_syndromes, num_words, 64) * weights).sum(axis=2, dtype=np.uint64)


WORKLOADS: dict[str, Callable[..., str]] = {
    "distance": run_distance,
    "hadamard": run_hadamard_distance,
    "count": run_count,
    "ccount": run_count,
    "coset": run_cosets,
    "first": run_first_heavy_coset,
}
