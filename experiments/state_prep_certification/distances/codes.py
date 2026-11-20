# SPDX-License-Identifier: Apache-2.0

"""The codes of the distance experiment: families drawn by a fixed size rule, and named anchors.

    python codes.py --out DIR [--write-sample] [--check]

**Families and sizes.**  Each family has endpoints n_lo and n_hi from its published source,
recorded with their sources in ``code_data/endpoints.json``.  The four target lengths are
n_j = n_lo (n_hi / n_lo)^(j / 3) for j = 0, ..., 3, and each target goes to the family's
admissible parameters whose length is nearest in log n; ties go to the smaller n, then to the
family's tie-break.  Targets that land on the same parameters merge, and the merge is reported.
No timing of any kind enters this rule.

**Draws.**  For each size, draws i = 0, 1, ... are made until 4 are structurally valid or 100 are
made.  Draw i of parameters ``params`` of family ``f`` uses ``numpy.random.default_rng(seed)`` with
``seed = int.from_bytes(sha256(json.dumps(["ftsp-distances-1", f, params, i], sort_keys=True,
separators=(",", ":"))).digest()[:8], "big") >> 1``.  The only screens are structural: the
construction's own rejection, and ``k = 0``.  A draw is also screened out if it duplicates a kept
code, an anchor, a CSS code of ``trial.WARM_UP_CODES``, or the Hadamard variant of such a code, by
the SHA-256 of the reduced row echelon forms of the check matrices.  ``trial.WARM_UP_CODES`` are the
codes on which ``experiments/distance_tuning`` warms up; no run of this experiment warms up on them,
but the screen against them is part of the rule that produced the frozen ``sample.json``.  Every
draw and its outcome goes to ``draws.jsonl``, a size with fewer valid draws than codes goes to
``shortfalls`` of the sample, and an unexpected exception stops the script.

**Anchors.**  The named codes of ``ANCHORS``, from ``experiments/distance_tuning/workloads.py``,
qLDPC's constructors, and the vendored constructions below, with a reference distance where an
independent source gives one.

**Storage.**  Every code is stored as plain matrices, ``H_X`` and ``H_Z`` for a CSS code and
``H = [x | z]`` otherwise, in ``codes/<code_hash>.npz``, where ``code_hash`` is the SHA-256 of
the stored arrays.  ``--write-sample`` writes ``code_data/sample.json``, with the Git tree of the
qLDPC source that generated it (``qldpc_src_tree``), and ``--check`` regenerates the codes and
compares their ``COMPARED_FIELDS`` with it, among them the anchors' reference distances, which the
analysis reads from ``sample.json``.  Git ignores ``code_data/``, which the offline bundle
carries.

The constructions of the families and of some anchors are vendored, with changes, from an earlier
unpublished version of this experiment by its authors.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import pathlib
import subprocess
import sys
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

import galois
import numpy as np
import numpy.typing as npt
from sympy.abc import x, y

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
sys.path.insert(1, str(pathlib.Path(__file__).resolve().parents[2] / "distance_tuning"))

from qldpc import abstract, codes

# Scripts in distance_tuning/ import each other as top-level modules.  mypy knows these files only
# as experiments.distance_tuning.*, and would silently treat a bare "import trial" as untyped.
if TYPE_CHECKING:
    from experiments.distance_tuning import trial, workloads
else:
    import trial
    import workloads

HERE = pathlib.Path(__file__).resolve().parent
# the experiment's data, which Git ignores: the endpoints of each family and the frozen sample
CODE_DATA = HERE / "code_data"
ENDPOINTS = CODE_DATA / "endpoints.json"
SAMPLE = CODE_DATA / "sample.json"
# part of every draw's seed: changing it redraws every code and invalidates sample.json
SEED_TAG = "ftsp-distances-1"
# the fields of each code that ``--check`` and the node's ``prepare`` compare with ``sample.json``
COMPARED_FIELDS = ("code_hash", "n", "k", "css", "equal_distance_xz", "reference_d")
CODES_PER_SIZE = 4
MAX_DRAWS = 100
NUM_SIZES = 4
GF2 = galois.GF(2)


def main() -> None:
    """Generate the codes, and write or check the sample."""
    args = parse_args()
    sample = generate(args.out)
    if args.write_sample:
        sample["qldpc_src_tree"] = subprocess.run(
            ["git", "-C", str(HERE), "rev-parse", "HEAD:src"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        SAMPLE.write_text(json.dumps(sample, indent=1, sort_keys=True) + "\n")
        print(f"wrote {SAMPLE}")
    if args.check:
        problems = compare_with_sample(sample)
        for problem in problems:
            print(problem)
        print(f"{len(problems)} differences from {SAMPLE.name}")
        if problems:
            sys.exit(1)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=pathlib.Path, required=True, help="where to store codes")
    parser.add_argument("--write-sample", action="store_true", help="write sample.json")
    parser.add_argument("--check", action="store_true", help="compare with sample.json")
    return parser.parse_args()


class Rejected(Exception):
    """A draw that its construction rejects, with the reason as its message."""


####################################################################################################
# families and the size rule


@dataclasses.dataclass(frozen=True)
class Family:
    """A family of drawn codes.

    Attributes:
        draw: Draw a code of the given parameters with a random generator.
        admissible: The admissible parameters with lengths up to a bound, as (params, n, tie-break).
        css: Whether the family's codes are CSS.
    """

    draw: Callable[[dict[str, Any], np.random.Generator], Any]
    admissible: Callable[[int], Iterator[tuple[dict[str, Any], int, Any]]]
    css: bool


def _bb_sizes(bound: int) -> Iterator[tuple[dict[str, Any], int, Any]]:
    for m in range(3, bound):
        for l in range(m, 5 * m + 1):
            if 2 * l * m <= bound:
                yield {"l": l, "m": m}, 2 * l * m, l - m


def _gb_sizes(bound: int) -> Iterator[tuple[dict[str, Any], int, Any]]:
    for l in range(9, bound // 2 + 1):
        yield {"l": l, "w": 4}, 2 * l, 0


def _hgp_sizes(bound: int) -> Iterator[tuple[dict[str, Any], int, Any]]:
    for bits in range(4, bound, 4):
        if 25 * bits * bits // 16 <= bound:
            yield {"bits": bits}, 25 * bits * bits // 16, 0


def _lp_sizes(bound: int) -> Iterator[tuple[dict[str, Any], int, Any]]:
    for lift in range(3, bound // 34 + 1):
        yield {"lift": lift}, 34 * lift, 0


# The published pair-partition arrays of the family, by instance name: their block columns and
# published lift size, from ``workloads.INSTANCES``.
PP_ARRAYS = ("110", "232", "276")


def _pp_sizes(bound: int) -> Iterator[tuple[dict[str, Any], int, Any]]:
    for array in PP_ARRAYS:
        _, published_lift, columns = _cpm_shape(workloads.INSTANCES[array][1])
        for lift in range(published_lift, bound // columns + 1):
            if _is_prime(lift):
                yield {"array": array, "P": lift}, columns * lift, array


def _qt_sizes(bound: int) -> Iterator[tuple[dict[str, Any], int, Any]]:
    for order in range(6, 33, 2):
        for length in (4, 5, 6, 7):
            if order * length * length <= bound:
                yield {"g": order, "c": length}, order * length * length, -length


def _dense_sizes(minimum: int) -> Callable[[int], Iterator[tuple[dict[str, Any], int, Any]]]:
    def sizes(bound: int) -> Iterator[tuple[dict[str, Any], int, Any]]:
        for n in range(minimum, bound + 1):
            yield {"n": n, "k": 2}, n, 0

    return sizes


def size_ladder(
    family: Family, n_lo: int, n_hi: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The parameters of the family's sizes, and a record of each target and where it went."""
    # admissible lengths up to 4 n_hi include the nearest one to every target, which is <= n_hi
    candidates = list(family.admissible(4 * n_hi))
    chosen: list[dict[str, Any]] = []
    record = []
    for j in range(NUM_SIZES):
        target = n_lo * (n_hi / n_lo) ** (j / (NUM_SIZES - 1))
        # nearest in log n, with distances rounded so that equal ones tie exactly
        params, n, _ = min(
            candidates,
            key=lambda item: (round(abs(math.log(item[1] / target)), 12), item[1], item[2]),
        )
        merged = params in chosen
        if not merged:
            chosen.append(params)
        record.append({"j": j, "target": target, "params": params, "n": n, "merged": merged})
    return chosen, record


####################################################################################################
# draws


def draw_bb(params: dict[str, Any], rng: np.random.Generator) -> codes.CSSCode:
    """Bivariate bicycle code with A = x^a + y^b + y^c and B = y^d + x^e + x^f (Bravyi et al.)."""
    l, m = params["l"], params["m"]
    a, d = int(rng.integers(l)), int(rng.integers(m))
    b, c = (int(v) for v in rng.choice(m, 2, replace=False))
    e, f = (int(v) for v in rng.choice(l, 2, replace=False))
    return codes.BBCode({x: l, y: m}, x**a + y**b + y**c, y**d + x**e + x**f)


def draw_gb(params: dict[str, Any], rng: np.random.Generator) -> codes.CSSCode:
    """Generalized bicycle code: two random polynomials of weight w in F_2[x]/(x^l - 1)."""
    l, weight = params["l"], params["w"]
    poly_a = _random_monomials(rng, [l], [x], weight)
    poly_b = _random_monomials(rng, [l], [x], weight)
    return codes.MBCode({x: l}, poly_a, poly_b)


def draw_hgp(params: dict[str, Any], rng: np.random.Generator) -> codes.CSSCode:
    """Hypergraph product of two independent random (3, 4)-regular LDPC codes."""
    bits = params["bits"]
    return codes.HGPCode(_regular_ldpc(rng, bits, 3, 4), _regular_ldpc(rng, bits, 3, 4))


def draw_lp(params: dict[str, Any], rng: np.random.Generator) -> codes.CSSCode:
    """Lifted product of a random 3 x 5 base matrix of monomials of Z_L with itself."""
    lift = params["lift"]
    group = abstract.CyclicGroup(lift)
    shift = group.generators[0]
    exponents = rng.integers(lift, size=(3, 5))
    matrix = abstract.RingArray.build([[shift ** int(e) for e in row] for row in exponents], group)
    return codes.LPCode(matrix)


def draw_pp(params: dict[str, Any], rng: np.random.Generator) -> codes.CSSCode:
    """A random member over F_P of a published pair-partition array of ``workloads.INSTANCES``."""
    matchings = published_matchings(workloads.INSTANCES[params["array"]][1])
    return codes.PairPartitionCode.from_pair_partition_array(
        matchings, params["P"], fix_gauge=True, seed=int(rng.integers(2**31))
    )


def published_matchings(text: str) -> list[list[list[list[int]]]]:
    """The pair-partition array of a published ``.cpm`` instance.

    Block columns u and v are paired in entry (i, j) when ``d_ju - e_iu = d_jv - e_iv`` (mod P).
    """
    _magic, _name, *numbers = text.split()
    lift, cols, rows_x, rows_z, *exponents = map(int, numbers)
    values = np.array(exponents)
    ex = values[: rows_x * cols].reshape(rows_x, cols)
    ez = values[rows_x * cols :].reshape(rows_z, cols)
    matchings = []
    for i in range(rows_x):
        row = []
        for j in range(rows_z):
            groups: dict[int, list[int]] = {}
            for col in range(cols):
                groups.setdefault(int(ez[j, col] - ex[i, col]) % lift, []).append(col)
            pairs = []
            for members in groups.values():
                if len(members) % 2:
                    raise ValueError("not a pair partition")
                pairs += [members[t : t + 2] for t in range(0, len(members), 2)]
            row.append(pairs)
        matchings.append(row)
    return matchings


def local_code(length: int) -> codes.ClassicalCode:
    """The local code of length c of the ``qt`` family: Hamming [7, 4] or a modified one.

    Lengths 6 and 5 shorten the [7, 4] Hamming code, and length 4 further punctures the length-5
    code, as the "Mittal" codes of ``experiments/quantum_tanner/saved_codes.py`` do.
    """
    hamming = codes.HammingCode(3)
    if length == 7:
        return hamming
    if length == 6:
        return hamming.shortened([3])
    if length == 5:
        return hamming.shortened([2, 3])
    if length == 4:
        return hamming.shortened([2, 3]).punctured([4])
    raise ValueError(f"No local code of length {length}")


def draw_qt(params: dict[str, Any], rng: np.random.Generator) -> codes.CSSCode:
    """Quantum Tanner code on a cyclic or dihedral group of order g, with a local code of length c.

    The group is ``CyclicGroup(g)`` or ``DihedralGroup(g // 2)``, chosen uniformly, and a code
    whose length is not g c^2 is rejected.
    """
    order, length = params["g"], params["c"]
    if rng.integers(2):
        group: abstract.Group = abstract.DihedralGroup(order // 2)
    else:
        group = abstract.CyclicGroup(order)
    # the row-reduced checks that qldpc built when sample.json was fixed
    code = workloads.get_row_reduced_qt_code(
        codes.QTCode.random(group, local_code(length), seed=int(rng.integers(2**31)))
    )
    if len(code) != order * length * length:
        raise Rejected(f"{len(code)} qubits, not g c^2 = {order * length * length}")
    return code


def draw_css_walk(params: dict[str, Any], rng: np.random.Generator) -> codes.CSSCode:
    """A dense CSS code: a walk of 30 n random CNOT gates from TrivCSS(k, s_X, s_Z).

    s_X = (n - k) // 2.  CNOT_{a,b} adds column a of H_X to column b, and column b of H_Z to
    column a, for a uniformly random ordered pair of distinct qubits.
    """
    n, k = params["n"], params["k"]
    s_x = (n - k) // 2
    s_z = n - k - s_x
    matrix_x = np.zeros((s_x, n), dtype=np.uint8)
    matrix_z = np.zeros((s_z, n), dtype=np.uint8)
    matrix_x[np.arange(s_x), k + np.arange(s_x)] = 1
    matrix_z[np.arange(s_z), k + s_x + np.arange(s_z)] = 1
    for _ in range(30 * n):
        a, b = (int(v) for v in rng.choice(n, 2, replace=False))
        matrix_x[:, b] ^= matrix_x[:, a]
        matrix_z[:, a] ^= matrix_z[:, b]
    return codes.CSSCode(matrix_x, matrix_z)


def draw_rstab(params: dict[str, Any], rng: np.random.Generator) -> codes.QuditCode:
    """A dense stabilizer code: commuting random Pauli strings, k = 2.

    Each new row is a random element of the symplectic complement of the rows so far, kept if it
    is independent of them.
    """
    n, k = params["n"], params["k"]
    rows: list[npt.NDArray[np.int_]] = []
    while len(rows) < n - k:
        if rows:
            matrix = GF2(np.array(rows))
            complement = GF2(np.hstack([matrix[:, n:], matrix[:, :n]])).null_space()
        else:
            complement = GF2(np.eye(2 * n, dtype=int))
        candidate = np.array(GF2(rng.integers(2, size=len(complement))) @ complement)
        if np.linalg.matrix_rank(GF2(np.array([*rows, candidate]))) == len(rows) + 1:
            rows.append(candidate)
    return codes.QuditCode(GF2(np.array(rows)))


def _random_monomials(
    rng: np.random.Generator, orders: list[int], symbols: list[Any], count: int
) -> Any:
    """A sum of ``count`` distinct random monomials of an abelian group with the given orders."""
    terms: set[tuple[int, ...]] = set()
    while len(terms) < count:
        terms.add(tuple(int(rng.integers(order)) for order in orders))
    return sum(
        math.prod(symbol**power for symbol, power in zip(symbols, exponents))
        for exponents in sorted(terms)
    )


def _regular_ldpc(
    rng: np.random.Generator, bits: int, column_weight: int, row_weight: int
) -> npt.NDArray[np.int_]:
    """A random (column_weight, row_weight)-regular parity check matrix, by the configuration model.

    Draws with repeated edges are redrawn, up to 1000 times.
    """
    checks = bits * column_weight // row_weight
    if checks * row_weight != bits * column_weight:
        raise Rejected("degrees do not match")
    for _ in range(1000):
        sockets = np.repeat(np.arange(checks), row_weight)
        rng.shuffle(sockets)
        edges = sockets.reshape(bits, column_weight)
        if any(len(set(row)) < column_weight for row in edges):
            continue
        matrix = np.zeros((checks, bits), dtype=int)
        for bit, rows in enumerate(edges):
            matrix[rows, bit] = 1
        return matrix
    raise Rejected("no simple regular graph found")


FAMILIES = {
    "bb": Family(draw_bb, _bb_sizes, css=True),
    "gb": Family(draw_gb, _gb_sizes, css=True),
    "hgp": Family(draw_hgp, _hgp_sizes, css=True),
    "lp": Family(draw_lp, _lp_sizes, css=True),
    "pp": Family(draw_pp, _pp_sizes, css=True),
    "qt": Family(draw_qt, _qt_sizes, css=True),
    "css_walk": Family(draw_css_walk, _dense_sizes(4), css=True),
    "rstab": Family(draw_rstab, _dense_sizes(3), css=False),
}


####################################################################################################
# anchors


def quadratic_residue_css(prime: int) -> codes.CSSCode:
    """CSS(Q, Q) for the binary quadratic-residue code Q of prime length p = -1 mod 8."""
    residues = {(i * i) % prime for i in range(1, prime)}
    nonresidues = set(range(1, prime)) - residues
    for support in (residues, nonresidues, residues | {0}, nonresidues | {0}):
        word = np.zeros(prime, dtype=int)
        word[list(support)] = 1
        generator = GF2(np.array([np.roll(word, shift) for shift in range(prime)]))
        basis = generator.row_reduce()
        basis = basis[np.any(basis, axis=1)]
        if len(basis) != (prime + 1) // 2:
            continue
        checks = basis.null_space()
        if not np.any(checks @ checks.T):
            return codes.CSSCode(checks, checks)
    raise ValueError(f"No dual-containing quadratic-residue code of length {prime}")


def color_666(distance: int) -> codes.CSSCode:
    """The triangular 6.6.6 color code of odd distance d: [[(3d^2 + 1) / 4, 1, d]].

    Points (a, b) of the triangular lattice with 0 <= b <= a <= 3(d - 1)/2 are 3-colored by
    (a + b) mod 3; points of color 1 are the centers of the faces, and the others are the qubits.
    Each face is the set of its lattice neighbors in the triangle.
    """
    size = 3 * (distance - 1) // 2
    points = [(a, b) for a in range(size + 1) for b in range(a + 1)]
    qubits = {point: index for index, point in enumerate(p for p in points if sum(p) % 3 != 1)}
    steps = [(1, 0), (0, 1), (-1, -1), (-1, 0), (0, -1), (1, 1)]
    faces = []
    for a, b in points:
        if (a + b) % 3 != 1:
            continue
        face = [qubits[(a + da, b + db)] for da, db in steps if (a + da, b + db) in qubits]
        row = np.zeros(len(qubits), dtype=int)
        row[face] = 1
        faces.append(row)
    checks = np.array(faces)
    return codes.CSSCode(checks, checks)


# Anchors: name -> (constructor, reference distance or None, source).  The quantum distance of
# CSS(Q, Q) for a quadratic-residue code Q of length p = -1 mod 8 is the minimum odd weight of Q.
ANCHORS: dict[str, tuple[Callable[[], Any], int | None, str]] = {
    "pp110": (lambda: workloads.get_code("pp110"), 12, "workloads.py; Lee et al. 2026"),
    "pp170": (lambda: workloads.get_code("pp170"), 20, "workloads.py; Lee et al. 2026"),
    "pp232": (lambda: workloads.get_code("pp232"), 12, "workloads.py; Lee et al. 2026"),
    "pp276": (lambda: workloads.get_code("pp276"), 14, "workloads.py; Okada and Kasai 2026"),
    "pp372": (lambda: workloads.get_code("pp372"), 18, "workloads.py; Lee et al. 2026"),
    "bb72": (
        lambda: codes.BBCode({x: 6, y: 6}, x**3 + y + y**2, y**3 + x + x**2),
        6,
        "Bravyi et al. 2024, Table 3",
    ),
    "bb90": (
        lambda: codes.BBCode({x: 15, y: 3}, x**9 + y + y**2, 1 + x**2 + x**7),
        10,
        "Bravyi et al. 2024, Table 3",
    ),
    "bb108": (
        lambda: codes.BBCode({x: 9, y: 6}, x**3 + y + y**2, y**3 + x + x**2),
        10,
        "Bravyi et al. 2024, Table 3",
    ),
    "gross": (lambda: workloads.get_code("gross"), 12, "workloads.py; Bravyi et al. 2024"),
    "bb288": (lambda: workloads.get_code("bb288"), 18, "workloads.py; Bravyi et al. 2024"),
    "gb46": (
        lambda: codes.MBCode({x: 23}, 1 + x**5 + x**8 + x**12, 1 + x + x**5 + x**7),
        9,
        "Panteleev and Kalachev 2021",
    ),
    "gb48": (
        lambda: codes.MBCode({x: 24}, 1 + x**2 + x**8 + x**15, 1 + x**2 + x**12 + x**17),
        8,
        "Panteleev and Kalachev 2021",
    ),
    "gb126": (
        lambda: codes.MBCode(
            {x: 63}, 1 + x + x**14 + x**16 + x**22, 1 + x**3 + x**13 + x**20 + x**42
        ),
        8,
        "Panteleev and Kalachev 2021",
    ),
    "lp13s5": (
        lambda: workloads.get_code("lp13s5"),
        None,
        "workloads.py; no independent reference",
    ),
    "lp13s8": (
        lambda: workloads.get_code("lp13s8"),
        None,
        "workloads.py; no independent reference",
    ),
    "qt7": (lambda: workloads.get_code("qt7"), None, "workloads.py; no independent reference"),
    "surface9": (lambda: codes.SurfaceCode(9), 9, "rotated surface code"),
    "surface13": (lambda: codes.SurfaceCode(13), 13, "rotated surface code"),
    "surface17": (lambda: codes.SurfaceCode(17), 17, "rotated surface code"),
    "toric10": (lambda: codes.ToricCode(10, rotated=True), 10, "rotated toric code"),
    "color9": (lambda: color_666(9), 9, "6.6.6 color code"),
    "color13": (lambda: color_666(13), 13, "6.6.6 color code"),
    "golay": (lambda: codes.QuantumGolayCode(), 7, "quantum Golay code"),
    "qr47": (
        lambda: quadratic_residue_css(47),
        11,
        "CSS(QR(47), QR(47)); QR(47) is [47, 24, 11] (He, Reed, and Truong 2001), of odd weight",
    ),
    "qr71": (
        lambda: quadratic_residue_css(71),
        11,
        (
            "CSS(QR(71), QR(71)); the extended code XQ71 is [72, 36, 12] (Rains and Sloane 1998, "
            "Sec. 12.1), and transitivity of its group gives QR(71) an odd word of weight 11"
        ),
    ),
    "qr79": (
        lambda: quadratic_residue_css(79),
        15,
        (
            "CSS(QR(79), QR(79)); the extended code XQ79 is [80, 40, 16] (Rains and Sloane 1998, "
            "Sec. 12.1), and transitivity of its group gives QR(79) an odd word of weight 15"
        ),
    ),
    "h_pp110": (lambda: workloads.get_hadamard_code("pp110"), 12, "Hadamard variant of pp110"),
    "h_pp232": (lambda: workloads.get_hadamard_code("pp232"), 12, "Hadamard variant of pp232"),
    "h_gross": (lambda: workloads.get_hadamard_code("gross"), 12, "Hadamard variant of gross"),
    "h_qt7": (lambda: workloads.get_hadamard_code("qt7"), None, "Hadamard variant of qt7"),
    "five_five": (
        lambda: codes.QuditCode.concatenate(codes.FiveQubitCode(), codes.FiveQubitCode()),
        9,
        "five-qubit code concatenated with itself (Knill and Laflamme 1996)",
    ),
}


####################################################################################################
# generation, storage, and hashes


def generate(out: pathlib.Path) -> dict[str, Any]:
    """Generate every anchor and drawn code, store them under ``out``, and return the sample."""
    (out / "codes").mkdir(parents=True, exist_ok=True)
    endpoints = json.loads(ENDPOINTS.read_text())["families"]
    seen: dict[str, str] = {}
    # screen out distance_tuning's warm-up codes and their Hadamard variants (see the docstring)
    for name in trial.WARM_UP_CODES:
        warm_up = workloads.get_code(name)
        if isinstance(warm_up, codes.CSSCode):
            seen[duplicate_key(warm_up)] = f"warm-up {name}"
            hadamard = workloads.get_hadamard_code(name)
            seen[duplicate_key(hadamard)] = f"warm-up {name} (Hadamard)"

    # anchors
    anchors = []
    for name, (build, reference, source) in ANCHORS.items():
        anchor = build()
        stored = store_code(out, anchor)
        key = duplicate_key(anchor)
        if key in seen:
            raise SystemExit(f"anchor {name} duplicates {seen[key]}")
        seen[key] = f"anchor {name}"
        anchors.append({"name": name, "reference_d": reference, "source": source, **stored})

    # families
    drawn = []
    ladders = {}
    shortfalls: list[dict[str, Any]] = []
    with open(out / "draws.jsonl", "w") as draws:
        for family_name, family in FAMILIES.items():
            n_lo, n_hi = endpoints[family_name]["n_lo"], endpoints[family_name]["n_hi"]
            sizes, ladders[family_name] = size_ladder(family, n_lo, n_hi)
            for params in sizes:
                kept = 0
                for index in range(MAX_DRAWS):
                    seed = draw_seed(family_name, params, index)
                    record: dict[str, Any] = {
                        "family": family_name,
                        "params": params,
                        "index": index,
                        "seed": seed,
                    }
                    try:
                        code = family.draw(params, np.random.default_rng(seed))
                    except Rejected as error:
                        record["outcome"] = f"rejected: {error}"
                    else:
                        key = duplicate_key(code)
                        if code.dimension == 0:
                            record["outcome"] = "rejected: k = 0"
                        elif key in seen:
                            record["outcome"] = f"rejected: duplicate of {seen[key]}"
                        else:
                            stored = store_code(out, code)
                            name = f"{family_name}.{_params_label(params)}.i{index}"
                            seen[key] = name
                            record["outcome"] = "kept"
                            record["code_hash"] = stored["code_hash"]
                            drawn.append(
                                {
                                    "name": name,
                                    "family": family_name,
                                    "params": params,
                                    "index": index,
                                    "seed": seed,
                                    **stored,
                                }
                            )
                            kept += 1
                    draws.write(json.dumps(record, sort_keys=True) + "\n")
                    if kept == CODES_PER_SIZE:
                        break
                if kept < CODES_PER_SIZE:
                    print(f"shortfall: {family_name} {params}: {kept} codes in {MAX_DRAWS} draws")
                    shortfalls.append({"family": family_name, "params": params, "kept": kept})
    return {
        "seed_tag": SEED_TAG,
        "endpoints": endpoints,
        "ladders": ladders,
        "shortfalls": shortfalls,
        "anchors": anchors,
        "codes": drawn,
    }


def store_code(out: pathlib.Path, code: Any) -> dict[str, Any]:
    """Store a code's plain matrices, and return its record."""
    if isinstance(code, codes.CSSCode):
        arrays = {
            "matrix_x": np.asarray(code.matrix_x, dtype=np.uint8),
            "matrix_z": np.asarray(code.matrix_z, dtype=np.uint8),
        }
        plain_css = codes.CSSCode(
            arrays["matrix_x"].astype(np.int_),
            arrays["matrix_z"].astype(np.int_),
            is_subsystem_code=False,
        )
        # whether the plain code searches only its X sector, which AT1 checks on two anchors
        equal_sectors = bool(plain_css._equal_distance_xz)
        plain: codes.QuditCode = plain_css
    else:
        arrays = {"matrix": np.asarray(code.matrix, dtype=np.uint8)}
        plain = codes.QuditCode(arrays["matrix"].astype(np.int_))
        equal_sectors = False
    code_hash = hash_arrays(arrays)
    contents: dict[str, Any] = arrays
    np.savez(out / "codes" / f"{code_hash}.npz", **contents)
    return {
        "code_hash": code_hash,
        "n": len(plain),
        "k": int(plain.dimension),
        "css": isinstance(code, codes.CSSCode),
        "equal_distance_xz": equal_sectors,
        "constructor": type(code).__name__,
    }


def load_code(out: pathlib.Path, code_hash: str) -> dict[str, npt.NDArray[Any]]:
    """The stored arrays of a code, after checking their hash."""
    with np.load(out / "codes" / f"{code_hash}.npz") as data:
        arrays = {key: data[key] for key in data.files}
    if hash_arrays(arrays) != code_hash:
        raise ValueError(f"The stored code {code_hash} does not match its hash")
    return arrays


def hash_arrays(arrays: dict[str, npt.NDArray[Any]]) -> str:
    """The SHA-256 of named arrays: names, dtypes, shapes, and bytes, in name order."""
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(f"{name}:{array.dtype.str}:{array.shape};".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def duplicate_key(code: Any) -> str:
    """The SHA-256 of the reduced row echelon forms of a code's check matrices."""
    if isinstance(code, codes.CSSCode):
        parts = [code.matrix_x, code.matrix_z]
    else:
        parts = [code.matrix]
    digest = hashlib.sha256()
    for matrix in parts:
        echelon = GF2(np.asarray(matrix, dtype=np.uint8)).row_reduce()
        rows = np.asarray(echelon[np.any(echelon, axis=1)], dtype=np.uint8)
        digest.update(f"{rows.shape};".encode() + rows.tobytes())
    return digest.hexdigest()


def draw_seed(family: str, params: dict[str, Any], index: int) -> int:
    """The seed of draw ``index`` of a family's parameters."""
    text = json.dumps([SEED_TAG, family, params, index], sort_keys=True, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") >> 1


def compare_with_sample(sample: dict[str, Any]) -> list[str]:
    """The differences between a regenerated sample and ``sample.json``, in ``COMPARED_FIELDS``."""
    frozen = json.loads(SAMPLE.read_text())
    problems = []
    for kind in ("anchors", "codes"):
        new = {entry["name"]: entry for entry in sample[kind]}
        old = {entry["name"]: entry for entry in frozen[kind]}
        for name in sorted(set(new) | set(old)):
            if name not in new or name not in old:
                problems.append(
                    f"{kind}: {name} is only in the {'new' if name in new else 'frozen'} sample"
                )
                continue
            for field in COMPARED_FIELDS:
                if new[name].get(field) != old[name].get(field):
                    problems.append(f"{kind}: {name} has another {field}")
    return problems


def _params_label(params: dict[str, Any]) -> str:
    """A short label of parameters, as in ``l6m6``."""
    return "".join(f"{key}{value}" for key, value in params.items())


def _cpm_shape(text: str) -> tuple[str, int, int]:
    """The name, lift size, and block columns of a ``.cpm`` instance."""
    _magic, name, lift, columns, *_ = text.split()
    return name, int(lift), int(columns)


def _is_prime(number: int) -> bool:
    """Whether a number is prime."""
    return number >= 2 and all(number % d for d in range(2, math.isqrt(number) + 1))


if __name__ == "__main__":
    main()
