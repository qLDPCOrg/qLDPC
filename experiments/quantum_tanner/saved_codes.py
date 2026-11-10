# SPDX-License-Identifier: Apache-2.0

"""Specifications, file names, and file contents of saved quantum Tanner codes.

The scripts in this directory share the helpers in this module.  A quantum Tanner code of the search
is specified by a CodeSpec, which identifies a group, a pair of classical local codes, and a choice
of Cayley complex, together with a random seed.  Each saved code is a text file, written by
qldpc.codes.QTCode.save, whose name identifies its specification and seed, and whose header records
the parameters computed by run_search.py.

Identifiers of groups (see get_group) are:
- "SmallGroup-{order}-{index}": a group in the GAP SmallGroups library, and
- "Dihedral-{m}": the dihedral group of order 2m, the symmetry group of a regular m-gon.

Identifiers of local codes (see get_local_code) are:
- "Hamming-{r}": the Hamming code with r parity checks and block length 2^r - 1,
- "CordaroWagner-{n}": a binary code of dimension 2 and block length n, with the largest minimum
  distance possible for its block length, as constructed by GUAVA,
- "Mittal-{n}": a modified [7, 4] Hamming code with block length n in {4, 5, 6} (see
  get_mittal_code),
- "Random-{n}-{m}-{index}": a random code with block length n and m parity checks, drawn by
  qldpc.codes.ClassicalCode.random with a seed determined by n, m, and index, and
- "Dual{identifier}": the dual of the code with the given identifier.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import functools
import hashlib
import math
import pathlib
import re

from qldpc import abstract, codes

RESULTS_DIR = pathlib.Path(__file__).parent / "results"  # default directory of all results
CODES_NAME = "codes"  # subdirectory of the results directory that holds saved codes
EXACT_DISTANCES_NAME = "exact_distances.csv"  # file of exact distances in the results directory

# columns of the file of exact distances; see compute_distances.py
EXACT_DISTANCE_FIELDS = ("code_key", "code_file", "status", "distance", "time_limit", "seconds")

# patterns of identifiers of groups and local codes, and of names of saved code files
_GROUP_ID = r"SmallGroup-\d+-\d+|Dihedral-\d+"
_LOCAL_CODE_ID = r"[A-Za-z]+(?:-\d+)+"
_CODE_FILE_NAME = re.compile(
    rf"qtcode_({_GROUP_ID})_({_LOCAL_CODE_ID})(?:\+({_LOCAL_CODE_ID}))?(_bipartite)?_s(\d+)\.txt"
)

####################################################################################################
# code specifications


@dataclasses.dataclass(frozen=True)
class CodeSpec:
    """The data that, together with a random seed, specifies a quantum Tanner code of the search.

    Attributes:
        group_id: An identifier of the group whose subsets define the Cayley complex; see
            get_group.
        code_a_id: An identifier of the local code on the first subset; see get_local_code.
        code_b_id: An identifier of the local code on the second subset; see get_local_code.
        bipartite: Whether the code lives on the bipartite Cayley complex of arXiv:2202.13641,
            rather than on the quadripartite complex of arXiv:2206.07571.
    """

    group_id: str
    code_a_id: str
    code_b_id: str
    bipartite: bool = False

    @property
    def has_dual_codes(self) -> bool:
        """Is the second local code the dual of the first?"""
        return self.code_b_id == get_dual_id(self.code_a_id)

    def get_seed(self, sample: int, attempt: int = 0) -> int:
        """Get the random seed of a sample with this specification.

        The seed is a deterministic hash of the group, the bytes of the parity check matrices of the
        local codes, the choice of complex, and the sample index.  The second local code and the
        choice of complex enter only if they differ from the defaults (dual local codes and the
        quadripartite complex), so that specifications with default choices keep the seeds of
        earlier versions of this module.  The bytes of a parity check matrix depend on its rows and
        data type, so a change to how a local code is constructed changes the seeds of its codes.

        A bipartite complex exists only if the subsets drawn for a seed satisfy the Total No
        Conjugacy condition (see qldpc.objects.CayleyComplex), so run_search.py tries the seeds of
        successive attempts until one does.
        """
        inputs: list[object] = [*get_group_hash_inputs(self.group_id)]
        inputs.append(get_local_code(self.code_a_id).matrix.tobytes())
        if not self.has_dual_codes:
            inputs.append(get_local_code(self.code_b_id).matrix.tobytes())
        if self.bipartite:
            inputs.append("bipartite")
        seed = get_deterministic_hash(*inputs, sample)
        return get_deterministic_hash(seed, attempt) if attempt else seed

    def build(self, seed: int) -> codes.QTCode:
        """Construct the quantum Tanner code with this specification and a given seed.

        Raises a ValueError if the complex is bipartite and the subsets drawn for this seed violate
        the Total No Conjugacy condition.
        """
        return codes.QTCode.random(
            get_group(self.group_id),
            get_local_code(self.code_a_id),
            get_local_code(self.code_b_id),
            bipartite=self.bipartite,
            seed=seed,
        )

    def get_path(self, codes_dir: pathlib.Path, seed: int) -> pathlib.Path:
        """Get the path of the file that saves the code with this specification and a given seed.

        The name of the file has the form "qtcode_{group}_{codes}{complex}_s{seed}.txt", where
        {codes} is the identifier of the first local code if the second is its dual, and
        "{code_a}+{code_b}" otherwise, and {complex} is "_bipartite" for a bipartite complex and
        empty otherwise.  See parse_code_path.
        """
        codes_id = self.code_a_id if self.has_dual_codes else f"{self.code_a_id}+{self.code_b_id}"
        complex_id = "_bipartite" if self.bipartite else ""
        return codes_dir / f"qtcode_{self.group_id}_{codes_id}{complex_id}_s{seed}.txt"


def parse_code_path(path: pathlib.Path) -> tuple[CodeSpec, int]:
    """Get the specification and seed of a saved code from its path; see CodeSpec.get_path."""
    match = _CODE_FILE_NAME.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unrecognized quantum Tanner code file name: {path}")
    group_id, code_a_id, code_b_id, complex_id, seed = match.groups()
    spec = CodeSpec(
        group_id, code_a_id, code_b_id or get_dual_id(code_a_id), bipartite=bool(complex_id)
    )
    return spec, int(seed)


####################################################################################################
# groups


def get_group_ids(family: str, max_order: int) -> list[str]:
    """Get identifiers of the groups of a family ("small" or "dihedral") up to a given order.

    The family "small" consists of all groups in the GAP SmallGroups library, and the family
    "dihedral" consists of the dihedral groups of order 2m for m >= 3.
    """
    if family == "small":
        return [
            f"SmallGroup-{order}-{index}"
            for order in range(1, max_order + 1)
            for index in range(1, abstract.SmallGroup.count_groups(order) + 1)
        ]
    if family == "dihedral":
        return [f"Dihedral-{sides}" for sides in range(3, max_order // 2 + 1)]
    raise ValueError(f"Group family not recognized: {family}")


@functools.cache
def get_group(group_id: str) -> abstract.Group:
    """Construct a group from its identifier; see the module docstring."""
    family, *parameters = group_id.split("-")
    if family == "SmallGroup":
        order, index = map(int, parameters)
        return abstract.SmallGroup(order, index)
    if family == "Dihedral":
        return abstract.DihedralGroup(int(parameters[0]))
    raise ValueError(f"Group not recognized: {group_id}")


def get_group_order(group_id: str) -> int:
    """Get the order of a group from its identifier, without constructing the group."""
    family, *parameters = group_id.split("-")
    return int(parameters[0]) * (2 if family == "Dihedral" else 1)


def get_group_hash_inputs(group_id: str) -> tuple[int | str, ...]:
    """Get the inputs that identify a group in the hash of a seed; see CodeSpec.get_seed."""
    family, *parameters = group_id.split("-")
    if family == "SmallGroup":
        return tuple(map(int, parameters))
    return (family, *map(int, parameters))


####################################################################################################
# local codes


@functools.cache
def get_local_code(code_id: str) -> codes.ClassicalCode:
    """Construct a classical local code from its identifier; see the module docstring.

    The local codes that the search uses by default have parameters [n, k, d]:
    - Hamming-2: [3, 1, 3], and Hamming-3: [7, 4, 3],
    - CordaroWagner-3 to CordaroWagner-6: [3, 2, 2], [4, 2, 2], [5, 2, 3], and [6, 2, 4], and
    - Mittal-4 to Mittal-6: [4, 2, 2], [5, 2, 3], and [6, 3, 3].
    """
    if code_id.startswith("Dual"):
        return ~get_local_code(code_id.removeprefix("Dual"))
    family, *parameters = code_id.split("-")
    values = [int(parameter) for parameter in parameters]
    if family == "Hamming" and len(values) == 1:
        return codes.HammingCode(values[0])
    if family == "CordaroWagner" and len(values) == 1:
        return codes.ClassicalCode.from_name(f"CordaroWagnerCode({values[0]})")
    if family == "Mittal" and len(values) == 1:
        return get_mittal_code(values[0])
    if family == "Random" and len(values) == 3:
        bits, checks, _ = values
        seed = get_deterministic_hash("Random", *values)
        return codes.ClassicalCode.random(bits, checks, seed=seed)
    raise ValueError(f"Local code not recognized: {code_id}")


def get_dual_id(code_id: str) -> str:
    """Get the identifier of the dual of a local code."""
    return code_id.removeprefix("Dual") if code_id.startswith("Dual") else f"Dual{code_id}"


def get_mittal_code(length: int) -> codes.ClassicalCode:
    """Get a modified [7, 4] Hamming code with a given block length (4, 5, or 6).

    Lengths 6 and 5 are obtained by shortening the Hamming code, and length 4 is obtained by further
    puncturing the length-5 code.
    """
    full_code = codes.HammingCode(3)
    if length == 4:
        return full_code.shortened([2, 3]).punctured([4])
    if length == 5:
        return full_code.shortened([2, 3])
    if length == 6:
        return full_code.shortened([3])
    raise ValueError(f"Unrecognized length for a Mittal code: {length}")


####################################################################################################
# saved code files


def add_results_dir_argument(parser: argparse.ArgumentParser) -> None:
    """Add the --results-dir option, which is shared by all quantum Tanner code scripts."""
    parser.add_argument(
        "--results-dir",
        type=pathlib.Path,
        default=RESULTS_DIR,
        help="directory of saved codes and their summary"
        " (default: the 'results' directory beside this script)",
    )


def get_saved_code_paths(codes_dir: pathlib.Path) -> list[pathlib.Path]:
    """Get the sorted paths of all saved quantum Tanner codes in a directory.

    Raises a FileNotFoundError if there are none.
    """
    paths = sorted(codes_dir.glob("qtcode_*.txt"))
    if not paths:
        raise FileNotFoundError(f"No saved quantum Tanner codes found in {codes_dir}")
    return paths


def get_code_header(
    num_trials: int, code_params: tuple[int, int, int | float], weight: int
) -> list[str]:
    """Get the header lines of a saved quantum Tanner code file; see read_code_header."""
    return [
        f"distance trials: {num_trials}",
        f"code parameters: {code_params}",
        f"stabilizer weight: {weight}",
    ]


def read_code_header(path: pathlib.Path) -> tuple[int, tuple[int, int, int | float], int]:
    """Read the header of a saved quantum Tanner code file; see get_code_header.

    Returns the number of distance trials, the code parameters (n, k, d), and the maximum weight of
    a parity check of the code.  The distance d is an upper bound, or nan if the code has no logical
    qubits.
    """
    with open(path) as file:
        num_trials = int(file.readline().split(":")[-1])
        param_text = file.readline().split(":")[-1]
        weight = int(file.readline().split(":")[-1])

    # parse code parameters written as "(n, k, d)", where d is nan if k = 0
    num_qudits, dimension, distance = (
        float(value) for value in param_text.strip(" ()\n").split(",")
    )
    code_params = (
        int(num_qudits),
        int(dimension),
        distance if math.isnan(distance) else int(distance),
    )
    return num_trials, code_params, weight


def get_code_key(path: pathlib.Path) -> str:
    """Get a key that identifies the quantum Tanner code in a saved file.

    Two files have the same key if and only if they save the same subsets, local codes, field, and
    choice of complex, and thereby the same code.  The key ignores the header of the file and the
    order in which the members of each subset are listed, which can differ between Python sessions.
    """
    # split the generating data of the code into blocks, each starting with a comment line
    blocks: list[list[str]] = []
    with open(path) as file:
        for line in file.read().splitlines():
            if line == "# subset_a:" or (blocks and line.startswith("#")):
                blocks.append([line])
            elif blocks:
                blocks[-1].append(line)

    # sort the members of each subset, and hash the result
    for block in blocks:
        if block[0] in ("# subset_a:", "# subset_b:"):
            block[1:] = sorted(block[1:])
    text = "\n".join("\n".join(block) for block in blocks)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def read_exact_distance_records(results_dir: pathlib.Path) -> list[dict[str, str]]:
    """Read the records of exact distance computations in a results directory, if there are any.

    Each record is a dictionary with the keys in EXACT_DISTANCE_FIELDS; see compute_distances.py.
    """
    path = results_dir / EXACT_DISTANCES_NAME
    if not path.is_file():
        return []
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


####################################################################################################
# helper functions


def get_deterministic_hash(*inputs: object) -> int:
    """Hash inputs by their repr, consistently across Python sessions (unlike the built-in hash).

    Each input needs a repr that is the same in every Python session.  Seeds of saved codes are such
    hashes, so changing this function changes which codes a search samples, and stops a search from
    resuming from codes saved before the change.
    """
    input_bytes = repr(inputs).encode("utf-8")
    hash_bytes = hashlib.sha256(input_bytes).digest()
    return int.from_bytes(hash_bytes[:4], byteorder="big", signed=False)
