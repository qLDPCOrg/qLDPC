# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any

from qldpc._util import get_deprecated_alias

from . import ring_array, rings
from ._monomials import (
    get_coefficient_and_exponents,
    iter_monomial_terms,
)
from .groups import (
    GL,
    PGL,
    PSL,
    SL,
    AbelianGroup,
    AlternatingGroup,
    CyclicGroup,
    DihedralGroup,
    GeneralLinearGroup,
    Group,
    GroupMember,
    ProjectiveGeneralLinearGroup,
    ProjectiveSpecialLinearGroup,
    QuaternionGroup,
    SmallGroup,
    SpecialLinearGroup,
    SymmetricGroup,
    TrivialGroup,
    WreathProductGroup,
    resolve_field,
)
from .linalg import (
    block_diag,
    get_howell_dual,
    kron,
    matmul,
)
from .ring_array import (
    RingArray,
)
from .rings import (
    GroupRing,
    RingMember,
)
from .wedderburn_artin import (
    WedderburnArtinComponentTransformer,
    WedderburnArtinTransformer,
)

__all__ = [
    "GL",
    "PGL",
    "PSL",
    "SL",
    "AbelianGroup",
    "AlternatingGroup",
    "CyclicGroup",
    "DihedralGroup",
    "Element",
    "GeneralLinearGroup",
    "Group",
    "GroupMember",
    "GroupRing",
    "ProjectiveGeneralLinearGroup",
    "ProjectiveSpecialLinearGroup",
    "Protograph",
    "QuaternionGroup",
    "RingArray",
    "RingMember",
    "SmallGroup",
    "SpecialLinearGroup",
    "SymmetricGroup",
    "TrivialGroup",
    "WedderburnArtinComponentTransformer",
    "WedderburnArtinTransformer",
    "WreathProductGroup",
    "block_diag",
    "get_coefficient_and_exponents",
    "get_howell_dual",
    "iter_monomial_terms",
    "kron",
    "matmul",
    "resolve_field",
]

# Deprecated names remain importable (including by star imports, since they are listed in __all__),
# and resolve at runtime through a module-level __getattr__ (PEP 562) that warns when accessed.
# Type checkers instead see plain aliases, so that they still flag misspelled attributes.
DEPRECATED_ALIASES = ring_array.DEPRECATED_ALIASES | rings.DEPRECATED_ALIASES

if TYPE_CHECKING:
    from .ring_array import Protograph as Protograph
    from .rings import Element as Element
else:

    def __getattr__(name: str) -> Any:
        """Resolve deprecated names of abstract-algebra classes, with a DeprecationWarning."""
        return get_deprecated_alias(__name__, name, DEPRECATED_ALIASES)
