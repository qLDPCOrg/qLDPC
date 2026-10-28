# SPDX-License-Identifier: Apache-2.0

"""Registry of deprecated names and methods, and helpers that warn about deprecations and changes.

A module with deprecated names ends with this marked stub, which serves the names that ALIASES and
SILENT_ALIASES list for that module:

    if not TYPE_CHECKING:  # DEPRECATED: see qldpc._deprecations

        def __getattr__(name: str) -> object:
            return _deprecations.get_deprecated_attribute(__name__, name)

Python calls a module-level __getattr__ (PEP 562) only for names that the module does not define.
A deprecated name thereby refers to the same object as its replacement, which preserves isinstance
checks and subclassing.  Type checkers do not see the stub, so they reject deprecated names, and no
__all__ lists them, so that star imports do not warn.

A registry target without an attribute names a module that replaces a deprecated submodule, which
is served as an attribute of its package, as in ``from package import submodule``.

Registry targets are lazy "module:attribute" strings, and this module imports no qLDPC subpackage
at import time, so that every module can import it without an import cycle.  Deprecated methods are
defined at the bottom of this module, and attached to their classes by install_deprecated_methods,
which qldpc/__init__.py calls once all subpackages are imported.
Deprecated arguments and modes are handled where they are accepted, by calling warn_deprecated or
raising defunct(...).

A future change to the behavior of a released API is likewise handled where its keyword is
accepted: a keyword argument selects the old or the new behavior, and if the caller does not specify
it, the old behavior is used and warn_future_change issues a FutureWarning.  Python shows
FutureWarnings to end users by default, unlike DeprecationWarnings.  Such blocks are marked with the
comment "# FUTURE CHANGE: see qldpc._deprecations"; the "Future changes" section of AGENTS.md
describes the steps of a future change.
"""

from __future__ import annotations

import importlib
import sys
import warnings
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, TypeVar

from qldpc._utils import get_external_caller_stacklevel

if TYPE_CHECKING:
    from collections.abc import Collection, Iterator, Sequence
    from types import FrameType

    import galois
    import networkx as nx
    import numpy as np
    import numpy.typing as npt
    import stim

    from qldpc.abstract.groups import NestedSequence
    from qldpc.codes import QuditCode
    from qldpc.decoders.dems import ErrorMechanism
    from qldpc.objects import TannerGraph

# the type of physical error rates that qldpc.codes.monte_carlo.ErrorRateFunc formerly accepted
OneOrManyFloats = TypeVar("OneOrManyFloats", float, Iterable[float])

_FunctionT = TypeVar("_FunctionT", bound=Callable[..., object])


####################################################################################################
# registry and attribute lookup


# Deprecated names that warn when accessed, by module, mapped to "module:attribute" targets, or to
# "module" targets for deprecated submodules.
ALIASES: dict[str, dict[str, str]] = {
    "qldpc.abstract": {
        "Element": "qldpc.abstract.rings:RingMember",
        "Protograph": "qldpc.abstract.ring_array:RingArray",
        "resolve_field": "qldpc.math:resolve_field",
    },
    "qldpc.abstract.ring_array": {
        "Protograph": "qldpc.abstract.ring_array:RingArray",
    },
    "qldpc.abstract.rings": {
        "Element": "qldpc.abstract.rings:RingMember",
        "Protograph": "qldpc.abstract.ring_array:RingArray",
    },
    "qldpc.circuits": {
        "restrict_to_qubits": "qldpc.circuits.common:_restrict_to_qubits",
        "validate_syndrome_qubit_ids": (
            "qldpc.circuits.memory.syndrome_measurement:_validate_syndrome_qubit_ids"
        ),
    },
    "qldpc.circuits.memory": {
        "validate_syndrome_qubit_ids": (
            "qldpc.circuits.memory.syndrome_measurement:_validate_syndrome_qubit_ids"
        ),
    },
    "qldpc.codes": {
        "code_capacity": "qldpc.codes._code_capacity",
        "get_distance_css_brouwer_zimmermann": (
            "qldpc.codes.distance._brouwer_zimmermann:get_distance_css_brouwer_zimmermann"
        ),
        "validate_distance_backend": "qldpc.codes.distance.core:_validate_distance_backend",
        "validate_distance_method": "qldpc.codes.distance.core:_validate_distance_method",
    },
    "qldpc.codes.distance": {
        "get_distance_css_brouwer_zimmermann": (
            "qldpc.codes.distance._brouwer_zimmermann:get_distance_css_brouwer_zimmermann"
        ),
        "validate_distance_backend": "qldpc.codes.distance.core:_validate_distance_backend",
        "validate_distance_method": "qldpc.codes.distance.core:_validate_distance_method",
    },
    "qldpc.codes.monte_carlo": {
        "OneOrManyFloats": "qldpc._deprecations.core:OneOrManyFloats",
    },
    "qldpc.decoders": {
        "BatchDecoder": "qldpc.decoders.protocols:BatchErrorDecoder",
        "CompiledSequentialWindowDecoder": (
            "qldpc.decoders.sinter.window:CompiledSequentialWindowDecoder"
        ),
        "CompiledSinterDecoder": "qldpc.decoders.sinter.core:CompiledSinterDecoder",
        "CompiledSubgraphDecoder": "qldpc.decoders.sinter.subgraph:CompiledSubgraphDecoder",
        "CompiledTrivialDecoder": "qldpc.decoders.sinter.core:CompiledTrivialDecoder",
        "CompositeDecoder": "qldpc.decoders.custom.composition:CompositeDecoder",
        "Decoder": "qldpc.decoders.protocols:ErrorDecoder",
        "DirectDecoder": "qldpc.decoders.custom.composition:DirectDecoder",
        "ExpandedErrorDecoder": ("qldpc.decoders.adapters.error_decoders:_ExpandedErrorDecoder"),
        "FlipPattern": "qldpc.decoders.dems:FlipPattern",
        "FrontierObservableDecoder": "qldpc.decoders.external.frontier:FrontierObservableDecoder",
        "GUFDecoder": "qldpc.decoders.custom.guf:GUFDecoder",
        "ILPDecoder": "qldpc.decoders.custom.ilp:ILPDecoder",
        "LookupDecoder": "qldpc.decoders.custom.lookup:LookupDecoder",
        "ObservableLookupDecoder": "qldpc.decoders.custom.lookup:ObservableLookupDecoder",
        "RelayBPDecoder": "qldpc.decoders.external.relay_bp:RelayBPDecoder",
        "SequentialSinterDecoder": "qldpc.decoders.sinter.window:SequentialWindowDecoder",
        "SubgraphSinterDecoder": "qldpc.decoders.sinter.subgraph:SubgraphDecoder",
        "TesseractDecoder": "qldpc.decoders.external.tesseract:TesseractDecoder",
        "WeightedLookupDecoder": "qldpc.decoders.custom.lookup:WeightedLookupDecoder",
        "WeightedObservableLookupDecoder": (
            "qldpc.decoders.custom.lookup:WeightedObservableLookupDecoder"
        ),
        "WrappedErrorDecoder": "qldpc.decoders.protocols:_WrappedErrorDecoder",
        "as_error_decoder": "qldpc.decoders.protocols:_as_error_decoder",
        "batch_decode_errors": "qldpc.decoders.protocols:_batch_decode_errors",
        "compiles_for_dem": "qldpc.decoders._capabilities:compiles_for_dem",
        "get_decoder_BF": "qldpc._deprecations.decoders:get_decoder_bf",
        "get_decoder_BP_LSD": "qldpc._deprecations.decoders:get_decoder_bp_lsd",
        "get_decoder_BP_OSD": "qldpc._deprecations.decoders:get_decoder_bp_osd",
        "get_decoder_GUF": "qldpc._deprecations.decoders:get_decoder_guf",
        "get_decoder_ILP": "qldpc._deprecations.decoders:get_decoder_ilp",
        "get_decoder_MWPM": "qldpc._deprecations.decoders:get_decoder_mwpm",
        "get_decoder_RBP": "qldpc._deprecations.decoders:get_decoder_rbp",
        "get_legacy_decoder_migration_message": (
            "qldpc._deprecations.decoders:get_legacy_decoder_migration_message"
        ),
        "is_prebuilt_decoder": "qldpc.decoders._capabilities:is_prebuilt_decoder",
        "is_prebuilt_observable_decoder": (
            "qldpc.decoders._capabilities:is_prebuilt_observable_decoder"
        ),
        "lookup_table": "qldpc.decoders.custom.lookup:lookup",
        "match_error_decoder_to_dem": (
            "qldpc.decoders.adapters.error_decoders:_match_error_decoder_to_dem"
        ),
        "reject_prebuilt_decoder": (
            "qldpc.decoders.construction._resolution:reject_prebuilt_decoder"
        ),
        "reject_removed_decoder_args": "qldpc._deprecations.decoders:reject_removed_decoder_args",
        "resolve_decoder": "qldpc._deprecations.decoders:resolve_decoder",
        "resolve_observable_decoder": "qldpc._deprecations.decoders:resolve_observable_decoder",
        "supports_batch_decoding": "qldpc.decoders.protocols:_supports_batch_decoding",
    },
    "qldpc.decoders.adapters": {
        "BatchErrorDecoder": "qldpc.decoders:BatchErrorDecoder",
        "BitPackedObservableDecoder": (
            "qldpc.decoders.adapters._observable_decoders:BitPackedObservableDecoder"
        ),
        "ErrorDecoder": "qldpc.decoders:ErrorDecoder",
        "ErrorsToFieldObservablesDecoder": (
            "qldpc.codes._code_capacity:ErrorsToFieldObservablesDecoder"
        ),
        "ExpandedErrorDecoder": ("qldpc.decoders.adapters.error_decoders:_ExpandedErrorDecoder"),
        "match_error_decoder_to_dem": (
            "qldpc.decoders.adapters.error_decoders:_match_error_decoder_to_dem"
        ),
        "validate_decoder_output": (
            "qldpc.decoders.adapters._observable_decoders:validate_decoder_output"
        ),
        "validate_observable_decoder": (
            "qldpc.decoders.adapters._observable_decoders:validate_observable_decoder"
        ),
    },
    "qldpc.decoders.custom": {
        "BatchDecoder": "qldpc.decoders.protocols:BatchErrorDecoder",
        "Decoder": "qldpc.decoders.protocols:ErrorDecoder",
        "PLACEHOLDER_ERROR_RATE": "qldpc.decoders.common:_PLACEHOLDER_ERROR_RATE",
        "RelayBPDecoder": "qldpc.decoders.external.relay_bp:RelayBPDecoder",
    },
    "qldpc.decoders.dems": {
        "CircuitError": "qldpc.decoders.dems:ErrorMechanism",
    },
    "qldpc.decoders.sinter": {
        "Decoder": "qldpc.decoders.protocols:ErrorDecoder",
        "SequentialSinterDecoder": "qldpc.decoders.sinter.window:SequentialWindowDecoder",
        "SubgraphSinterDecoder": "qldpc.decoders.sinter.subgraph:SubgraphDecoder",
    },
    "qldpc.external": {
        "codes": "qldpc.external._codes",
        "groups": "qldpc.external._groups",
    },
    # renamed functions of the deprecated qldpc.external.codes, which now refers to this module
    "qldpc.external._codes": {
        "get_classical_code": "qldpc.external._codes:get_guava_code",
        "get_distance_bound": "qldpc._deprecations.core:get_distance_bound",
        "get_quantum_code": "qldpc.external._codes:get_qecdb_code",
    },
}

# Deprecated functions, which are served without warning when accessed, because they warn when
# called instead.
SILENT_ALIASES: dict[str, dict[str, str]] = {
    "qldpc.decoders": {
        "decode": "qldpc._deprecations.decoders:decode",
        "decode_observables": "qldpc._deprecations.decoders:decode_observables",
        "get_decoder": "qldpc._deprecations.decoders:get_decoder",
        "get_decoder_bf": "qldpc._deprecations.decoders:get_decoder_bf",
        "get_decoder_bp_lsd": "qldpc._deprecations.decoders:get_decoder_bp_lsd",
        "get_decoder_bp_osd": "qldpc._deprecations.decoders:get_decoder_bp_osd",
        "get_decoder_guf": "qldpc._deprecations.decoders:get_decoder_guf",
        "get_decoder_ilp": "qldpc._deprecations.decoders:get_decoder_ilp",
        "get_decoder_lookup": "qldpc._deprecations.decoders:get_decoder_lookup",
        "get_decoder_mwpm": "qldpc._deprecations.decoders:get_decoder_mwpm",
        "get_decoder_rbp": "qldpc._deprecations.decoders:get_decoder_rbp",
        "get_error_decoder": "qldpc._deprecations.decoders:get_error_decoder",
        "get_observable_decoder": "qldpc._deprecations.decoders:get_observable_decoder",
    },
}


def get_deprecated_attribute(module_name: str, name: str) -> object:
    """Retrieve a deprecated attribute of a module, warning if ALIASES lists it.

    The warning names the replacement: by its new name if it was renamed, by its full path if it
    only moved, or not at all if its new home is private.

    Args:
        module_name: The name of the module whose attribute is being retrieved.
        name: The name of the attribute being retrieved.

    Returns:
        The object that the deprecated name refers to.

    Raises:
        AttributeError: If the registry lists no such name for the module.
    """
    if (target := SILENT_ALIASES.get(module_name, {}).get(name)) is not None:
        return _import_target(target)
    if (target := ALIASES.get(module_name, {}).get(name)) is None:
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
    if not _is_import_probe(sys._getframe(2)):
        warn_deprecated(
            _get_alias_warning(name, target), stacklevel=get_external_caller_stacklevel()
        )
    return _import_target(target)


def _import_target(target: str) -> object:
    """Import the object at a "module:attribute" target, or the module at a "module" target."""
    module_name, _, attribute = target.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, attribute) if attribute else module


def _get_alias_warning(name: str, target: str) -> str:
    """Describe the replacement for a deprecated name with the given registry target."""
    module_name, _, attribute = target.partition(":")
    path = f"{module_name}.{attribute}" if attribute else module_name
    if any(part.startswith("_") for part in path.split(".")):
        return f"{name} is deprecated and will be removed from the public API"
    replacement = attribute if attribute and attribute != name else path
    return f"{name} is deprecated; use {replacement} instead"


def _is_import_probe(frame: FrameType | None) -> bool:
    """Is a frame the probe that checks for names before a ``from package import ...`` statement?

    Python checks that a package provides each name in ``from package import name`` before it
    retrieves the name, so a module-level __getattr__ is called twice for one such statement.
    """
    return (
        frame is not None
        and frame.f_code.co_name == "_handle_fromlist"
        and frame.f_globals.get("__name__") == "importlib._bootstrap"
    )


####################################################################################################
# warnings and defunct errors


def warn_deprecated(message: str, *, stacklevel: int) -> None:
    """Issue a DeprecationWarning, with a stacklevel relative to the caller of this function."""
    warnings.warn(message, DeprecationWarning, stacklevel=stacklevel + 1)


def warn_future_change(message: str, *, stacklevel: int) -> None:
    """Issue a FutureWarning about a future change, with a stacklevel relative to the caller.

    Pass stacklevel=get_external_caller_stacklevel() from qldpc._utils, so that the warning points
    at the user code that called into qLDPC.
    """
    warnings.warn(message, FutureWarning, stacklevel=stacklevel + 1)


def defunct(message: str) -> ValueError:
    """Return the error that a defunct method or argument raises when it is used."""
    return ValueError(message)


####################################################################################################
# deprecated functions


def get_distance_bound(
    code: QuditCode,
    num_trials: int = 1,
    *,
    cutoff: int | None = None,
    maxav: str = "fail",
) -> int:
    """Deprecated: estimate the distance of a quantum code using GAP's QDistRnd package.

    If given a CSSCode, estimate its Z-distance.  This function was formerly
    qldpc.external.codes.get_distance_bound.
    """
    from qldpc.codes import CSSCode
    from qldpc.external._codes import get_qdistrnd_distance_bound
    from qldpc.objects import Pauli

    if isinstance(code, CSSCode):
        return get_qdistrnd_distance_bound(
            code.get_stabilizer_ops(Pauli.X),
            num_trials,
            matrix_z=code.matrix_z,
            cutoff=cutoff,
            maxav=maxav,
        )
    return get_qdistrnd_distance_bound(
        code.matrix,
        num_trials,
        is_subsystem_code=code.is_subsystem_code,
        cutoff=cutoff,
        maxav=maxav,
    )


####################################################################################################
# deprecated methods


def install_deprecated_methods() -> None:
    """Attach deprecated methods to their classes."""
    from qldpc.abstract import SmallGroup, TrivialGroup
    from qldpc.codes import AbstractCode, BBCode, ClassicalCode, CSSCode, QuditCode
    from qldpc.decoders.custom import DirectDecoder, GUFDecoder
    from qldpc.decoders.dems import DetectorErrorModelArrays

    _attach_tanner_graph_methods(ClassicalCode, symplectic=False)
    _attach_tanner_graph_methods(QuditCode, symplectic=True)

    @_deprecated_method(ClassicalCode)
    def puncture(self: ClassicalCode, bits: Collection[int]) -> ClassicalCode:
        """Deprecated alias for ClassicalCode.punctured."""
        warn_deprecated(
            "ClassicalCode.puncture is DEPRECATED; use ClassicalCode.punctured instead",
            stacklevel=get_external_caller_stacklevel(),
        )
        return self.punctured(bits)

    @_deprecated_method(ClassicalCode)
    def shorten(self: ClassicalCode, bits: Collection[int]) -> ClassicalCode:
        """Deprecated alias for ClassicalCode.shortened."""
        warn_deprecated(
            "ClassicalCode.shorten is DEPRECATED; use ClassicalCode.shortened instead",
            stacklevel=get_external_caller_stacklevel(),
        )
        return self.shortened(bits)

    @_deprecated_method(QuditCode)
    def get_dual_subsystem_code(self: QuditCode) -> QuditCode:
        """Deprecated alias for self.dual()."""
        warn_deprecated(
            "QuditCode.get_dual_subsystem_code is DEPRECATED; use QuditCode.dual instead",
            stacklevel=get_external_caller_stacklevel(),
        )
        return self.dual()

    @_deprecated_method(QuditCode)
    def conjugate(self: QuditCode) -> QuditCode:
        """Deprecated alias for ``conjugated()``."""
        name = type(self).__name__
        warn_deprecated(
            f"{name}.conjugate is DEPRECATED; use {name}.conjugated instead",
            stacklevel=get_external_caller_stacklevel(),
        )
        code = self.conjugated()
        return code.to_css() if isinstance(self, CSSCode) else code

    @_deprecated_method(BBCode)
    def modular_inverse(
        self: BBCode, basis: tuple[tuple[int, int], tuple[int, int]], aa: int, bb: int
    ) -> tuple[int, int]:
        """Deprecated method that solves for the coefficients of a vector in a basis."""
        warn_deprecated(
            "BBCode.modular_inverse is DEPRECATED and will be removed from the public API",
            stacklevel=get_external_caller_stacklevel(),
        )
        return self._get_basis_coefficients(basis, aa, bb)

    @_deprecated_method(DirectDecoder)
    def decode(self: DirectDecoder, word: npt.NDArray[np.int_]) -> None:
        """DEFUNCT alias for DirectDecoder.correct."""
        raise defunct(
            "DirectDecoder.decode is DEFUNCT; use DirectDecoder.correct instead.  A DirectDecoder"
            " corrects code words, and is not a syndrome decoder"
        )

    @_deprecated_method(DirectDecoder)
    def decode_batch(self: DirectDecoder, words: npt.NDArray[np.int_]) -> None:
        """DEFUNCT alias for DirectDecoder.correct_batch."""
        raise defunct(
            "DirectDecoder.decode_batch is DEFUNCT; use DirectDecoder.correct_batch instead.  A"
            " DirectDecoder corrects code words, and is not a syndrome decoder"
        )

    @_deprecated_method(SmallGroup, static=True)
    def number(order: int) -> int:
        """Deprecated alias for SmallGroup.count_groups."""
        warn_deprecated(
            "SmallGroup.number is DEPRECATED; use SmallGroup.count_groups instead",
            stacklevel=get_external_caller_stacklevel(),
        )
        return SmallGroup.count_groups(order)

    @_deprecated_method(SmallGroup, static=True)
    def generator(order: int) -> Iterator[SmallGroup]:
        """Deprecated alias for SmallGroup.iter_groups."""
        warn_deprecated(
            "SmallGroup.generator is DEPRECATED; use SmallGroup.iter_groups instead",
            stacklevel=get_external_caller_stacklevel(),
        )
        return SmallGroup.iter_groups(order)

    @_deprecated_method(TrivialGroup, static=True)
    def to_ring_array(data: npt.NDArray[np.int_] | NestedSequence) -> None:
        """DEFUNCT alias for qldpc.abstract.RingArray.build(data)."""
        raise defunct(
            "TrivialGroup.to_ring_array(data) is DEFUNCT; use"
            " qldpc.abstract.RingArray.build(data) instead"
        )

    @_deprecated_method(DetectorErrorModelArrays, static=True)
    def get_circuit_errors(
        dem: stim.DetectorErrorModel, *, decompose_errors: bool = False
    ) -> list[ErrorMechanism]:
        """Deprecated alias for DetectorErrorModelArrays.get_error_mechanisms."""
        warn_deprecated(
            "DetectorErrorModelArrays.get_circuit_errors is DEPRECATED; use"
            " DetectorErrorModelArrays.get_error_mechanisms instead",
            stacklevel=get_external_caller_stacklevel(),
        )
        return DetectorErrorModelArrays.get_error_mechanisms(dem, decompose_errors=decompose_errors)

    @_deprecated_method(DetectorErrorModelArrays, static=True)
    def get_merged_circuit_errors(errors: list[ErrorMechanism]) -> list[ErrorMechanism]:
        """Deprecated alias for DetectorErrorModelArrays.get_merged_error_mechanisms."""
        warn_deprecated(
            "DetectorErrorModelArrays.get_merged_circuit_errors is DEPRECATED; use"
            " DetectorErrorModelArrays.get_merged_error_mechanisms instead",
            stacklevel=get_external_caller_stacklevel(),
        )
        return DetectorErrorModelArrays.get_merged_error_mechanisms(errors)

    @_deprecated_method(GUFDecoder, attribute=True)
    def code(self: GUFDecoder) -> AbstractCode:
        """Deprecated code whose parity check matrix this decoder decodes."""
        warn_deprecated(
            "GUFDecoder.code is DEPRECATED and will be removed from the public API",
            stacklevel=get_external_caller_stacklevel(),
        )
        return QuditCode(self._matrix) if self.symplectic else ClassicalCode(self._matrix)


def _attach_tanner_graph_methods(cls: type, *, symplectic: bool) -> None:
    """Attach deprecated static methods that convert between matrices and Tanner graphs."""
    import galois

    from qldpc.objects import TannerGraph

    name = cls.__name__

    @_deprecated_method(cls, static=True)
    def matrix_to_graph(matrix: npt.NDArray[np.int_] | Sequence[Sequence[int]]) -> TannerGraph:
        """Deprecated alias for qldpc.objects.TannerGraph.from_matrix."""
        warn_deprecated(
            f"{name}.matrix_to_graph is DEPRECATED; use qldpc.objects.TannerGraph.from_matrix"
            " instead",
            stacklevel=get_external_caller_stacklevel(),
        )
        return TannerGraph.from_matrix(matrix, symplectic=symplectic)

    @_deprecated_method(cls, static=True)
    def graph_to_matrix(graph: nx.DiGraph) -> galois.FieldArray:
        """Deprecated alias for qldpc.objects.TannerGraph.to_matrix."""
        warn_deprecated(
            f"{name}.graph_to_matrix is DEPRECATED; use qldpc.objects.TannerGraph.to_matrix"
            " instead",
            stacklevel=get_external_caller_stacklevel(),
        )
        field = getattr(graph, "field", galois.GF2)
        return TannerGraph(graph, field=field, symplectic=symplectic).to_matrix()


def _deprecated_method(
    cls: type, *, static: bool = False, attribute: bool = False
) -> Callable[[_FunctionT], _FunctionT]:
    """Attach a function to a class as a method, with the module and qualified name of a member.

    A static function is attached as a static method, and an attribute function as a read-only
    property.
    """

    def attach(function: _FunctionT) -> _FunctionT:
        function.__module__ = cls.__module__
        function.__qualname__ = f"{cls.__qualname__}.{function.__name__}"
        function.__annotations__.pop("self", None)  # methods in class bodies leave self implicit
        member: object = function
        if static:
            member = staticmethod(function)
        elif attribute:
            member = property(function)
        setattr(cls, function.__name__, member)
        return function

    return attach
