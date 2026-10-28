# SPDX-License-Identifier: Apache-2.0

"""Unit tests for core.py."""

from __future__ import annotations

import importlib
import inspect
import unittest.mock
import warnings

import galois
import networkx as nx
import numpy as np
import pytest
import stim
from sympy.abc import x, y

from qldpc import _deprecations, abstract, codes, decoders, math
from qldpc.objects import TannerGraph


def test_aliases() -> None:
    """Each deprecated name warns once, naming its replacement, and is the replacement itself."""
    for module_name, aliases in _deprecations.ALIASES.items():
        module = importlib.import_module(module_name)
        for name, target in aliases.items():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                value = getattr(module, name)
            target_module, _, attribute = target.partition(":")
            replacement = importlib.import_module(target_module)
            replacement = getattr(replacement, attribute) if attribute else replacement
            assert value is replacement

            # a private replacement is not named, a renamed object is named by its new name, and a
            # moved one by its full path
            path = f"{target_module}.{attribute}" if attribute else target_module
            if any(part.startswith("_") for part in path.split(".")):
                expected = f"{name} is deprecated and will be removed from the public API"
            elif attribute != name:
                expected = f"{name} is deprecated; use {attribute} instead"
                assert _get_from_package(module_name, attribute) is replacement
            else:
                expected = f"{name} is deprecated; use {path} instead"
            assert [str(warning.message) for warning in caught] == [expected]
            assert caught[0].category is DeprecationWarning
            assert caught[0].filename == __file__


def test_silent_aliases() -> None:
    """Deprecated functions that warn when called do not warn when accessed."""
    for module_name, aliases in _deprecations.SILENT_ALIASES.items():
        module = importlib.import_module(module_name)
        for name, target in aliases.items():
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                value = getattr(module, name)
            target_module, _, attribute = target.partition(":")
            assert value is getattr(importlib.import_module(target_module), attribute)


def test_deprecated_names() -> None:
    """Deprecated names are importable but unlisted, and unknown names are not deprecated."""
    for module_name in _deprecations.ALIASES.keys() | _deprecations.SILENT_ALIASES.keys():
        names = set(_deprecations.ALIASES.get(module_name, {}))
        names |= set(_deprecations.SILENT_ALIASES.get(module_name, {}))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            module = importlib.import_module(module_name)
            namespace: dict[str, object] = {}
            exec(f"from {module_name} import {', '.join(sorted(names))}", namespace)  # noqa: S102
        assert names <= namespace.keys()
        assert not names & set(getattr(module, "__all__", ()))
    with pytest.raises(AttributeError, match=r"module 'qldpc\.decoders' has no attribute 'Other'"):
        _deprecations.get_deprecated_attribute("qldpc.decoders", "Other")


def test_deprecated_import_warns_once() -> None:
    """Importing a deprecated name from a package warns once, although Python looks it up twice."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from qldpc.decoders import Decoder  # type: ignore[attr-defined]
    assert Decoder is decoders.ErrorDecoder
    assert [str(warning.message) for warning in caught] == [
        "Decoder is deprecated; use ErrorDecoder instead"
    ]
    assert caught[0].filename == __file__


def test_star_imports_do_not_warn() -> None:
    """Star imports of modules that serve deprecated names do not warn."""
    for module_name in _deprecations.ALIASES.keys() | _deprecations.SILENT_ALIASES.keys():
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # a star import is only valid at module level, so execute it in a fresh namespace
            exec(f"from {module_name} import *", {})  # noqa: S102


def test_deprecated_methods() -> None:
    """Deprecated methods warn and delegate to their replacements."""
    classical_code = codes.RepetitionCode(3)
    qudit_code = codes.QuditCode(codes.FiveQubitCode())
    css_code = codes.SteaneCode()
    bb_code = codes.BBCode((3, 3), 1 + x + x * y, 1 + y + x * y)
    cases: list[tuple[object, str, tuple[object, ...], object, str]] = [
        (classical_code, "puncture", ([0],), classical_code.punctured([0]), "ClassicalCode"),
        (classical_code, "shorten", ([0],), classical_code.shortened([0]), "ClassicalCode"),
        # 1 * (1, 1) + 2 * (0, 1) == (1, 0) modulo 3
        (bb_code, "modular_inverse", (((1, 1), (0, 1)), 1, 0), (1, 2), "BBCode"),
        (qudit_code, "get_dual_subsystem_code", (), qudit_code.dual(), "QuditCode"),
        (qudit_code, "conjugate", (), qudit_code.conjugated(), "QuditCode"),
        (css_code, "conjugate", (), css_code.conjugated().to_css(), "SteaneCode"),
    ]
    for code, method_name, args, expected, class_name in cases:
        with pytest.warns(DeprecationWarning, match="DEPRECATED") as caught:
            result = getattr(code, method_name)(*args)
        assert len(caught) == 1
        assert caught[0].filename == __file__
        assert str(caught[0].message).startswith(f"{class_name}.{method_name} is DEPRECATED")
        assert result == expected
    assert isinstance(result, codes.CSSCode)

    # an attached method reads like one defined in the class body
    puncture = vars(codes.ClassicalCode)["puncture"]
    assert puncture.__module__ == codes.ClassicalCode.__module__
    assert puncture.__qualname__ == "ClassicalCode.puncture"
    assert str(inspect.signature(puncture)).startswith("(self, bits")


def test_deprecated_tanner_graph_methods() -> None:
    """Deprecated conversions between matrices and Tanner graphs delegate to TannerGraph."""
    matrix = galois.GF(3)([[1, 2, 0, 0], [0, 1, 1, 2]])
    for cls, symplectic in [(codes.ClassicalCode, False), (codes.QuditCode, True)]:
        name = cls.__name__
        with pytest.warns(
            DeprecationWarning, match=f"{name}.matrix_to_graph is DEPRECATED"
        ) as caught:
            graph = cls.matrix_to_graph(matrix)  # type: ignore[attr-defined]
        assert caught[0].filename == __file__
        expected = TannerGraph.from_matrix(matrix, symplectic=symplectic)
        assert nx.utils.graphs_equal(graph, expected)

        # a plain graph with a field attribute, as these methods formerly built, converts back
        graph = nx.DiGraph(expected)
        graph.field = expected.field
        with pytest.warns(
            DeprecationWarning, match=f"{name}.graph_to_matrix is DEPRECATED"
        ) as caught:
            assert np.array_equal(cls.graph_to_matrix(graph), matrix)  # type: ignore[attr-defined]
        assert caught[0].filename == __file__

    # subclasses inherit the deprecated methods
    with pytest.warns(DeprecationWarning, match="QuditCode.matrix_to_graph is DEPRECATED"):
        graph = codes.CSSCode.matrix_to_graph(matrix)  # type: ignore[attr-defined]
    assert graph.symplectic


def test_deprecated_guf_decoder_code() -> None:
    """The deprecated code of a GUF decoder warns and rebuilds the code that it decodes."""
    code = codes.FiveQubitCode()
    for decoder, expected in [
        (decoders.custom.GUFDecoder(code.matrix), codes.ClassicalCode(code.matrix)),
        (
            decoders.custom.GUFDecoder(code.matrix, symplectic=True),
            codes.QuditCode(-math.symplectic_conjugate(code.matrix)),
        ),
    ]:
        with pytest.warns(DeprecationWarning, match="GUFDecoder.code is DEPRECATED") as caught:
            assert decoder.code == expected  # type: ignore[attr-defined]
        assert caught[0].filename == __file__


def test_deprecated_get_distance_bound() -> None:
    """The deprecated QDistRnd distance bound sends GAP the same commands as the code methods."""
    stabilizer_code, css_code = codes.FiveQubitCode(), codes.SurfaceCode(2)
    subsystem_code = codes.QuditCode(codes.SHYPSCode(2).matrix)
    with (
        unittest.mock.patch("qldpc.external.gap.require_package"),
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=True),
        unittest.mock.patch("qldpc.external.gap.get_output", return_value="3") as get_output,
    ):
        for code, get_bound in [
            (stabilizer_code, lambda: stabilizer_code.get_distance_bound(1, backend="gap")),
            (css_code, lambda: css_code.get_distance_bound(1, pauli="Z", backend="gap")),
        ]:
            code.forget_distance()
            assert get_bound() == _deprecations.core.get_distance_bound(code) == 3
            assert get_output.call_args_list[-2] == get_output.call_args_list[-1]
        with pytest.raises(ValueError, match="non-CSS subsystem codes"):
            _deprecations.core.get_distance_bound(subsystem_code)


def test_deprecated_small_group_methods() -> None:
    """Deprecated SmallGroup methods warn and delegate to their replacements."""
    group = abstract.CyclicGroup(2)
    generators = [tuple(gen.array_form) for gen in group.generators]
    with (
        unittest.mock.patch("qldpc.external._groups.get_small_group_number", return_value=1),
        unittest.mock.patch("qldpc.external._groups.get_generators", return_value=generators),
    ):
        with pytest.warns(DeprecationWarning, match="use SmallGroup.count_groups") as caught:
            assert abstract.SmallGroup.number(2) == 1  # type: ignore[attr-defined]
        assert caught[0].filename == __file__
        with pytest.warns(DeprecationWarning, match="use SmallGroup.iter_groups") as caught:
            groups = list(abstract.SmallGroup.generator(2))  # type: ignore[attr-defined]
        assert caught[0].filename == __file__
    assert len(groups) == 1 and groups[0].equiv(group)


def test_deprecated_dem_methods() -> None:
    """Deprecated DetectorErrorModelArrays methods warn and delegate to their replacements."""
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 ^ D1 L0
        error(0.2) D0 D1 L0
        error(0.1) D0 ^ D1 L0
    """)
    dem_arrays = decoders.DetectorErrorModelArrays
    for decompose_errors in [False, True]:
        with pytest.warns(DeprecationWarning, match="use .*get_error_mechanisms") as caught:
            errors = dem_arrays.get_circuit_errors(  # type: ignore[attr-defined]
                dem, decompose_errors=decompose_errors
            )
        assert caught[0].filename == __file__
        assert errors == dem_arrays.get_error_mechanisms(dem, decompose_errors=decompose_errors)
    with pytest.warns(DeprecationWarning, match="use .*get_merged_error_mechanisms") as caught:
        merged = dem_arrays.get_merged_circuit_errors(errors)  # type: ignore[attr-defined]
    assert caught[0].filename == __file__
    assert merged == dem_arrays.get_merged_error_mechanisms(errors)


def test_defunct_methods() -> None:
    """Defunct methods raise an error that names their replacements."""
    direct_decoder = decoders.custom.DirectDecoder(lambda word: word)
    words = np.zeros((2, 3), dtype=int)
    for obj, method_name, args, replacement in [
        (direct_decoder, "decode", (words[0],), "DirectDecoder.correct"),
        (direct_decoder, "decode_batch", (words,), "DirectDecoder.correct_batch"),
        (abstract.TrivialGroup(), "to_ring_array", ([[1]],), "RingArray.build"),
    ]:
        with pytest.raises(ValueError, match=f"DEFUNCT; use .*{replacement}"):
            getattr(obj, method_name)(*args)


def _get_from_package(module_name: str, name: str) -> object:
    """Get a name from a module, or else from the closest package in qldpc that provides it."""
    parts = module_name.split(".")
    modules = [importlib.import_module(".".join(parts[:end])) for end in range(len(parts), 1, -1)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return next(getattr(module, name) for module in modules if hasattr(module, name))
