"""Unit tests for groups.py

Copyright 2023 The qLDPC Authors and Infleqtion Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import unittest.mock
import urllib

import galois
import numpy as np
import pytest
from sympy.combinatorics import Permutation

from qldpc import abstract, external

# define global testing variables
ORDER, INDEX = 2, 1
GENERATORS = [[(0, 1)]]
GROUP = f"SmallGroup({ORDER},{INDEX})"
GROUP_URL = external.groups.GROUPNAMES_URL + "1/C2.html"
MOCK_INDEX_HTML = """<table class="gptable" columns="6" style='width: 70%;'>
<tr><th width="12%"></th><th width="60%"></th><th width="5%"><a href='T.html'>d</a></th><th width="5%"><a href='R.html'>&rho;</a></th><th width="12%">Label</th><th width="7%">ID</th></tr><tr><td id="c2"><a href="1/C2.html">C<sub>2</sub></a></td><td><a href="cyclic.html">Cyclic</a> group</td><td><a href="T15.html#c2">2</a></td><td><a href="R.html#dim1+">1+</a></td><td>C2</td><td>2,1</td></tr>
</table>"""
MOCK_GROUP_HTML = """<b><a href='https://en.wikipedia.org/wiki/Group actions' title='See wikipedia' class='wiki'>Permutation representations of C<sub>2</sub></a></b><br><a id='shl1' class='shl' href="javascript:showhide('shs1','shl1','Regular action on 2 points');"><span class="nsgpn">&#x25ba;</span>Regular action on 2 points</a> - transitive group <a href="../T15.html#2t1">2T1</a><div id='shs1' class='shs'>Generators in S<sub>2</sub><br><pre class='pre' id='textgn1'>(1 2)</pre>&emsp;<button class='copytext' id='copygn1'>Copy</button><br>"""


def get_mock_page(text: str) -> unittest.mock.MagicMock:
    """Fake webpage with the given text."""
    mock_page = unittest.mock.MagicMock()
    mock_page.read.return_value = text.encode("utf-8")
    return mock_page


def test_get_group_url() -> None:
    """Retrieve url for group webpage on GroupNames.org."""
    # cannot connect to general webpage
    with unittest.mock.patch(
        "urllib.request.urlopen", side_effect=urllib.error.URLError("message")
    ):
        assert external.groups.get_group_url(ORDER, INDEX) is None

    # cannot find group in the index
    mock_page = get_mock_page(MOCK_INDEX_HTML.replace(f"{ORDER},{INDEX}", ""))
    with (
        unittest.mock.patch("urllib.request.urlopen", return_value=mock_page),
        pytest.raises(ValueError, match="Group .* not found"),
    ):
        external.groups.get_group_url(ORDER, INDEX)

    # cannot find link to group webpage
    mock_page = get_mock_page(MOCK_INDEX_HTML.replace("href", ""))
    with (
        unittest.mock.patch("urllib.request.urlopen", return_value=mock_page),
        pytest.raises(ValueError, match="Webpage .* not found"),
    ):
        external.groups.get_group_url(ORDER, INDEX)

    # everything works as expected
    mock_page = get_mock_page(MOCK_INDEX_HTML)
    with unittest.mock.patch("urllib.request.urlopen", return_value=mock_page):
        assert external.groups.get_group_url(ORDER, INDEX) == GROUP_URL


def test_maybe_get_generators_from_groupnames() -> None:
    """Retrieve generators from group webpage on GroupNames.org."""
    # group not indexed
    assert external.groups.maybe_get_generators_from_groupnames("") is None

    # group url not found
    with unittest.mock.patch("qldpc.external.groups.get_group_url", return_value=None):
        assert external.groups.maybe_get_generators_from_groupnames(GROUP) is None

    # cannot find generators
    mock_page = get_mock_page(MOCK_GROUP_HTML.replace("pre", ""))
    with (
        unittest.mock.patch("qldpc.external.groups.get_group_url", return_value=GROUP_URL),
        unittest.mock.patch("urllib.request.urlopen", return_value=mock_page),
        pytest.raises(ValueError, match="Generators .* not found"),
    ):
        external.groups.maybe_get_generators_from_groupnames(GROUP)

    # everything works as expected
    mock_page = get_mock_page(MOCK_GROUP_HTML)
    with (
        unittest.mock.patch("qldpc.external.groups.get_group_url", return_value=GROUP_URL),
        unittest.mock.patch("urllib.request.urlopen", return_value=mock_page),
    ):
        assert external.groups.maybe_get_generators_from_groupnames(GROUP) == GENERATORS


def test_maybe_get_generators_from_gap() -> None:
    """Retrieve generators from GAP 4."""
    external.gap.require_package.cache_clear()
    with unittest.mock.patch("qldpc.external.gap.is_installed", return_value=False):
        assert external.groups.maybe_get_generators_from_gap(GROUP) is None

    # cannot extract cycle from string
    with (
        unittest.mock.patch("qldpc.external.gap.require_package", return_value=None),
        unittest.mock.patch("qldpc.external.gap.get_output", return_value="\n(1, 2a)\n"),
        pytest.raises(ValueError, match="Cannot extract cycle"),
    ):
        assert external.groups.maybe_get_generators_from_gap(GROUP) is None

    # everything works as expected
    with (
        unittest.mock.patch("qldpc.external.gap.require_package", return_value=None),
        unittest.mock.patch("qldpc.external.gap.get_output", return_value="\n(1, 2)\n"),
        pytest.warns(UserWarning, match="_TEST_"),
    ):
        assert external.groups.maybe_get_generators_from_gap(GROUP, warning="_TEST_") == GENERATORS


def test_get_generators() -> None:
    """Retrieve generators for a GAP group somehow."""
    # retrieve known groups
    for group, generators in external.groups.KNOWN_GROUPS.items():
        assert external.groups.get_generators(group) == generators

    # retrieve from GAP
    with unittest.mock.patch(
        "qldpc.external.groups.maybe_get_generators_from_gap", return_value=GENERATORS
    ):
        assert external.groups.get_generators(GROUP) == GENERATORS

    # retrieve from GroupNames.org
    with (
        unittest.mock.patch(
            "qldpc.external.groups.maybe_get_generators_from_gap", return_value=None
        ),
        unittest.mock.patch(
            "qldpc.external.groups.maybe_get_generators_from_groupnames", return_value=GENERATORS
        ),
    ):
        assert external.groups.get_generators(GROUP) == GENERATORS

    # fail to retrieve from anywhere :(
    with (
        unittest.mock.patch(
            "qldpc.external.groups.maybe_get_generators_from_gap", return_value=None
        ),
        unittest.mock.patch(
            "qldpc.external.groups.maybe_get_generators_from_groupnames", return_value=None
        ),
    ):
        with (
            unittest.mock.patch("qldpc.external.gap.require_package", return_value=None),
            pytest.raises(ValueError, match="Cannot build GAP group"),
        ):
            external.groups.get_generators(GROUP)

        with (
            unittest.mock.patch("qldpc.external.gap.require_package", return_value=None),
            pytest.raises(ValueError, match="Cannot build GAP group"),
        ):
            external.groups.get_generators("CyclicGroup(2)")


def test_get_generators_from_magma(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Retrieve generators for a MAGMA group."""
    # define the automorphism group of a two-bit repetition code
    group = "AutomorphismGroup(LinearCode(Matrix(GF(2),1,2,[[1,1]])));"
    generators = [[(0, 1)]]

    # mock user inputs
    inputs = iter(
        ["Permutation group acting on a set of cardinality 2", "Order = 2", "    (1, 2)", ""]
    )
    monkeypatch.setattr("builtins.input", lambda: next(inputs))

    cache: dict[str, str] = {}
    with (
        unittest.mock.patch("qldpc.cache.get_disk_cache", return_value=cache),
        unittest.mock.patch("pyperclip.copy", return_value=None),
    ):
        # compute generators with MAGMA
        assert external.groups.get_generators_from_magma(group) == generators
        terminal_output, error_message = capsys.readouterr()
        assert not error_message
        assert terminal_output.startswith("Run the following command in MAGMA:")

        # now use the cache!
        assert external.groups.get_generators_from_magma(group) == generators
        terminal_output, error_message = capsys.readouterr()
        assert not error_message
        assert terminal_output.startswith("Run the following command in MAGMA:")
        assert "NOTICE: group found in the local MAGMA group cache" in terminal_output

    # mock invalid user input / MAGMA output
    inputs = iter(["There are no cycles in this output!", ""])
    monkeypatch.setattr("builtins.input", lambda: next(inputs))
    with pytest.raises(ValueError, match="Invalid MAGMA output"):
        external.groups.get_generators_from_magma(group)
    capsys.readouterr()  # intercept print statements


def test_get_small_group_number() -> None:
    """Retrieve the number of groups of some order."""
    order, number = 16, 14
    text = rf"<td>{order},{number}</td>"

    # fail to determine group number
    with (
        unittest.mock.patch("qldpc.external.groups.maybe_get_webpage", return_value=None),
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=False),
        pytest.raises(ValueError, match="Cannot determine"),
    ):
        external.groups.get_small_group_number(order)

    # retrieve from GAP
    with (
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=True),
        unittest.mock.patch("qldpc.external.gap.get_output", return_value=str(number)),
    ):
        assert external.groups.get_small_group_number(order) == number

    # retrieve from GroupNames.org
    with (
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=False),
        unittest.mock.patch("qldpc.external.groups.maybe_get_webpage", return_value=text),
    ):
        assert external.groups.get_small_group_number(order) == number


def test_get_small_group_structure() -> None:
    """Retrieve a description of the structure of a group."""
    order, index = 12, 3
    structure = "C3 : C4"

    # retrieve a structure from cache
    cache = {(order, index): structure}
    with unittest.mock.patch("qldpc.cache.get_disk_cache", return_value=cache):
        assert external.groups.get_small_group_structure(order, index) == structure

    # fail to retrieve structure from GAP
    with (
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=True),
        unittest.mock.patch("qldpc.external.gap.get_output", return_value=""),
        pytest.raises(ValueError, match="Group not recognized"),
    ):
        external.groups.get_small_group_structure(order, index)

    # retrieve structure from GAP
    with (
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=True),
        unittest.mock.patch("qldpc.external.gap.get_output", return_value=structure),
    ):
        assert external.groups.get_small_group_structure(order, index) == structure

    # GAP is not installed
    with (
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=False),
    ):
        structure = f"SmallGroup({order},{index})"
        assert external.groups.get_small_group_structure(order, index) == structure


def test_idempotents() -> None:
    """Find primitive central idempotents of a group algebra."""
    z_2 = galois.GF(2).primitive_element
    z_2_2 = galois.GF(4).primitive_element
    field = galois.GF(4)
    fake_output = "[ (Z(2))*(), (Z(2^2)^2)*(1,2)+(Z(2)^0)*(3,4)(5,6) ]"
    expected_idempotents = (
        ((int(field(z_2)), ((),)),),
        ((int(field(z_2_2**2)), ((0, 1),)), (int(field(z_2**0)), ((2, 3), (4, 5)))),
    )
    with (
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=True),
        unittest.mock.patch("qldpc.external.gap.require_package", return_value=None),
        unittest.mock.patch("qldpc.external.gap.get_output", return_value=fake_output),
    ):
        idempotents = external.groups.get_primitive_central_idempotents("fake_group", field.order)
        assert idempotents == expected_idempotents


def test_known_groups() -> None:
    """Retrieve known groups."""
    for group, generators in external.groups.KNOWN_GROUPS.items():
        assert external.groups.get_generators(group) == generators

        gap_generators = external.groups.get_generators_with_gap(group)
        assert gap_generators is None or gap_generators == generators


def test_find_permutations() -> None:
    mocked_outputs = [
        "(1,4,3,2)\n(1,4,2,3)\n(1,2,4,3)\n(1,2,3,4)\n(1,3,4,2)\n(1,3,2,4)\n",
        "(1,4,3,2)\n(1,4,2,3)\n(1,2,4,3)\n(1,2,3,4)\n(1,3,4,2)\n(1,3,2,4)\n",
        "(2,3)\n(1,3)\n(1,2)\n",
        "(1,2)\n",
        "\n",
        "\n",
    ]

    expected_4x4_order4 = (
        [
            abstract.GroupMember(0, 3, 2, 1),
            abstract.GroupMember(0, 3, 1, 2),
            abstract.GroupMember(0, 1, 3, 2),
            abstract.GroupMember(0, 1, 2, 3),
            abstract.GroupMember(0, 2, 3, 1),
            abstract.GroupMember(0, 2, 1, 3),
        ],
        [
            abstract.GroupMember(0, 3, 2, 1),
            abstract.GroupMember(0, 3, 1, 2),
            abstract.GroupMember(0, 1, 3, 2),
            abstract.GroupMember(0, 1, 2, 3),
            abstract.GroupMember(0, 2, 3, 1),
            abstract.GroupMember(0, 2, 1, 3),
        ],
    )

    expected_3x2_order2 = (
        [abstract.GroupMember(1, 2), abstract.GroupMember(0, 2), abstract.GroupMember(0, 1)],
        [abstract.GroupMember(0, 1)],
    )

    expected_empty: tuple[list[abstract.GroupMember], list[abstract.GroupMember]] = ([], [])

    with unittest.mock.patch("qldpc.external.gap.get_output", side_effect=mocked_outputs):
        actual = external.groups.get_permutation_symmetry_of_matrix(4, 4, 4)
        assert actual == expected_4x4_order4
        actual = external.groups.get_permutation_symmetry_of_matrix(2, 3, 2)
        assert actual == expected_3x2_order2
        actual = external.groups.get_permutation_symmetry_of_matrix(5, 3, 3)
        assert actual == expected_empty


def test_find_balanced_permutations() -> None:
    mocked_output = [
        "(4,6,5)\n(4,5,6)\n(3,6,5)\n(3,6,4)\n(3,4,6)\n(3,4,5)\n(3,5,6)\n(3,5,4)\n(2,6,5)\n(2,6,4)\n(2,6,3)\n(2,3,6)\n(2,3,4)\n(2,3,5)\n(2,4,6)\n(2,4,5)\n(2,4,3)\n(2,5,6)\n(2,5,4)\n(2,5,3)\n(1,6,5)\n(1,6,4)\n(1,6,3)\n(1,6,2)\n(1,6,2)(3,4,5)\n(1,6,2)(3,5,4)\n(1,6,5)(2,3,4)\n(1,6,4)(2,3,5)\n(1,6,5)(2,4,3)\n(1,6,3)(2,4,5)\n(1,6,4)(2,5,3)\n(1,6,3)(2,5,4)\n(1,2,6)\n(1,2,6)(3,4,5)\n(1,2,6)(3,5,4)\n(1,2,3)\n(1,2,3)(4,6,5)\n(1,2,3)(4,5,6)\n(1,2,4)\n(1,2,4)(3,6,5)\n(1,2,4)(3,5,6)\n(1,2,5)\n(1,2,5)(3,6,4)\n(1,2,5)(3,4,6)\n(1,3,6)\n(1,3,4)\n(1,3,5)\n(1,3,2)\n(1,3,2)(4,6,5)\n(1,3,2)(4,5,6)\n(1,3,4)(2,6,5)\n(1,3,5)(2,6,4)\n(1,3,6)(2,4,5)\n(1,3,5)(2,4,6)\n(1,3,6)(2,5,4)\n(1,3,4)(2,5,6)\n(1,4,6)\n(1,4,5)\n(1,4,3)\n(1,4,2)\n(1,4,2)(3,6,5)\n(1,4,2)(3,5,6)\n(1,4,5)(2,3,6)\n(1,4,6)(2,3,5)\n(1,4,5)(2,6,3)\n(1,4,3)(2,6,5)\n(1,4,6)(2,5,3)\n(1,4,3)(2,5,6)\n(1,5,6)\n(1,5,4)\n(1,5,3)\n(1,5,2)\n(1,5,2)(3,4,6)\n(1,5,2)(3,6,4)\n(1,5,6)(2,3,4)\n(1,5,4)(2,3,6)\n(1,5,6)(2,4,3)\n(1,5,3)(2,4,6)\n(1,5,4)(2,6,3)\n(1,5,3)(2,6,4)\n",
        "(4,6,5)\n(4,5,6)\n(3,6,5)\n(3,6,4)\n(3,4,6)\n(3,4,5)\n(3,5,6)\n(3,5,4)\n(2,6,5)\n(2,6,4)\n(2,6,3)\n(2,3,6)\n(2,3,4)\n(2,3,5)\n(2,4,6)\n(2,4,5)\n(2,4,3)\n(2,5,6)\n(2,5,4)\n(2,5,3)\n(1,6,5)\n(1,6,4)\n(1,6,3)\n(1,6,2)\n(1,6,2)(3,4,5)\n(1,6,2)(3,5,4)\n(1,6,5)(2,3,4)\n(1,6,4)(2,3,5)\n(1,6,5)(2,4,3)\n(1,6,3)(2,4,5)\n(1,6,4)(2,5,3)\n(1,6,3)(2,5,4)\n(1,2,6)\n(1,2,6)(3,4,5)\n(1,2,6)(3,5,4)\n(1,2,3)\n(1,2,3)(4,6,5)\n(1,2,3)(4,5,6)\n(1,2,4)\n(1,2,4)(3,6,5)\n(1,2,4)(3,5,6)\n(1,2,5)\n(1,2,5)(3,6,4)\n(1,2,5)(3,4,6)\n(1,3,6)\n(1,3,4)\n(1,3,5)\n(1,3,2)\n(1,3,2)(4,6,5)\n(1,3,2)(4,5,6)\n(1,3,4)(2,6,5)\n(1,3,5)(2,6,4)\n(1,3,6)(2,4,5)\n(1,3,5)(2,4,6)\n(1,3,6)(2,5,4)\n(1,3,4)(2,5,6)\n(1,4,6)\n(1,4,5)\n(1,4,3)\n(1,4,2)\n(1,4,2)(3,6,5)\n(1,4,2)(3,5,6)\n(1,4,5)(2,3,6)\n(1,4,6)(2,3,5)\n(1,4,5)(2,6,3)\n(1,4,3)(2,6,5)\n(1,4,6)(2,5,3)\n(1,4,3)(2,5,6)\n(1,5,6)\n(1,5,4)\n(1,5,3)\n(1,5,2)\n(1,5,2)(3,4,6)\n(1,5,2)(3,6,4)\n(1,5,6)(2,3,4)\n(1,5,4)(2,3,6)\n(1,5,6)(2,4,3)\n(1,5,3)(2,4,6)\n(1,5,4)(2,6,3)\n(1,5,3)(2,6,4)\n",
        "\n",
        "\n",
    ]
    with unittest.mock.patch("qldpc.external.gap.get_output", side_effect=mocked_output):
        matrix = np.array(
            [
                [1, 1, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 1, 1],
                [1, 0, 0, 0, 0, 1],
            ]
        )
        R, C = external.groups.get_balanced_permutations_of_matrix(matrix, 3)
        assert R == abstract.GroupMember.from_sympy(Permutation([2, 3, 4, 5, 0, 1]))
        assert C == abstract.GroupMember.from_sympy(Permutation([4, 5, 0, 1, 2, 3]))
        matrix = np.array(
            [
                [1, 1, 0],
                [0, 1, 1],
                [0, 0, 1],
            ]
        )
        with pytest.raises(ValueError):
            external.groups.get_balanced_permutations_of_matrix(matrix, 4)
