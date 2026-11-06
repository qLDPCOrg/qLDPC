# SPDX-License-Identifier: Apache-2.0

"""Unit tests for _groups.py."""

from __future__ import annotations

import unittest.mock
import urllib

import galois
import pytest

from qldpc import external
from qldpc.external import _groups

# define global testing variables
ORDER, INDEX = 2, 1
GENERATORS = [[(0, 1)]]
GROUP = f"SmallGroup({ORDER},{INDEX})"
GROUP_URL = _groups._GROUPNAMES_URL + "1/C2.html"
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
        assert _groups._get_group_url(ORDER, INDEX) is None

    # cannot find group in the index
    mock_page = get_mock_page(MOCK_INDEX_HTML.replace(f"{ORDER},{INDEX}", ""))
    with unittest.mock.patch("urllib.request.urlopen", return_value=mock_page):
        assert _groups._get_group_url(ORDER, INDEX) is None

    # cannot find link to group webpage
    mock_page = get_mock_page(MOCK_INDEX_HTML.replace("href", ""))
    with unittest.mock.patch("urllib.request.urlopen", return_value=mock_page):
        assert _groups._get_group_url(ORDER, INDEX) is None

    # everything works as expected
    mock_page = get_mock_page(MOCK_INDEX_HTML)
    with unittest.mock.patch("urllib.request.urlopen", return_value=mock_page):
        assert _groups._get_group_url(ORDER, INDEX) == GROUP_URL


def test_maybe_get_generators_from_groupnames() -> None:
    """Retrieve generators from group webpage on GroupNames.org."""
    # group not indexed
    assert _groups._maybe_get_generators_from_groupnames("") is None

    # group url not found
    with unittest.mock.patch("qldpc.external._groups._get_group_url", return_value=None):
        assert _groups._maybe_get_generators_from_groupnames(GROUP) is None

    # cannot find generators
    mock_page = get_mock_page(MOCK_GROUP_HTML.replace("pre", ""))
    with (
        unittest.mock.patch("qldpc.external._groups._get_group_url", return_value=GROUP_URL),
        unittest.mock.patch("urllib.request.urlopen", return_value=mock_page),
    ):
        assert _groups._maybe_get_generators_from_groupnames(GROUP) is None

    # group webpage is unreachable, or times out while loading
    with (
        unittest.mock.patch("qldpc.external._groups._get_group_url", return_value=GROUP_URL),
        unittest.mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("message")),
    ):
        assert _groups._maybe_get_generators_from_groupnames(GROUP) is None
    mock_page = unittest.mock.MagicMock()
    mock_page.read.side_effect = TimeoutError
    with (
        unittest.mock.patch("qldpc.external._groups._get_group_url", return_value=GROUP_URL),
        unittest.mock.patch("urllib.request.urlopen", return_value=mock_page),
    ):
        assert _groups._maybe_get_generators_from_groupnames(GROUP) is None

    # everything works as expected
    mock_page = get_mock_page(MOCK_GROUP_HTML)
    with (
        unittest.mock.patch("qldpc.external._groups._get_group_url", return_value=GROUP_URL),
        unittest.mock.patch("urllib.request.urlopen", return_value=mock_page),
    ):
        assert _groups._maybe_get_generators_from_groupnames(GROUP) == GENERATORS


def test_parse_gap_permutations() -> None:
    """Parse GAP permutations, tolerating blank lines and the identity."""
    # a blank line between permutations must not create a spurious identity generator
    assert _groups._parse_gap_permutations("(1,2)\n\n(3,4)") == [[(0, 1)], [(2, 3)]]
    # the identity permutation parses to a generator with no cycles
    assert _groups._parse_gap_permutations("()") == [[]]


def test_maybe_get_generators_from_gap() -> None:
    """Retrieve generators from GAP 4."""
    external.gap.require_package.cache_clear()
    with unittest.mock.patch("qldpc.external.gap.is_installed", return_value=False):
        assert _groups._maybe_get_generators_from_gap(GROUP) is None

    # cannot extract cycle from string
    with (
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=True),
        unittest.mock.patch("qldpc.external.gap.require_package", return_value=None),
        unittest.mock.patch("qldpc.external.gap.get_output", return_value="\n(1, 2a)\n"),
        pytest.raises(ValueError, match="Cannot extract cycle"),
    ):
        _groups._maybe_get_generators_from_gap(GROUP)

    # everything works as expected
    with (
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=True),
        unittest.mock.patch("qldpc.external.gap.require_package", return_value=None),
        unittest.mock.patch("qldpc.external.gap.get_output", return_value="\n(1, 2)\n"),
        pytest.warns(UserWarning, match="_TEST_"),
    ):
        assert _groups._maybe_get_generators_from_gap(GROUP, warning="_TEST_") == GENERATORS


def test_get_generators() -> None:
    """Retrieve generators for a GAP group somehow."""
    # retrieve a known group without calling GAP
    group, generators = next(iter(_groups._KNOWN_GROUPS.items()))
    with unittest.mock.patch(
        "qldpc.external.gap.is_callable", side_effect=AssertionError("GAP should not be called")
    ):
        assert _groups.get_generators(group) == generators

    # retrieve from GAP
    with (
        unittest.mock.patch("qldpc.external.gap.is_callable", return_value=True),
        unittest.mock.patch(
            "qldpc.external._groups._maybe_get_generators_from_gap", return_value=GENERATORS
        ),
    ):
        assert _groups.get_generators(GROUP) == GENERATORS

    # retrieve from GroupNames.org
    with (
        unittest.mock.patch("qldpc.external.gap.is_callable", return_value=True),
        unittest.mock.patch(
            "qldpc.external._groups._maybe_get_generators_from_gap", return_value=None
        ),
        unittest.mock.patch(
            "qldpc.external._groups._maybe_get_generators_from_groupnames", return_value=GENERATORS
        ),
    ):
        assert _groups.get_generators(GROUP) == GENERATORS

    # without a callable GAP, try GroupNames.org before the interactive GAP workflow
    with (
        unittest.mock.patch("qldpc.external.gap.is_callable", return_value=False),
        unittest.mock.patch(
            "qldpc.external._groups._maybe_get_generators_from_gap",
            side_effect=AssertionError("GAP should not be prompted for"),
        ),
        unittest.mock.patch(
            "qldpc.external._groups._maybe_get_generators_from_groupnames", return_value=GENERATORS
        ),
    ):
        assert _groups.get_generators(GROUP) == GENERATORS

    # without a callable GAP or GroupNames.org, fall back to the interactive GAP workflow
    with (
        unittest.mock.patch("qldpc.external.gap.is_callable", return_value=False),
        unittest.mock.patch(
            "qldpc.external._groups._maybe_get_generators_from_gap", return_value=GENERATORS
        ),
        unittest.mock.patch(
            "qldpc.external._groups._maybe_get_generators_from_groupnames", return_value=None
        ),
    ):
        assert _groups.get_generators(GROUP) == GENERATORS

    # fail to retrieve from anywhere :(
    with (
        unittest.mock.patch("qldpc.external.gap.is_callable", return_value=False),
        unittest.mock.patch(
            "qldpc.external._groups._maybe_get_generators_from_gap", return_value=None
        ),
        unittest.mock.patch(
            "qldpc.external._groups._maybe_get_generators_from_groupnames", return_value=None
        ),
    ):
        with (
            unittest.mock.patch("qldpc.external.gap.require_package", return_value=None),
            pytest.raises(ValueError, match="Cannot build GAP group"),
        ):
            _groups.get_generators(GROUP)

        with (
            unittest.mock.patch("qldpc.external.gap.require_package", return_value=None),
            pytest.raises(ValueError, match="Cannot build GAP group"),
        ):
            _groups.get_generators("CyclicGroup(2)")


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
    with unittest.mock.patch("qldpc.cache.get_disk_cache", return_value=cache):
        # compute generators with MAGMA
        assert _groups.get_generators_from_magma(group) == generators
        terminal_output, error_message = capsys.readouterr()
        assert not error_message
        assert terminal_output.startswith("Run the following command in MAGMA:")

        # now use the cache!
        assert _groups.get_generators_from_magma(group) == generators
        terminal_output, error_message = capsys.readouterr()
        assert not error_message
        assert terminal_output.startswith("Run the following command in MAGMA:")
        assert "NOTICE: group found in the local MAGMA group cache" in terminal_output

    # mock invalid user input / MAGMA output
    inputs = iter(["There are no cycles in this output!", ""])
    monkeypatch.setattr("builtins.input", lambda: next(inputs))
    with pytest.raises(ValueError, match="Invalid MAGMA output"):
        _groups.get_generators_from_magma(group)
    capsys.readouterr()  # intercept print statements


def test_get_small_group_number() -> None:
    """Retrieve the number of groups of some order."""
    order, number = 16, 14
    text = rf"<td>{order},{number}</td>"

    # fail to determine group number: webpage is unreachable
    with (
        unittest.mock.patch("qldpc.external._groups._maybe_get_webpage", return_value=None),
        unittest.mock.patch("qldpc.external.gap.is_callable", return_value=False),
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=False),
        pytest.raises(ValueError, match="Cannot determine"),
    ):
        _groups.get_small_group_number(order)

    # fail to determine group number: webpage has no matching entries
    with (
        unittest.mock.patch(
            "qldpc.external._groups._maybe_get_webpage", return_value="<html></html>"
        ),
        unittest.mock.patch("qldpc.external.gap.is_callable", return_value=False),
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=False),
        pytest.raises(ValueError, match="Cannot determine"),
    ):
        _groups.get_small_group_number(order)

    # retrieve from GAP
    with (
        unittest.mock.patch("qldpc.external.gap.is_callable", return_value=True),
        unittest.mock.patch("qldpc.external.gap.require_package", return_value=True),
        unittest.mock.patch("qldpc.external.gap.get_output", return_value=str(number)),
    ):
        assert _groups.get_small_group_number(order) == number

    # retrieve from GroupNames.org without prompting for manual GAP use
    with (
        unittest.mock.patch("qldpc.external.gap.is_callable", return_value=False),
        unittest.mock.patch(
            "qldpc.external.gap.is_installed",
            side_effect=AssertionError("GAP should not be prompted for"),
        ),
        unittest.mock.patch("qldpc.external._groups._maybe_get_webpage", return_value=text),
    ):
        assert _groups.get_small_group_number(order) == number

    # fall back to the interactive GAP workflow when GroupNames.org is unreachable
    with (
        unittest.mock.patch("qldpc.external.gap.is_callable", return_value=False),
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=True),
        unittest.mock.patch("qldpc.external._groups._maybe_get_webpage", return_value=None),
        unittest.mock.patch("qldpc.external.gap.require_package", return_value=True),
        unittest.mock.patch("qldpc.external.gap.get_output", return_value=str(number)),
    ):
        assert _groups.get_small_group_number(order) == number


def test_get_small_group_structure() -> None:
    """Retrieve a description of the structure of a group."""
    order, index = 12, 3
    structure = "C3 : C4"

    # retrieve a structure from cache
    cache = {(order, index): structure}
    with unittest.mock.patch("qldpc.cache.get_disk_cache", return_value=cache):
        assert _groups.get_small_group_structure(order, index) == structure

    # fail to retrieve structure from GAP
    with (
        unittest.mock.patch("qldpc.external.gap.is_callable", return_value=True),
        unittest.mock.patch("qldpc.external.gap.get_output", return_value=""),
        pytest.raises(ValueError, match="Group not recognized"),
    ):
        _groups.get_small_group_structure(order, index)

    # retrieve structure from GAP
    with (
        unittest.mock.patch("qldpc.external.gap.is_callable", return_value=True),
        unittest.mock.patch("qldpc.external.gap.get_output", return_value=structure),
    ):
        assert _groups.get_small_group_structure(order, index) == structure

    # GAP cannot be called without prompting the user
    with (
        unittest.mock.patch("qldpc.external.gap.is_callable", return_value=False),
        unittest.mock.patch(
            "qldpc.external.gap.is_installed",
            side_effect=AssertionError("GAP should not be prompted for"),
        ),
    ):
        structure = f"SmallGroup({order},{index})"
        assert _groups.get_small_group_structure(order, index) == structure


def test_idempotents() -> None:
    """Find primitive central idempotents of a group algebra."""
    # retrieve known PCIs without calling GAP
    (group, field_order), pcis = next(iter(_groups._KNOWN_PRIMITIVE_CENTRAL_IDEMPOTENTS.items()))
    with unittest.mock.patch(
        "qldpc.external.gap.require_package", side_effect=AssertionError("GAP should not be called")
    ):
        assert _groups.get_primitive_central_idempotents(group, field_order) == pcis

    z_2 = galois.GF(2).primitive_element
    z_2_2 = galois.GF(4).primitive_element
    field = galois.GF(4)
    fake_output = "[ (Z(2))*(), (Z(2^2)^2)*(1,2)+(Z(2)^0)*(3,4)(5,6) ]"
    expected_idempotents = [
        ((int(field(z_2)), ((),)),),
        ((int(field(z_2_2**2)), ((0, 1),)), (int(field(z_2**0)), ((2, 3), (4, 5)))),
    ]
    with (
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=True),
        unittest.mock.patch("qldpc.external.gap.require_package", return_value=None),
        unittest.mock.patch("qldpc.external.gap.get_output", return_value=fake_output),
    ):
        idempotents = _groups.get_primitive_central_idempotents("fake_group", field.order)
        assert idempotents == expected_idempotents


def test_idempotents_over_intermediate_subfield() -> None:
    """Coefficients whose GAP field is a proper intermediate subfield are embedded, not reindexed.

    Over GF(16) the primitive central idempotents of the cyclic group of order 3 have coefficients
    that are cube roots of unity, which GAP reports as powers of Z(2^2) -- the generator of the
    subfield GF(4).  The generator of GF(4) must be mapped into GF(16) through the subfield
    embedding (a power of the GF(16) primitive element), not by reinterpreting the integer that
    galois uses to store the GF(4) generator (which is a primitive element of GF(16), of order 15).
    """
    field = galois.GF(16)
    gap_output = (
        "[ (Z(2)^0)*()+(Z(2)^0)*(1,2,3)+(Z(2)^0)*(1,3,2), "
        "(Z(2)^0)*()+(Z(2^2))*(1,2,3)+(Z(2^2)^2)*(1,3,2), "
        "(Z(2)^0)*()+(Z(2^2)^2)*(1,2,3)+(Z(2^2))*(1,3,2) ]"
    )

    # Z(2^2) is the GF(4) generator; its image in GF(16) is primitive_element ** ((16-1)//(4-1)).
    z4 = field.primitive_element ** ((field.order - 1) // (galois.GF(4).order - 1))
    # {1, z4, z4**2} are the cube roots of unity in GF(16).  Computing them here without the
    # embedding formula pins the expected coefficients to the characters of the cyclic group of
    # order 3, the property that makes each nontrivial idempotent square to itself.  Reusing the
    # integers galois stores the GF(4) coefficients as -- {1, 2, 3} -- would be wrong: 2 has order
    # 15 in GF(16), so it is not a cube root of unity.
    assert {int(root) for root in field.elements if root**3 == field(1)} == {1, int(z4), int(z4**2)}
    expected_idempotents = [
        ((1, ((),)), (1, ((0, 1, 2),)), (1, ((0, 2, 1),))),
        ((1, ((),)), (int(z4), ((0, 1, 2),)), (int(z4**2), ((0, 2, 1),))),
        ((1, ((),)), (int(z4**2), ((0, 1, 2),)), (int(z4), ((0, 2, 1),))),
    ]
    with (
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=True),
        unittest.mock.patch("qldpc.external.gap.require_package", return_value=None),
        unittest.mock.patch("qldpc.external.gap.get_output", return_value=gap_output),
    ):
        idempotents = _groups.get_primitive_central_idempotents("group", field.order)
        assert idempotents == expected_idempotents
