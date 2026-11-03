# SPDX-License-Identifier: Apache-2.0

"""Unit tests for cache.py."""

from __future__ import annotations

import copy
import pathlib
import pickle
import unittest.mock
from collections.abc import Hashable

import diskcache_community
import galois
import numpy as np
import pytest

import qldpc.cache
import qldpc.external._codes


def test_pytest() -> None:
    """Disk caching is bypassed when running under pytest."""
    assert qldpc.cache._running_with_pytest()
    assert qldpc.cache.get_disk_cache("test") == {}

    calls = []

    @qldpc.cache.use_disk_cache("test")
    def test_func(value: int) -> int:
        """Test docstring."""
        calls.append(value)
        return value

    assert test_func.__doc__ == "Test docstring."
    assert test_func(1) == test_func(1) == test_func.refresh(1) == 1
    assert calls == [1, 1, 1]


def test_use_disk_cache(tmp_path: pathlib.Path) -> None:
    """Cache function outputs."""

    cache: dict[Hashable, int] = {}
    with (
        unittest.mock.patch("qldpc.cache._running_with_pytest", return_value=False),
        unittest.mock.patch("diskcache_community.Cache", return_value=cache),
    ):

        @qldpc.cache.use_disk_cache("test_name")
        def get_five(_: str) -> int:
            return 5

        # use cache to save/retrieve results
        get_five("test_arg")  # save results to cache
        assert cache == {"test_arg": 5}
        assert cache["test_arg"] == get_five("test_arg")

        # post-process inputs to determine the cache key
        @qldpc.cache.use_disk_cache("test_name", key_func=lambda _: None)
        def get_six(_: str) -> int:
            return 6

        assert get_six("test_arg") == 6
        assert cache == {"test_arg": 5, None: 6}

        # delete an entry from the cache
        qldpc.cache.clear_entry("test_name", None)
        assert cache == {"test_arg": 5}

        # raise a warning if trying to delete an entry that does not exist in the cache
        with pytest.warns(UserWarning, match="entry does not exist"):
            qldpc.cache.clear_entry("test_name", "some_key")
        with pytest.warns(UserWarning, match="located at"):
            qldpc.cache.clear_entry("test_name", "some_key", cache_dir=tmp_path)


def test_refresh() -> None:
    """Refresh cached function outputs."""

    cache: dict[Hashable, int] = {}
    with (
        unittest.mock.patch("qldpc.cache._running_with_pytest", return_value=False),
        unittest.mock.patch("diskcache_community.Cache", return_value=cache),
    ):
        calls: list[tuple[int, int]] = []

        @qldpc.cache.use_disk_cache("test_name")
        def count(value: int, *, offset: int = 0) -> int:
            calls.append((value, offset))
            return len(calls)

        # the first call computes a result, and the second call retrieves it from the cache
        assert count(1) == count(1) == 1
        assert calls == [(1, 0)]

        # refreshing recomputes the result and overwrites the cache entry
        assert count.refresh(1) == 2
        assert cache == {1: 2}
        assert count(1) == 2
        assert calls == [(1, 0), (1, 0)]

        # refreshing a missing entry computes and saves a result without warnings
        assert count.refresh(2, offset=3) == 3
        assert cache == {1: 2, (2, ("offset", 3)): 3}

        # refreshing uses the custom cache key
        @qldpc.cache.use_disk_cache("test_name", key_func=lambda value: f"key-{value}")
        def double(value: int) -> int:
            calls.append((value, 0))
            return 2 * value

        cache["key-4"] = 0
        assert double(4) == 0
        assert double.refresh(4) == 8
        assert cache["key-4"] == 8
        assert double(4) == 8


def test_planted_entry(tmp_path: pathlib.Path) -> None:
    """A cache entry that would run code when it is unpickled is recomputed instead."""

    class Planted:
        def __reduce__(self) -> tuple[object, ...]:
            return pytest.fail, ("a planted cache entry ran code",)

    calls: list[str] = []

    @qldpc.cache.use_disk_cache("test_name", cache_dir=tmp_path)
    def get_five(value: str) -> int:
        calls.append(value)
        return 5

    # anyone who can write to the cache directory can store a pickle with the default disk
    with diskcache_community.Cache(tmp_path / "test_name", disk=diskcache_community.Disk) as cache:
        cache["test_arg"] = Planted()

    with unittest.mock.patch("qldpc.cache._running_with_pytest", return_value=False):
        assert get_five("test_arg") == get_five("test_arg") == 5
    assert calls == ["test_arg"]


def test_uncached_calls(tmp_path: pathlib.Path) -> None:
    """Calls whose key or result the disk cache cannot store are computed, but not cached."""
    calls: list[int] = []

    @qldpc.cache.use_disk_cache("ones", cache_dir=tmp_path)
    def get_ones(size: int) -> galois.FieldArray:
        calls.append(size)
        return galois.GF(2).Ones(size)

    @qldpc.cache.use_disk_cache("doubles", cache_dir=tmp_path)
    def double(value: int) -> int:
        calls.append(value)
        return 2 * value

    with unittest.mock.patch("qldpc.cache._running_with_pytest", return_value=False):
        qldpc.cache.get_disk_cache("ones", cache_dir=tmp_path)[2] = "an older result"

        # a galois.FieldArray result is returned and never cached, and refreshing it drops the older
        # result; only the first such call warns
        with pytest.warns(UserWarning, match="subclass of ndarray") as record:
            arrays = [get_ones.refresh(2), get_ones(2), get_ones(2)]
        assert len(record) == 1
        assert all(type(ones) is galois.GF(2) and ones.tolist() == [1, 1] for ones in arrays)
        assert calls == [2, 2, 2]

        # a NumPy integer argument, which cannot be part of a key, gives the right result
        calls.clear()
        (three,) = np.array([3])
        with pytest.warns(UserWarning, match="key of type numpy") as record:
            doubles = [double(three), double(three)]
        assert len(record) == 1
        assert doubles == [6, 6]
        assert calls == [3, 3]


def test_pickle() -> None:
    """Cached module-level functions are pickled by reference."""
    function = qldpc.external._codes.get_guava_code
    assert pickle.dumps(function)
    assert copy.deepcopy(function) is function
    assert function.__wrapped__.__name__ == "get_guava_code"
