"""Unit tests for cache.py

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
from collections.abc import Hashable

import pytest

import qldpc.cache


def test_pytest() -> None:
    """We are running with pytest."""
    assert qldpc.cache.running_with_pytest()

    def test_func() -> None: ...

    assert qldpc.cache.use_disk_cache("test")(test_func) is test_func
    assert qldpc.cache.get_disk_cache("test") == {}


def test_use_disk_cache() -> None:
    """Cache function outputs."""

    cache: dict[Hashable, int] = {}
    with (
        unittest.mock.patch("qldpc.cache.running_with_pytest", return_value=False),
        unittest.mock.patch("diskcache.Cache", return_value=cache),
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
