"""Helper function(s) for caching results

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

import functools
import os
import sys
import warnings
from collections.abc import Callable, Hashable
from typing import Any, ParamSpec

import diskcache
import platformdirs

Params = ParamSpec("Params")


def get_disk_cache_path(cache_name: str, *, cache_dir: str | None = None) -> str:
    """Retrieve the path of a cache."""
    cache_dir = cache_dir or os.path.join(platformdirs.user_cache_dir(), "qldpc")
    return os.path.join(cache_dir, cache_name)


def get_disk_cache(cache_name: str, *, cache_dir: str | None = None) -> diskcache.Cache:
    """Retrieve a dictionary-like cache object."""
    if running_with_pytest():
        return {}
    return diskcache.Cache(get_disk_cache_path(cache_name, cache_dir=cache_dir))


def use_disk_cache(
    cache_name: str,
    *,
    cache_dir: str | None = None,
    key_func: Callable[..., Hashable] | None = None,
) -> Callable[[Callable[Params, Any]], Callable[Params, Any]]:
    """Decorator to cache results to disk."""

    def decorator(function: Callable[Params, Any]) -> Callable[Params, Any]:
        if running_with_pytest():
            return function

        @functools.wraps(function)
        def function_with_cache(*args: Params.args, **kwargs: Params.kwargs) -> Any:
            # retrieve results from cache, if available
            cache = get_disk_cache(cache_name, cache_dir=cache_dir)
            if key_func is not None:
                key = key_func(*args, **kwargs)
            else:
                key = args + tuple(kwargs.items())
                key = key if len(key) != 1 else key[0]  # unpack length-1 tuples
            if key in cache:
                return cache[key]

            # compute results and save to cache
            result = function(*args, **kwargs)
            cache[key] = result
            return result

        return function_with_cache

    return decorator


def running_with_pytest() -> bool:
    """Are we currently running  with pytest?"""
    return "pytest" in sys.modules


def clear_entry(cache_name: str, key: Hashable, *, cache_dir: str | None = None) -> None:
    """Clear an entry from a local cache."""
    cache = get_disk_cache(cache_name, cache_dir=cache_dir)
    if key in cache:
        del cache[key]
    else:
        cache_path = get_disk_cache_path(cache_name, cache_dir=cache_dir)
        warnings.warn(
            f"Attempted to delete the entry '{key}' from the cache '{cache_name}'"
            + ("" if cache_dir is None else f" (located at '{cache_path}')")
            + ", but this entry does not exist.",
            stacklevel=2,
        )
