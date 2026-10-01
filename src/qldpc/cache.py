# SPDX-License-Identifier: Apache-2.0

"""Helper function(s) for caching results."""

from __future__ import annotations

import functools
import pathlib
import sys
import warnings
from collections.abc import Callable, Hashable
from typing import ParamSpec, Protocol, TypeVar, cast

import diskcache
import platformdirs

Params = ParamSpec("Params")
Result = TypeVar("Result")
Result_co = TypeVar("Result_co", covariant=True)


class CachedFunction(Protocol[Params, Result_co]):
    """A function whose results are cached to disk.

    Calling the function retrieves its result from the cache if available, and otherwise computes
    and caches the result.  Calling CachedFunction.refresh ignores any existing cache entry,
    recomputes the result, and saves the new result to the cache.
    """

    def __call__(self, *args: Params.args, **kwargs: Params.kwargs) -> Result_co: ...

    def refresh(self, *args: Params.args, **kwargs: Params.kwargs) -> Result_co:
        """Recompute the result for these arguments and overwrite the cached value."""


def get_disk_cache_path(
    cache_name: str, *, cache_dir: pathlib.Path | str | None = None
) -> pathlib.Path:
    """Retrieve the path of a cache."""
    cache_dir = cache_dir or pathlib.Path(platformdirs.user_cache_dir()) / "qldpc"
    return pathlib.Path(cache_dir) / cache_name


def get_disk_cache(
    cache_name: str, *, cache_dir: pathlib.Path | str | None = None
) -> diskcache.Cache:
    """Retrieve a dictionary-like cache object."""
    if running_with_pytest():
        return {}
    return diskcache.Cache(get_disk_cache_path(cache_name, cache_dir=cache_dir))


def use_disk_cache(
    cache_name: str,
    *,
    cache_dir: pathlib.Path | str | None = None,
    key_func: Callable[..., Hashable] | None = None,
) -> Callable[[Callable[Params, Result]], CachedFunction[Params, Result]]:
    """Decorator to cache results to disk.

    By default, the cache key is the tuple of positional arguments followed by (keyword, value)
    pairs, unpacked if this tuple has length 1.  A custom key_func, called with the same arguments
    as the decorated function, overrides this default.

    The decorated function has a .refresh method with the same signature, which ignores any
    existing cache entry, recomputes the result, and saves the new result to the cache.  Refreshing
    only affects this cache, and does not clear other caches used internally by the function.

    Disk caching is bypassed when running with pytest.
    """

    def decorator(function: Callable[Params, Result]) -> CachedFunction[Params, Result]:
        def get_key(*args: Params.args, **kwargs: Params.kwargs) -> Hashable:
            if key_func is not None:
                return key_func(*args, **kwargs)
            key = args + tuple(kwargs.items())
            return key if len(key) != 1 else key[0]  # unpack length-1 tuples

        @functools.wraps(function)
        def function_with_cache(*args: Params.args, **kwargs: Params.kwargs) -> Result:
            # retrieve results from cache, if available
            cache = get_disk_cache(cache_name, cache_dir=cache_dir)
            key = get_key(*args, **kwargs)
            if key in cache:
                return cast(Result, cache[key])

            # compute results and save to cache
            result = function(*args, **kwargs)
            cache[key] = result
            return result

        def refresh(*args: Params.args, **kwargs: Params.kwargs) -> Result:
            cache = get_disk_cache(cache_name, cache_dir=cache_dir)
            key = get_key(*args, **kwargs)
            result = function(*args, **kwargs)
            cache[key] = result
            return result

        refresh.__doc__ = CachedFunction.refresh.__doc__
        function_with_cache.refresh = refresh  # type: ignore[attr-defined]
        return cast(CachedFunction[Params, Result], function_with_cache)

    return decorator


def running_with_pytest() -> bool:
    """Are we currently running with pytest?"""
    return "pytest" in sys.modules


def clear_entry(
    cache_name: str, key: Hashable, *, cache_dir: pathlib.Path | str | None = None
) -> None:
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
