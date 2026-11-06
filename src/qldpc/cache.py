# SPDX-License-Identifier: Apache-2.0

"""Persistent disk caching for the results of expensive computations."""

from __future__ import annotations

import functools
import pathlib
import sys
import warnings
from collections.abc import Callable, Hashable, MutableMapping
from typing import Any, Generic, ParamSpec, TypeVar, cast

import diskcache
import platformdirs

_Params = ParamSpec("_Params")
_Result = TypeVar("_Result")


def use_disk_cache(
    cache_name: str,
    *,
    cache_dir: pathlib.Path | str | None = None,
    key_func: Callable[..., Hashable] | None = None,
) -> Callable[[Callable[_Params, _Result]], CachedFunction[_Params, _Result]]:
    """Decorator to cache results to disk.

    By default, the cache key is the tuple of positional arguments followed by (keyword, value)
    pairs, unpacked if this tuple has length 1.  A custom key_func, called with the same arguments
    as the decorated function, overrides this default.

    The decorated function has a .refresh method with the same signature, which ignores any existing
    cache entry, recomputes the result, and saves the new result to the cache.  Refreshing only
    affects this cache, and does not clear other caches used internally by the function.

    The decorator is intended for module-level functions: a decorated function is pickled by
    reference to its qualified name, and it does not bind as a method.

    Disk caching is bypassed when running with pytest.
    """

    def decorator(function: Callable[_Params, _Result]) -> CachedFunction[_Params, _Result]:
        return CachedFunction(function, cache_name, cache_dir=cache_dir, key_func=key_func)

    return decorator


class CachedFunction(Generic[_Params, _Result]):
    """A function whose results are cached to disk.

    Calling the function retrieves its result from the cache if available, and otherwise computes
    and caches the result.  Calling CachedFunction.refresh ignores any existing cache entry,
    recomputes the result, and saves the new result to the cache.
    """

    __wrapped__: Callable[_Params, _Result]

    def __init__(
        self,
        function: Callable[_Params, _Result],
        cache_name: str,
        *,
        cache_dir: pathlib.Path | str | None = None,
        key_func: Callable[..., Hashable] | None = None,
    ) -> None:
        self._function = function
        self._cache_name = cache_name
        self._cache_dir = cache_dir
        self._key_func = key_func
        functools.update_wrapper(self, function)

    def __call__(self, *args: _Params.args, **kwargs: _Params.kwargs) -> _Result:
        """Retrieve a result from the cache if available, and otherwise compute and cache it."""
        cache = get_disk_cache(self._cache_name, cache_dir=self._cache_dir)
        key = self._get_key(*args, **kwargs)
        if key in cache:
            return cast(_Result, cache[key])
        result = self._function(*args, **kwargs)
        cache[key] = result
        return result

    def refresh(self, *args: _Params.args, **kwargs: _Params.kwargs) -> _Result:
        """Recompute the result for these arguments and overwrite the cached value."""
        cache = get_disk_cache(self._cache_name, cache_dir=self._cache_dir)
        key = self._get_key(*args, **kwargs)
        result = self._function(*args, **kwargs)
        cache[key] = result
        return result

    def __reduce__(self) -> str:
        """Pickle by reference to the module-level name of the decorated function."""
        return self._function.__qualname__

    def _get_key(self, *args: _Params.args, **kwargs: _Params.kwargs) -> Hashable:
        """Get the cache key for the given arguments."""
        if self._key_func is not None:
            return self._key_func(*args, **kwargs)
        key = args + tuple(kwargs.items())
        return key if len(key) != 1 else key[0]  # unpack length-1 tuples


def get_disk_cache(
    cache_name: str, *, cache_dir: pathlib.Path | str | None = None
) -> MutableMapping[Hashable, Any]:
    """Retrieve a dictionary-like cache object.

    While pytest is imported, this returns a new, empty dict rather than a persistent disk cache.
    """
    if _running_with_pytest():
        return {}
    return diskcache.Cache(get_disk_cache_path(cache_name, cache_dir=cache_dir))


def get_disk_cache_path(
    cache_name: str, *, cache_dir: pathlib.Path | str | None = None
) -> pathlib.Path:
    """Retrieve the path of a cache."""
    cache_dir = cache_dir or pathlib.Path(platformdirs.user_cache_dir()) / "qldpc"
    return pathlib.Path(cache_dir) / cache_name


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


def _running_with_pytest() -> bool:
    """Are we currently running with pytest?"""
    return "pytest" in sys.modules
