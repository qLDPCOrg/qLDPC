# Workarounds for bugs of numba

This guide lists the code in qLDPC that exists only because of a bug of [numba](https://github.com/numba/numba), with one section for each bug, and says how to remove the code once numba is fixed.

Each site carries a marker that names its section, so `git grep "TEMPORARY until"` lists every site, apart from passages of docstrings that describe a workaround, which the section's steps name.
A function or class with a docstring that the migration deletes has the marker right after its docstring's summary line.
Anything else that the migration deletes has the marker as a comment above it:

```python
# TEMPORARY until <trigger>: see numba_workarounds.md, "<section title>".
```

A trigger holds once qLDPC's `pyproject.toml` requires a numba release that does what it says.
To migrate, follow the section's steps, delete the imports that become unused, and delete the section.

## Clashing names of code compiled in different processes

Numba names compiled code by function name, argument types, and a count that starts at one in each process ([numba/numba#11016](https://github.com/numba/numba/issues/11016)).
Functions that numba generates from one template, such as the sorts behind `np.sort` and `np.argsort`, can thus get the same name for different code in different processes, and a process that loads both from the disk cache runs one in place of the other.
So `_compilation` starts the count at a random number in each process.
The count is not part of the key by which numba's disk cache finds compiled code, so starting it at a random number does not stop later processes from reusing cached code.

**Trigger:** numba fixes numba/numba#11016.

**Migration:**

1. In `_compilation`, delete `_work_around_numba_name_clashes`, `_NUMBA_NAME_CLASHES_AVOIDED`, the `if` statement that calls `os.register_at_fork`, and the check that opens `_compile_with_cache_in`.
2. In `_compilation_test`, delete `test_code_compiled_in_different_processes_has_distinct_names`, `test_compiled_code_is_not_cached_without_the_workaround_for_a_numba_bug`, `_PRINT_SYMBOL_NAMES`, and `_IN_PARENT_AND_FORKED_CHILD`.

## Data files that hold code for another key

Numba saves the code that it compiles for each key, which includes the argument types, in a data file, and keeps an index that maps each key to its data file.
It names the data file of new code by the lowest number that the index it read does not use, and nothing stops another process from writing the index between its read and its write ([numba/numba#10862](https://github.com/numba/numba/issues/10862)).
Two processes that compile a function for different argument types at once can thus write the same data file, and leave the index pointing at code compiled for the other argument types, which can raise an error or silently return wrong results.
Environments with different versions of numba that share a cache directory also write each other's data files, since each reads only an index that its own version wrote.
So `_KeyCheckedCacheFile` saves numba's version and the key with the code, and loads the code only if both match.

**Trigger:** numba loads a data file only if its own version saved it for the key that it loads (numba/numba#10862).

**Migration:**

1. In `_compilation`, delete `_KeyCheckedCacheFile`, and the paragraph of `_compile_with_cache_in` that changes the class of numba's cache-file object, except its `return dispatcher`.
2. Delete what the docstrings of `always_compiled` and `_compile_with_cache_in` say about that object, so that `always_compiled` says "If the directory cannot be used, qLDPC warns once, and every process compiles the function again.", and delete the paragraph of the module docstring that points to `_KeyCheckedCacheFile`.
3. In `_compilation_test`, delete `test_versions_compiled_at_the_same_time_are_loaded_only_for_their_argument_types`, `test_code_saved_by_another_version_of_numba_is_compiled_again`, `test_a_function_is_not_cached_if_numba_reads_its_files_through_another_class`, and `_get_top_right_entry`.
