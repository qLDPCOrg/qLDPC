# SPDX-License-Identifier: Apache-2.0

"""Copies of modules of checks-superstaq that its latest release lacks (see README.md).

Importing this package checks that the copies are unchanged, and puts the directory of the check
scripts on ``PYTHONPATH``, so that the pytest processes of the coverage check can import the pytest
plugin of this package.
"""

import hashlib
import os
import pathlib

# The git blob hash of each copy, as in the upstream commit that README.md names.
UPSTREAM_HASHES = {
    "coverage_.py": "1cd15d331dec7aa7e308856f86f3307b3cfa8cce",
    "_coverage_contexts.py": "a6120c42310e64a174c84f6ced49318acfec286d",
}


def get_blob_hash(path: pathlib.Path) -> str:
    """The git blob hash of a file, with Windows line endings read as Unix ones."""
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()


# Fail if a copy was edited.
_PACKAGE = pathlib.Path(__file__).parent
for _name, _hash in UPSTREAM_HASHES.items():
    if get_blob_hash(_PACKAGE / _name) != _hash:
        raise ImportError(
            f"checks/_superstaq_coverage/{_name} differs from the upstream file that it"
            " copies; change that file upstream, and copy it anew (see"
            " checks/_superstaq_coverage/README.md)"
        )

# Let the pytest processes import the plugin of this package.
_CHECKS = str(_PACKAGE.parent)
if _CHECKS not in os.environ.get("PYTHONPATH", "").split(os.pathsep):
    os.environ["PYTHONPATH"] = os.pathsep.join(
        filter(None, [_CHECKS, os.environ.get("PYTHONPATH")])
    )
