# SPDX-License-Identifier: Apache-2.0

"""The settings and paths that the modules of the certification experiment share.

``CONFIG`` is the experiment's ``config.json``.  ``DATA`` is its data directory,
``circuit_data/``, which Git ignores, and ``INPUTS`` the directory of the certification inputs in
it, with their record ``circuits.json`` (see ``circuits.py``).
"""

from __future__ import annotations

import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text())
DATA = HERE / "circuit_data"
INPUTS = DATA / "inputs"
CIRCUITS_JSON = INPUTS / "circuits.json"
