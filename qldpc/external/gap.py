"""Module for communicating with the GAP computer algebra system

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
import re
import subprocess
from collections.abc import Sequence

import qldpc

GAP_ROOT = os.path.join(os.path.dirname(os.path.dirname(qldpc.__file__)), "gap")


@functools.cache
def is_installed() -> bool:
    """Is GAP 4 installed?"""
    commands = ["gap", "-q", "-c", r'Print(GAPInfo.Version, "\n"); QUIT;']
    try:
        result = subprocess.run(commands, capture_output=True, text=True)
        return bool(re.match(r"4\.[0-9]+\.[0-9]", result.stdout.strip()))
    except Exception:
        return False


def sanitize_commands(commands: Sequence[str]) -> tuple[str, ...]:
    """Sanitize GAP commands: don't format Print statements, and quit at the end."""
    stream = "__stream__"
    prefix = [
        f"{stream} := OutputTextUser();",
        f"SetPrintFormattingStatus({stream},false);",
    ]
    suffix = ["QUIT;"]
    commands = [cmd.replace("Print(", f"PrintTo({stream}, ") for cmd in commands]
    return tuple(prefix + commands + suffix)


def get_output(*commands: str) -> str:
    """Get the output from the given GAP commands."""
    commands = sanitize_commands(commands)
    shell_commands = ["gap", "-l", f";{GAP_ROOT}", "-q", "--quitonbreak", "-c", " ".join(commands)]
    result = subprocess.run(shell_commands, capture_output=True, text=True)
    if result.stderr:
        raise ValueError(
            f"Error encountered when running GAP:\n{result.stderr}\n\n"
            f"GAP command:\n{' '.join(commands)}"
        )
    return result.stdout


@functools.cache
def require_package(name: str, repo: str | None = None) -> None:
    """Enforce the installation of a GAP package.

    Args:
        name: The GAP package name.
        repo: The repository from which to git clone the package, if necessary.
            Defaults to f"https://github.com/gap-packages/{name}" if no repository is provided.

    Raises:
        ValueError: If the package is not installed and an attempt to install it fails.
    """
    if not is_installed():
        raise FileNotFoundError(
            "GAP 4 is required, but an installation cannot be found\n"
            "GAP should be accessible from the command line with the 'gap' command"
        )

    availability = get_output(f'Print(TestPackageAvailability("{name.lower()}"));')
    if availability == "fail":
        response = (
            input(f"GAP package {name} is required but not installed.  Try to install it? (Y/n)")
            .strip()
            .lower()
        )
        if not response or response == "y":
            repo = repo or f"https://github.com/gap-packages/{name}"
            commands = ["git", "clone", repo, os.path.join(GAP_ROOT, "pkg", name.lower())]
            print(" ".join(commands))
            install_result = subprocess.run(commands, capture_output=True, text=True)
            if install_result.stderr:
                raise ValueError(f"Failed to install {name}\n\n{install_result.stderr}")
        else:
            raise ValueError(f"Cannot proceed without the required package ({name})")
