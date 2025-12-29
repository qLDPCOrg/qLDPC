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

import pyperclip

import qldpc

GAP_ROOT = os.path.join(os.path.dirname(os.path.dirname(qldpc.__file__)), "gap")


@functools.cache
def is_callable() -> bool:
    """Can we call GAP 4 from the command line?"""
    commands = ["gap", "-q", "-c", r'Print(GAPInfo.Version, "\n"); QUIT;']
    try:
        result = subprocess.run(commands, capture_output=True, text=True)
        version = result.stdout.strip()
        return bool(version) and bool(re.match(r"4\.\d+\.\d+", version))
    except FileNotFoundError:
        pass
    return False


@functools.cache
def is_installed() -> bool:
    """Is GAP 4 installed?"""
    if is_callable():
        return True
    print("GAP 4 cannot be called from the command line (with 'gap').")
    print("Can you manually copy/paste commands and outputs between here and GAP? [y/N]")
    answer = input().lower()
    return answer == "y" or answer == "yes"


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
    if not is_installed():
        raise FileNotFoundError("GAP 4 is required to proceed, but is not installed")

    if is_callable():
        commands = sanitize_commands(commands)
        shell_commands = [
            "gap",
            "-l",
            f";{GAP_ROOT}",
            "-q",
            "--quitonbreak",
            "-c",
            " ".join(commands),
        ]
        result = subprocess.run(shell_commands, capture_output=True, text=True)
        if result.stderr:
            raise ValueError(
                f"Error encountered when running GAP:\n{result.stderr}\n\n"
                f"GAP command:\n{' '.join(commands)}"
            )
        return result.stdout

    command = " ".join(commands)
    print("Run the following command in GAP:")
    print()
    print(command)
    print()

    cache_name = "gap_output"
    cache = qldpc.cache.get_disk_cache(cache_name)

    if output := cache.get(command, None):
        print("NOTICE: GAP command and output found in the local cache.  Retrieved output:")
        print("=" * 80)
        print(output)
        print("=" * 80)
        print()
        print(
            "If you think that the cached result is incorrect, you can remove it from the cache"
            " by running the following commands:\n"
            f'\nimport qldpc\nqldpc.cache.clear_entry("{cache_name}", """{command}""")\n'
        )
        return output

    pyperclip.copy(command)
    print("===============================================================================")
    print("NOTE:")
    try:
        pyperclip.copy(command)
        print("The above command has been copied to your system clipboard.")
        print("You can paste the command into GAP with ctrl+v or cmd+v.")
    except pyperclip.PyperclipException:  # pragma: no cover
        print("Failed to automatically copy the above command into your system clipboard.")
        print("See https://pyperclip.readthedocs.io/en/latest/index.html#not-implemented-error")
        print("Manually copy/paste the above command into GAP.")
    print("In turn, copy the resulting output from GAP and paste it here to continue.")
    print("Type an empty line (hit Enter twice) to finish.")
    print("===============================================================================")
    print()

    # read in GAP output
    lines = []
    while line := input():
        lines.append(line)
    output = "\n".join(lines)

    # save output to cache and return
    cache[command] = output
    return output


@functools.cache
def require_package(name: str, repo: str | None = None) -> bool:
    """Enforce the installation of a GAP package.

    Args:
        name: The GAP package name.
        repo: The repository from which to git clone the package, if necessary.
            Defaults to f"https://github.com/gap-packages/{name}" if no repository is provided.

    Raises:
        ValueError: If the package is not installed and an attempt to install it fails.

    Returns:
        True if the requirement is satisfied (raises an error otherwise).
    """
    availability = get_output(f'Print(TestPackageAvailability("{name.lower()}"));')

    if availability.strip() == "fail":
        repo = repo or f"https://github.com/gap-packages/{name}"
        if not is_callable():
            raise ModuleNotFoundError(
                f"GAP package '{name}' is required but not installed.\n"
                f"You may be able to find this pacakge at {repo}"
            )

        response = (
            input(f"GAP package '{name}' is required but not installed.  Try to install it? (Y/n)")
            .strip()
            .lower()
        )
        if not response or response == "y":
            commands = ["git", "clone", repo, os.path.join(GAP_ROOT, "pkg", name.lower())]
            print(" ".join(commands))
            install_result = subprocess.run(commands, capture_output=True, text=True)
            if install_result.returncode:
                raise ValueError(f"Failed to install {name}\n\n{install_result.stderr}")
        else:
            raise ValueError(f"Cannot proceed without the required package, {name}")

    return True
