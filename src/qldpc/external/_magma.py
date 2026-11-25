# SPDX-License-Identifier: Apache-2.0

"""Running the MAGMA computer algebra system from the command line.

See https://magma.maths.usyd.edu.au.  The command-line options that this module uses are documented
in the "Command Line Options" section of the MAGMA handbook.
"""

from __future__ import annotations

import functools
import os
import re
import shutil
import subprocess
import tempfile

# MAGMA prints these markers around the output of a command.  The script builds each marker from
# two strings, so that the script's own text never contains a marker.
_BEGIN_MARKER = "__qldpc_begin__"
_END_MARKER = "__qldpc_end__"

# The script fixes the random seed, so that runs of MAGMA's randomized algorithms are reproducible.
_SEED = 1


class MAGMAError(ValueError):
    """MAGMA reported an error in a command."""


class MAGMARunError(RuntimeError):
    """MAGMA did not finish running a command, without reporting an error in the command."""


####################################################################################################
# calling MAGMA


def is_on_path() -> bool:
    """Is there a ``magma`` executable on ``PATH``?  Unlike :func:`is_callable`, never runs it."""
    return shutil.which("magma") is not None


@functools.cache
def is_callable() -> bool:
    """Can MAGMA be called from the command line (with ``magma``)?

    This check runs MAGMA once, so it also fails if MAGMA cannot start, such as for lack of a
    licence.  It never prompts for input.
    """
    if not is_on_path():
        return False
    try:
        return get_output("1 + 1;", timeout=60).split() == ["2"]
    except (OSError, subprocess.TimeoutExpired, MAGMAError, MAGMARunError):
        return False


def get_output(command: str, *, timeout: float | None = None) -> str:
    """Get the output that MAGMA prints when it runs the given command.

    This function writes the command to a temporary script file and runs ``magma`` on that file in
    a subprocess, with a fixed random seed, without the user's startup file, and with output lines
    of unlimited length.

    Raises:
        MAGMAError: If MAGMA reports an error in the command.
        MAGMARunError: If MAGMA does not finish running the command for any other reason.
    """
    command = command.strip()
    if not command.endswith(";"):
        command += ";"
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "command.m")
        with open(path, "w") as file:
            file.write(_get_script(command))
        # -b suppresses MAGMA's banner and exit summary, -n skips the user's startup file, and -S
        # sets the random seed
        result = subprocess.run(
            ["magma", "-b", "-n", "-S", str(_SEED), path],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    return _parse_output(command, result.returncode, result.stdout, result.stderr)


####################################################################################################
# helper functions


def _get_script(command: str) -> str:
    """A MAGMA script that runs a command, prints markers around its output, and quits.

    The script stops at the first error, does not echo itself, and does not wrap output lines.
    """
    lines = [
        "SetQuitOnError(true);",
        "SetEchoInput(false);",
        "SetColumns(0);",
        _get_print_marker_statement(_BEGIN_MARKER),
        command,
        _get_print_marker_statement(_END_MARKER),
        "quit;",
    ]
    return "\n".join(lines) + "\n"


def _get_print_marker_statement(marker: str) -> str:
    """A MAGMA statement that prints a marker, without the marker appearing in its own text."""
    middle = len(marker) // 2
    return f'print "{marker[:middle]}" cat "{marker[middle:]}";'


def _parse_output(command: str, returncode: int, stdout: str, stderr: str) -> str:
    """Extract the output of a command from the output of a MAGMA script that runs it.

    The markers show that MAGMA ran the command to the end, so the output of a finished command is
    returned whatever MAGMA's exit status, which the MAGMA handbook does not document.
    """
    match = re.search(
        rf"^{_BEGIN_MARKER}$\n(.*?)^{_END_MARKER}$", stdout, flags=re.MULTILINE | re.DOTALL
    )
    if match:
        return match.group(1)

    # MAGMA reports errors in commands on lines such as "User error: ..." or "Runtime error ..."
    details = f"MAGMA output:\n{stdout}"
    if stderr:
        details += f"\n\nMAGMA standard error:\n{stderr}"
    details += f"\n\nMAGMA command:\n{command}"
    if re.search(r"^\w+ error\b", stdout, flags=re.MULTILINE):
        raise MAGMAError(f"MAGMA reported an error\n\n{details}")
    raise MAGMARunError(
        f"MAGMA did not finish running a command (exit code {returncode})\n\n{details}"
    )
