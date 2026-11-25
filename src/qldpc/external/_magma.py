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
import warnings

from qldpc._utils import get_external_caller_stacklevel

# MAGMA prints these markers around the output of a command.  The script builds each marker from
# two strings, so that the script's own text never contains a marker.
_BEGIN_MARKER = "__qldpc_begin__"
_END_MARKER = "__qldpc_end__"

# MAGMA runs with this random seed, so that its randomized algorithms give reproducible results.
_SEED = 1

# MAGMA reports errors on lines such as "User error: ...", "Runtime error in ...", or "System error"
_ERROR_PATTERN = re.compile(r"^\w+ error\b", flags=re.MULTILINE)


class MAGMAError(ValueError):
    """MAGMA reported an error while running a command."""


class MAGMARunError(RuntimeError):
    """MAGMA stopped before finishing a command, without reporting an error."""


####################################################################################################
# calling MAGMA


def is_on_path() -> bool:
    """Is there a ``magma`` executable on ``PATH``?  Unlike :func:`is_callable`, never runs it."""
    return shutil.which("magma") is not None


@functools.cache
def is_callable() -> bool:
    """Can MAGMA be called from the command line (with ``magma``)?

    This check runs MAGMA once, so it also fails if MAGMA cannot start, such as for lack of a
    licence.  If ``magma`` is on ``PATH`` but fails the check, this function warns with the reason.
    It never prompts for input.
    """
    if not is_on_path():
        return False
    try:
        output = get_output("1 + 1;", timeout=60)
    except (subprocess.TimeoutExpired, MAGMAError, MAGMARunError) as error:
        reason = str(error)
    else:
        if output.split() == ["2"]:
            return True
        reason = f"MAGMA printed {output!r} for 1 + 1"
    warnings.warn(
        "A magma executable is on PATH, but a test run of it failed, so qLDPC does not run MAGMA"
        f" automatically.\n\n{reason}",
        stacklevel=get_external_caller_stacklevel(),
    )
    return False


def get_output(command: str, *, timeout: float | None = None) -> str:
    """Get the output that MAGMA prints when it runs the given command.

    This function writes the command to a temporary script file and runs ``magma`` on that file in
    a subprocess, with a fixed random seed, without the user's startup file, and with output lines
    of unlimited length.

    Raises:
        MAGMAError: If MAGMA reports an error, such as an error in the command.
        MAGMARunError: If MAGMA cannot start, or stops before finishing the command without
            reporting an error.
    """
    command = command.strip()
    if not command.endswith(";"):
        command += ";"
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "command.m")
        with open(path, "w") as file:
            file.write(_get_script(command))
        try:
            result = subprocess.run(
                [*_get_arguments(), path],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )
        except OSError as error:
            raise MAGMARunError(
                f"Cannot run MAGMA: {error}\n\nMAGMA command:\n{command}"
            ) from error
    return _parse_output(command, result.returncode, result.stdout, result.stderr)


####################################################################################################
# helper functions


def _get_arguments() -> list[str]:
    """The command line that runs MAGMA, without the script file.

    The option -b suppresses MAGMA's banner and exit summary, -n skips the user's startup file, and
    -S sets the random seed.
    """
    return ["magma", "-b", "-n", "-S", str(_SEED)]


def _get_script(command: str) -> str:
    """A MAGMA script that runs a command, prints markers around its output, and quits.

    The script stops at the first error, does not echo itself, and neither wraps nor pages output.
    """
    lines = [
        "SetQuitOnError(true);",
        "SetEchoInput(false);",
        "SetAutoColumns(false);",
        "SetColumns(0);",
        "SetRows(0);",
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
    details = f"MAGMA output:\n{stdout}"
    if stderr:
        details += f"\n\nMAGMA standard error:\n{stderr}"
    details += f"\n\nMAGMA command:\n{command}"

    match = re.search(
        rf"^{_BEGIN_MARKER}$\n(.*?)^{_END_MARKER}$", stdout, flags=re.MULTILINE | re.DOTALL
    )
    if match and not _ERROR_PATTERN.search(match.group(1)):
        return match.group(1)
    if _ERROR_PATTERN.search(stdout):
        raise MAGMAError(f"MAGMA reported an error\n\n{details}")
    raise MAGMARunError(
        f"MAGMA stopped before finishing a command (exit code {returncode})\n\n{details}"
    )
