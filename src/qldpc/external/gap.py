# SPDX-License-Identifier: Apache-2.0

"""Module for communicating with the GAP computer algebra system.

See https://www.gap-system.org.
"""

from __future__ import annotations

import functools
import importlib
import os
import re
import subprocess
from collections.abc import Callable, Sequence
from typing import Protocol, cast

import pyperclip

import qldpc

GAP_ROOT = os.path.join(os.path.dirname(os.path.dirname(qldpc.__file__)), "gap")


class _LibGap(Protocol):
    """Subset of the optional ``passagemath-gap`` API used by qLDPC."""

    def eval(self, command: str) -> object:
        """Evaluate a GAP expression."""

    def function_factory(self, function_name: str) -> Callable[..., object]:
        """Return a callable GAP function."""


_libgap_packages: dict[str, tuple[str, str | None]] = {}
_subprocess_packages: set[str] = set()

# PassageMath retains warnings from successful evaluations in its error stream.
_RESET_LIBGAP_ERROR_OUTPUT = """CallFuncList(function()
    CloseStream(ERROR_OUTPUT);
    MakeReadWriteGlobal("ERROR_OUTPUT");
    libgap_errout := "";
    ERROR_OUTPUT := OutputTextString(libgap_errout, false);
    MakeReadOnlyGlobal("ERROR_OUTPUT");
end, [])"""


def _get_libgap() -> _LibGap | None:
    """Return the optional direct GAP binding, if it is importable.

    PassageMath's own library-level GAP feature uses ``sage.all__sagemath_gap`` before importing
    ``sage.libs.gap.libgap``.  Follow that supported order rather than importing the
    interactive-only ``passagemath_gap`` convenience namespace.  See the import-cycle workaround
    in PassageMath's ``sage.features.gap`` (commit
    https://github.com/passagemath/passagemath/commit/f35b8847b4226074d84c4645ea2c6eacab69e70b).
    """
    try:
        importlib.import_module("sage.all__sagemath_gap")
        module = importlib.import_module("sage.libs.gap.libgap")
    except (ImportError, OSError):
        return None
    return cast(_LibGap | None, getattr(module, "libgap", None))


@functools.cache
def _is_gap_executable() -> bool:
    """Can the GAP executable be used for package installation and evaluation?"""
    commands = ["gap", "-q", "-c", r'Print(GAPInfo.Version, "\n");; QUIT;;']
    try:
        result = subprocess.run(commands, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return False
    return bool(re.match(r"4\.\d+\.\d+", result.stdout.strip()))


@functools.cache
def is_callable() -> bool:
    """Can we call GAP 4, directly through libgap or from the command line?"""
    if _get_libgap() is not None:
        return True

    commands = ["gap", "-q", "-c", r'Print(GAPInfo.Version, "\n");; QUIT;;']
    try:
        result = subprocess.run(commands, capture_output=True, text=True, check=False)
        version = result.stdout.strip()
        return bool(version) and bool(re.match(r"4\.\d+\.\d+", version))
    except FileNotFoundError:
        pass
    return False


@functools.cache
def is_installed() -> bool:
    """Is GAP 4 available?

    When neither libgap nor the command-line program is available, this function prompts on standard
    input, so it blocks in a non-interactive context (and raises ``EOFError`` if standard input is
    closed).
    """
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
        f"{stream} := OutputTextUser();;",
        f"SetPrintFormattingStatus({stream},false);;",
    ]
    suffix = ["QUIT;;"]
    commands = [cmd.replace("Print(", f"PrintTo({stream}, ") for cmd in commands]
    return tuple(prefix + commands + suffix)


def _gap_string(value: object) -> str:
    """Convert a libgap value to the corresponding Python string."""
    if isinstance(value, str):
        return value
    try:
        converted = value.sage()  # type: ignore[attr-defined]
    except (AttributeError, NotImplementedError, ValueError):
        converted = str(value)
    return converted if isinstance(converted, str) else str(converted)


def _get_output_libgap(commands: Sequence[str], libgap: _LibGap) -> str:
    """Evaluate commands through libgap while capturing GAP's printed output."""
    stream = "__qldpc_output__"
    stream_object = f"{stream}_stream"
    body = [
        f"local {stream}, {stream_object};",
        f'{stream} := "";',
        f"{stream_object} := OutputTextString({stream}, false);",
        f"SetPrintFormattingStatus({stream_object}, false);",
    ]
    for command in commands:
        command = command.rstrip()
        if not command.endswith(";") and not re.search(r"(?:do|then|else|repeat)$", command):
            command += ";"
        body.append(command.replace("Print(", f"PrintTo({stream_object}, "))
    body.extend([f"CloseStream({stream_object});", f"return {stream};"])
    direct_command = "CallFuncList(function()\n" + "\n".join(body) + "\nend, [])"

    try:
        libgap.eval(_RESET_LIBGAP_ERROR_OUTPUT)
        result = libgap.eval(direct_command)
        libgap.eval(_RESET_LIBGAP_ERROR_OUTPUT)
    except Exception as error:
        raise ValueError(
            "Error encountered when running GAP through libgap\n\n"
            f"{error}\n\nGAP command:\n{' '.join(commands)}"
        ) from error
    return _gap_string(result)


def get_output(*commands: str, use_pipe: bool = False) -> str:
    """Get the output from the given GAP commands.

    When ``passagemath-gap`` is installed, this function evaluates commands through its in-process
    libgap binding.  Otherwise, when GAP is callable, it runs the commands in a subprocess.
    Finally, it falls back to a manual workflow that prints the commands, copies them to the system
    clipboard, and reads the pasted output from standard input (blocking, and raising ``EOFError``
    if standard input is closed), caching the result to disk.

    Raises:
        FileNotFoundError: If GAP 4 is not installed.
        ValueError: If GAP reports an error (a nonzero exit code or output on standard error).
    """
    if not is_installed():
        raise FileNotFoundError("GAP 4 is required to proceed, but is not installed")

    if not _subprocess_packages and (libgap := _get_libgap()) is not None:
        return _get_output_libgap(commands, libgap)

    if is_callable():
        return _get_output_subprocess(commands, use_pipe)

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


def _get_output_subprocess(commands: Sequence[str], use_pipe: bool = False) -> str:
    """Evaluate commands through the GAP executable."""
    commands = sanitize_commands(commands)
    shell_commands = [
        "gap",
        "-l",
        f";{GAP_ROOT}",
        "-q",
        "--quitonbreak",
    ]
    script_input = " ".join(commands)
    if not use_pipe:
        shell_commands.extend(["-c", script_input])
        pipe_input = None
    else:
        pipe_input = script_input
    result = subprocess.run(
        shell_commands,
        input=pipe_input,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode or result.stderr:
        parts = [f"Error encountered when running GAP (exit code {result.returncode})"]
        if result.stderr:
            parts.append(result.stderr)
        parts.append(f"GAP command:\n{' '.join(commands)}")
        raise ValueError("\n\n".join(parts))
    return result.stdout


def _confirm_package_install(name: str) -> None:
    """Ask the user for permission to install a missing GAP package."""
    response = (
        input(f"GAP package '{name}' is required but not installed.  Try to install it? (Y/n)")
        .strip()
        .lower()
    )
    if response not in {"", "y", "yes"}:
        raise ValueError(f"Cannot proceed without the required package, {name}")


def _install_package_subprocess(name: str, repo: str) -> None:
    """Install a GAP package for the executable backend."""
    commands = ["git", "clone", repo, os.path.join(GAP_ROOT, "pkg", name.lower())]
    print(" ".join(commands))
    install_result = subprocess.run(commands, capture_output=True, text=True, check=False)
    if install_result.returncode:
        raise ValueError(f"Failed to install {name}\n\n{install_result.stderr}")


def _require_package_subprocess(
    name: str,
    repo: str | None,
    availability: str | None = None,
    *,
    confirm_install: bool = True,
) -> bool:
    """Ensure a GAP package is available through the executable backend."""
    if availability is None:
        availability = get_output(f'Print(TestPackageAvailability("{name.lower()}"));;')
    if availability.strip() == "fail":
        repo = repo or f"https://github.com/gap-packages/{name}"
        if confirm_install:
            _confirm_package_install(name)
        _install_package_subprocess(name, repo)
    return True


def _get_package_manager_source(name: str, repo: str | None) -> str:
    """Return a source that GAP's PackageManager recognizes."""
    if repo is None:
        return name
    source = repo.rstrip("/")
    recognized_suffixes = (".git", ".hg", ".tar.gz", ".tar.bz2", "PackageInfo.g")
    return source if source.endswith(recognized_suffixes) else f"{source}.git"


def _get_libgap_install_instructions(name: str, source: str) -> str:
    """Explain how to install a package into libgap's user package directory."""
    return (
        f"To install '{name}' for libgap manually, run `sage -gap` from the Python environment "
        "where qldpc[gap] is installed, then enter:\n"
        'LoadPackage("PackageManager");\n'
        f'InstallPackage("{source}");'
    )


def _install_package_libgap(name: str, repo: str | None, libgap: _LibGap) -> bool:
    """Install a GAP package into the user package directory visible to libgap."""
    source = _get_package_manager_source(name, repo)
    instructions = _get_libgap_install_instructions(name, source)
    try:
        manager = libgap.function_factory("LoadPackage")("PackageManager")
    except Exception as error:
        raise ModuleNotFoundError(
            f"Could not load GAP's PackageManager through libgap.\n{instructions}"
        ) from error
    if _gap_string(manager).strip().lower() == "fail":
        raise ModuleNotFoundError(
            f"GAP's PackageManager is unavailable through libgap.\n{instructions}"
        )

    try:
        installed = libgap.function_factory("InstallPackage")(source, libgap.eval("false"))
    except Exception as error:
        raise ValueError(f"Failed to install {name} through libgap.\n{instructions}") from error
    if _gap_string(installed).strip().lower() in {"fail", "false"}:
        raise ValueError(f"Failed to install {name} through libgap.\n{instructions}")

    try:
        availability = libgap.function_factory("TestPackageAvailability")(name.lower())
    except Exception as error:
        raise ValueError(f"Could not verify the libgap installation of {name}") from error
    if _gap_string(availability).strip().lower() == "fail":
        raise ValueError(
            f"GAP installed {name}, but the package is still unavailable to libgap.\n{instructions}"
        )
    return True


@functools.cache
def require_package(name: str, repo: str | None = None) -> bool:
    """Enforce the installation of a GAP package.

    With the direct libgap backend, this checks the package through GAP. A missing package prompts
    on standard input and, with the user's consent, is installed through GAP's PackageManager into
    the user package directory visible to libgap. If that installation fails but a GAP executable
    is available, qLDPC prints instructions for installing the package for libgap before falling
    back to the executable backend. With the executable backend, qLDPC installs a missing package
    by cloning its repository into the GAP root's ``pkg`` directory.

    Args:
        name: The GAP package name.
        repo: The package repository to install, if necessary. With libgap, this is passed to GAP's
            PackageManager; with the executable backend, it is cloned with Git. Defaults to the
            package name for PackageManager or f"https://github.com/gap-packages/{name}" for Git.

    Raises:
        ModuleNotFoundError: If the package is missing and GAP cannot be called to install it.
        ValueError: If the user declines to install a missing package, or the installation fails.

    Returns:
        True if the requirement is satisfied (raises an error otherwise).
    """
    libgap = None if _subprocess_packages else _get_libgap()
    if libgap is not None:
        try:
            availability = libgap.function_factory("TestPackageAvailability")(name.lower())
        except Exception as error:
            raise ValueError(f"Could not check GAP package availability for {name}") from error
        if str(availability).lower() != "fail":
            _libgap_packages[name.lower()] = (name, repo)
            return True

        _confirm_package_install(name)
        try:
            installed = _install_package_libgap(name, repo, libgap)
        except (ModuleNotFoundError, ValueError) as error:
            if not _is_gap_executable():
                raise
            print(error)
            print("Falling back to the GAP executable for this package.")
        else:
            _libgap_packages[name.lower()] = (name, repo)
            return installed

        requirements = dict(_libgap_packages)
        requirements[name.lower()] = (name, repo)
        for package, package_repo in requirements.values():
            availability = _get_output_subprocess(
                (f'Print(TestPackageAvailability("{package.lower()}"));;',)
            )
            _require_package_subprocess(
                package,
                package_repo,
                availability,
                confirm_install=package.lower() != name.lower(),
            )
        _subprocess_packages.update(requirements)
        _libgap_packages.clear()
        require_package.cache_clear()
        return True

    availability = get_output(f'Print(TestPackageAvailability("{name.lower()}"));;')
    if availability.strip() == "fail":
        repo = repo or f"https://github.com/gap-packages/{name}"
        if not is_callable():
            raise ModuleNotFoundError(
                f"GAP package '{name}' is required but not installed.\n"
                f"You may be able to find this package at {repo}"
            )
        return _require_package_subprocess(name, repo, availability)
    return True
