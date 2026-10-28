# SPDX-License-Identifier: Apache-2.0

"""Where the experiment keeps its files, and how its runs are named.

Every file that a script writes goes to a subdirectory of ``results/``, which Git ignores:

- ``logs/`` holds the files of annealing runs (chains, code records, logs) and the jobs file.
- ``data/`` holds the best code of each size, and the tables that are computed from those codes.
- ``figures/`` holds the PDF files of the plotting scripts.
- ``cache/`` holds simulations that the plotting scripts reuse.

``cli.py`` has the options that move the results directory, and that select its ``test/``
subdirectory for runs at smoke-test scale.
"""

from __future__ import annotations

from pathlib import Path

from .proposals import CodeType

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
TEST_DIR_NAME = "test"

# The codes and the qecdb.org snapshot that the paper reports, which no script writes, and the
# subdirectory of the results directory in which ``reproduce_paper`` works with copies of them.
PAPER_DATA_DIR = EXPERIMENT_DIR / "paper_data"
PAPER_DIR_NAME = "paper"

# Names of the text files with the best code of each code size, by code type.
CODE_FILE_NAMES: dict[CodeType, str] = {
    "css": "annealed_codes.txt",
    "swel": "annealed_codes_swel.txt",
}


def get_logs_dir(results_dir: Path) -> Path:
    """The directory of the files of annealing runs."""
    return results_dir / "logs"


def get_data_dir(results_dir: Path) -> Path:
    """The directory of the best codes and of tables computed from them."""
    return results_dir / "data"


def get_figures_dir(results_dir: Path) -> Path:
    """The directory of the figures."""
    return results_dir / "figures"


def get_cache_dir(results_dir: Path) -> Path:
    """The directory of the simulations that the plotting scripts reuse."""
    return results_dir / "cache"


def get_half_supports_file(results_dir: Path) -> Path:
    """The text file of the counts of half supports of the best codes."""
    return get_data_dir(results_dir) / "half_supports.txt"


def get_code_files(results_dir: Path) -> dict[CodeType, Path]:
    """The text files of the best codes, by code type, whether or not they exist yet."""
    return {
        code_type: get_data_dir(results_dir) / name for code_type, name in CODE_FILE_NAMES.items()
    }


def find_code_files(results_dir: Path) -> dict[CodeType, Path]:
    """The text files of the best codes that exist, by code type.

    Raises:
        SystemExit: If no file exists, because ``collect_data`` has not been run.
    """
    files = {
        code_type: path for code_type, path in get_code_files(results_dir).items() if path.exists()
    }
    if not files:
        raise SystemExit(
            f"No files of annealed codes in {get_data_dir(results_dir)}: run collect_data first"
        )
    return files


def get_run_name(
    code_type: CodeType, num_qubits: int, num_logicals: int, seed: int, tag: str = ""
) -> str:
    """The file stem of a run, such as ``css_n20_k04_s003`` or ``swel_n20_k04_s003_batch2``."""
    name = f"{code_type}_n{num_qubits}_k{num_logicals:02d}_s{seed:03d}"
    return f"{name}_{tag}" if tag else name
