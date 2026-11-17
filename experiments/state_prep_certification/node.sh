#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# The single node entry point of the experiments of state_prep_certification.
#
#   bash experiments/state_prep_certification/node.sh EXPERIMENT --approved-cpu-hours H [--jobs N]
#   bash experiments/state_prep_certification/node.sh --check
#   bash experiments/state_prep_certification/node.sh EXPERIMENT --status | --cleanup
#
# Run it from the unpacked bundle's qLDPC/ directory.  It never uses the network.
#
# 1. Preflight: checks and records in results/preflight.json the architecture (x86_64), glibc
#    (>= the pinned minimum), uv, the pinned CPython (or the bundle's standalone build), perl,
#    sha256sum, lscpu, tar, and GNU parallel (the system's if at least the pinned version, else
#    the bundle's).  --check stops here; it needs only this script and pins.json beside it, so it
#    can run before the bundle exists.
# 2. Install, once per bundle: a virtual environment in results/venv-<bundle> from the bundle's
#    wheels alone, with hashes required, marked complete only after the install succeeds; and
#    GNU parallel's script in results/ext/bin if the system has none.
# 3. Run: <venv>/bin/python experiments/state_prep_certification/run_node.py EXPERIMENT ..., with the numba
#    cache in results/numba-cache.  Repeating the command resumes the run.
#
# Before every start, the files of the bundle are checked against its SHA256SUMS, so that a tree
# edited on the node is not run under the bundle's label.  Every other option passes through to
# run_node.py; `run_node.py --help` lists them.  The environment variable FTSP_RESULTS chooses
# another results directory, which bundles may share, since each bundle installs its own
# environment.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PINS="$HERE/pins.json"
RESULTS="${FTSP_RESULTS:-$HERE/results}"
export UV_OFFLINE=1

fail() {
    echo "node.sh: $*" >&2
    exit 1
}

# read a string field of pins.json, such as "version" inside "python", without a Python interpreter
pin() {
    tr -d '\n' < "$PINS" | sed -E "s/.*\"$1\"[[:space:]]*:[[:space:]]*\\{[^}]*\"$2\"[[:space:]]*:[[:space:]]*\"([^\"]*)\".*/\\1/"
}

# 0 if version $1 >= version $2 (dotted or plain integers)
version_at_least() {
    [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n 1)" = "$2" ]
}

####################################################################################################
# 1. preflight

preflight() {
    [ -f "$PINS" ] || fail "pins.json must sit beside node.sh"
    local python_version min_glibc parallel_version missing=""
    python_version="$(pin python version)"
    min_glibc="$(pin platform min_glibc)"
    parallel_version="$(pin gnu_parallel version)"

    local arch glibc uv_version python_path="" perl_version parallel_source parallel_found=""
    arch="$(uname -m)"
    [ "$arch" = "x86_64" ] || fail "this node is $arch; the bundle needs x86_64 Linux"
    glibc="$(ldd --version 2>/dev/null | head -n 1 | grep -oE '[0-9]+\.[0-9]+$' || true)"
    [ -n "$glibc" ] || fail "cannot read the glibc version from 'ldd --version'"
    version_at_least "$glibc" "$min_glibc" ||
        fail "glibc $glibc is older than $min_glibc, which the bundle's wheels need"
    command -v uv > /dev/null || missing="$missing uv"
    uv_version="$(uv --version 2>/dev/null || true)"
    if command -v uv > /dev/null; then
        python_path="$(uv python find "$python_version" 2>/dev/null || true)"
    fi
    command -v perl > /dev/null || missing="$missing perl"
    perl_version="$(perl -e 'print $]' 2>/dev/null || true)"
    if [ -n "$perl_version" ] && ! version_at_least "$perl_version" 5.008; then
        fail "perl $perl_version is older than 5.8, which GNU parallel needs"
    fi
    for tool in sha256sum lscpu tar sed sort; do
        command -v "$tool" > /dev/null || missing="$missing $tool"
    done
    [ -z "$missing" ] || fail "missing:$missing.  Install them (lscpu is in util-linux)."

    if command -v parallel > /dev/null; then
        parallel_found="$(parallel --version 2>/dev/null | head -n 1 | grep -oE '[0-9]{8}' || true)"
    fi
    if [ -n "$parallel_found" ] && version_at_least "$parallel_found" "$parallel_version"; then
        parallel_source="system $parallel_found"
    else
        parallel_source="bundle $parallel_version"
    fi

    mkdir -p "$RESULTS"
    cat > "$RESULTS/preflight.json" << EOF
{
 "arch": "$arch",
 "glibc": "$glibc",
 "uv": "$uv_version",
 "python_version": "$python_version",
 "python_found": "$python_path",
 "python_source": "$([ -n "$python_path" ] && echo uv || echo bundle)",
 "perl": "$perl_version",
 "parallel": "$parallel_source",
 "kernel": "$(uname -sr)",
 "checked_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
    echo "node.sh: preflight passed; recorded in $RESULTS/preflight.json"
    cat "$RESULTS/preflight.json"
}

####################################################################################################
# 2. install

# the virtual environment of this bundle
venv_dir() {
    local name
    name="$(sed -n 's/.*"name": *"\([^"]*\)".*/\1/p' "$HERE/bundle.json" 2>/dev/null | head -n 1)"
    [ -n "$name" ] || fail "no bundle.json beside node.sh: run this script from an unpacked bundle"
    echo "$RESULTS/venv-$name"
}

install() {
    local bundle python_version asset python_path venv
    venv="$(venv_dir)"
    bundle="$(cd "$HERE/../../.." && pwd)"
    [ -d "$bundle/wheels" ] && [ -f "$bundle/requirements-node.txt" ] ||
        fail "no bundle around $HERE: expected $bundle/wheels and $bundle/requirements-node.txt"
    python_version="$(pin python version)"
    asset="$(pin python standalone_asset)"

    if [ ! -f "$venv/.installed" ]; then
        python_path="$(uv python find "$python_version" 2>/dev/null || true)"
        if [ -z "$python_path" ]; then
            mkdir -p "$RESULTS/python-$python_version"
            tar -xzf "$bundle/python/$asset" -C "$RESULTS/python-$python_version" --strip-components=1
            python_path="$RESULTS/python-$python_version/bin/python${python_version%.*}"
        fi
        rm -rf "$venv"
        uv venv --offline --python "$python_path" "$venv"
        uv pip install --offline --no-index --find-links "$bundle/wheels" --require-hashes \
            --python "$venv/bin/python" -r "$bundle/requirements-node.txt"
        touch "$venv/.installed"
    fi

    if [[ "$(grep -o '"parallel": "[a-z]*' "$RESULTS/preflight.json")" == *bundle ]]; then
        if [ ! -x "$RESULTS/ext/bin/parallel" ]; then
            mkdir -p "$RESULTS/ext/bin"
            tar -xjf "$bundle"/parallel/parallel-*.tar.bz2 -C "$RESULTS/ext" --wildcards '*/src/parallel'
            cp "$RESULTS"/ext/parallel-*/src/parallel "$RESULTS/ext/bin/parallel"
            chmod +x "$RESULTS/ext/bin/parallel"
        fi
        export PATH="$RESULTS/ext/bin:$PATH"
    fi
    parallel --version | head -n 1
}

####################################################################################################
# 3. run

case " $* " in
    *" --check "*)
        preflight
        exit 0
        ;;
    *" --help "* | *" -h "*)
        sed -n '3,/^$/p' "$0"
        exit 0
        ;;
    *" --status "* | *" --cleanup "*)
        exec "$(venv_dir)/bin/python" "$HERE/run_node.py" "$@" --results-dir "$RESULTS"
        ;;
esac

preflight
(cd "$HERE/../../.." && sha256sum --check --quiet SHA256SUMS) ||
    fail "the bundle's files differ from its SHA256SUMS; unpack the bundle again"
install
export NUMBA_CACHE_DIR="$RESULTS/numba-cache"
cd "$HERE/../.."
exec "$(venv_dir)/bin/python" experiments/state_prep_certification/run_node.py "$@" --results-dir "$RESULTS"
