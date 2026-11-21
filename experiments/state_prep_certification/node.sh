#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# The single node entry point of the experiments of state_prep_certification.
#
#   bash experiments/state_prep_certification/node.sh EXPERIMENT --approved-cpu-hours H [--jobs N]
#   bash experiments/state_prep_certification/node.sh --check
#   bash experiments/state_prep_certification/node.sh EXPERIMENT --status | --cleanup [--subset S]
#
# Run it from the unpacked bundle's qLDPC/ directory.  It never uses the network.
#
# 1. Preflight: checks the architecture (x86_64), glibc (>= the pinned minimum), uv, the pinned
#    CPython (or the bundle's standalone build), perl, sha256sum, lscpu, ps, tar, bzip2, and GNU
#    parallel (the system's if at least the pinned version, else the bundle's).  --check stops
#    here and records the checks in results/preflight.json, or in preflight-<bundle>.json from a
#    bundle; it needs only this script and pins.json beside it, so it can run before the bundle
#    exists.
# 2. Install, once per bundle, in results/: a virtual environment in venv-<bundle> from the
#    bundle's wheels alone, with hashes required, marked complete only after the install
#    succeeds; the bundle's CPython in python-<bundle>, if uv finds no pinned one; and GNU
#    parallel's script in ext-<bundle>/bin, if the system has none of the pinned version.  Then
#    results/preflight-<bundle>.json records the checks and the GNU parallel that runs.
# 3. Run: <venv>/bin/python experiments/state_prep_certification/run_node.py EXPERIMENT ...,
#    with the numba cache in results/numba-cache.  Repeating the command resumes the run.
#
# Before every start, and before --status and --cleanup, the files of the bundle are checked
# against its SHA256SUMS, so that a tree edited on the node is not run under the bundle's label.
# Every other option passes through to run_node.py; `run_node.py --help` lists them.  --status and
# --cleanup act on one label directory, so they need the run's --subset, if it was not main.  The
# environment variable FTSP_RESULTS chooses another results directory, relative to the directory
# where this script starts; bundles may share it, since each keeps its own environment, CPython,
# GNU parallel, and preflight record there.  This script passes it to run_node.py as
# --results-dir, which it refuses as an option.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PINS="$HERE/pins.json"
RESULTS="${FTSP_RESULTS:-$HERE/results}"
case "$RESULTS" in
    /*) ;;
    *) RESULTS="$PWD/$RESULTS" ;;
esac
BUNDLE=""
if [ -f "$HERE/bundle.json" ]; then
    BUNDLE="$(sed -n 's/.*"name": *"\([^"]*\)".*/\1/p' "$HERE/bundle.json" | head -n 1 || true)"
fi
VENV="$RESULTS/venv-$BUNDLE"
PREFLIGHT="$RESULTS/preflight${BUNDLE:+-$BUNDLE}.json"
export UV_OFFLINE=1

fail() {
    echo "node.sh: $*" >&2
    exit 1
}

need_bundle() {
    [ -n "$BUNDLE" ] || fail "no bundle.json beside node.sh: run it from an unpacked bundle"
}

# read a string field of pins.json, such as "version" inside "python", without a Python interpreter
pin() {
    local colon='[[:space:]]*:[[:space:]]*' value
    value="$(tr -d '\n' < "$PINS" |
        sed -nE "s/.*\"$1\"$colon\\{[^}]*\"$2\"$colon\"([^\"]*)\".*/\\1/p")"
    [ -n "$value" ] || fail "pins.json has no string \"$2\" in \"$1\""
    echo "$value"
}

# 0 if version $1 >= version $2 (dotted or plain integers)
version_at_least() {
    [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n 1)" = "$2" ]
}

# a value as a JSON string: control characters dropped, backslashes and quotes escaped
json_string() {
    printf '"%s"' "$(printf '%s' "$1" | tr '\t\r\n' '   ' | tr -d '\000-\037' |
        sed 's/\\/\\\\/g; s/"/\\"/g')"
}

####################################################################################################
# 1. preflight

preflight() {
    [ -f "$PINS" ] || fail "pins.json must sit beside node.sh"
    local min_glibc parallel_found="" missing=""
    PYTHON_VERSION="$(pin python version)"
    min_glibc="$(pin platform min_glibc)"
    PARALLEL_VERSION="$(pin gnu_parallel version)"

    ARCH="$(uname -m)"
    [ "$ARCH" = "x86_64" ] || fail "this node is $ARCH; the bundle needs x86_64 Linux"
    GLIBC="$(ldd --version 2>/dev/null | head -n 1 | grep -oE '[0-9]+\.[0-9]+$' || true)"
    [ -n "$GLIBC" ] || fail "cannot read the glibc version from 'ldd --version'"
    version_at_least "$GLIBC" "$min_glibc" ||
        fail "glibc $GLIBC is older than $min_glibc, which the bundle's wheels need"
    command -v uv > /dev/null || missing="$missing uv"
    UV_VERSION="$(uv --version 2>/dev/null || true)"
    PYTHON_FOUND=""
    if command -v uv > /dev/null; then
        PYTHON_FOUND="$(uv python find "$PYTHON_VERSION" 2>/dev/null || true)"
    fi
    command -v perl > /dev/null || missing="$missing perl"
    PERL_VERSION="$(perl -e 'print $]' 2>/dev/null || true)"
    if [ -n "$PERL_VERSION" ] && ! version_at_least "$PERL_VERSION" 5.008; then
        fail "perl $PERL_VERSION is older than 5.8, which GNU parallel needs"
    fi
    for tool in sha256sum lscpu ps tar bzip2 sed sort; do
        command -v "$tool" > /dev/null || missing="$missing $tool"
    done
    [ -z "$missing" ] || fail "missing:$missing.  Install them (lscpu is in util-linux)."

    if command -v parallel > /dev/null; then
        parallel_found="$(parallel --version 2>/dev/null | head -n 1 | grep -oE '[0-9]{8}' || true)"
    fi
    if [ -n "$parallel_found" ] && version_at_least "$parallel_found" "$PARALLEL_VERSION"; then
        PARALLEL_SOURCE="system"
        PARALLEL_VERSION="$parallel_found"
    else
        PARALLEL_SOURCE="bundle"
    fi
}

# write the preflight's record; $1: the path and version of the GNU parallel that runs, if any
write_preflight() {
    mkdir -p "$RESULTS"
    cat > "$PREFLIGHT.tmp" << EOF
{
 "bundle": $(json_string "$BUNDLE"),
 "arch": $(json_string "$ARCH"),
 "glibc": $(json_string "$GLIBC"),
 "uv": $(json_string "$UV_VERSION"),
 "python_version": $(json_string "$PYTHON_VERSION"),
 "python_found": $(json_string "$PYTHON_FOUND"),
 "python_source": $(json_string "$([ -n "$PYTHON_FOUND" ] && echo uv || echo bundle)"),
 "perl": $(json_string "$PERL_VERSION"),
 "parallel": $(json_string "$PARALLEL_SOURCE $PARALLEL_VERSION"),
 "parallel_used": $(json_string "$1"),
 "kernel": $(json_string "$(uname -sr)"),
 "checked_at": $(json_string "$(date -u +%Y-%m-%dT%H:%M:%SZ)")
}
EOF
    mv "$PREFLIGHT.tmp" "$PREFLIGHT"
    echo "node.sh: preflight passed; recorded in $PREFLIGHT"
    cat "$PREFLIGHT"
}

check_sums() {
    (cd "$HERE/../../.." && sha256sum --check --quiet SHA256SUMS) ||
        fail "the bundle's files differ from its SHA256SUMS; unpack the bundle again"
}

####################################################################################################
# 2. install

install() {
    local bundle asset python_path python_dir="$RESULTS/python-$BUNDLE" ext="$RESULTS/ext-$BUNDLE"
    bundle="$(cd "$HERE/../../.." && pwd)"
    [ -d "$bundle/wheels" ] && [ -f "$bundle/requirements-node.txt" ] ||
        fail "no bundle around $HERE: expected $bundle/wheels and $bundle/requirements-node.txt"
    asset="$(pin python standalone_asset)"

    if [ ! -f "$VENV/.installed" ]; then
        python_path="$(uv python find "$PYTHON_VERSION" 2>/dev/null || true)"
        if [ -z "$python_path" ]; then
            rm -rf "$python_dir"
            mkdir -p "$python_dir"
            tar -xzf "$bundle/python/$asset" -C "$python_dir" --strip-components=1
            python_path="$python_dir/bin/python${PYTHON_VERSION%.*}"
        fi
        rm -rf "$VENV"
        uv venv --offline --python "$python_path" "$VENV"
        uv pip install --offline --no-index --find-links "$bundle/wheels" --require-hashes \
            --python "$VENV/bin/python" -r "$bundle/requirements-node.txt"
        touch "$VENV/.installed"
    fi

    if [ "$PARALLEL_SOURCE" = bundle ]; then
        if [ ! -x "$ext/bin/parallel" ]; then
            rm -rf "$ext"
            mkdir -p "$ext/bin"
            tar -xjf "$bundle"/parallel/parallel-*.tar.bz2 -C "$ext" --wildcards '*/src/parallel'
            cp "$ext"/parallel-*/src/parallel "$ext/bin/parallel"
            chmod +x "$ext/bin/parallel"
        fi
        export PATH="$ext/bin:$PATH"
    fi
}

####################################################################################################
# 3. run

check=false help=false report=false
for arg in "$@"; do
    case "$arg" in
        --check) check=true ;;
        --help | -h) help=true ;;
        --status | --cleanup) report=true ;;
        --results-dir | --results-dir=*) fail "set FTSP_RESULTS instead of passing --results-dir" ;;
    esac
done
if [ $# -eq 0 ] || $help; then
    sed -n '3,/^$/p' "$0"
    if $help; then exit 0; fi
    exit 1
fi
if $check; then
    preflight
    write_preflight ""
    exit 0
fi
need_bundle
if $report; then
    check_sums
    [ -x "$VENV/bin/python" ] || fail "nothing is installed in $VENV yet: start a run first"
    exec "$VENV/bin/python" "$HERE/run_node.py" "$@" --results-dir "$RESULTS"
fi

preflight
check_sums
install
write_preflight "$(command -v parallel) $(parallel --version 2>/dev/null | head -n 1 || true)"
export NUMBA_CACHE_DIR="$RESULTS/numba-cache"
cd "$HERE/../.."
exec "$VENV/bin/python" experiments/state_prep_certification/run_node.py "$@" \
    --results-dir "$RESULTS"
