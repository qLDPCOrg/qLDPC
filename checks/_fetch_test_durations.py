#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Download .test_durations from the latest successful run of the "Store test durations" workflow.

CI balances its coverage shards by the durations in .test_durations (see ``coverage_.py
--shard``).  To re-tune them, start the workflow (``.github/workflows/test-durations.yaml``) from
the Actions tab, wait for it to finish, run this script, and commit the updated file.  The script
reports the test files that the new file lacks, and the names in it that are not test files.

Downloading an artifact needs a GitHub token, even from a public repository.  The script uses
GITHUB_TOKEN or GH_TOKEN if one is set, or else the token that git uses for github.com.
"""

import io
import json
import os
import pathlib
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from typing import Any

REPOSITORY = "qLDPCOrg/qLDPC"
WORKFLOW = "test-durations.yaml"
ARTIFACT = "test-durations"
DURATIONS_FILE = ".test_durations"
ROOT = pathlib.Path(__file__).resolve().parent.parent


def main() -> int:
    """Write the durations of the latest successful run, and report mismatches with the tests."""
    token = _get_token()
    runs = _get_json(f"actions/workflows/{WORKFLOW}/runs?branch=main&status=success", token)
    if not runs["workflow_runs"]:
        sys.exit(f"No successful run of {WORKFLOW} on main.")
    run = runs["workflow_runs"][0]
    artifacts = _get_json(f"actions/runs/{run['id']}/artifacts", token)["artifacts"]
    artifact = next((item for item in artifacts if item["name"] == ARTIFACT), None)
    if artifact is None or artifact["expired"]:
        sys.exit(f"The run {run['html_url']} has no {ARTIFACT} artifact to download.")
    with zipfile.ZipFile(io.BytesIO(_download(artifact["archive_download_url"], token))) as archive:
        contents = archive.read(DURATIONS_FILE)
    durations = json.loads(contents)
    (ROOT / DURATIONS_FILE).write_bytes(contents)
    print(f"Wrote {DURATIONS_FILE} from {run['html_url']}, started {run['run_started_at']}.")

    listed = subprocess.run(
        ["git", "ls-files", "--", "src/*_test.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    test_files = set(listed.stdout.split())
    missing = sorted(test_files - durations.keys())
    unknown = sorted(durations.keys() - test_files)
    for name in missing:
        print(f"Missing test file: {name}")
    for name in unknown:
        print(f"Not a test file: {name}")
    return 1 if missing or unknown else 0


def _get_token() -> str:
    """A GitHub token from GITHUB_TOKEN or GH_TOKEN, or else the token that git uses for GitHub."""
    for variable in ("GITHUB_TOKEN", "GH_TOKEN"):
        if token := os.environ.get(variable):
            return token
    credential = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\n\n",
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    for line in credential.stdout.splitlines():
        if line.startswith("password="):
            return line.removeprefix("password=")
    sys.exit("Set GITHUB_TOKEN, or let git store a credential for github.com.")


def _get_json(path: str, token: str) -> Any:
    """The JSON response of a GitHub API request for this repository."""
    request = urllib.request.Request(
        f"https://api.github.com/repos/{REPOSITORY}/{path}", headers=_get_headers(token)
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def _download(url: str, token: str) -> bytes:
    """Download an artifact, whose URL redirects to storage that must not receive the token."""
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        opener.open(urllib.request.Request(url, headers=_get_headers(token)))
    except urllib.error.HTTPError as error:
        if error.code != 302:
            raise
        with urllib.request.urlopen(error.headers["Location"]) as response:
            return response.read()
    raise RuntimeError(f"{url} did not redirect to the artifact.")


def _get_headers(token: str) -> dict[str, str]:
    """The headers of an authenticated GitHub API request."""
    return {"Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}"}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """A handler that reports a redirect as an error rather than following it."""

    def redirect_request(self, *args: object, **kwargs: object) -> None:
        """Follow no redirect."""


if __name__ == "__main__":
    sys.exit(main())
