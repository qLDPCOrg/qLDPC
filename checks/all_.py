#!/usr/bin/env python3
import sys

import _common
import checks_superstaq

if __name__ == "__main__":
    # The docs build runs here too, so the pandoc that nbsphinx needs has to be discoverable.
    _common.ensure_pandoc_on_path()
    skip_files = ["--sysmon", "--skip", "configs", "requirements"]
    sys.exit(checks_superstaq.all_.run(*skip_files, *sys.argv[1:]))
