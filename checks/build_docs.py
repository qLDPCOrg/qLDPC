#!/usr/bin/env python3
import sys

import _common
import checks_superstaq as checks

if __name__ == "__main__":
    _common.ensure_pandoc_on_path()
    sys.exit(checks.build_docs.run(*sys.argv[1:]))
