#!/usr/bin/env python3
import sys

import checks_superstaq as checks

if __name__ == "__main__":
    sys.exit(checks.build_docs.run(*sys.argv[1:]))
