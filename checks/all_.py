#!/usr/bin/env python3
import sys

import checks_superstaq

if __name__ == "__main__":
    skip_files = ["--sysmon", "--skip", "configs", "requirements", "build_docs"]
    exit(checks_superstaq.all_.run(*skip_files, *sys.argv[1:]))
