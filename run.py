#!/usr/bin/env python3
"""
Convenience runner for executing the HERALD CLI directly from the source code.
Usage: python run.py investigate google.com
"""
import sys
from herald.cli import main

if __name__ == "__main__":
    sys.exit(main())
