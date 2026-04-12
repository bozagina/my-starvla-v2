#!/usr/bin/env python3
from __future__ import annotations

"""
Compatibility entrypoint for A-outputs audit.

Canonical implementation lives in `tools/fasa_a_outputs_audit.py`.
This wrapper keeps legacy invocations stable while we standardize the
public toolchain names.
"""

from fasa_a_outputs_audit import main


if __name__ == "__main__":
    main()
