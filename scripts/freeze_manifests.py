#!/usr/bin/env python
"""Freeze (or check) the split / SameDist / general-order manifests. See contam.data.freeze.

Example:
    python scripts/freeze_manifests.py           # write manifests/MANIFEST.lock.json
    python scripts/freeze_manifests.py --add     # add general_order.jsonl after the B1 audit
    python scripts/freeze_manifests.py --check   # before every training / eval run
"""

from contam.data.freeze import main

if __name__ == "__main__":
    main()
