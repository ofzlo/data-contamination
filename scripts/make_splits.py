#!/usr/bin/env python
"""Make deterministic NoCaps exposure splits (A = explore, B = confirm). See contam.data.splits.

Example:
    python scripts/make_splits.py                  # splits A and B from configs/default.yaml seeds
    python scripts/make_splits.py --split A --split-seed 7 --ref-seed 8 --out-dir /tmp/splits
"""

from contam.data.splits import main

if __name__ == "__main__":
    main()
