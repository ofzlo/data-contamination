#!/usr/bin/env python
"""Build the SameDist control (D4) for splits A and B from COCO train2017 captions. See contam.data.samedist.

Example:
    python scripts/make_samedist.py                     # samedist_A.json, samedist_B.json next to the splits
    python scripts/make_samedist.py --split A --exclude-coco-ids b1_coco_exclude.txt
"""

from contam.data.samedist import main

if __name__ == "__main__":
    main()
