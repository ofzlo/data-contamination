#!/usr/bin/env python
"""Write the fixed order of the LLaVA-665K general pool (general_order.jsonl). See contam.data.general_pool.

Example:
    python scripts/general_pool.py [--exclude-images b1_exclude_images.txt]
"""

from contam.data.general_pool import main

if __name__ == "__main__":
    main()
