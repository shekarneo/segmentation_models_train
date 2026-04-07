#!/usr/bin/env python3
"""CLI for SAM2 on a single tile (used by compare stage via --sam2-command). Runs sam2_infer_tile.main()."""
import sys
from pathlib import Path

# When run as src/utils/run_sam2_tile.py, project root is parents[2]
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.sam2_infer_tile import main
sys.exit(main())
