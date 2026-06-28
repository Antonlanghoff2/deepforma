
from __future__ import annotations

import sys
from pathlib import Path


project_root = Path(__file__).resolve().parent
root_path = str(project_root)
if root_path not in sys.path:
    sys.path.insert(0, root_path)
