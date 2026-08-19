#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

# Source checkout: <repo>/scripts/dataops_agent.py -> <repo>/platform_agent
# Runtime:         <runtime>/opt_airflow/scripts/dataops_agent.py -> sibling packages
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from platform_agent.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
