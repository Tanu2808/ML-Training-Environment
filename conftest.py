"""Root conftest.py — ensures the project root is on sys.path.

This mirrors the ``pythonpath = ["."]`` setting in ``pyproject.toml`` and
allows IDEs / type-checkers that discover conftest.py to correctly resolve
``from src.data.loader import ...`` style imports in tests.
"""

import sys
from pathlib import Path

# Add the project root (directory containing this file) to sys.path so that
# ``import src.data.loader`` works regardless of how the test runner is invoked.
_PROJECT_ROOT = Path(__file__).parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
