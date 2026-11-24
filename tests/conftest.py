import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGACY_DIR = os.path.join(ROOT, "legacy")

sys.path.insert(0, ROOT)
os.chdir(ROOT)  # configs/ and map paths are relative to the repository root


def add_legacy_to_path():
    """Makes the legacy PyTorch package importable (its modules use both "src.<x>" and "<x>" imports)."""
    for p in (LEGACY_DIR, os.path.join(LEGACY_DIR, "src")):
        if p not in sys.path:
            sys.path.insert(0, p)
