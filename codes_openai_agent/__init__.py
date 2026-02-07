import os
import sys

# Ensure this package's directory is on sys.path for intra-package imports
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)
