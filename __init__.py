"""Repo-root shim that forwards imports to the real simphony package."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

_PKG_DIR = Path(__file__).resolve().parent / "simphony"
_SPEC = spec_from_file_location(__name__, _PKG_DIR / "__init__.py", submodule_search_locations=[str(_PKG_DIR)])
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load simphony package from {_PKG_DIR}")
_MODULE = module_from_spec(_SPEC)
sys.modules[__name__] = _MODULE
_SPEC.loader.exec_module(_MODULE)
