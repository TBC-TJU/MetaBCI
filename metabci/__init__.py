# -*- coding: utf-8 -*-
"""
MetaBCI: An open-source platform for brain-computer interfaces.

MetaBCI has 3 main modules:
- brainda: datasets, algorithms, and deep learning
- brainflow: signal acquisition framework
- brainstim: stimulus presentation
"""


def _get_version() -> str:
    """Get version from installed package or pyproject.toml."""
    # Try to get version from installed package metadata
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version("metabci")
        except PackageNotFoundError:
            pass
    except ImportError:
        pass

    # Fallback: read from pyproject.toml (for development/source installs)
    try:
        from pathlib import Path
        import re

        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        if pyproject_path.exists():
            content = pyproject_path.read_text(encoding="utf-8")
            match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
            if match:
                return match.group(1)
    except Exception:
        pass

    # Final fallback
    return "unknown"


__version__ = _get_version()
