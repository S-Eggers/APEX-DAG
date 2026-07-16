from __future__ import annotations

from dataclasses import dataclass, field

@dataclass(frozen=True)
class LibrarySpec:
    """How to find (search) and verify (modules) one library."""

    name: str
    search: list[str] = field(default_factory=list)
    modules: frozenset[str] = field(default_factory=frozenset)

def _default_spec(name: str) -> LibrarySpec:
    """Derive a spec from the raw name (pip foo-bar -> import foo_bar)."""
    module = name.strip().lower().replace("-", "_")
    return LibrarySpec(name=name, search=[f"import {module}"], modules=frozenset({module}))

_OVERRIDES: dict[str, LibrarySpec] = {
    "polars": LibrarySpec("polars", ["import polars"], frozenset({"polars"})),
    "apache-beam": LibrarySpec("apache-beam", ["import apache_beam"], frozenset({"apache_beam"})),
    "apache_beam": LibrarySpec("apache-beam", ["import apache_beam"], frozenset({"apache_beam"})),
    "beam": LibrarySpec("apache-beam", ["import apache_beam"], frozenset({"apache_beam"})),
    "scikit-learn": LibrarySpec("scikit-learn", ["import sklearn", "from sklearn"], frozenset({"sklearn"})),
    "opencv-python": LibrarySpec("opencv-python", ["import cv2"], frozenset({"cv2"})),
    "pillow": LibrarySpec("pillow", ["from PIL import"], frozenset({"PIL"})),
    "beautifulsoup4": LibrarySpec("beautifulsoup4", ["from bs4 import"], frozenset({"bs4"})),
    "pytorch": LibrarySpec("pytorch", ["import torch"], frozenset({"torch"})),
}

def resolve_spec(name: str) -> LibrarySpec:
    """Return the spec for name, falling back to a name-derived default."""
    key = name.strip().lower()
    return _OVERRIDES.get(key, _default_spec(name))
