"""
Gesture template library: CRUD over data/gestures.json.

Storage format:
{
  "fist": [[x0,y0], [x1,y1], ...],  // 21 normalized (x,y) pairs
  ...
}
"""

import json
import os
import numpy as np

_LIBRARY_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'gestures.json')


def load() -> dict[str, np.ndarray]:
    """Returns {name: (21,2) float32 array}."""
    if not os.path.exists(_LIBRARY_PATH):
        return {}
    with open(_LIBRARY_PATH) as f:
        raw = json.load(f)
    return {name: np.array(template, dtype=np.float32) for name, template in raw.items()}


def save(library: dict[str, np.ndarray]) -> None:
    os.makedirs(os.path.dirname(_LIBRARY_PATH), exist_ok=True)
    with open(_LIBRARY_PATH, 'w') as f:
        json.dump({name: template.tolist() for name, template in library.items()}, f, indent=2)


def add(name: str, template: np.ndarray) -> None:
    """Add or overwrite a gesture template."""
    lib = load()
    lib[name] = template
    save(lib)


def delete(name: str) -> None:
    lib = load()
    if name not in lib:
        raise KeyError(f"Gesture '{name}' not found.")
    del lib[name]
    save(lib)


def list_gestures() -> list[str]:
    return list(load().keys())
