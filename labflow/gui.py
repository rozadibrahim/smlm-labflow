"""
labflow.gui

Open a backend's GUI through labflow so the user gets hands-on access to all of
that tool's features. Driven by the method's `gui` field in config/methods.yaml:

    gui: napari                       -> open the data in napari (labflow.review)
    gui: ["cellpose"]                 -> launch the tool's own GUI (subprocess)
    gui: ["fiji", "{input}"]          -> launch with the input substituted
    (no gui field)                    -> fall back to napari, so every backend
                                         still gets a hands-on viewer

This is the bidirectional companion to headless `run`: classical backends get
napari for inspection; GUI-first tools (Cellpose, Fiji/ThunderSTORM/TrackMate,
Picasso, SMAP) get their native window with the run's data loaded.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from .registry import REPO_ROOT, resolve


def open_gui(method: str, input_path: str | Path, reg: Optional[Dict[str, Any]] = None) -> None:
    spec = resolve(method, reg)
    gui = spec.get("gui")

    if gui in (None, "napari"):
        try:
            from .review import open_in_napari
        except Exception as exc:                          # pragma: no cover
            raise RuntimeError(f"could not import the napari bridge: {exc}") from exc
        if gui is None:
            print(f"'{method}' has no native GUI — opening its data in napari instead.")
        open_in_napari(input_path)
        return

    if isinstance(gui, (list, tuple)):
        args = [str(a).format(input=str(input_path), repo=str(REPO_ROOT)) for a in gui]
        print(f"launching {method} GUI: {' '.join(args)}")
        try:
            subprocess.run(args)                          # blocks until the GUI closes
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"could not launch {method}'s GUI ({args[0]!r} not found). "
                f"Install/activate its environment first (see docs/methods.md)."
            ) from exc
        return

    raise RuntimeError(f"method {method!r} has an unrecognized 'gui' spec: {gui!r}")
