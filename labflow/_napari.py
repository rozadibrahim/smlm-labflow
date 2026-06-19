"""
labflow._napari -- minimal napari plugin (a GUI front door for biologists).

Adds a "LabFlow: open run" dock widget that loads a run directory (or a single
localizations / clusters CSV) as napari point layers, reusing labflow.review.gather_layers
(clusters coloured by cluster_id). This is an *additional* entry point -- the CLI, the
method registry, and the Snakemake/Nextflow pipelines (the developer path) are unchanged.

Needs napari + magicgui (the `gui` extra) and a display. The module imports only the
stdlib at top level (napari/magicgui are imported lazily, inside napari), so it is
import-safe everywhere; the widget itself is exercised when napari loads the plugin.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple


def _layer_tuples(layers: List[dict]) -> List[Tuple[Any, dict, str]]:
    """Convert review.gather_layers() specs into napari LayerDataTuples."""
    tuples: List[Tuple[Any, dict, str]] = []
    for layer in layers:
        kwargs: dict = {"name": layer["name"], "size": 6}
        if layer.get("cluster_id") is not None:
            kwargs["features"] = {"cluster_id": layer["cluster_id"]}
            kwargs["face_color"] = "cluster_id"
        tuples.append((layer["data"], kwargs, layer.get("type", "points")))
    return tuples


def open_run():
    """Return a magicgui dock widget that loads a LabFlow run / CSV into napari layers."""
    from magicgui import magicgui
    from napari.types import LayerDataTuple

    @magicgui(call_button="Load", run={"label": "Run dir or CSV", "mode": "d"})
    def widget(run: Path) -> List[LayerDataTuple]:
        from .review import gather_layers
        return _layer_tuples(gather_layers(str(run)))

    return widget
