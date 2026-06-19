"""Provenance manifest written beside each output (labflow.provenance)."""

import json
from pathlib import Path

from labflow import provenance


def test_manifest_records_method_and_hashes(tmp_path):
    inp = tmp_path / "in.csv"
    inp.write_text("frame,x,y\n0,1,2\n", encoding="utf-8")
    out = tmp_path / "out.csv"
    out.write_text("track_id,frame,x,y\n1,0,1,2\n", encoding="utf-8")

    spec = {"name": "trackpy", "stage": "track", "params": {}}
    mp = provenance.write(out, spec=spec, params={"radius": 3}, input_path=inp,
                          runtime="python")

    assert mp == provenance.manifest_path(out)
    rec = json.loads(mp.read_text(encoding="utf-8"))
    assert rec["method"] == "trackpy"
    assert rec["stage"] == "track"
    assert rec["isolation"] == "in-core"
    assert rec["params"] == {"radius": 3}
    assert rec["input"]["sha256"].startswith("sha256:")
    assert rec["output"]["sha256"].startswith("sha256:")
    assert rec["input"]["sha256"] != rec["output"]["sha256"]


def test_isolation_level_tracks_runtime(tmp_path):
    out = tmp_path / "o.csv"
    out.write_text("track_id\n1\n", encoding="utf-8")
    inp = tmp_path / "i.csv"
    inp.write_text("track_id\n1\n", encoding="utf-8")

    docker_spec = {"name": "magik", "stage": "track",
                   "image": "ghcr.io/x/smlm-magik:latest", "gpu": True}
    rec = json.loads(provenance.write(
        out, spec=docker_spec, params={}, input_path=inp,
        runtime="docker", engine="docker").read_text(encoding="utf-8"))
    assert rec["isolation"] == "container"
    assert rec["image"]["ref"].endswith("smlm-magik:latest")
    assert rec["image"]["gpu"] is True

    extra_spec = {"name": "locan", "stage": "cluster", "install": {"extra": "stats"}}
    rec2 = json.loads(provenance.write(
        out, spec=extra_spec, params={}, input_path=inp,
        runtime="python").read_text(encoding="utf-8"))
    assert rec2["isolation"] == "core-extra"
