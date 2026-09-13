"""Contract validation at stage boundaries (labflow.contract)."""

from pathlib import Path

import pytest

from labflow.contract import ContractError, required_for, validate


def _write(p: Path, header: str) -> Path:
    p.write_text(header + "\n1,2,3,4\n", encoding="utf-8")
    return p


def test_stage_without_contract_is_noop(tmp_path):
    p = _write(tmp_path / "x.csv", "anything,goes,here")
    validate(p, "report")        # no CSV contract -> must not raise
    validate(p, None)
    assert required_for("report") is None


def test_track_output_must_carry_ids(tmp_path):
    good = (tmp_path / "good.csv")
    good.write_text("track_id,frame,x,y\n1,0,1.0,2.0\n", encoding="utf-8")
    validate(good, "track")      # conforms -> ok

    bad = (tmp_path / "bad.csv")
    bad.write_text("frame,x,y\n0,1.0,2.0\n", encoding="utf-8")   # no track_id
    with pytest.raises(ContractError) as e:
        validate(bad, "track")
    assert "track_id" in str(e.value)


def test_cluster_requires_cluster_id(tmp_path):
    bad = (tmp_path / "c.csv")
    bad.write_text("frame,x,y,z\n0,1,2,0\n", encoding="utf-8")
    with pytest.raises(ContractError):
        validate(bad, "cluster")


def test_truncated_segment_tiff_is_rejected(tmp_path):
    tif = tmp_path / "masks.tif"
    tif.write_bytes(b"II*\x00")   # a mask image, not a table
    with pytest.raises(ContractError, match="readable TIFF"):
        validate(tif, "segment")


def test_segment_integer_labels_and_float_rejection(tmp_path):
    import numpy as np
    import tifffile
    path = tmp_path / "labels.tif"
    tifffile.imwrite(path, np.array([[0, 1], [70000, 70000]], dtype=np.uint32))
    validate(path, "segment")
    tifffile.imwrite(path, np.ones((2, 2), dtype=np.float32))
    with pytest.raises(ContractError, match="integer labels"):
        validate(path, "segment")
    validate(path, "cluster")      # non-CSV outputs use their own stage contract


def test_missing_file_raises(tmp_path):
    with pytest.raises(ContractError):
        validate(tmp_path / "nope.csv", "drift")
