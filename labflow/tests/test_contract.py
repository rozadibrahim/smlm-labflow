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


def test_non_csv_output_is_skipped(tmp_path):
    tif = tmp_path / "masks.tif"
    tif.write_bytes(b"II*\x00")   # a mask image, not a table
    validate(tif, "segment")      # segment has no CSV contract anyway
    validate(tif, "cluster")      # even a contract stage: non-CSV artifact -> skip


def test_missing_file_raises(tmp_path):
    with pytest.raises(ContractError):
        validate(tmp_path / "nope.csv", "drift")
