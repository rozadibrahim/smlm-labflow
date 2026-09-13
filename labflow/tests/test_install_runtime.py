import json
import sys
import subprocess
import venv

import pytest

from labflow import install
from labflow.runner import run_method


def test_empty_environment_is_not_installed(tmp_path, monkeypatch):
    monkeypatch.setenv("LABFLOW_ENV_ROOT", str(tmp_path))
    venv.EnvBuilder(with_pip=False).create(tmp_path / "empty")
    assert not install.is_installed({"name": "empty", "env": "empty", "runtime": "venv"})


def test_real_install_isolation_and_foreign_working_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("LABFLOW_ENV_ROOT", str(tmp_path / "environments with spaces"))
    poison = tmp_path / "poison"
    poison.mkdir()
    (poison / "core_only.py").write_text("raise RuntimeError('core leaked into backend')")
    monkeypatch.setenv("PYTHONPATH", str(poison))
    spec = {"name": "fixture", "runtime": "venv", "env": "envs/fixture", "stage": "test",
            "install": {"probe": "json"}, "command": ["python", "-c",
            "import sys,pathlib,importlib.util; assert importlib.util.find_spec('core_only') is None; "
            "pathlib.Path(sys.argv[1]).write_text(sys.prefix)", "{output}"]}
    reg = {"methods": {"fixture": spec}}
    install.install_tool("fixture", reg=reg)
    assert install.is_installed(spec)
    monkeypatch.chdir(tmp_path)
    inp = tmp_path / "input.csv"
    inp.write_text("x,y\n1,2\n")
    output = tmp_path / "output.csv"
    run_method("fixture", input_path=inp, output_path=output, reg=reg)
    assert output.read_text() == str(install._envdir("fixture"))
    assert json.loads((tmp_path / "output.csv.labflow.json").read_text())["runtime"] == "venv"
    broken = dict(spec, install={"probe": "missing_backend_module"})
    assert not install.is_installed(broken)
    def fail_install(command, **kwargs):
        assert not (install._envdir("fixture") / ".labflow-install.json").exists()
        return subprocess.CompletedProcess(command, 1)
    monkeypatch.setattr(install.subprocess, "run", fail_install)
    with pytest.raises(RuntimeError, match="install step failed"):
        install.install_tool("fixture", force=True, reg=reg)
    assert not install.is_installed(spec)


def test_stub_install_fails_before_creating_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("LABFLOW_ENV_ROOT", str(tmp_path))
    spec = {"runtime": "venv", "env": "unfinished", "implementation": "stub"}
    with pytest.raises(RuntimeError, match="not implemented"):
        install.install_tool("unfinished", reg={"methods": {"unfinished": spec}})
    assert not (tmp_path / "unfinished").exists()


def test_requirement_edit_invalidates_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(install, "REPO_ROOT", tmp_path)
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("package==1\n")
    spec = {"install": {"requirements": "requirements.txt"}}
    before = install._recipe_hash(spec)
    requirements.write_text("package==2\n")
    assert install._recipe_hash(spec) != before


def test_missing_conda_recipe_is_actionable():
    with pytest.raises(ValueError, match="missing environment recipe"):
        install.plan({"name": "broken", "runtime": "conda", "install": {"conda_file": "missing.yml"}})


def test_external_recipe_is_not_treated_as_already_in_core():
    commands = install.plan({"name": "backend", "runtime": "external",
                             "install": {"script": "scripts/install_liteloc.py"}})
    assert commands == [[sys.executable, str(install.REPO_ROOT / "scripts/install_liteloc.py")]]


def test_cli_stub_error_is_actionable():
    from click.testing import CliRunner
    from labflow.cli import cli
    result = CliRunner().invoke(cli, ["install", "miro"])
    assert result.exit_code == 1
    assert "adapter is not implemented" in result.output
    assert "Traceback" not in result.output
