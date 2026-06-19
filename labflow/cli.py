"""
labflow.cli

The single, registry-driven CLI, built on Click.

    labflow list [--stage drift] [--json]
    labflow run <stage> --method <name> -i <input> -o <output> [--param k=v ...]
    labflow <stage> --method <name> -i <input> -o <output>     # registry alias
    labflow pipeline [--cores N] [--configfile config/config.yaml] [-n]

Per-stage subcommands (drift, track, analyze, ...) are generated dynamically from
the method registry, so adding a method to config/methods.yaml makes it available
here and in the pipeline with no CLI code changes.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any, Dict, Optional, Tuple

import click

from .registry import load_registry, methods, resolve, stages, validate_registry
from .runner import REPO_ROOT, run_method


def _parse_params(pairs: Tuple[str, ...]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in pairs or ():
        if "=" not in item:
            raise click.ClickException(f"--param expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def _do_run(stage, method, input_path, output_path, pixel_size, units, param, gui=False) -> None:
    spec = resolve(method)
    if spec.get("stage") != stage:
        raise click.ClickException(
            f"method {method!r} is in stage {spec.get('stage')!r}, not {stage!r}"
        )
    if gui:
        from .gui import open_gui
        open_gui(method, input_path)
        return
    if not output_path:
        raise click.ClickException("-o/--output is required (omit it only with --gui).")
    params = _parse_params(param)
    if pixel_size is not None:
        params.setdefault("pixel_size_nm", pixel_size)
    if units:
        params.setdefault("units", units)
    out = run_method(method, input_path=input_path, output_path=output_path, params=params)
    click.echo(f"{stage}:{method} -> {out}")


def _run_options(func):
    """Shared options for `run` and the per-stage aliases."""
    func = click.option("--gui", is_flag=True, default=False,
                        help="Open the backend's own GUI (or napari) for hands-on "
                             "use instead of running headless.")(func)
    func = click.option("--param", multiple=True,
                        help="Method parameter key=value (JSON value). Repeatable.")(func)
    func = click.option("--units", type=click.Choice(["nm", "pixel"]), default=None)(func)
    func = click.option("--pixel-size", "pixel_size", type=float, default=None)(func)
    func = click.option("-o", "--output", "output_path", default=None,
                        help="Output path (required unless --gui).")(func)
    func = click.option("-i", "--input", "input_path", required=True,
                        help="Input path (CSV, or movie folder for localize).")(func)
    func = click.option("--method", "-b", required=True,
                        help="Backend/method name (see `labflow list`).")(func)
    return func


def _make_stage_command(stage: str) -> click.Command:
    @click.command(name=stage, help=f"Run the {stage} stage (alias for `run {stage}`).")
    @_run_options
    def _cmd(method, input_path, output_path, pixel_size, units, param, gui):
        _do_run(stage, method, input_path, output_path, pixel_size, units, param, gui)
    return _cmd


class LabflowGroup(click.Group):
    """Group that exposes one dynamic subcommand per registry stage."""

    def list_commands(self, ctx):
        static = super().list_commands(ctx)
        reg = load_registry()
        dynamic = [s for s in stages(reg) if methods(s, reg)]
        return static + [s for s in dynamic if s not in static]

    def get_command(self, ctx, name):
        cmd = super().get_command(ctx, name)
        if cmd is not None:
            return cmd
        reg = load_registry()
        if name in stages(reg) and methods(name, reg):
            return _make_stage_command(name)
        return None


@click.group(cls=LabflowGroup, context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="smlm-labflow", prog_name="labflow")
def cli():
    """SMLM LabFlow - reproducible orchestration for localization microscopy."""


@cli.command("list")
@click.option("--stage", default=None, help="Filter to one stage.")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")
def list_cmd(stage: Optional[str], as_json: bool):
    """List registered methods, grouped by stage."""
    reg = load_registry()
    grouped = {st: methods(st, reg) for st in stages(reg)
               if methods(st, reg) and (stage is None or st == stage)}
    if as_json:
        payload = {st: {n: {"runtime": s.get("runtime", "python"),
                            "description": s.get("description", "")}
                        for n, s in ms.items()}
                   for st, ms in grouped.items()}
        click.echo(json.dumps(payload, indent=2))
        return
    for st, ms in grouped.items():
        click.echo(f"\n[{st}]")
        for name, spec in ms.items():
            click.echo(f"  {name:16s} {str(spec.get('runtime', 'python')):9s} "
                       f"{spec.get('description', '')}")
    click.echo()


@cli.command("run")
@click.argument("stage")
@_run_options
def run_cmd(stage, method, input_path, output_path, pixel_size, units, param, gui):
    """Run one method of STAGE over the file contract (or --gui for hands-on)."""
    _do_run(stage, method, input_path, output_path, pixel_size, units, param, gui)


@cli.command("install")
@click.argument("tool")
@click.option("-n", "--dry-run", "dry_run", is_flag=True,
              help="Print the install plan without running it.")
@click.option("--force", is_flag=True, help="Reinstall even if already present.")
@click.option("--build", is_flag=True,
              help="Build the image locally instead of pulling the prebuilt one.")
def install_cmd(tool, dry_run, force, build):
    """Install a tool's isolated env / container (pull prebuilt; automatic)."""
    from .install import install_tool
    install_tool(tool, dry_run=dry_run, force=force, build=build)


@cli.command("installed")
def installed_cmd():
    """Show which tools are installed (core / installed / not installed)."""
    from .install import IN_CORE, is_installed
    reg = load_registry()
    for stage in stages(reg):
        ms = methods(stage, reg)
        if not ms:
            continue
        click.echo(f"\n[{stage}]")
        for name, spec in ms.items():
            s = dict(spec); s["name"] = name
            rt = str(s.get("runtime", "python"))
            planned = str(s.get("status", "ready")).lower() != "ready"
            if rt in IN_CORE and not (s.get("install") or {}).get("extra"):
                if not is_installed(s):
                    mark = "planned (not implemented)"
                else:
                    mark = "planned (stub)" if planned else "core"
            else:
                mark = "installed" if is_installed(s) else f"-> labflow install {name}"
            click.echo(f"  {name:16s} {rt:8s} {mark}")
    click.echo()


@cli.command("doctor")
def doctor_cmd():
    """Report isolation health: core env, container engine, per-tool placement."""
    import shutil
    import sys
    from .install import IN_CORE, _in_isolated_env, engine

    click.echo("labflow doctor\n")
    iso = _in_isolated_env()
    click.echo(f"  core env isolated : {'yes' if iso else 'NO  (running on the global Python)'}")
    if not iso:
        click.echo("      -> create envs/labflow and run labflow from it "
                   "(see `labflow install -h`).")
    click.echo(f"  interpreter       : {sys.prefix}")
    eng = engine()
    click.echo(f"  container engine  : {eng}  ({'found' if shutil.which(eng) else 'NOT on PATH'})")

    reg = load_registry()
    order = ["in-core", "core-extra", "venv", "conda", "container"]
    levels: Dict[str, list] = {k: [] for k in order}
    for stage in stages(reg):
        for name, spec in methods(stage, reg).items():
            rt = str(spec.get("runtime", "python"))
            if rt in IN_CORE:
                lvl = "core-extra" if (spec.get("install") or {}).get("extra") else "in-core"
            else:
                lvl = {"venv": "venv", "conda": "conda", "docker": "container"}.get(rt, rt)
            levels.setdefault(lvl, []).append(name)
    click.echo("\n  isolation levels (placement avoids dependency conflicts):")
    for lvl in order:
        names = levels.get(lvl) or []
        if names:
            click.echo(f"    {lvl:11s} {len(names):2d}  {', '.join(sorted(names))}")
    click.echo("\n  (each non-core tool runs in its own env/container; only files cross "
               "the boundary)")

    issues = validate_registry(reg)
    errors = [i for i in issues if i[0] == "error"]
    warns = [i for i in issues if i[0] == "warn"]
    click.echo("\n  registry lint:")
    if not errors:
        click.echo("    no errors - every method entry is well-formed")
    for _lvl, name, msg in errors:
        click.echo(f"    ERROR  {name}: {msg}")
    if warns:
        names = ", ".join(sorted({n for _l, n, _m in warns}))
        click.echo(f"    {len(warns)} scaffold(s) not yet runnable (expected for "
                   f"planned methods): {names}")
    click.echo()


@cli.command("conformance")
@click.option("--stage", default=None, help="Only test methods of this stage.")
def conformance_cmd(stage: Optional[str]):
    """Smoke-test every installed method end-to-end over a synthetic fixture.

    Reports PASS / FAIL / SKIP per tool so a lab can confirm what actually runs on
    this machine. Exits non-zero if an installed, ready tool fails (a CI gate).
    """
    from .conformance import print_report, run_conformance
    raise SystemExit(1 if print_report(run_conformance(stage)) else 0)


@cli.command("review")
@click.argument("run")
@click.option("--with", "viewer", default="napari",
              help="GUI viewer to open the run in (currently: napari).")
def review_cmd(run, viewer):
    """Open a run in a GUI viewer for inspection (the napari exception)."""
    if viewer != "napari":
        raise click.ClickException(f"unsupported viewer {viewer!r}; only 'napari' is wired.")
    from .review import open_in_napari
    open_in_napari(run)


@cli.command("pipeline", context_settings={"ignore_unknown_options": True})
@click.option("--engine", type=click.Choice(["snakemake", "nextflow"]), default="snakemake",
              help="Workflow engine. Both read config/methods.yaml and emit the same "
                   "output tree; nextflow adds cluster/cloud executors (needs Java).")
@click.option("--cores", default=4, type=int, help="snakemake: cores.")
@click.option("--profile", "nf_profile", default=None,
              help="nextflow: executor profile (e.g. slurm, awsbatch).")
@click.option("--configfile", default=None,
              help="snakemake: --configfile; nextflow: -params-file.")
@click.option("-n", "--dry-run", "dry_run", is_flag=True)
@click.argument("extra", nargs=-1, type=click.UNPROCESSED)
def pipeline_cmd(engine, cores, nf_profile, configfile, dry_run, extra):
    """Run the whole DAG with the chosen engine (default: snakemake)."""
    import shutil as _shutil

    wf = REPO_ROOT / "workflow"
    if engine == "nextflow":
        if not _shutil.which("nextflow"):
            raise click.ClickException(
                "nextflow is not on PATH (it needs Java 11+). Install it:\n"
                "    curl -s https://get.nextflow.io | bash   # then move ./nextflow onto PATH\n"
                "or use the Python-native engine:  labflow pipeline --engine snakemake")
        cmd = ["nextflow", "run", str(wf / "main.nf"), "-c", str(wf / "nextflow.config")]
        if nf_profile:
            cmd += ["-profile", nf_profile]
        if configfile:
            cmd += ["-params-file", configfile]
        if dry_run:
            cmd.append("-preview")          # nextflow's closest equivalent to a dry run
    else:
        cmd = ["snakemake", "-s", str(wf / "Snakefile"), "--cores", str(cores)]
        if dry_run:
            cmd.append("-n")
        if configfile:
            cmd += ["--configfile", configfile]
    cmd += list(extra)
    raise SystemExit(subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode)


def main():
    cli()


if __name__ == "__main__":
    main()
