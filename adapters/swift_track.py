"""
swift multi-state SPT adapter (Endesfelder / Heilemann lab) -- runtime `local`; needs
the `swift` binary on PATH (request it from the authors -- not freely redistributable,
so it is detected via install.requires_cmd rather than auto-installed).

Contract: localizations.csv (frame, x, y) in -> tracks.csv (track_id, frame, x, y) out.

BINDING POINT: swift is an external binary with its own config/CLI. Wire `_run_swift()`
to your swift install (invoke it via subprocess, then map its output to the contract).
"""

import argparse
import json
import shutil

import pandas as pd  # noqa: F401  (for parsing swift output in the binding-point impl)


def _run_swift(inp: str, out: str, params: dict) -> None:
    if shutil.which("swift") is None:
        raise SystemExit("swift binary not on PATH. Obtain swift from the authors "
                         "(Endesfelder/Heilemann lab) and add it to PATH.")
    # BINDING POINT: invoke swift on `inp` with your config, then map its output to
    # tracks.csv (track_id, frame, x, y), e.g.:
    #   subprocess.run(["swift", inp, "--config", params.get("config"), "--out", tmp])
    #   parse(tmp) -> DataFrame[track_id, frame, x, y] -> to_csv(out)
    raise NotImplementedError(
        "swift invocation is the binding point: wire adapters/swift_track.py:_run_swift "
        "to your swift CLI + config, then write tracks.csv from its output.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--params", default="{}")
    args = ap.parse_args()
    _run_swift(args.inp, args.out, json.loads(args.params))


if __name__ == "__main__":
    main()
