"""
DECODE localizer adapter  (Speiser et al., Nat. Methods 2021; TuragaLab/DECODE).

Runs INSIDE the `decode` conda env (built by `labflow install decode`). Reads a frame
stack, runs DECODE inference with a trained model, and writes canonical localizations;
labflow validates the output contract on the core side.

    labflow run localize -b decode -i frames.tif -o locs.csv \
        --param model=/path/model.pt --param param_file=/path/param.yaml

BINDING POINT
-------------
`_infer()` follows the documented DECODE "Fit" workflow (decode.readthedocs.io,
~v0.10). The exact constructor kwargs have changed across DECODE releases, so the
call is wrapped: if your installed version differs it raises a clear message telling
you to adjust this one function -- it does not silently produce wrong coordinates.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _localize_io import read_frames, write_localizations


def _infer(frames, params):
    """Return per-localization arrays (frame_ix, x_nm, y_nm, z_nm, photons)."""
    import torch
    import decode
    import decode.utils

    model_path = params.get("model") or ""
    param_file = params.get("param_file") or ""
    if not model_path or not param_file:
        raise SystemExit("DECODE needs --param model=<ckpt.pt> and "
                         "--param param_file=<param.yaml> (the training config).")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    param = decode.utils.param_io.ParamHandling().load_params(param_file)

    # --- DECODE Fit pipeline (confirm against your installed decode version) -----
    model = decode.neuralfitter.models.SigmaMUNet.parse(param)
    model = decode.utils.model_io.LoadSaveModel(
        model, output_file=None, input_file=model_path).load_init(device=device)

    camera = decode.simulation.camera.Photon2Camera.parse(param)
    camera.device = "cpu"

    frame_proc = decode.neuralfitter.utils.processing.TransformSequence([
        decode.neuralfitter.frame_processing.AutoCenterCrop(8),
        decode.neuralfitter.scale_transform.AmplitudeRescale.parse(param),
    ])
    post_proc = decode.neuralfitter.utils.processing.TransformSequence([
        decode.neuralfitter.coord_transform.Offset2Coordinate.parse(param),
        decode.neuralfitter.post_processing.SpatialIntegration.parse(param),
    ])
    infer = decode.neuralfitter.Infer(
        model=model, ckpt_path=None, device=device,
        frame_proc=frame_proc, post_proc=post_proc, camera=camera)

    frames_t = camera.backward(torch.from_numpy(frames.astype("float32")))
    emitter = infer.forward(frames_t)
    # ----------------------------------------------------------------------------

    xyz = emitter.xyz_nm.cpu().numpy()
    return (emitter.frame_ix.cpu().numpy(),
            xyz[:, 0], xyz[:, 1], xyz[:, 2],
            emitter.phot.cpu().numpy())


def main():
    ap = argparse.ArgumentParser(description="DECODE localizer adapter (labflow).")
    ap.add_argument("--in", dest="inp", required=True, help="frame stack (TIFF)")
    ap.add_argument("--out", dest="out", required=True, help="canonical localizations CSV")
    ap.add_argument("--params", default="{}", help="JSON params (model, param_file)")
    args = ap.parse_args()

    params = json.loads(args.params) if args.params else {}
    frames = read_frames(args.inp)
    try:
        frame, x, y, z, photons = _infer(frames, params)
    except (ImportError, AttributeError, TypeError) as exc:
        raise SystemExit(
            "DECODE inference binding point did not match your install: " + repr(exc) +
            "\nEdit adapters/decode_localize.py:_infer for your decode version "
            "(decode.readthedocs.io).")
    write_localizations(args.out, frame=frame, x=x, y=y, z=z, photons=photons,
                        backend="decode", source_file=args.inp)
    print(f"decode: {len(frame)} localizations -> {args.out}")


if __name__ == "__main__":
    main()
