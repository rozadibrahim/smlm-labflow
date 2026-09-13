"""Small upstream LiteLoc checks, without trained models or a microscope dataset.

Run with the LiteLoc interpreter: python scripts/check_liteloc_engineering.py
--source /workspace/backends/LiteLoc --output <record.json>
Random weights establish execution/serialization only, not localization quality.
"""
import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    source = Path(args.source).resolve()
    sys.path.insert(0, str(source))
    import torch
    from network.liteloc import LiteLoc

    for name in ("network.loc_model", "network.multi_process", "utils.help_utils", "spline"):
        importlib.import_module(name)
    torch.set_num_threads(4)
    torch.manual_seed(7)
    torch.backends.cudnn.allow_tf32 = False
    model = LiteLoc().eval()
    frames = torch.randn(5, 32, 32)
    with torch.no_grad():
        cpu = model(frames)
        model = model.cuda()
        gpu = model(frames.cuda())
        for expected, actual in zip(cpu, gpu):
            torch.testing.assert_close(actual.cpu(), expected, atol=2e-4, rtol=2e-4)
        assert gpu[0].shape == (3, 32, 32)
        detections = model.post_process(gpu[0].clone(), gpu[1].clone())
        assert detections.ndim == 2 and detections.shape[1] == 6
        assert torch.isfinite(detections).all()
    model.train()
    outputs = model(torch.randn(6, 32, 32, device="cuda"), test=False)
    loss = sum(t.square().mean() for t in outputs)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    torch.optim.NAdam(model.parameters(), lr=8e-4).step()
    model.eval()
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = Path(tmp) / "state.pt"
        torch.save(model.state_dict(), checkpoint)
        restored = LiteLoc().cuda().eval()
        restored.load_state_dict(torch.load(checkpoint, weights_only=True))
        with torch.no_grad():
            for expected, actual in zip(model(frames.cuda()), restored(frames.cuda())):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.cuda.synchronize()
    record = {"upstream_commit": subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
        "status": "PASS", "checks": ["upstream analysis/training imports", "CPU/CUDA forward parity",
        "postprocessing shape and finite values", "backward and NAdam step", "state-dict checkpoint round-trip"],
        "scope": "Random-weight engineering check; does not validate trained localization or full movie pipeline."}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
