"""Exercise CUDA kernels, cuBLAS and cuDNN, including backward propagation.

Run with the GPU backend's interpreter, not the lightweight LabFlow core.
This checks runtime compatibility, not scientific model accuracy.
"""
import json
import platform

import torch
import torchvision
import torchaudio


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this interpreter")
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    checks = []

    values = torch.arange(16, dtype=torch.float32, device=device)
    assert (values * values).sum().item() == 1240.0
    checks.append("CUDA elementwise/reduction")

    a, b = torch.randn(128, 128), torch.randn(128, 128)
    torch.testing.assert_close((a.to(device) @ b.to(device)).cpu(), a @ b,
                               atol=1e-4, rtol=1e-4)
    checks.append("cuBLAS matrix multiplication vs CPU")

    model = torch.nn.Sequential(torch.nn.Conv2d(1, 8, 3, padding=1),
                                torch.nn.ReLU(), torch.nn.Conv2d(8, 1, 3, padding=1))
    inputs = torch.randn(2, 1, 32, 32)
    reference = model(inputs).detach()
    model = model.to(device)
    prediction = model(inputs.to(device))
    torch.testing.assert_close(prediction.detach().cpu(), reference, atol=1e-4, rtol=1e-4)
    loss = prediction.square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all().item()
               for p in model.parameters())
    torch.optim.Adam(model.parameters(), lr=1e-3).step()
    checks.append("cuDNN convolution vs CPU, backward and optimizer step")

    boxes = torch.tensor([[0., 0., 2., 2.], [0., 0., 2., 2.]], device=device)
    scores = torch.tensor([0.9, 0.8], device=device)
    assert torchvision.ops.nms(boxes, scores, 0.5).tolist() == [0]
    checks.append("torchvision CUDA extension")
    torch.cuda.synchronize()
    print(json.dumps({"python": platform.python_version(), "torch": torch.__version__,
                      "torchvision": torchvision.__version__, "torchaudio": torchaudio.__version__,
                      "cuda_build": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                      "gpu": torch.cuda.get_device_name(0),
                      "capability": torch.cuda.get_device_capability(0),
                      "checks_passed": checks}, indent=2))


if __name__ == "__main__":
    main()
