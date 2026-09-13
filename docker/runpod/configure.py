"""Configure only the image's copy; the source registry remains portable."""
from pathlib import Path

import yaml

root = Path("/opt/smlm-labflow")
registry_path = root / "config/methods.yaml"
registry = yaml.safe_load(registry_path.read_text())
registry["methods"]["liteloc"]["command"][0] = (
    "/opt/conda/envs/smlm-labflow/bin/python"
)
registry_path.write_text(yaml.safe_dump(registry, sort_keys=False))

# Never ship the maintainer's machine-specific backend paths.
paths = yaml.safe_load((root / "adapters/backend_paths.example.yml").read_text())
paths["liteloc"]["root"] = "/workspace/backends/LiteLoc"
(root / "adapters/backend_paths.yml").write_text(yaml.safe_dump(paths, sort_keys=False))

config_path = root / "config/config.yaml"
config = yaml.safe_load(config_path.read_text())
config.update(input_dir="/workspace/data/movies", run_dir="/workspace/outputs/snakemake_run",
              calibrate_input="/workspace/data/beads", train_input="/workspace/data/train")
config_path.write_text(yaml.safe_dump(config, sort_keys=False))

# Per-tool installations persist on the attached volume. The baked core does
# not depend on that volume and is available immediately.
envs = root / "envs"
if envs.exists():
    raise RuntimeError("envs must be excluded from the Docker build context")
envs.symlink_to("/workspace/envs", target_is_directory=True)
