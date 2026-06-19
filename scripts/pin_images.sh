#!/usr/bin/env bash
# Pin the GHCR tool images in config/methods.yaml to immutable @sha256 digests, so every
# lab pulls the EXACT bits that passed CI (reproducibility). Run AFTER the images are
# built + published (push a v* tag -> .github/workflows/build-images.yml), then commit
# the methods.yaml diff.
#
#   bash scripts/pin_images.sh
#   OWNER=rozadibrahim bash scripts/pin_images.sh
set -uo pipefail
OWNER="${OWNER:-rozadibrahim}"
YAML="config/methods.yaml"
TOOLS="cellpose stardist microsam omnipose magik trackmate deeptrace deepspt"

command -v docker >/dev/null 2>&1 || { echo "need docker (buildx) to read image digests"; exit 1; }

for t in $TOOLS; do
  ref="ghcr.io/$OWNER/smlm-$t"
  dig=$(docker buildx imagetools inspect "$ref:latest" --format '{{.Manifest.Digest}}' 2>/dev/null)
  if [ -z "$dig" ]; then echo "  skip $t (no published :latest yet)"; continue; fi
  python - "$YAML" "$ref" "$dig" <<'PY'
import re, sys
yaml, ref, dig = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(yaml, encoding="utf-8").read()
# replace  ghcr.io/<owner>/smlm-<tool>(:tag | @sha256:...)?  with the pinned digest
s = re.sub(re.escape(ref) + r"(:[\w.\-]+|@sha256:[0-9a-f]+)?", f"{ref}@{dig}", s)
open(yaml, "w", encoding="utf-8").write(s)
print(f"  pinned {ref}@{dig[:23]}...")
PY
done
echo "done -> review with:  git diff $YAML"
