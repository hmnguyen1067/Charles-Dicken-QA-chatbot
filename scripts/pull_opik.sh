#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/comet-ml/opik.git"
DEST_DIR="opik"

if [ -d "$DEST_DIR" ]; then
  echo "Directory '$DEST_DIR' exists. Skip cloning."
else
  echo "Cloning Opik from $REPO_URL into '$DEST_DIR'..."
  git clone --depth 1 "$REPO_URL" "$DEST_DIR"
fi
