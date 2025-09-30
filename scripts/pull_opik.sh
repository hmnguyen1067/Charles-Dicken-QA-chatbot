# /bin/bash

REPO_URL="https://github.com/comet-ml/opik.git"
DEST_DIR="opik"

@if [ -d "$$DEST_DIR" ]; then
    echo "Directory $$DEST_DIR already exists. Skipping clone."
    else
    git clone "$$REPO_URL" "$$DEST_DIR"
fi
