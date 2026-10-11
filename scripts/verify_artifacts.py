"""Verify that all historical sample checkpoints retain their baseline bytes."""

import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "artifacts/manifest.json").read_text(encoding="utf-8"))
    for record in manifest["checkpoints"]:
        path = root / record["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Historical checkpoint changed: {path}")
        for key in ("config", "prediction_fixture"):
            if not (root / record[key]).is_file():
                raise FileNotFoundError(record[key])
    print(f"Verified {len(manifest['checkpoints'])} unchanged historical checkpoints")


if __name__ == "__main__":
    main()
