from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the pinned MediaPipe hand model")
    parser.add_argument("--force", action="store_true", help="Replace an existing model")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    destination = project_root / "models" / "hand_landmarker.task"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not args.force:
        print(f"Model already exists: {destination}")
        return 0

    temporary = destination.with_suffix(".task.download")
    try:
        print(f"Downloading {MODEL_URL}")
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out)
        if temporary.stat().st_size < 1_000_000:
            raise RuntimeError("Downloaded model is unexpectedly small")
        temporary.replace(destination)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        print(f"Model download failed: {exc}", file=sys.stderr)
        return 1

    print(f"Saved model to {destination} ({destination.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

