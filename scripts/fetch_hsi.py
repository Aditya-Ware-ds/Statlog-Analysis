"""Download Indian Pines, Pavia University and Salinas (.mat) and verify their SHA-256 digests.

The canonical host (www.ehu.eus, Grupo de Inteligencia Computacional, UPV/EHU)
is tried first; public GitHub mirrors are used as fall-backs. The digests
below were computed from files whose git blob hashes agree across independent
mirrors and whose dimensions and per-class counts match the published values.

    python scripts/fetch_hsi.py [--out data/hsi]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

EHU = "https://www.ehu.eus/ccwintco/uploads"
GH = "https://raw.githubusercontent.com"
FILES = {
    "Indian_pines_corrected.mat": (
        "ec2f8808710919d566f70f0d4aa885aae1ddfd42b734aba71c5e12ca65450939",
        [f"{EHU}/6/67/Indian_pines_corrected.mat",
         f"{GH}/researcher111/LearningHyperspectral/master/starter-Pines/Indian_pines_corrected.mat"],
    ),
    "Indian_pines_gt.mat": (
        "65c4687a8ab04f6da4789799bc3bc4f6e88bccac3ed6a2e6ae367e5e6b9e429c",
        [f"{EHU}/c/c4/Indian_pines_gt.mat",
         f"{GH}/researcher111/LearningHyperspectral/master/starter-Pines/Indian_pines_gt.mat"],
    ),
    "PaviaU.mat": (
        "28447fa87f7a5797845e9a189c0da85e23b1d06a4ba7361e5ff44efbf834d2fb",
        [f"{EHU}/e/ee/PaviaU.mat", f"{GH}/SphrGhfri/noisy-hyperspectral-classification/HEAD/Pavia/PaviaU.mat"],
    ),
    "PaviaU_gt.mat": (
        "23f6a426928f9b32984adffe659e29f554f9fb6c93b5a107528d308d5087a829",
        [f"{EHU}/5/50/PaviaU_gt.mat", f"{GH}/SphrGhfri/noisy-hyperspectral-classification/HEAD/Pavia/PaviaU_gt.mat"],
    ),
    "Salinas_corrected.mat": (
        "5ec1c0d22f56d18ecd336f8e35735863c0f160682e04e0c18ef3f89a3334d87d",
        [f"{EHU}/a/a3/Salinas_corrected.mat",
         f"{GH}/SphrGhfri/noisy-hyperspectral-classification/HEAD/Salinas/Salinas_corrected.mat"],
    ),
    "Salinas_gt.mat": (
        "ecfab4d31ef5553f097943235d8ea502038eb4a2067b2ad10b33e37c949955e2",
        [f"{EHU}/f/fa/Salinas_gt.mat", f"{GH}/SphrGhfri/noisy-hyperspectral-classification/HEAD/Salinas/Salinas_gt.mat"],
    ),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "hsi")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for name, (digest, urls) in FILES.items():
        dest = a.out / name
        if dest.exists() and sha256(dest) == digest:
            print(f"{name}: present, digest OK")
            continue
        for url in urls:
            try:
                urllib.request.urlretrieve(url, dest)
            except Exception as exc:
                print(f"{name}: {url} failed ({exc})", file=sys.stderr)
                continue
            if sha256(dest) == digest:
                print(f"{name}: downloaded from {url}, digest OK")
                break
            print(f"{name}: digest mismatch from {url}", file=sys.stderr)
        else:
            raise SystemExit(f"could not obtain a verified copy of {name}")


if __name__ == "__main__":
    main()
