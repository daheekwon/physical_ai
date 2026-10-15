#!/usr/bin/env python3
"""Provision the public, pinned inputs excluded from version control.

The command never downloads generated results or internal documents.  It only
clones public source repositories and downloads public data/checkpoints listed
below.  Each download is written to a ``.partial`` path, size- and SHA-256-
verified, then atomically moved into place.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]

SOURCES = {
    "stable-worldmodel": ("external/stable-worldmodel", "https://github.com/galilai-group/stable-worldmodel.git", "21446f1ede6d5284e981bd7b47f432b994e6d812"),
    "le-wm": ("external/le-wm", "https://github.com/lucas-maes/le-wm.git", "8edfeb336732b5f3ce7b8b210d0ba370a09e2cac"),
    "diffusion-policy": ("external/diffusion-policy", "https://github.com/real-stanford/diffusion_policy.git", "5ba07ac6661db573af695b419a7947ecb704690f"),
    "dino-wm": ("external/dino-wm", "https://github.com/gaoyuezhou/dino_wm.git", "0a9492fa12044b852ae9e001cc74604b79c8bb0c"),
}

ASSETS = [
    # Official LeWorldModel PushT release, revision 655cd446b9929369d7d406001da85c15d1457850 (MIT).
    {"id": "pusht-archive", "profiles": {"core"}, "url": "https://huggingface.co/datasets/quentinll/lewm-pusht/resolve/655cd446b9929369d7d406001da85c15d1457850/pusht_expert_train.h5.zst", "path": "data/downloads/lewm-pusht/pusht_expert_train.h5.zst", "bytes": 13136247974, "sha256": "7cfbd6d90fa2f27876379a5ff169715a36ed82edbda64f9e5b5bfa34d212f318"},
    {"id": "pusht-dataset", "profiles": {"core"}, "extract": "zstd-copy", "archive": "pusht-archive", "path": "data/datasets/pusht_expert_train.h5", "bytes": 46300921856},
    # Official LeWM model release, revision 22b330c28c27ead4bfd1888615af1340e3fe9052 (MIT).
    {"id": "lewm-pusht-config", "profiles": {"core"}, "url": "https://huggingface.co/quentinll/lewm-pusht/resolve/22b330c28c27ead4bfd1888615af1340e3fe9052/config.json", "path": "data/checkpoints/lewm-pusht/config.json", "bytes": 1313, "sha256": "2e534ee5c6286721c0ef967e24bddbfde482abf9"},
    {"id": "lewm-pusht-weights", "profiles": {"core"}, "url": "https://huggingface.co/quentinll/lewm-pusht/resolve/22b330c28c27ead4bfd1888615af1340e3fe9052/weights.pt", "path": "data/checkpoints/lewm-pusht/weights.pt", "bytes": 72290721, "sha256": "48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607"},
    # Official LeWorldModel Cube release, revision 02a19a67a0dc8c9d6215f89c19e0a597691e152a (MIT).
    {"id": "cube-archive", "profiles": {"cube"}, "url": "https://huggingface.co/datasets/quentinll/lewm-cube/resolve/02a19a67a0dc8c9d6215f89c19e0a597691e152a/cube_single_expert.tar.zst", "path": "data/downloads/lewm-cube/cube_single_expert.tar.zst", "bytes": 46184624478, "sha256": "3725d6a01abd492164441ef0a27e588f52b94a118fab56b96987b1a34a6c2600"},
    {"id": "cube-dataset", "profiles": {"cube"}, "extract": "zstd-tar-member", "archive": "cube-archive", "member": "cube_single_expert.h5", "path": "data/datasets/lewm-cube/cube_single_expert.h5", "bytes": 101942558720},
    # Official LeWM model release, revision b0747c5002e86d2ce8f3cd8178004b97524c587d (MIT).
    {"id": "lewm-cube-config", "profiles": {"cube"}, "url": "https://huggingface.co/quentinll/lewm-cube/resolve/b0747c5002e86d2ce8f3cd8178004b97524c587d/config.json", "path": "data/checkpoints/lewm-cube/config.json", "bytes": 1313, "sha256": "f9a946cdc87e3f7139de9e8eecccc2a436007425"},
    {"id": "lewm-cube-weights", "profiles": {"cube"}, "url": "https://huggingface.co/quentinll/lewm-cube/resolve/b0747c5002e86d2ce8f3cd8178004b97524c587d/weights.pt", "path": "data/checkpoints/lewm-cube/weights.pt", "bytes": 72291425, "sha256": "2839a907362f403f9136383016e91774373a295d958ae75121791f22a9fddf89"},
    # DINOv2 ViT-S/14 (Apache-2.0) and Point-LeWM's public retrained DINO-WM releases (CC-BY-4.0).
    {"id": "dinov2-vits14", "profiles": {"dinowm"}, "url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth", "path": "data/checkpoints/dinov2-frozen-audit/dinov2_vits14_pretrain.pth", "bytes": 88283115, "sha256": "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"},
    {"id": "dinowm-pusht-model", "profiles": {"dinowm"}, "url": "https://huggingface.co/fafraob/point-cloud-pusht/resolve/664278a4c0b4c5f607d0993744986b2d10b4168e/dino-wm/checkpoints/model_latest.pth", "path": "data/checkpoints/dinowm-retrained-fafraob/pusht/checkpoints/model_latest.pth", "bytes": 88837855, "sha256": "14489cfdf4132a126fd1cb3cac39370a1fb2141503cd3e41aeb3c2dead271dd0"},
    {"id": "dinowm-pusht-config", "profiles": {"dinowm"}, "url": "https://huggingface.co/fafraob/point-cloud-pusht/resolve/664278a4c0b4c5f607d0993744986b2d10b4168e/dino-wm/hydra.yaml", "path": "data/checkpoints/dinowm-retrained-fafraob/pusht/hydra.yaml", "bytes": 904, "sha256": "1f2c7767f06a82ae8d86a8be168e303b5941b5105e8cff5060e9e5cb54f3f3a2"},
    {"id": "dinowm-pusht-stats", "profiles": {"dinowm"}, "url": "https://huggingface.co/fafraob/point-cloud-pusht/resolve/664278a4c0b4c5f607d0993744986b2d10b4168e/dino-wm/swm_h5_norm_stats.json", "path": "data/checkpoints/dinowm-retrained-fafraob/pusht/swm_h5_norm_stats.json", "bytes": 571, "sha256": "90619b3b9e0534f223e19eac595330dd426e3319b832ec3717f7ec90d2fe5de5"},
    {"id": "dinowm-cube-model", "profiles": {"dinowm"}, "url": "https://huggingface.co/fafraob/point-cloud-ogb-cube/resolve/ebf42b0905f83634a5bfa5f721fca7b8c70f355a/dino-wm/checkpoints/model_latest.pth", "path": "data/checkpoints/dinowm-retrained-fafraob/ogb-cube/checkpoints/model_latest.pth", "bytes": 88838431, "sha256": "f1c103a876becd0041bbecc41503260fc9ab4d34ec8877f30d5014cf3e63e325"},
    {"id": "dinowm-cube-config", "profiles": {"dinowm"}, "url": "https://huggingface.co/fafraob/point-cloud-ogb-cube/resolve/ebf42b0905f83634a5bfa5f721fca7b8c70f355a/dino-wm/hydra.yaml", "path": "data/checkpoints/dinowm-retrained-fafraob/ogb-cube/hydra.yaml", "bytes": 903, "sha256": "44021feff43776e7ef62a764597efdce6be75d72152e75125d2bdefed28e269c"},
    {"id": "dinowm-cube-stats", "profiles": {"dinowm"}, "url": "https://huggingface.co/fafraob/point-cloud-ogb-cube/resolve/ebf42b0905f83634a5bfa5f721fca7b8c70f355a/dino-wm/swm_h5_norm_stats.json", "path": "data/checkpoints/dinowm-retrained-fafraob/ogb-cube/swm_h5_norm_stats.json", "bytes": 1807, "sha256": "4024831da0a5d3a27f315ae25428bf23c03dbe781e0ec2fc9f19ffd803256f59"},
]

PROFILE_SOURCES = {"core": {"stable-worldmodel", "le-wm", "diffusion-policy"}, "cube": {"stable-worldmodel", "le-wm"}, "dinowm": {"dino-wm"}}
MANUAL = "Diffusion Policy epoch0500.ckpt is intentionally manual: its source URL/training run was not retained (expected SHA-256 bac7221f7e34cd51162dc1972e1a39ffcddc87de1dc1780c44ffa61b88c4ff76). The local DINOv2 source copy also has no recorded Git revision, so it is not replaced by a moving upstream branch."


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(asset: dict, hashes: bool) -> bool:
    path = ROOT / asset["path"]
    if not path.is_file() or path.stat().st_size != asset["bytes"]:
        return False
    return not hashes or "sha256" not in asset or sha256(path) == asset["sha256"]


def selected_assets(profiles: set[str]) -> list[dict]:
    return [asset for asset in ASSETS if asset["profiles"] & profiles]


def clone(source_id: str) -> None:
    relative, url, revision = SOURCES[source_id]
    path = ROOT / relative
    if path.exists():
        if not (path / ".git").is_dir():
            raise RuntimeError(f"Refusing to replace non-Git directory: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", url, str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "fetch", "--quiet", "origin", revision], check=True)
    subprocess.run(["git", "-C", str(path), "checkout", "--detach", "--quiet", revision], check=True)
    actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if actual != revision:
        raise RuntimeError(f"{source_id}: expected {revision}, got {actual}")
    print(f"SOURCE OK  {source_id} @ {actual}")


def download(asset: dict, force: bool) -> None:
    target = ROOT / asset["path"]
    if target.exists():
        if verify(asset, hashes=True):
            print(f"ASSET OK   {asset['id']}")
            return
        if not force:
            raise RuntimeError(f"Mismatched existing file: {target}; inspect it or rerun with --force")
        target.unlink()
    partial = target.with_name(target.name + ".partial")
    target.parent.mkdir(parents=True, exist_ok=True)
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(asset["url"], headers={"User-Agent": "zj-wam-bootstrap", **({"Range": f"bytes={offset}-"} if offset else {})})
    with urllib.request.urlopen(request, timeout=90) as response:
        resumed = offset and response.getcode() == 206
        if offset and not resumed:
            partial.unlink()
            offset = 0
        mode = "ab" if resumed else "wb"
        with partial.open(mode) as output:
            for block in iter(lambda: response.read(8 * 1024 * 1024), b""):
                output.write(block)
    if partial.stat().st_size != asset["bytes"] or sha256(partial) != asset["sha256"]:
        raise RuntimeError(f"Verification failed for {asset['id']}; partial file retained at {partial}")
    os.replace(partial, target)
    print(f"DOWNLOADED {asset['id']}")


def extract(asset: dict, by_id: dict[str, dict], force: bool) -> None:
    target = ROOT / asset["path"]
    if target.exists():
        if target.stat().st_size == asset["bytes"]:
            print(f"EXTRACT OK  {asset['id']}")
            return
        if not force:
            raise RuntimeError(f"Mismatched existing file: {target}; inspect it or rerun with --force")
        target.unlink()
    if shutil.which("zstd") is None:
        raise RuntimeError("zstd is required to extract this asset")
    archive = ROOT / by_id[asset["archive"]]["path"]
    partial = target.with_name(target.name + ".partial")
    if partial.exists():
        raise RuntimeError(f"Remove or inspect incomplete extraction first: {partial}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if asset["extract"] == "zstd-copy":
        with partial.open("xb") as output:
            subprocess.run(["zstd", "-dc", str(archive)], stdout=output, check=True)
    else:
        process = subprocess.Popen(["zstd", "-dc", str(archive)], stdout=subprocess.PIPE)
        assert process.stdout is not None
        with tarfile.open(fileobj=process.stdout, mode="r|") as tar:
            member = next((item for item in tar if item.name == asset["member"]), None)
            if member is None or not member.isfile() or PurePosixPath(member.name).is_absolute() or ".." in PurePosixPath(member.name).parts:
                raise RuntimeError("Expected safe archive member was not found")
            source = tar.extractfile(member)
            assert source is not None
            with partial.open("xb") as output:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
        if process.wait() != 0:
            raise RuntimeError("zstd failed while extracting Cube archive")
    if partial.stat().st_size != asset["bytes"]:
        raise RuntimeError(f"Extraction size check failed for {asset['id']}; partial retained")
    os.replace(partial, target)
    print(f"EXTRACTED  {asset['id']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("sources", "assets", "verify"))
    parser.add_argument("--profile", action="append", choices=("core", "cube", "dinowm"), default=[])
    parser.add_argument("--force", action="store_true", help="replace a mismatched downloaded/extracted file")
    parser.add_argument("--verify-hashes", action="store_true", help="hash all selected files during verification")
    args = parser.parse_args()
    profiles = set(args.profile or ["core"])
    if args.action == "sources":
        for source_id in sorted(set().union(*(PROFILE_SOURCES[p] for p in profiles))):
            clone(source_id)
        print("NOTE:", MANUAL)
        return
    assets = selected_assets(profiles)
    if args.action == "verify":
        missing = [asset["id"] for asset in assets if not verify(asset, args.verify_hashes)]
        print("NOTE:", MANUAL)
        if missing:
            raise SystemExit("Missing or invalid: " + ", ".join(missing))
        print("ALL SELECTED ASSETS VERIFIED")
        return
    by_id = {asset["id"]: asset for asset in ASSETS}
    for asset in assets:
        if "url" in asset:
            download(asset, args.force)
        else:
            extract(asset, by_id, args.force)


if __name__ == "__main__":
    main()
