# Reproducing Third-Party Sources and Assets

The repository excludes `data/`, `external/`, generated results, and internal research documents. The public sources and immutable release revisions are encoded in [`scripts/bootstrap_assets.py`](../scripts/bootstrap_assets.py).

## Provenance

| Component | Public source | Locked revision or status |
| --- | --- | --- |
| PushT dataset and LeWM checkpoints | `quentinll/lewm-pusht` on Hugging Face | Dataset `655cd446…`; model `22b330c2…` |
| Cube dataset and LeWM checkpoint | `quentinll/lewm-cube` on Hugging Face | Dataset `02a19a67…`; model `b0747c50…` |
| Runtime and reference code | `galilai-group/stable-worldmodel`, `lucas-maes/le-wm`, and `real-stanford/diffusion_policy` on GitHub | Exact commits in the bootstrap script |
| Retrained DINO-WM checkpoints | `fafraob/point-cloud-pusht` and `fafraob/point-cloud-ogb-cube` on Hugging Face | `664278a4…` and `ebf42b09…` |
| DINOv2 ViT-S/14 weights | Meta DINOv2 public release | Direct URL plus SHA-256 |

## Profiles

| Profile | Public inputs | Approximate post-download storage |
| --- | --- | --- |
| `core` | PushT dataset, LeWM PushT checkpoint, and three upstream repositories | 60 GB after extraction |
| `cube` | Cube dataset, LeWM Cube checkpoint, and runtime repositories | 148 GB after extraction |
| `dinowm` | DINOv2 ViT-S/14 weights plus released retrained DINO-WM models | 0.3 GB |

Archives are retained after extraction so their checksum can be independently verified. Plan for disk space above these figures while an archive and its unpacked dataset coexist.

## Commands

```bash
# Clone exact commits for the main PushT environment.
python scripts/bootstrap_assets.py sources --profile core

# Download, verify, and unpack the public inputs.
python scripts/bootstrap_assets.py assets --profile core

# Add Cube and retrained DINO-WM inputs as needed.
python scripts/bootstrap_assets.py sources --profile cube --profile dinowm
python scripts/bootstrap_assets.py assets --profile cube --profile dinowm

# Verify installed inputs without downloading.
python scripts/bootstrap_assets.py verify --profile core --verify-hashes
```

The script uses only Python's standard library plus the `zstd` executable for dataset extraction. It downloads public, revision-pinned Hugging Face resources directly and checks file size and SHA-256 before making a download visible at its final path.

## Manual inputs kept out of automation

The 4.3 GB Diffusion Policy checkpoint (`data/checkpoints/diffusion-policy-pusht/epoch0500.ckpt`) has no retained public source URL or training-run identifier. Obtain the exact project copy from the owner; its required SHA-256 is recorded by the bootstrap script. Do not substitute a similarly named checkpoint.

The original local DINOv2 source copy also lacks a recorded Git revision. The script intentionally does not replace it with the moving upstream default branch. Record and review a specific DINOv2 commit before attempting a strict reproduction of that extension.
