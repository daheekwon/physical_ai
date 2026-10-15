# ZJ-WAM: Executed-Outcome Validation for Action Corrections

This repository contains the experiment code for ZJ-WAM, a research project on reducing robot-action jerk while preserving the intended object trajectory. It evaluates locally generated action corrections with frozen world models, controller-based baselines, and executed outcomes in PushT and Cube-style environments.

The code focuses on reproducible experiment logic: perturbation construction, response and sensitivity analysis, candidate selection, physical-execution checks, diagnostics, and statistical summaries. Model checkpoints, datasets, generated artifacts, and internal research documents are intentionally not part of this repository.

## Repository layout

- `zj_wam/` — reusable experiment, simulation, metric, selection, and analysis modules.
- `scripts/` — runnable experiment pipelines, audits, analyses, and reports.
- `configs/` — versioned JSON configuration files for the experiment suites.
- `tests/` — unit and behavioral tests for core experimental logic.
- `requirements-experiment.txt` — pinned Python dependencies used by the experiments.

## Setup

Use Python 3.11 or a compatible environment. Install the project dependencies, including a PyTorch build appropriate for your platform and CUDA setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-experiment.txt
```

Several pipelines require external simulators, pretrained checkpoints, and expert datasets. Those assets are not distributed here; update the paths in the relevant file under `configs/` before running a pipeline.

## Running checks

Run the test suite from the repository root:

```bash
pytest -q
```

Experiment entry points are in `scripts/`. Each pipeline reads its corresponding JSON configuration from `configs/` and writes generated files below `artifacts/` or `outputs/`, both of which are ignored by Git.

## Data and artifact policy

To keep the repository lightweight and shareable, Git ignores datasets, checkpoints, third-party working copies, generated results, caches, and internal documentation. Reproduce an experiment by provisioning the required assets locally and configuring their paths; do not commit those assets or generated outputs.

## Scope

The repository provides research code and experimental infrastructure. It does not package the underlying world-model weights, benchmark datasets, or third-party repositories, which remain subject to their own licenses and access terms.
