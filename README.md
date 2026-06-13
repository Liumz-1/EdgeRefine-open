# EdgeRefine

Official open-source implementation of **EdgeRefine**, a privacy-preserving graph learning framework for node classification, graph classification, link prediction, attack evaluation, and calibration-oriented graph analysis.

## Overview

This repository is the public release of the `EdgeRefine 4openscience (2)` codebase prepared for paper submission and academic reproducibility.

The release includes:

- the shared model definitions for `GAT`, `GCN`, and `GIN`
- the complete graph perturbation and post-processing pipeline used by EdgeRefine
- public experiment scripts organized by paper experiment category
- a dataset preparation workflow for all released experiments, including `dblp`, `acm`, `cora`, `amap`, and `MUTAG`

This release does **not** include:

- experiment result files
- internal statistical testing scripts
- internal scripts for exporting or reloading preprocessed matrices

## Repository Structure

```text
EdgeRefine/
|- dataset/                     # Create this directory and extract the released dataset archive here
|  |- acm/
|  |- amap/
|  |- cora/
|  |- dblp/
|  `- TUDataset/MUTAG/
|- experiment_utils.py          # Shared utilities for loading data, preprocessing, training, and seeding
|- model.py                     # GAT, GCN, and GIN implementations
|- structure.py                 # EdgeRefine, Blink, DPRR, LAPGRAPH, LDPGen, and graph post-processing
|- grand_attack_core.py         # Shared GRAND attack implementation
|- exp_*.py                     # Public experiment entrypoints
`- requirements.txt             # Minimal Python dependencies for the released code
```

All experiment outputs are written to `result/` at runtime. The `result/` directory is not included in the public release.

## Installation

### Environment

- Python `3.10` is recommended
- PyTorch `2.5.0`
- PyTorch Geometric `2.6.1`

### Dependencies

Install the required packages with:

```bash
pip install -r requirements.txt
```

## Datasets

The released code expects a local `dataset/` directory. The datasets are provided as a compressed archive `EdgeRefine-datasets.zip` in the repository root.

Before running the experiments, unzip it into the repository root. This will create the expected `dataset/` directory used by the experiment scripts.

- `EdgeRefine-datasets.zip`

After downloading the archive, create a `dataset/` directory under the repository root if it does not already exist, then extract the archive **into `dataset/`**.

Example target layout after extraction:

```text
EdgeRefine/
`- dataset/
   |- acm/
   |- amap/
   |- cora/
   |- dblp/
   `- TUDataset/MUTAG/
```

In other words, after extraction the following paths should exist:

- `dataset/acm/acm_adj.npy`
- `dataset/amap/amap_adj.npy`
- `dataset/cora/cora_adj.npy`
- `dataset/dblp/dblp_adj.npy`
- `dataset/TUDataset/MUTAG/raw/MUTAG_A.txt`

### Node Classification

- `dblp`
- `acm`
- `cora`
- `amap`

These datasets are used by the EdgeRefine main experiments, baseline comparisons, method-variant experiments, runtime analysis, attack evaluation, and link prediction experiments.

### Graph Classification

- `MUTAG`

`MUTAG` should be available under `dataset/TUDataset/MUTAG` and is used by the graph classification script.

## Public Experiment Scripts

Each public script corresponds to a paper experiment category or a public evaluation entrypoint.

### Main Node Classification

- `exp_single_edgerefine.py`
  Single-run public EdgeRefine benchmark over the default node-classification datasets, privacy budgets, and GNN backbones.
- `exp_main_edgerefine.py`
  Main EdgeRefine experiment entrypoint for the released node-classification setting.
- `exp_no_noise_baseline.py`
  Original-graph baseline without privacy perturbation.

### Baseline Comparisons

- `exp_single_baselines.py`
  Single-run public baseline benchmark over `blink_hard`, `blink_hybrid`, `dprr`, `lapgraph`, and `ldpgen`.
- `exp_blink_hard_comparison.py`
  Blink-hard comparison experiment.
- `exp_blink_hybrid_comparison.py`
  Blink-hybrid comparison experiment.
- `exp_other_baselines_comparison.py`
  Comparison against other released baselines.

### Method Variants and Analysis

- `exp_method_variant_comparison.py`
  Variant comparison for:
  - `jaccard + simple`
  - `jaccard + isotonic`
  - `jaccard + beta`
  - `jaccard + temperature`
  - `jaccard + histogram`
  - `adamic_adar + histogram`
- `exp_error_metrics.py`
  Probability calibration and graph-estimation error analysis, including `Brier`, `ECE`, and `pMAD`.
- `exp_runtime_publish_time.py`
  Runtime and timing-oriented evaluation.

### Attack and Link Prediction

- `exp_grand_attack.py`
  GRAND attack evaluation.
- `exp_link_prediction.py`
  Link prediction on privacy-processed graphs.
- `exp_link_prediction_base.py`
  Link prediction on the original graph as a reference baseline.

### Graph Classification

- `exp_mutag_graph_classification.py`
  MUTAG graph classification experiment. This script keeps its original repeated-run evaluation protocol and is intentionally preserved as-is.

## Running Experiments

Below are minimal public commands for the released experiments.

### Main EdgeRefine

```bash
python exp_single_edgerefine.py
```

### Main Released Baselines

```bash
python exp_single_baselines.py
```

### No-Noise Baseline

```bash
python exp_no_noise_baseline.py
```

### Main Released Entry

```bash
python exp_main_edgerefine.py
```

### Method Variants

```bash
python exp_method_variant_comparison.py
```

### Error Metrics

```bash
python exp_error_metrics.py
```

### Runtime Evaluation

```bash
python exp_runtime_publish_time.py
```

### GRAND Attack

```bash
python exp_grand_attack.py
```

### Link Prediction

```bash
python exp_link_prediction.py
python exp_link_prediction_base.py
```

### MUTAG Graph Classification

```bash
python exp_mutag_graph_classification.py
```

## Reproducing Main Results

For the released node-classification experiments, the public default setup uses a **single reproducible run with `seed=42`**.

This applies to the released single-run public entrypoints:

- `exp_single_edgerefine.py`
- `exp_single_baselines.py`
- `exp_no_noise_baseline.py`
- `exp_main_edgerefine.py`

The `MUTAG` graph classification script is an intentional exception and preserves its original repeated-run protocol.

## Output Files

Every public script creates its own subdirectory under `result/` and writes CSV outputs there.

Examples:

- `result/exp_single_edgerefine/edgerefine_single_results.csv`
- `result/exp_single_baselines/baseline_single_results.csv`
- `result/exp_main_edgerefine/main_edgerefine_results.csv`

Result files are generated locally when you run experiments, but they are **not** included in the public GitHub release.

## Citation

This repository is the artifact for the EdgeRefine paper. The final paper citation will be added after the proceedings version becomes available.

For now, please cite the archived artifact DOI generated by Zenodo for the CCS 2026 artifact release.

```bibtex
@software{edgerefine_artifact_2026,
  title  = {EdgeRefine Artifact},
  author = {Liu, Mingzhe and coauthors},
  year   = {2026},
  url    = {https://github.com/Liumz-1/EdgeRefine-open},
  note   = {Artifact for ACM CCS 2026 submission}
}

## Contact

For questions about the released code, datasets, or experiment scripts, please contact the repository maintainer through the project page or the corresponding paper contact information.
