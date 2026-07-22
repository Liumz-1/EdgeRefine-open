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

Before running the experiments, create a `dataset/` directory and extract the archive contents into it.

```bash
mkdir dataset
unzip EdgeRefine-datasets.zip -d dataset
```

The current archive uses POSIX-style `/` separators and can also be extracted portably with Python:

```bash
python -m zipfile -e EdgeRefine-datasets.zip dataset
```

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
  Preprocessing time, normalized density, and optional GNN training-time evaluation for Tables 5-6.
- `exp_sampling_rate_sweep.py`
  Sampling-rate sweep used by Table 7.
- `exp_rr_only_ablation.py`
  Randomized-response-only ablation used by Figure 5.
- `exp_summary_metrics.py`
  Post-processing for the stability and privacy-utility metrics in Tables 3-4.

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

### Main EdgeRefine (recommended)

```bash
python exp_main_edgerefine.py
```

`exp_single_edgerefine.py` remains available as a more verbose single-seed grid entrypoint. Both scripts now preprocess each `(dataset, epsilon, seed)` graph once and reuse it across GAT, GCN, and GIN.

### Main Released Baselines

```bash
python exp_single_baselines.py
```

### No-Noise Baseline

```bash
python exp_no_noise_baseline.py
```

### Verbose Single-Seed Entry

```bash
python exp_single_edgerefine.py
```

### Method Variants

```bash
python exp_method_variant_comparison.py
```

### Error Metrics

```bash
python exp_error_metrics.py
```

This script directly outputs MAE, Brier score, ECE, pMAD, and the ECE/pMAD ratio. Use `--seed`, `--datasets`, and `--epsilons` to restrict a run.

### Runtime Evaluation

```bash
python exp_runtime_publish_time.py
```

Use `--skip-training` when only preprocessing time and graph density are needed. The default training-time run uses GAT; select another backbone with `--model`.

### GRAND Attack

```bash
python exp_grand_attack.py
```

The default command sweeps all seven privacy budgets. A smaller run can be selected explicitly, for example `python exp_grand_attack.py --epsilons 0.5 1.0`.

### Table 7 Sampling-Rate Sweep

```bash
python exp_sampling_rate_sweep.py
```

### Figure 5 RR-Only Ablation

```bash
python exp_rr_only_ablation.py
```

Pass multiple values to `--seeds` when repeated runs are required.

### Table 3/4 Summary Metrics

After generating EdgeRefine, baseline, and Origin CSV files, run:

```bash
python exp_summary_metrics.py \
  --edgerefine-csv result/exp_main_edgerefine/main_edgerefine_results.csv \
  --baseline-csvs result/exp_single_baselines/baseline_single_results.csv \
  --origin-csv result/exp_no_noise_baseline/no_noise_baseline_results.csv
```

This produces variance, CV, AUR, MCF, and PUBI for each method, dataset, and architecture.

### Link Prediction

```bash
python exp_link_prediction.py
python exp_link_prediction_base.py
```

Both scripts use a fixed, disjoint split of unique undirected edges. Positive test edges are removed from the graph before representation learning and are excluded from training negative sampling.

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

The single-seed workflow is intended as a deterministic, runnable artifact check. Paper values that report means, standard deviations, or significance tests require the corresponding repeated-seed configuration.

## Paper Result Map

| Paper result | Public entrypoint |
|---|---|
| Figure 3 main accuracy grid | `exp_main_edgerefine.py`, `exp_single_baselines.py`, `exp_no_noise_baseline.py` |
| Tables 3-4 stability and PUBI | `exp_summary_metrics.py` |
| Tables 5-6 density and timing | `exp_runtime_publish_time.py` |
| Table 7 sampling-rate analysis | `exp_sampling_rate_sweep.py` |
| Figure 4 probability errors | `exp_error_metrics.py` |
| Figure 5 RR-only ablation | `exp_rr_only_ablation.py` |
| Table 8 MUTAG | `exp_mutag_graph_classification.py` |
| Table 9 GRAND attack | `exp_grand_attack.py` |

## Runtime Notes

Jaccard estimation is the dominant preprocessing cost on the larger graphs. The implementation uses blockwise matrix multiplication to avoid the original Python all-pairs intersection loop, but full multi-dataset and multi-seed sweeps can still take substantial time. Use each script's `--help` options to run a small subset first. Preprocessed graphs are reused across the three GNN backbones whenever the graph construction does not depend on the model.

## Output Files

Every public script creates its own subdirectory under `result/` and writes CSV outputs there.

Examples:

- `result/exp_single_edgerefine/edgerefine_single_results.csv`
- `result/exp_single_baselines/baseline_single_results.csv`
- `result/exp_main_edgerefine/main_edgerefine_results.csv`

Result files are generated locally when you run experiments, but they are **not** included in the public GitHub release.

## Citation

This repository is the artifact for the EdgeRefine paper. The final paper citation will be added after the proceedings version becomes available.

Until the proceedings citation becomes available, please cite the repository URL.
```bibtex
@software{edgerefine_artifact_2026,
  title  = {EdgeRefine Artifact},
  author = {Liu and coauthors},
  year   = {2026},
  url    = {https://github.com/Liumz-1/EdgeRefine-open},
  note   = {Artifact for ACM CCS 2026 submission}
}
```
## Contact

For questions about the released code, datasets, or experiment scripts, please contact the repository maintainer through the project page or the corresponding paper contact information.
