# Installation (Conda)

```bash
git clone https://github.com/mittnenzweiglab/mapping-sc-to-st.git
cd mapping_sc_to_st

conda create -n mapping_sc_to_st python=3.10.18 -y
conda activate mapping_sc_to_st

pip install -r requirements.txt
```
# mapping_sc_to_st

Tools to map a reference scRNA-seq atlas onto image-based spatial transcriptomics (ST) data (WeMERFISH zebrafish time course).

## Background
scRNA-seq has rich cell-type structure but no coordinates; ST has coordinates but noisier expression.  
The pipeline links the two by solving an optimal-transport problem that balances **expression similarity** and **neighborhood/geometry consistency** (Fused Gromov-Wasserstein (FGW) Optimal Transport (OT)), then refines the match using anchors and cell-type–aware steps.

## How the code is organized
Core modules live in `mapping_sc_to_st/`:

- `prep.py`, `precomp.py` – filtering, gene alignment, normalization, and cached representations needed by the solver
- `m_cost.py`, `geometry.py` – cost terms (expression / spatial structure) used in mapping
- `fgw_solver.py`, `transport_engine.py`, `solver_adapter.py` – FGW / OT solvers and wrappers
- `global_map.py` – global 1:1 mapping (best-match ST location per reference cell)
- `pairwise_refine.py`, `unpaired_single_refine.py` – refinement within / between cell types
- `anchors.py`, `final_global_anchor_fgw.py`, `alignment.py` – anchor selection and anchor-based FGW updates
- `merge.py`, `bleeding.py`, `gene_correlation.py` – merging/refinement utilities and evaluation helpers
- `run_pipeline.py` – end-to-end orchestration used by the example notebooks

Dependencies are listed in `requirements.txt`.

## Notebooks (00–04)
- `00.make_clusters_75_E1.ipynb`  
  Loads data, applies basic filters, and prepares clustering / embeddings used downstream.
- `01.explanation_75_E1.ipynb`  
  Detailed pipeline and code walk-through. This notebook is the reference for how `mapping_sc_to_st` runs.
- `02.example_75_E1_CV.ipynb`  
  Runs the pipeline end-to-end and performs leave-one-out (LOO) cross-validation.
- `03.cv_figure_75_E1.ipynb`  
  Plots CV results and compares against a baseline (e.g., moscot).
- `04.predicted_observed_75_E1.ipynb`  
  Predicted-vs-observed evaluation (all genes) and method comparisons (moscot / tangram).


