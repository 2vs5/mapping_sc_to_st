# Installation (Conda)

```bash
git clone https://github.com/2vs5/mapping_sc_to_st.git
cd mapping_sc_to_st

conda create -n mapping_sc_to_st python=3.10.18 -y
conda activate mapping_sc_to_st

pip install -r requirements.txt
```
# mapping_sc_to_st

Tools to map a reference scRNA-seq atlas onto image-based spatial transcriptomics (ST) data (WeMERFISH zebrafish time course).

## abstract
Integrating single-cell RNA sequencing (scRNA-seq) with imaging-based spatial transcriptomics is essential for reconstructing spatially organized cellular states in developing tissues. However, this remains challenging in early embryogenesis, where signal mixing, segmentation uncertainty, and dynamically forming boundaries can distort local expression patterns and obscure biologically meaningful spatial structure.

This thesis presents a framework for robust single-cell-spatial transcriptomics mapping in early zebrafish embryogenesis. The approach begins with an initial similarity-based mapping and then refines it in two stages. First, spatially adjacent but transcriptionally non-proximal cell-type pairs are identified as candidate boundary regions affected by signal mixing. Directional signal-mixing genes are then detected based on attenuation of fold-change between the single-cell and spatial domains. Using these pair-specific gene sets, local pairwise refinement is performed with Fused Gromov Wasserstein (FGW) alignment under direction-specific gene exclusion. In the second stage, high-confidence spatial anchor cells are selected from the locally refined mapping and used to guide a global FGW refinement step, improving structural consistency across the tissue.

The framework was evaluated on weMERFISH data from three zebrafish developmental stages: 50\% epiboly, 75\% epiboly, and 6-somite. Across stages, the proposed framework more faithfully preserved spatial coherence, marker-associated expression structure, and biologically plausible tissue organization than the baseline methods. In quantitative reconstruction analysis, it consistently outperformed Moscot, while Tangram achieved the highest overall gene-wise Pearson correlation. However, when quantitative agreement was interpreted together with qualitative spatial mapping, marker-level evaluation, and robustness under gene hold-out, the proposed framework showed the strongest overall performance across stages. Leave-one-out validation further showed that the inferred correspondences remained comparatively stable when individual genes were excluded, supporting the robustness of the proposed approach.

Overall, these results suggest that reliable single-cell-spatial transcriptomics mapping in developing tissues requires not only transcriptomic agreement but also preservation of boundary-related and broader spatial structure. By explicitly accounting for local distortion near tissue boundaries during alignment, the proposed framework provides a biologically informed strategy for more robust spatial reconstruction and interpretation in embryonic systems.


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

## Tutorial
A minimal example based on publicly available data is provided in the `tutorial/` directory.
It is configured without absolute paths, so the full pipeline can be executed without modifying any local directory settings.

### Execution order
Run the notebooks in `example/` in the following order:
- `01.data_import.ipynb`

  Loads public scRNA-seq and spatial transcriptomics datasets, constructs AnnData objects, performs basic preprocessing and gene alignment, and saves the processed data under a configurable BASE_DIR.
- `02.run_example.ipynb`

  Loads the processed data, computes expression and geometry costs, runs global FGW mapping, applies anchor-based refinement, and generates example outputs and visualizations.

## Notebooks (00–04)
- `00.make_clusters_75_E1.ipynb`  
  Loads data, applies basic filters, and prepares clustering/embeddings used downstream.
- `01.explanation_75_E1.ipynb`  
  Detailed pipeline and code walk-through. This notebook is the reference for how `mapping_sc_to_st` runs.
- `02.example_75_E1_CV.ipynb`  
  Runs the pipeline end-to-end and performs leave-one-out (LOO) cross-validation.
- `03.cv_figure_75_E1.ipynb`  
  Plots CV results and compares against a baseline (e.g., moscot).
- `04.predicted_observed_75_E1.ipynb`  
  Predicted-vs-observed evaluation (all genes) and method comparisons (moscot / tangram).


