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
Integrating single-cell RNA sequencing with imaging-based spatial transcriptomics is essential for reconstructing spatially organized cellular states in developing tissues. In imaging-based spatial transcriptomics of developing embryos, signal mixing between adjacent cell populations can distort measured expression profiles near tissue boundaries, causing existing integration methods to misassign cells at these interfaces. 

This thesis presents a signal-mixing robust framework for mapping single-cell states to spatial transcriptomics data during early zebrafish embryogenesis. The framework consists of three main components. First, spatially adjacent but transcriptionally non-proximal cell-type pairs are identified as candidate boundary regions likely to be affected by signal mixing. Second, directional signal-mixing-associated genes are identified based on attenuation of fold-change contrast between the single-cell and spatial domains. Third, the initial mapping is refined through a two-stage Fused Gromov–Wasserstein (FGW) strategy that combines local pairwise correction with anchor-guided global refinement to improve structural consistency across the tissue. The framework was evaluated on whole-embryo MERFISH data from three zebrafish developmental stages: 50\% epiboly, 75\% epiboly, and 6-somite. 

Across stages, the proposed framework better preserved the spatial organization of cell types near tissue boundaries, including the EVL at epiboly stages and the notochord at the 6-somite stage. In quantitative reconstruction analysis, it consistently outperformed Moscot, whereas Tangram achieved the highest overall gene-wise Pearson correlation. However, Tangram achieved high correlation by distributing mapping weights broadly across the single-cell reference, resulting in an improved correlation between imputed and observed spatial gene expression values, but strong differences in the observed and predicted dynamic expression ranges. Leave-one-out validation further showed that the inferred correspondences remained more stable than those of Moscot when individual genes were excluded. These results show that gene-wise correlation alone is not sufficient to evaluate SC–ST mapping quality under signal contamination, and that cell-type-specific spatial structure should be assessed independently. By explicitly accounting for local distortion near tissue boundaries during alignment, the proposed framework provides a biologically informed strategy for robust spatial reconstruction.


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


