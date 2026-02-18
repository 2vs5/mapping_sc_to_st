"""
Single source of truth for AnnData keys used across the pipeline.

Rule
----
- All modules must import keys from here.
- Raw vs normalized representations must be explicitly separated.
"""

# ============================================================
# Prefixes
# ============================================================
GLOBAL_PREFIX = "global_sc"
FINAL_PREFIX = "final_pairwise"

# ============================================================
# AnnData .obsm keys (SPATIAL)UNS_SC_KNN_EDGES
# ============================================================

# --- ST spatial coordinates ---
OBSM_ST_SPATIAL = "spatial"
"""
Raw spatial coordinates of ST spots.

- NEVER normalized
- Dataset-dependent units (pixel / micron)
- Used for visualization and reference only
"""

OBSM_ST_SPATIAL_NORMED = "spatial_normed"
OBSM_ST_UMAP = "X_umap" # ST UMAP (scanpy default)
"""
Normalized spatial coordinates of ST spots.

- Typically normalized by median kNN distance
- Used for distance computation, FGW geometry, anchor cost, proximity
"""

# Optional metadata for normalization
UNS_ST_SPATIAL_NORM_INFO = "spatial_norm_info"

# ============================================================
# AnnData .obsm keys (ST) — Final global anchor FGW (optional)
# ============================================================

# FGW-imputed expression panel stored on ST
OBSM_IMPUTED_FINAL_GLOBAL_ANCHOR = "X_imputed_final_global_anchor"

# ============================================================
# AnnData .obsm keys (SC)
# ============================================================

OBSM_SC_LATENT = "X_scvi" # SC latent space
OBSM_SC_ST_REP = "st_anchor_rep" # SC representative ST coordinate

# ============================================================
# AnnData .obs keys (SC)
# ============================================================

OBS_SC_HAS_REP = "has_st_anchor_rep"

# ============================================================
# AnnData .obs keys (ST) — Global mapping
# ============================================================

OBS_GLOBAL_SIM = "global_sc_sim"
OBS_GLOBAL_TYPE = "global_sc_type"
OBS_GLOBAL_BEST_CELL = "global_sc_best_cell"
OBS_GLOBAL_BEST_CELL_IDX = "global_sc_best_cell_idx"
OBS_GLOBAL_BEST_CELL_TYPE = "global_sc_best_cell_type"

OBS_GLOBAL_ANCHOR = "global_sc_anchor"
OBS_GLOBAL_ANCHOR_TYPE = "global_sc_anchor_type"

# ============================================================
# AnnData .obs keys (ST) — Final outputs
# ============================================================

OBS_FINAL_TYPE = "final_pairwise_type"
OBS_FINAL_SCORE = "final_pairwise_score"

OBS_FINAL_ANCHOR_TYPE = "final_pairwise_anchor_type"
OBS_FINAL_ANCHOR      = "final_pairwise_anchor"

OBS_FINAL_GLOBAL_TYPE = 'global_final_type'
OBS_FINAL_GLOBAL_SCORE = 'global_final_score'

# Mask of spots that were updated by final global anchor FGW
OBS_FGW_UPDATED_MASK = "final_global_anchor_fgw_updated"

# ============================================================
# AnnData .uns keys
# ============================================================

UNS_SC_KNN_EDGES = "sc_knn_edges"

UNS_TYPEPAIR_RESULTS = "typepair_proximity_results"
UNS_TYPEPAIR_PAIRS = "typepair_pairs_selected"
UNS_TYPEPAIR_PAIRS_ALL = "typepair_pairs_all_neighbors"

UNS_FINAL_WEIGHTS = "final_pairwise_weights"
UNS_FINAL_PAIR_WEIGHTS = "final_pairwise_pair_weights"

UNS_GLOBAL_GENES_UNION = "global_sc_genes_union"

UNS_GLOBAL_FINAL_WEIGHTS = "final_global_weights"

# ============================================================
# AnnData .obs keys (SC) — Anchor usage stats (optional)
# ============================================================

OBS_SC_IS_ANCHOR = "is_anchor_sc"
OBS_SC_ANCHOR_N_SPOTS = "anchor_n_spots"
OBS_SC_ANCHOR_MAX_SCORE = "anchor_max_score"

# ============================================================
# Compatibility aliases (legacy constant names)
# ============================================================

# Older downstream scripts used names with the substring "_SC_" even though
# they refer to ST .obs fields.
OBS_GLOBAL_SC_BEST_CELL_IDX = OBS_GLOBAL_BEST_CELL_IDX
OBS_GLOBAL_SC_SIM = OBS_GLOBAL_SIM
UNS_GLOBAL_SC_GENES_UNION = UNS_GLOBAL_GENES_UNION
