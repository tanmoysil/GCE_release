"""
Centralised file paths for analysis scripts.

All paths are relative to the repository root. Analysis scripts should
import from here rather than defining their own CONFIG dicts.
"""
import os

# Repository root — one level up from this config/ directory
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset paths
DATASETS_DIR = os.path.join(REPO_ROOT, "datasets")
HALFAMP_MAT = os.path.join(DATASETS_DIR, "X_y_wo_tremor_10_wfooof_halfamp.mat")
FULLAMP_MAT = os.path.join(DATASETS_DIR, "X_y_wo_tremor_10_wfooof_fullamp.mat")
HALFAMP_ALL_CONDITIONS_MAT = os.path.join(DATASETS_DIR, "X_y_all_conditions_85_55_wfooof_halfamp_test.mat")
FULLAMP_ALL_CONDITIONS_MAT = os.path.join(DATASETS_DIR, "X_y_all_conditions_85_55_wfooof_fullamp.mat")
WOFOOOF_HALFAMP_MAT = os.path.join(DATASETS_DIR, "X_y_wo_tremor_10_wofooof_halfamp.mat")
P11_HALFAMP_MAT = os.path.join(DATASETS_DIR, "X_y_P11_test_halfamp.mat")
P11_FULLAMP_MAT = os.path.join(DATASETS_DIR, "X_y_P11_test_fullamp.mat")
P11_SCORES_CSV = os.path.join(DATASETS_DIR, "P11_scores.csv")

# Kinematic cohort (separate study, same center; three patients)
_KINEMATIC_DIR = os.path.join(
    DATASETS_DIR,
    "new_test_cohort",
    "Scores-20260616T191500Z-3-001",
    "Scores",
)
KINEMATIC_MAT          = os.path.join(_KINEMATIC_DIR, "X_y_London_P3P4P5.mat")
KINEMATIC_FT_DELTA_CSV = os.path.join(_KINEMATIC_DIR, "London_FingerTap_delta_PC_scores.csv")
KINEMATIC_HM_DELTA_CSV = os.path.join(_KINEMATIC_DIR, "London_HandMovement_delta_PC_scores.csv")

# Clinical scores
SCORES_HALFAMP_XLSX = os.path.join(DATASETS_DIR, "norm_wo_tremor_halfamp.xlsx")
SCORES_FULLAMP_XLSX = os.path.join(DATASETS_DIR, "norm_wo_tremor_fullamp.xlsx")

# Pretrained models — point these at your own trained checkpoints.
# (Encoder filenames follow the convention
#  <model>_encoder_best_K{K}_L{L}_lamb{lambda}_causal-{score}.pt)
PRETRAINED_MODELS_DIR = os.path.join(REPO_ROOT, "pretrained_models")
BEST_ENCODER_PT = os.path.join(
    PRETRAINED_MODELS_DIR, "wfooof", "Run1",
    "pd_all_encoder_best_K1_L5_lamb0.068_causal-0.4808.pt",
)
BEST_CLASSIFIER = os.path.join(PRETRAINED_MODELS_DIR, "pd_all")
