import pandas as pd
from utils.cascade_labels import (
    STAGE1_MAP,
    STAGE2_MAP,
    ARM_STAGE3_MAP,
    LEG_STAGE3_MAP
)

# =============================
# Stage 1 dataset
# =============================
def build_stage1_df(df):
    stage_df = df[df["label"].isin(STAGE1_MAP.keys())].copy()
    stage_df["label"] = stage_df["label"].map(STAGE1_MAP)
    return stage_df


# =============================
# Stage 2 dataset (arm vs leg)
# =============================
def build_stage2_df(df):
    stage_df = df[df["label"].isin(STAGE2_MAP.keys())].copy()
    stage_df["label"] = stage_df["label"].map(STAGE2_MAP)
    return stage_df


# =============================
# Stage 3A dataset (arm side)
# =============================
def build_stage3_arm_df(df):
    stage_df = df[df["label"].isin(ARM_STAGE3_MAP.keys())].copy()
    stage_df["label"] = stage_df["label"].map(ARM_STAGE3_MAP)
    return stage_df


# =============================
# Stage 3B dataset (leg side)
# =============================
def build_stage3_leg_df(df):
    stage_df = df[df["label"].isin(LEG_STAGE3_MAP.keys())].copy()
    stage_df["label"] = stage_df["label"].map(LEG_STAGE3_MAP)
    return stage_df
