import os
import shutil
from pathlib import Path

# ============================
# CONFIG
# ============================

RAW_DATASET = r"D:\GEI\gait-model\data2"  # original images
OUTPUT_ROOT = r"processed_dataset"

VALID_CLASSES = {
    'nm', 'l-r0.5', 'l-l0.5', 'fb', 'a-r0.5', 'a-l0.5', 'a-r0', 'a-l0'
}

# ============================
# LABEL PARSER
# ============================

def parse_label(filename):
    """
    Extract class token from filename.

    Example:
    101_90_fb_02_Seq_3.png -> fb
    105_90_a-r0_01_Seq_2.png -> a-r0
    """

    parts = filename.split("_")
    if len(parts) < 3:
        return None

    label = parts[2]

    if label in VALID_CLASSES:
        return label

    return None


# ============================
# CLASS MAPPERS
# ============================

def stage1_binary(label):
    if label == "nm":
        return "normal"
    if label == "fb":
        return "fullbody"
    return None


def stage2_region(label):
    if label.startswith("a-"):
        return "arm"
    if label.startswith("l-"):
        return "leg"
    return None


def stage3_arm_side(label):
    if label.startswith("a-l"):
        return "left_arm"
    if label.startswith("a-r"):
        return "right_arm"
    return None


def stage3_leg_side(label):
    if label.startswith("l-l"):
        return "left_leg"
    if label.startswith("l-r"):
        return "right_leg"
    return None


# ============================
# HELPERS
# ============================

def copy_to_stage(src, stage_dir, class_name):
    dst_dir = Path(stage_dir) / class_name
    dst_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(src, dst_dir / Path(src).name)


# ============================
# MAIN BUILDER
# ============================

def build_dataset():

    stage1_dir = Path(OUTPUT_ROOT) / "stage1_binary"
    stage2_dir = Path(OUTPUT_ROOT) / "stage2_region"
    stage3_arm_dir = Path(OUTPUT_ROOT) / "stage3_arm_side"
    stage3_leg_dir = Path(OUTPUT_ROOT) / "stage3_leg_side"

    image_count = 0

    for root, _, files in os.walk(RAW_DATASET):

        for file in files:

            if not file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            full_path = os.path.join(root, file)

            label = parse_label(file)
            if label is None:
                continue

            # ---------- Stage 1 ----------
            s1 = stage1_binary(label)
            if s1:
                copy_to_stage(full_path, stage1_dir, s1)

            # ---------- Stage 2 ----------
            s2 = stage2_region(label)
            if s2:
                copy_to_stage(full_path, stage2_dir, s2)

            # ---------- Stage 3 ARM ----------
            s3_arm = stage3_arm_side(label)
            if s3_arm:
                copy_to_stage(full_path, stage3_arm_dir, s3_arm)

            # ---------- Stage 3 LEG ----------
            s3_leg = stage3_leg_side(label)
            if s3_leg:
                copy_to_stage(full_path, stage3_leg_dir, s3_leg)

            image_count += 1

    print(f"\nDone. Processed {image_count} images.")
    print("Output:", OUTPUT_ROOT)


if __name__ == "__main__":
    build_dataset()
