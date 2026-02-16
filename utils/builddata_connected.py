import os
import shutil
from pathlib import Path

SOURCE = r"D:\GEI\gait-model\data\Multiclass6"

OUT_BINARY = r"D:\GEI\gait-model\data\CascadeBinary"
OUT_REGION = r"D:\GEI\gait-model\data\CascadeRegion"
OUT_SIDE = r"D:\GEI\gait-model\data\CascadeSide"


STAGE1_MAP = {'nm': 'nm', 'fb': 'fb'}

STAGE2_MAP = {
    'a-r0': 'arm', 'a-l0': 'arm',
    'a-r0.5': 'arm', 'a-l0.5': 'arm',
    'l-r0.5': 'leg', 'l-l0.5': 'leg'
}

STAGE3_MAP = {
    'a-l0': 'left', 'a-l0.5': 'left', 'l-l0.5': 'left',
    'a-r0': 'right', 'a-r0.5': 'right', 'l-r0.5': 'right'
}


def extract_label(filename):
    name = filename.lower()

    for label in ['nm','fb','a-r0.5','a-l0.5','a-r0','a-l0','l-r0.5','l-l0.5']:
        if label in name:
            return label
    return None


def copy(dst_root, class_name, src):
    dst = Path(dst_root) / class_name
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)


for root, _, files in os.walk(SOURCE):
    for file in files:
        if not file.endswith(".png"):
            continue

        path = os.path.join(root, file)
        label = extract_label(file)

        if label is None:
            continue

        if label in STAGE1_MAP:
            copy(OUT_BINARY, STAGE1_MAP[label], path)

        if label in STAGE2_MAP:
            copy(OUT_REGION, STAGE2_MAP[label], path)

        if label in STAGE3_MAP:
            copy(OUT_SIDE, STAGE3_MAP[label], path)

print("Cascade datasets created.")
