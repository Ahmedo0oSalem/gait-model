import torch
from pathlib import Path

from data.data_preprocessor import load_gait_sequences
from data.dataset import GaitFrameDataset
from models.gait2DCNNDescending import Gait2DCNNParam
from models.cascade_2D import Gait2DCNNDescendingCascade
from training.training import run_kfold_training
from utils.cascade_df_builder import (
    build_stage1_df,
    build_stage2_df,
    build_stage3_arm_df,
    build_stage3_leg_df
)
from utils.model_utils import load_partial_weights

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Train a single stage
# -------------------------------
from pathlib import Path
import torch

def train_stage(df, num_classes, save_name, pretrained_path=None,epochs=10):

    
    save_dir = Path.cwd() / "saved_models"
    save_dir.mkdir(parents=True, exist_ok=True)

    def model_builder(num_classes=num_classes):
        model = Gait2DCNNDescendingCascade(num_classes=num_classes)

        if pretrained_path:
            load_partial_weights(model, pretrained_path, DEVICE)

        return model

    # run training
    accuracies = run_kfold_training(
        df=df,
        model_class=model_builder,
        dataset_class=GaitFrameDataset,
        num_classes=num_classes,
        k_folds=5,
        epochs=epochs,
        batch_size=32,
        lr=1e-3,
        num_workers=2,
        use_tqdm=True,
        aggregate_lr_labels=False
    )

    # ------------------------------------------------
    # SELECT BEST FOLD MODEL
    # ------------------------------------------------
    best_fold = accuracies.index(max(accuracies)) + 1

    best_fold_path = save_dir / f"model_builder_fold_{best_fold}_best.pth"
    final_stage_path = save_dir / save_name

    torch.save(torch.load(best_fold_path), final_stage_path)

    print(f"\n✅ Best fold = {best_fold}")
    print(f"✅ Stage model saved → {final_stage_path}")

    return final_stage_path



# -------------------------------
# Full Cascade Training
# -------------------------------
if __name__ == "__main__":
    import os
    print(Path.cwd())


    processed_path = r"D:\GEI\gait-model\processed_dataset"
    Path("saved_modelscascade").mkdir(exist_ok=True)

    print("\n====================")
    print("STAGE 1: Normal vs Fullbody")
    print("====================")
    df_stage1 = load_gait_sequences(r"D:\GEI\processed_dataset\stage1_binary", load_images=False)
    print(df_stage1.columns)
    print(df_stage1.sample(10))
    print("Total frames:", len(df_stage1))
    print("Class distribution:\n", df_stage1['label'].value_counts())
    df_stage1 = build_stage1_df(df_stage1)
    print("Class distribution:\n", df_stage1['label'].value_counts())
    stage1_weights = train_stage(
        df_stage1,
        num_classes=2,
        save_name="stage1_best.pth",
        pretrained_path=None,
        epochs=10
    )

    print("\n====================")
    print("STAGE 2: Arm vs Leg")
    print("====================")
    df_stage2 = load_gait_sequences(r"\GEI\processed_dataset\stage2_region", load_images=False)
    print(df_stage2.columns)
    print(df_stage2.sample(10))
    print("Total frames:", len(df_stage2))
    print("Class distribution:\n", df_stage2['label'].value_counts())
    df_stage2 = build_stage2_df(df_stage2)
    print("Class distribution:\n", df_stage2['label'].value_counts())

    stage2_weights = train_stage(
        df_stage2,
        num_classes=2,
        save_name="stage2_best.pth",
        pretrained_path=stage1_weights,
        epochs=10
    )

    print("\n====================")
    print("STAGE 3: Arm Side (Left vs Right)")
    print("====================")
    df_stage3_arm = load_gait_sequences(r"\GEI\processed_dataset\stage3_arm_side", load_images=False)
    print(df_stage3_arm.columns)
    print(df_stage3_arm.sample(10))
    print("Total frames:", len(df_stage3_arm))
    print("Class distribution:\n", df_stage3_arm['label'].value_counts())
    df_stage3_arm = build_stage3_arm_df(df_stage3_arm)
    print("Class distribution:\n", df_stage3_arm['label'].value_counts())

    stage3_arm_weights = train_stage(
        df_stage3_arm,
        num_classes=2,
        save_name="stage3_arm_best.pth",
        pretrained_path=stage2_weights,
        epochs=10
    )

    print("\n====================")
    print("STAGE 3: Leg Side (Left vs Right)")
    print("====================")
    df_stage3_leg = load_gait_sequences(r"\GEI\processed_dataset\stage3_leg_side", load_images=False)
    print(df_stage3_leg.columns)
    print(df_stage3_leg.sample(10))
    print("Total frames:", len(df_stage3_leg))
    print("Class distribution:\n", df_stage3_leg['label'].value_counts())
    df_stage3_leg = build_stage3_leg_df(df_stage3_leg)
    print("Class distribution:\n", df_stage3_leg['label'].value_counts())
    stage3_leg_weights = train_stage(
        df_stage3_leg,
        num_classes=2,
        save_name="stage3_leg_best.pth",
        pretrained_path=stage2_weights,
        epochs=10
    )

    print("\n🎯 CASCADE TRAINING COMPLETE")
