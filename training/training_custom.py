import torch
from sklearn.metrics import accuracy_score
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import inspect
import os
import pandas as pd  # ✅ for CSV logging

from data.data_preprocessor import load_gait_sequences
from data.dataset import GaitFrameDataset
from models.gait2DCNNDescending import Gait2DCNNParam
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

# =========================
#  Evaluation Function
# =========================
def evaluate_model(model, data_loader, device, use_seq_len, use_tqdm=True, label="Validation"):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        loop = tqdm(data_loader, desc=f"Evaluating {label}", leave=False) if use_tqdm else data_loader
        for x, y, *extras in loop:
            x, y = x.to(device), y.to(device)
            seq_lengths = extras[0].to(device) if use_seq_len and extras else None
            outputs = model(x, seq_lengths) if use_seq_len and seq_lengths is not None else model(x)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    print(f"{label} Accuracy: {acc:.4f}")
    return acc


# =========================
#  Training Function
# =========================
def train_model(model, train_loader, val_loader, criterion, optimizer, device, scheduler=None, num_epochs=10, use_tqdm=True):
    model = model.to(device)
    use_seq_len = "seq_lengths" in inspect.signature(model.forward).parameters
    train_acc, val_acc = 0, 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False) if use_tqdm else train_loader

        for x, y, *extras in loop:
            x, y = x.to(device), y.to(device)
            seq_lengths = extras[0].to(device) if use_seq_len and extras else None
            optimizer.zero_grad()
            outputs = model(x, seq_lengths) if use_seq_len and seq_lengths is not None else model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if scheduler:
            scheduler.step()

        avg_loss = running_loss / len(train_loader)
        train_acc = evaluate_model(model, train_loader, device, use_seq_len, use_tqdm=False, label="Train")
        val_acc = evaluate_model(model, val_loader, device, use_seq_len, use_tqdm=False, label="Val")
        print(f"Epoch {epoch}/{num_epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    return train_acc, val_acc


# =========================
#  K-Fold Function
# =========================
def run_kfold_training(
    df,
    model_class,
    dataset_class,
    num_classes,
    k_folds=5,
    epochs=20,
    batch_size=4,
    lr=1e-3,
    num_workers=1,
    use_tqdm=True,
    use_tvl1=False,
    flow_augment=None,
    aggregate_lr_labels=False,
    config_name="Unknown_Config",
    csv_path="results.csv"
):
    device = torch.device("cuda" if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    train_accuracies = []
    val_accuracies = []
    labels = df['label'].values
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    os.makedirs("saved_models", exist_ok=True)

    # ✅ Initialize CSV if it doesn’t exist
    if not os.path.exists(csv_path):
        pd.DataFrame(columns=["Config", "Fold", "Train_Acc", "Val_Acc", "Avg_Train_Acc", "Avg_Val_Acc"]).to_csv(csv_path, index=False)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, labels)):
        print(f"\n--- Fold {fold_idx + 1} ---")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        train_dataset = dataset_class(dataframe=train_df, aggregate_lr_labels=aggregate_lr_labels)
        print("Unique labels in dataset:", train_dataset.data['label'].unique())
        val_dataset = dataset_class(dataframe=val_df, aggregate_lr_labels=aggregate_lr_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        model = model_class(num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=lr)

        train_acc, val_acc = train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=epochs, use_tqdm=use_tqdm)
        model_path = f"saved_models/{config_name}_fold_{fold_idx + 1}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # ✅ Log each fold result immediately
        pd.DataFrame([{
            "Config": config_name,
            "Fold": fold_idx + 1,
            "Train_Acc": train_acc,
            "Val_Acc": val_acc,
            "Avg_Train_Acc": np.mean(train_accuracies),
            "Avg_Val_Acc": np.mean(val_accuracies)
        }]).to_csv(csv_path, mode='a', header=False, index=False)

    print(f"\n📊 Config: {config_name}")
    print(f"Average Train Acc: {np.mean(train_accuracies):.4f}")
    print(f"Average Val Acc: {np.mean(val_accuracies):.4f}")

    return val_accuracies


# =========================
#  MAIN SCRIPT
# =========================
if __name__ == "__main__":
    df = load_gait_sequences(r"D:\GEI\gait-model\data\Multiclass6", load_images=False)
    print("Unique labels in dataset:", df['label'].unique())
    print("Number of unique labels:", df['label'].nunique())

    configs = [
        {"filters": [128, 64, 32], "dropouts": [0.1, 0.2, 0.3]},
        {"filters": [64, 128, 256], "dropouts": [0.2, 0.3, 0.4]},
        {"filters": [32, 64, 128], "dropouts": [0.3, 0.4, 0.5]},
    ]

    results_csv = "results.csv"

    for i, cfg in enumerate(configs, 1):
        config_name = f"Config_{i}_filters_{cfg['filters']}_dropouts_{cfg['dropouts']}"
        print(f"\n🚀 Running {config_name}")

        #Num_classes is the parameter you pass when calling this lambda function
        model_class = lambda num_classes: Gait2DCNNParam(
            num_classes=num_classes,
            in_channels=1,
            filters=cfg["filters"],
            dropouts=cfg["dropouts"]
        )

        accuracies = run_kfold_training(
            df=df,
            model_class=model_class,
            dataset_class=GaitFrameDataset,
            num_classes=6,
            k_folds=5,
            epochs=10,
            batch_size=32,
            lr=1e-3,
            num_workers=2,
            use_tqdm=True,
            aggregate_lr_labels=False,
            config_name=config_name,
            csv_path=results_csv
        )

        print(f"✅ {config_name} | Avg Val Acc: {np.mean(accuracies):.4f}")
