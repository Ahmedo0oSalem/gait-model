import torch
from sklearn.metrics import accuracy_score
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import inspect
import os

from data.data_preprocessor import load_gait_sequences
from data.dataset import GaitFrameDataset  # Updated dataset import
from models.gaitLSTM import GEIConvLSTMClassifier
from models.gait3DCNN import Gait3DCNNClassifier
from models.gaitFlow3DCNN import Flow3DCNNClassifier
from utils.visualization import visualize_fold_accuracies
from data.kcv import run_kfold_cross_validation
from models.gait2DCNN import Gait2DCNN
from data.dataset import GaitFrameSequenceDataset
from data.dataset import GaitOpticalFlowDataset


def evaluate_model(model, data_loader, device, use_seq_len, use_tqdm=True, label="Validation"):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        loop = tqdm(data_loader, desc=f"Evaluating {label}", leave=False) if use_tqdm else data_loader
        for x, y, *extras in loop:
            x = x.to(device)
            y = y.to(device)
            seq_lengths = extras[0].to(device) if use_seq_len and extras else None

            outputs = model(x, seq_lengths) if use_seq_len and seq_lengths is not None else model(x)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"{label} Accuracy: {acc:.4f}")
    return acc

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    scheduler=None,
    num_epochs=10,
    use_tqdm=True
):
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

        # 👇 Evaluate training & validation accuracy for this epoch
        train_acc = evaluate_model(model, train_loader, device, use_seq_len, use_tqdm=False, label="Train")
        val_acc = evaluate_model(model, val_loader, device, use_seq_len, use_tqdm=False, label="Val")

        print(f"Epoch {epoch}/{num_epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    return train_acc, val_acc  # from final epoch

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
    aggregate_lr_labels=False
):
    from torch.utils.data import DataLoader
    from sklearn.model_selection import StratifiedKFold

    device = torch.device("cuda" if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    train_accuracies = []
    val_accuracies = []

    labels = df['label'].values
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    # 🔨 ADDED: Setup save directory
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, labels)):
        print(f"\n--- Fold {fold_idx + 1} ---")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        aggregate = aggregate_lr_labels
        if isinstance(dataset_class, GaitOpticalFlowDataset):
            train_dataset = dataset_class(
                dataframe=train_df,
                train_augmentations=flow_augment,
                use_tvl1=use_tvl1
            )
            val_dataset = dataset_class(
                dataframe=val_df,
                train_augmentations=None,
                use_tvl1=use_tvl1,
                label_to_index=train_dataset.label_to_index  # Ensure same label mapping
            )
        else:
            train_dataset = dataset_class(
                dataframe=train_df,
                aggregate_lr_labels=aggregate
            )
            print("Unique labels in dataset:", train_dataset.data['label'].unique())


            val_dataset = dataset_class(
                dataframe=val_df,
                aggregate_lr_labels=aggregate
            )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        model = model_class(num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=lr)

        train_acc, val_acc = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            device,
            num_epochs=epochs,
            use_tqdm=use_tqdm
        )

        # 💾 ADDED: Model Saving Logic
        model_filename = os.path.join(save_dir, f"{model_class.__name__}_fold_{fold_idx + 1}.pth")
        torch.save(model.state_dict(), model_filename)
        print(f"Model saved to {model_filename}")
        # 💾 END Model Saving Logic

        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

    print(f"\nAverage Training Accuracy: {np.mean(train_accuracies):.4f}")
    print(f"Average Validation Accuracy: {np.mean(val_accuracies):.4f}")

    return val_accuracies

if __name__ == "__main__":
    df = load_gait_sequences(r"D:\GEI\gait-model\data\Multiclass4", load_images=False)
    for root, dirs, files in os.walk(r"D:\GEI\gait-model\data\Multiclass4"):
        print(root)
    print(df['label'].nunique())
    print("Unique labels in dataset:", df['label'].unique())
    print("Number of unique labels:", df['label'].nunique())
    accuracies = run_kfold_training(
        df=df,
        model_class=Gait2DCNN,
        dataset_class=GaitFrameDataset,
        num_classes=6,
        k_folds=5,
        epochs=10,
        batch_size=32,
        lr=1e-3,
        num_workers=2,
        use_tqdm=True,
        use_tvl1=False,
        flow_augment=...,
        aggregate_lr_labels=True
    )

    visualize_fold_accuracies(accuracies)