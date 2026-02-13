import torch
from sklearn.metrics import accuracy_score
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import inspect
import os
from pathlib import Path

# Imports from your project structure
from data.data_preprocessor import load_gait_sequences
from data.dataset import GaitFrameDataset, GaitFrameSequenceDataset, GaitOpticalFlowDataset
from models.gaitLSTM import GEIConvLSTMClassifier
from models.gait3DCNN import Gait3DCNNClassifier
from models.gaitFlow3DCNN import Flow3DCNNClassifier
from utils.visualization import visualize_fold_accuracies
from data.kcv import run_kfold_cross_validation
from models.gait2DCNN import Gait2DCNN
from models.gait2DCNNDes import Gait2DCNNDescending
from models.gait2DCNNDescending import Gait2DCNNParam
from models.cascade_2D import Gait2DCNNDescendingCascade

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
    
    best_val_acc = 0.0
    best_model_state = None
    # We track the training accuracy that corresponded to the BEST validation moment
    associated_train_acc = 0.0

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

        # Evaluate at the end of every epoch
        train_acc = evaluate_model(model, train_loader, device, use_seq_len, use_tqdm=False, label="Train")
        val_acc = evaluate_model(model, val_loader, device, use_seq_len, use_tqdm=False, label="Val")

        # 👇 Checkpoint: Update best weights and record the associated accuracies
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            associated_train_acc = train_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"🌟 Epoch {epoch}: Best Val Acc updated to {best_val_acc:.4f}")

        print(f"Epoch {epoch}/{num_epochs} | Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    return best_model_state, best_val_acc, associated_train_acc

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
    
    # Trackers for the summary
    fold_results = []

    labels = df['label'].values
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    save_dir = Path("saved_models")
    save_dir.mkdir(exist_ok=True)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, labels)):
        print(f"\n" + "="*20)
        print(f"STARTING FOLD {fold_idx + 1}")
        print("="*20)

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        
        if dataset_class == GaitOpticalFlowDataset:
            train_dataset = dataset_class(dataframe=train_df, train_augmentations=flow_augment, use_tvl1=use_tvl1)
            val_dataset = dataset_class(dataframe=val_df, train_augmentations=None, use_tvl1=use_tvl1, 
                                        label_to_index=train_dataset.label_to_index)
        else:
            train_dataset = dataset_class(dataframe=train_df, aggregate_lr_labels=aggregate_lr_labels)
            val_dataset = dataset_class(dataframe=val_df, aggregate_lr_labels=aggregate_lr_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_loader_dataset:=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        model = model_class(num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=lr)

        # Train and get the best weights found during this fold
        best_weights, best_v_acc, best_t_acc = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            device,
            num_epochs=epochs,
            use_tqdm=use_tqdm
        )

        # 💾 Save best weights for this fold
        model_filename = save_dir / f"{model_class.__name__}_fold_{fold_idx + 1}_best.pth"
        torch.save(best_weights, model_filename)
        
        fold_results.append({'fold': fold_idx + 1, 'train_acc': best_t_acc, 'val_acc': best_v_acc})
        print(f"✅ Fold {fold_idx + 1} Complete. Weights saved to {model_filename}")

    # --- FINAL SUMMARY PRINTING ---
    print(f"\n" + "!"*40)
    print(f"{'FOLD':<10} | {'TRAIN ACC':<12} | {'VAL ACC':<12}")
    print("-" * 40)
    for res in fold_results:
        print(f"Fold {res['fold']:<5} | {res['train_acc']:<12.4f} | {res['val_acc']:<12.4f}")
    
    avg_train = np.mean([r['train_acc'] for r in fold_results])
    avg_val = np.mean([r['val_acc'] for r in fold_results])
    
    print("-" * 40)
    print(f"{'AVERAGE':<10} | {avg_train:<12.4f} | {avg_val:<12.4f}")
    print("!" * 40 + "\n")

    return [r['val_acc'] for r in fold_results]

if __name__ == "__main__":
    data_path = r"D:\GEI\gait-model\data\CascadeBinary"
    df = load_gait_sequences(data_path, load_images=False)
    df_stage1 = df[df['label'].isin(['nm', 'fb'])].copy()
    
    accuracies = run_kfold_training(
        df=df_stage1,
        model_class=Gait2DCNNDescendingCascade,
        dataset_class=GaitFrameDataset,
        num_classes=2,
        k_folds=5,
        epochs=10,
        batch_size=32,
        lr=1e-3,
        num_workers=2,
        use_tqdm=True,
        aggregate_lr_labels=False
    )

    visualize_fold_accuracies(accuracies)