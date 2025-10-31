import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import os
from torchvision import models
import matplotlib.pyplot as plt
import random

from data.data_preprocessor import load_gait_sequences
from data.dataset import GaitFrameDataset
from models.gait2DCNN import Gait2DCNN


# ----------------------------
# 🔹 Evaluation function
# ----------------------------
def evaluate_model(model, dataloader, device, class_names):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"✅ Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(all_labels, all_preds, target_names=class_names))
    print("\nConfusion Matrix:\n", confusion_matrix(all_labels, all_preds))
    return acc, np.array(all_preds), np.array(all_labels)


# ----------------------------
# 🔹 Pretrained ResNet baseline
# ----------------------------
def get_pretrained_resnet(num_classes):
    model = models.resnet18(pretrained=True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ----------------------------
# 🔹 Visualization helper
# ----------------------------
def visualize_predictions(imgs, labels, preds, class_names, title):
    plt.figure(figsize=(12, 4))
    indices = random.sample(range(len(imgs)), min(8, len(imgs)))
    for i, idx in enumerate(indices):
        plt.subplot(2, 4, i + 1)
        img = imgs[idx].squeeze(0).cpu().numpy()
        true_label = class_names[labels[idx]]
        pred_label = class_names[preds[idx]]
        plt.imshow(img, cmap='gray')
        plt.title(f"T:{true_label}\nP:{pred_label}", fontsize=10)
        plt.axis('off')
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


# ----------------------------
# 🔹 Main testing function
# ----------------------------
def test_trained_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tests = [
        {
            "name": "Multiclass6",
            "path": r"D:\GEI\gait-model\data\Multiclass6",
            "model_path": r"D:\GEI\gait-model\saved_models\Multi6\Gait2DCNN_fold_5.pth",
            "num_classes": 6
        },
        {
            "name": "Multiclass4",
            "path": r"D:\GEI\gait-model\data\Multiclass4",
            "model_path": r"D:\GEI\gait-model\saved_models\Multi4\Gait2DCNN_fold_2.pth",
            "num_classes": 6  # trained with 6 outputs
        },
        {
            "name": "Binary",
            "path": r"D:\GEI\gait-model\data\Binary",
            "model_path": r"D:\GEI\gait-model\saved_models\Binary\Gait2DCNN_fold_3.pth",
            "num_classes": 2
        }
    ]

    for test in tests:
        print("\n" + "=" * 80)
        print(f"🧠 Testing models on: {test['name']}")

        # Load dataset
        df = load_gait_sequences(test["path"], load_images=False)
        dataset = GaitFrameDataset(dataframe=df, return_metadata=False)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        class_names = sorted(dataset.label_to_index.keys())
        print("Detected classes:", class_names)

        # ----------------------
        # 🔹 Load YOUR model
        # ----------------------
        model_custom = Gait2DCNN(num_classes=test["num_classes"]).to(device)
        model_custom.load_state_dict(torch.load(test["model_path"], map_location=device))
        print("\nEvaluating your trained Gait2DCNN:")
        acc_custom, preds_custom, labels_custom = evaluate_model(model_custom, dataloader, device, class_names)

        # ----------------------
        # 🔹 Load pretrained ResNet baseline
        # ----------------------
        model_resnet = get_pretrained_resnet(num_classes=test["num_classes"]).to(device)
        print("\nEvaluating pretrained ResNet18 baseline:")
        acc_resnet, preds_resnet, labels_resnet = evaluate_model(model_resnet, dataloader, device, class_names)

        # ----------------------
        # 🔹 Visualization: compare predictions
        # ----------------------
        imgs, labels = next(iter(dataloader))
        visualize_predictions(imgs, labels, preds_custom[:len(labels)], class_names, f"{test['name']} – Gait2DCNN Predictions")
        visualize_predictions(imgs, labels, preds_resnet[:len(labels)], class_names, f"{test['name']} – ResNet18 Predictions")

        # ----------------------
        # 🔹 Print Summary
        # ----------------------
        print(f"\n📊 Comparison on {test['name']}:")
        print(f"   Gait2DCNN accuracy: {acc_custom:.4f}")
        print(f"   ResNet18 baseline : {acc_resnet:.4f}")


if __name__ == "__main__":
    test_trained_models()
