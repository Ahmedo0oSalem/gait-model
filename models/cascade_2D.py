import torch
import torch.nn as nn
import torch.nn.functional as F


class Gait2DCNNDescendingCascade(nn.Module):
    """
    CNN for hierarchical cascade learning with fine-tuning support.

    Usage:
    - Train Level 1 (normal vs fullbody)
    - Load weights → fine-tune Level 2 (arm vs leg)
    - Load weights → fine-tune Level 3 (left vs right)

    Input shape: [B, 1, H, W]
    """

    def __init__(self, num_classes=2, in_channels=1):
        super().__init__()

        # ----------------------------
        # FEATURE EXTRACTOR (BACKBONE)
        # ----------------------------
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        # ----------------------------
        # CLASSIFIER HEAD (BINARY)
        # ----------------------------
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)  # always 2 for your case
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    # --------------------------------------------------
    # LOAD WEIGHTS FROM PREVIOUS STAGE (FOR CASCADE)
    # --------------------------------------------------
    def load_previous_stage(self, path):
        """
        Load weights from previous model stage and fine-tune.

        Example:
        model.load_previous_stage("model_level1.pth")
        """

        state_dict = torch.load(path, map_location="cpu")
        self.load_state_dict(state_dict)

        print(f"✅ Loaded weights from {path} (ready for fine-tuning)")
