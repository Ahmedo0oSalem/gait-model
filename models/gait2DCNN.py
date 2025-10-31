import torch
import torch.nn as nn

class Gait2DCNN(nn.Module):
    """
    CNN for single-frame gait recognition.
    Input shape: [B, 1, H, W]
    """
    def __init__(self, num_classes=5, in_channels=1):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: [B, 1, H, W]
        # ✅ No permutation needed
        x = self.features(x)
        x = self.classifier(x)
        return x


# 🔍 Test
if __name__ == "__main__":
    model = Gait2DCNN(num_classes=4)
    x = torch.randn(8, 1, 224, 224)  # from your GaitFrameDataset
    out = model(x)
    print(out.shape)  # ✅ [8, 4]
