import torch
import torch.nn as nn
import torch.nn.functional as F

class Gait2DCNNDescending(nn.Module):
    """
    CNN for single-frame gait recognition.
    Input shape: [B, 1, H, W]
    """
    def __init__(self, num_classes=5, in_channels=1):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1 - start with high number of filters
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3 - fewer filters at the deepest layer
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Adaptive pooling to fixed size output (1x1)
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# 🔍 Test
if __name__ == "__main__":
    model = Gait2DCNNDescending(num_classes=4)
    x = torch.randn(8, 1, 224, 224)  # Example input tensor
    out = model(x)
    print(out.shape)  # Should output: torch.Size([8, 4])
