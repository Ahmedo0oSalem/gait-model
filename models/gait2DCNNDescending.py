import torch
import torch.nn as nn

class Gait2DCNNParam(nn.Module):
    """
    Parametric CNN for single-frame gait recognition.
    You can control filters, dropouts, etc.
    """
    def __init__(self, num_classes=5, in_channels=1, filters=[128, 64, 32], dropouts=[0.1, 0.2, 0.3]):
        super().__init__()

        layers = []
        input_c = in_channels

        # Build each block dynamically
        for out_c, drop in zip(filters, dropouts):
            layers += [
                nn.Conv2d(input_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout(drop)
            ]
            input_c = out_c

        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.features = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters[-1], 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))
