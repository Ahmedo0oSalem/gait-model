import torch
import torch.nn as nn

class Gait2DCNNParam(nn.Module):
    """
    Parametric CNN for single-frame gait recognition.
    Supports:
    - custom kernel size
    - activation choice
    - pooling type
    - batchnorm toggle
    - custom classifier hidden size
    """

    def __init__(
        self,
        num_classes=5,
        in_channels=1,
        filters=[128, 64, 32],
        dropouts=[0.1, 0.1, 0.1],  # ✅ LOW default dropout
        kernel_size=3,
        activation="relu",
        pool_type="max",
        use_batchnorm=True,
        classifier_hidden=64,
        global_pool=True
    ):
        super().__init__()

        assert len(filters) == len(dropouts), \
            "filters and dropouts must be same length"

        activations = {
            "relu": nn.ReLU(inplace=True),
            "leaky": nn.LeakyReLU(0.1, inplace=True),
            "elu": nn.ELU(inplace=True)
        }
        act_layer = activations[activation]

        if pool_type == "max":
            pool_layer = nn.MaxPool2d(2)
        else:
            pool_layer = nn.AvgPool2d(2)

        layers = []
        input_c = in_channels

        for out_c, drop in zip(filters, dropouts):
            layers.append(nn.Conv2d(
                input_c, out_c,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ))

            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_c))

            layers.append(act_layer)
            layers.append(pool_layer)

            if drop > 0:
                layers.append(nn.Dropout(drop))

            input_c = out_c

        if global_pool:
            layers.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.features = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters[-1], classifier_hidden),
            act_layer,
            nn.Dropout(0.1),  # ✅ reduced & consistent
            nn.Linear(classifier_hidden, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))
