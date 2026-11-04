
import torch
import torch.nn as nn


class ECGCNN(nn.Module):

    def __init__(self, num_classes=4, dropout=0.3):
        super().__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv1d(
                1, 32,
                kernel_size=7,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            # Block 2
            nn.Conv1d(
                32, 64,
                kernel_size=5,
                padding=2,
                bias=False,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            # Block 3
            nn.Conv1d(
                64, 128,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):

        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)

        return x


if __name__ == "__main__":

    model = ECGCNN(num_classes=4)

    dummy_input = torch.randn(256, 1, 216)

    output = model(dummy_input)

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    print("ECG CNN architecture test")
    print("Input:", dummy_input.shape)
    print("Output:", output.shape)
    print(f"Parameters: {total_params:,}")

    assert output.shape == (256, 4)

    print("Model test passed.")
