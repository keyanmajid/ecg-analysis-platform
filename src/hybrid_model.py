
import torch
import torch.nn as nn

from src.model import ECGCNN


class ECGHybridCNN(nn.Module):

    def __init__(self, num_classes=4, dropout=0.3):
        super().__init__()

        self.waveform_branch = ECGCNN(
            num_classes=num_classes,
            dropout=dropout
        )

        self.rhythm_branch = nn.Sequential(
            nn.Linear(4, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(160, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, waveform, rhythm):

        # Extract 128-dimensional CNN features directly.
        x = self.waveform_branch.features(waveform)
        x = self.waveform_branch.pool(x)
        waveform_features = torch.flatten(x, start_dim=1)

        # Extract 32-dimensional rhythm features.
        rhythm_features = self.rhythm_branch(rhythm)

        # Combine 128 + 32 = 160 features.
        combined = torch.cat(
            [waveform_features, rhythm_features],
            dim=1
        )

        return self.classifier(combined)
