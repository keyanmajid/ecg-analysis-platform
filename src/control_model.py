
import torch
import torch.nn as nn

from src.hybrid_model import ECGHybridCNN


class ECGWaveformControl(nn.Module):
    """
    Hybrid-matched waveform-only control.

    Reuses the hybrid's exact waveform branch and classifier
    initialization. Rhythm input is replaced by 32 zeros.
    """

    def __init__(self, num_classes=4, dropout=0.3):
        super().__init__()

        reference = ECGHybridCNN(
            num_classes=num_classes,
            dropout=dropout
        )

        self.waveform_branch = reference.waveform_branch
        self.classifier = reference.classifier

        del reference

    def forward(self, waveform):
        x = self.waveform_branch.features(waveform)
        x = self.waveform_branch.pool(x)
        waveform_features = torch.flatten(x, start_dim=1)

        # No rhythm information is supplied.
        rhythm_placeholder = waveform_features.new_zeros(
            waveform_features.size(0), 32
        )

        combined = torch.cat(
            [waveform_features, rhythm_placeholder],
            dim=1
        )

        return self.classifier(combined)
