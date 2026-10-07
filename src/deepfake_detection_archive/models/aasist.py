"""Typed boundary for the MIT-licensed upstream AASIST-L architecture.

Source: clovaai/aasist at a04c9863f63d44471dde8a6abcb3b082b07cd1d1.
Only models/AASIST.py is vendored, with its LICENSE and complete NOTICE.
The reference uses class 1=bonafide; this adapter returns class-0 minus class-1
logits. This gives fake-oriented log odds, not the original class-1 raw score.
No upstream dataset, evaluator or pretrained checkpoint is included.
"""

from typing import cast

import torch
from torch import Tensor, nn

from ._vendor.aasist import Model
from .core import validate_waveforms


AASIST_L_SAMPLES = 64600
AASIST_L_SAMPLE_RATE = 16000


class AASISTLite(nn.Module):
    """Exact upstream small configuration with deterministic input adaptation."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = Model(
            {  # type: ignore[no-untyped-call]
                "architecture": "AASIST",
                "nb_samp": AASIST_L_SAMPLES,
                "first_conv": 128,
                "filts": [70, [1, 32], [32, 32], [32, 24], [24, 24]],
                "gat_dims": [24, 32],
                "pool_ratios": [0.4, 0.5, 0.7, 0.5],
                "temperatures": [2.0, 2.0, 100.0, 100.0],
            }
        )

    def prepare_input(self, waveforms: Tensor, lengths: Tensor | None = None) -> Tensor:
        """Repeat short valid input and crop the first 64600 samples, as upstream.

        Coverage of long media belongs to the deterministic segmentation caller;
        a single forward covers one fixed-duration segment only.
        """
        lengths = validate_waveforms(waveforms, lengths)
        indices = torch.arange(AASIST_L_SAMPLES, device=waveforms.device)
        return waveforms.gather(1, indices[None, :] % lengths[:, None])

    def extract_features(
        self, waveforms: Tensor, lengths: Tensor | None = None
    ) -> Tensor:
        embeddings, _ = self.backbone(self.prepare_input(waveforms, lengths))
        return cast(Tensor, embeddings)

    def forward(self, waveforms: Tensor, lengths: Tensor | None = None) -> Tensor:
        _, logits = self.backbone(self.prepare_input(waveforms, lengths))
        return cast(Tensor, logits[:, 0] - logits[:, 1])
