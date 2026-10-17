"""Local, pinned SSL Anti-spoofing LA loader and upstream waveform frontend."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from transformers import Wav2Vec2Config, Wav2Vec2Model

from .checkpoint import sha256_file


CUT = 64600
PREFIX = "ssl_model.model."
DROPPED = ("quantizer.", "project_q.", "final_proj.")
UPSTREAM_COMMIT = "4acaa61dcef5f7610f43aa4d0b29c4559b970cd2"
UPSTREAM_MODEL_SHA256 = (
    "08b2b99b9cc0e90732746471325185f2eb144795ee35338e0a02951015a856c6"
)
RULES = (
    (
        r"^feature_extractor\.conv_layers\.(\d+)\.0\.",
        r"feature_extractor.conv_layers.\1.conv.",
    ),
    (
        r"^feature_extractor\.conv_layers\.(\d+)\.2\.1\.",
        r"feature_extractor.conv_layers.\1.layer_norm.",
    ),
    (r"^post_extract_proj\.", "feature_projection.projection."),
    (r"^layer_norm\.", "feature_projection.layer_norm."),
    (
        r"^encoder\.pos_conv\.0\.weight_g$",
        "encoder.pos_conv_embed.conv.parametrizations.weight.original0",
    ),
    (
        r"^encoder\.pos_conv\.0\.weight_v$",
        "encoder.pos_conv_embed.conv.parametrizations.weight.original1",
    ),
    (r"^encoder\.pos_conv\.0\.bias$", "encoder.pos_conv_embed.conv.bias"),
    (
        r"^encoder\.layers\.(\d+)\.self_attn_layer_norm\.",
        r"encoder.layers.\1.layer_norm.",
    ),
    (r"^encoder\.layers\.(\d+)\.self_attn\.", r"encoder.layers.\1.attention."),
    (
        r"^encoder\.layers\.(\d+)\.fc1\.",
        r"encoder.layers.\1.feed_forward.intermediate_dense.",
    ),
    (
        r"^encoder\.layers\.(\d+)\.fc2\.",
        r"encoder.layers.\1.feed_forward.output_dense.",
    ),
)


def fairseq_to_hf(key: str) -> str | None:
    """Map one fairseq XLS-R key to transformers; None drops pre-training heads."""
    if key.startswith(DROPPED):
        return None
    if key == "mask_emb":
        return "masked_spec_embed"
    for pattern, replacement in RULES:
        if re.search(pattern, key):
            return re.sub(pattern, replacement, key)
    return key


def pad_or_tile(wave: NDArray[np.float32], max_len: int = CUT) -> NDArray[np.float32]:
    """Upstream ``data_utils_SSL.pad``: head crop, or tile the whole clip."""
    if wave.ndim != 1 or wave.size == 0:
        raise ValueError("expected a nonempty mono waveform")
    if wave.shape[0] >= max_len:
        return wave[:max_len]
    repeats = max_len // wave.shape[0] + 1
    return np.tile(wave, repeats)[:max_len]


class SSLModel(nn.Module):
    """transformers XLS-R with the attribute layout the checkpoint expects."""

    config: Wav2Vec2Config | None = None

    def __init__(self, device: object) -> None:
        super().__init__()
        if SSLModel.config is None:
            raise RuntimeError("SSLModel.config must be set before building")
        self.model = Wav2Vec2Model(SSLModel.config)
        self.device = device
        self.out_dim = 1024

    def extract_feat(self, input_data: torch.Tensor) -> torch.Tensor:
        if input_data.ndim == 3:
            input_data = input_data[:, :, 0]
        hidden: torch.Tensor = self.model(input_data).last_hidden_state
        return hidden


def upstream_model_class(source: Path, expected_sha256: str) -> type[nn.Module]:
    """Execute pinned upstream model.py with SSLModel swapped for the HF port."""
    actual = sha256_file(source)
    if actual != expected_sha256:
        raise ValueError(f"upstream model.py sha256 {actual} != {expected_sha256}")
    text = source.read_text(encoding="utf-8")
    start = text.index("class SSLModel")
    end = text.index("#---------AASIST back-end")
    text = text[:start] + text[end:]
    text = text.replace("import fairseq\n", "")
    namespace: dict[str, Any] = {
        "__name__": "ssl_antispoofing_upstream",
        "SSLModel": SSLModel,
    }
    exec(compile(text, str(source), "exec"), namespace)  # noqa: S102 -- pinned MIT source
    model_class: type[nn.Module] = namespace["Model"]
    return model_class


def build_model(
    weights: Path,
    upstream: Path,
    config: Path,
    device: str,
    *,
    expected_weights_sha256: str | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Build the detector and load the checkpoint strictly through the key map."""
    weights_sha256 = sha256_file(weights)
    if (
        expected_weights_sha256 is not None
        and weights_sha256 != expected_weights_sha256
    ):
        raise ValueError(f"SHA-256 mismatch: {weights}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    SSLModel.config = Wav2Vec2Config.from_dict(
        json.loads(config.read_text(encoding="utf-8"))
    )
    model = upstream_model_class(upstream, UPSTREAM_MODEL_SHA256)(None, device)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    mapped: dict[str, torch.Tensor] = {}
    dropped: list[str] = []
    for key, value in state.items():
        target = key
        if key.startswith(PREFIX):
            renamed = fairseq_to_hf(key[len(PREFIX) :])
            if renamed is None:
                dropped.append(key)
                continue
            target = PREFIX + renamed
        if target in mapped:
            raise ValueError(f"two checkpoint keys map to {target}")
        mapped[target] = value
    expected = model.state_dict()
    mismatched = [
        k for k, v in mapped.items() if k in expected and expected[k].shape != v.shape
    ]
    if mismatched:
        raise ValueError(f"shape mismatch: {mismatched[:5]}")
    model.load_state_dict(mapped, strict=True)
    model.to(device).eval()
    report = {
        "architecture": "audio_ssl_antispoofing_v1",
        "schema_version": 1,
        "weights_sha256": weights_sha256,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_model_sha256": UPSTREAM_MODEL_SHA256,
        "xlsr_config_sha256": sha256_file(config),
        "checkpoint_keys": len(state),
        "dropped_pretraining_keys": dropped,
        "loaded_keys": len(mapped),
    }
    return model, report
