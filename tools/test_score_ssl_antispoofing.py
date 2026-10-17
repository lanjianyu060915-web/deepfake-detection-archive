"""Behavior checks for the SSL_Anti-spoofing scorer; run with unittest discovery."""

from __future__ import annotations

# ruff: noqa: PT009, PT027 - standalone unittest checks alongside the CLI tool
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
from transformers import Wav2Vec2Config, Wav2Vec2Model

from deepfake_detection_archive.models import ssl_antispoofing as ssl


TINY = {
    "hidden_size": 16,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "intermediate_size": 32,
    "conv_dim": [8, 8],
    "conv_stride": [5, 2],
    "conv_kernel": [10, 3],
    "conv_bias": True,
    "feat_extract_norm": "layer",
    "do_stable_layer_norm": True,
    "num_conv_pos_embeddings": 4,
    "num_conv_pos_embedding_groups": 2,
    "mask_time_prob": 0.075,
}


def fairseq_keys(config: dict[str, object]) -> list[str]:
    """fairseq Wav2Vec2 (layer_norm_first) parameter names for a tiny config."""
    keys = ["post_extract_proj.weight", "post_extract_proj.bias"]
    keys += ["layer_norm.weight", "layer_norm.bias", "mask_emb"]
    keys += ["encoder.pos_conv.0.weight_g", "encoder.pos_conv.0.weight_v"]
    keys += ["encoder.pos_conv.0.bias", "encoder.layer_norm.weight"]
    keys += ["encoder.layer_norm.bias", "quantizer.vars", "project_q.weight"]
    keys += ["final_proj.weight"]
    conv = config["conv_dim"]
    assert isinstance(conv, list)
    for index in range(len(conv)):
        stem = f"feature_extractor.conv_layers.{index}"
        keys += [f"{stem}.0.weight", f"{stem}.0.bias"]
        keys += [f"{stem}.2.1.weight", f"{stem}.2.1.bias"]
    layers = config["num_hidden_layers"]
    assert isinstance(layers, int)
    for index in range(layers):
        stem = f"encoder.layers.{index}"
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            keys += [f"{stem}.self_attn.{name}.weight", f"{stem}.self_attn.{name}.bias"]
        for name in ("self_attn_layer_norm", "fc1", "fc2", "final_layer_norm"):
            keys += [f"{stem}.{name}.weight", f"{stem}.{name}.bias"]
    return keys


class KeyMappingTests(unittest.TestCase):
    def test_maps_every_fairseq_key_onto_the_transformers_model(self) -> None:
        mapped = {ssl.fairseq_to_hf(key) for key in fairseq_keys(TINY)}
        mapped.discard(None)
        expected = set(Wav2Vec2Model(Wav2Vec2Config.from_dict(TINY)).state_dict())
        self.assertEqual(mapped, expected)

    def test_drops_only_pretraining_heads(self) -> None:
        dropped = [k for k in fairseq_keys(TINY) if ssl.fairseq_to_hf(k) is None]
        self.assertEqual(
            sorted(dropped), ["final_proj.weight", "project_q.weight", "quantizer.vars"]
        )
        self.assertEqual(ssl.fairseq_to_hf("mask_emb"), "masked_spec_embed")


class PadTests(unittest.TestCase):
    def test_crops_from_the_head_and_tiles_short_clips(self) -> None:
        long_wave = np.arange(70000, dtype=np.float32)
        self.assertTrue(
            np.array_equal(ssl.pad_or_tile(long_wave), long_wave[: ssl.CUT])
        )
        short = np.array([1, 2, 3], dtype=np.float32)
        tiled = ssl.pad_or_tile(short, 7)
        self.assertEqual(tiled.tolist(), [1, 2, 3, 1, 2, 3, 1])
        with self.assertRaisesRegex(ValueError, "nonempty mono"):
            ssl.pad_or_tile(np.zeros((2, 4), dtype=np.float32))


UPSTREAM = """import torch
import torch.nn as nn
import fairseq


class SSLModel(nn.Module):
    def __init__(self, device):
        raise RuntimeError("fairseq front-end must not be used")


#---------AASIST back-end------------------------#
class Model(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.ssl_model = SSLModel(device)
"""


class UpstreamLoadingTests(unittest.TestCase):
    def test_swaps_the_front_end_and_checks_the_source_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "model.py"
            source.write_text(UPSTREAM, encoding="utf-8")
            digest = hashlib.sha256(UPSTREAM.encode()).hexdigest()
            ssl.SSLModel.config = Wav2Vec2Config.from_dict(TINY)
            model = ssl.upstream_model_class(source, digest)(None, "cpu")
            self.assertIsInstance(model.ssl_model, ssl.SSLModel)
            with self.assertRaisesRegex(ValueError, "sha256"):
                ssl.upstream_model_class(source, "0" * 64)


if __name__ == "__main__":
    unittest.main()
