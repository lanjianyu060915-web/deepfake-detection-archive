"""Cross-media SSL batching preserves segment order and individual embeddings."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from deepfake_detection_archive.data.manifest import SampleRecord, file_sha256
from deepfake_detection_archive.features import (
    extract_audio_feature_batch,
    extract_features,
)
from deepfake_detection_archive.media import AudioConfig, VideoConfig
from deepfake_detection_archive.models.encoders import FrozenAudioEncoder


def test_audio_feature_batch_preserves_media_and_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transformers = pytest.importorskip("transformers")
    sf = pytest.importorskip("soundfile")
    bundle = tmp_path / "encoder"
    config = transformers.Wav2Vec2Config(
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        conv_dim=(8, 8),
        conv_stride=(5, 2),
        conv_kernel=(10, 3),
        num_conv_pos_embeddings=8,
        num_conv_pos_embedding_groups=2,
        mask_time_prob=0.0,
    )
    transformers.Wav2Vec2Model(config).save_pretrained(bundle, safe_serialization=True)
    transformers.Wav2Vec2FeatureExtractor().save_pretrained(bundle)
    encoder = FrozenAudioEncoder(
        bundle, expected_hashes={p.name: file_sha256(p) for p in bundle.iterdir()}
    )
    records = []
    for index in range(3):
        path = tmp_path / f"{index}.flac"
        samples = (5000 * np.sin(np.arange(1600) * (index + 1) / 13)).astype(np.int16)
        sf.write(path, samples, 16000, subtype="PCM_16")
        records.append(
            SampleRecord(
                sample_id=str(index),
                path=path.name,
                modality="audio",
                label=index % 2,
                source="synthetic",
                sha256=file_sha256(path),
            )
        )
    audio_config = AudioConfig(backend="soundfile")
    expected = [
        extract_features(
            record, tmp_path, encoder, audio_config, VideoConfig(), 400, 2, "cpu", 4
        )
        for record in records
    ]
    batch_sizes = []
    forward = encoder.forward

    def record_batch(values: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        batch_sizes.append(values.shape[0])
        return forward(values, **kwargs)

    monkeypatch.setattr(encoder, "forward", record_batch)
    actual = extract_audio_feature_batch(
        records, tmp_path, encoder, audio_config, 400, 2, "cpu", 4
    )
    assert batch_sizes == [4, 2]
    assert len(actual) == 3
    for value, reference in zip(actual, expected, strict=True):
        assert value.shape == (2, 8)
        assert value.device.type == "cpu"
        assert value.dtype == torch.float32
        torch.testing.assert_close(value, reference, rtol=1e-4, atol=1e-5)
