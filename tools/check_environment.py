"""Check optional research/media/ML dependencies with synthetic inputs only.

Run after installing media, research, and one of the cpu/cu126 extras.
This is an environment probe, not a detector, benchmark, or training run.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import cv2
import imagehash
import ipykernel
import librosa
import matplotlib
import numpy as np
import pandas as pd
import scipy
import soundfile as sf
import timm
import torch
import torchvision
import yaml
from PIL import Image
from sklearn.model_selection import GroupKFold
from transformers import Wav2Vec2Config, Wav2Vec2Model

from deepfake_detection_archive.media import VideoConfig, decode_video


def require(condition: object, message: str) -> None:
    """Keep probe checks active even when Python runs with optimization enabled."""
    if not condition:
        raise RuntimeError(message)


def media_checks(directory: Path) -> dict[str, object]:
    """Exercise audio transforms, image hashes, and deterministic video sampling."""
    sample_rate = 16_000
    wave = np.sin(2 * np.pi * 440 * np.arange(sample_rate) / sample_rate).astype(
        np.float32
    )
    audio = directory / "tone.wav"
    sf.write(audio, wave, sample_rate, subtype="FLOAT")
    decoded, actual_rate = sf.read(audio, dtype="float32")
    np.testing.assert_allclose(decoded, wave)
    require(actual_rate == sample_rate, "Audio sample rate changed")
    resampled = librosa.resample(decoded, orig_sr=sample_rate, target_sr=8_000)
    mel = librosa.feature.melspectrogram(y=resampled, sr=8_000, n_mels=16)
    require(resampled.shape == (8_000,), "Unexpected resampled length")
    require(np.isfinite(mel).all(), "Nonfinite mel features")

    image = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
    encoded_ok, encoded = cv2.imencode(".png", image)
    require(encoded_ok, "PNG encoding failed")
    np.testing.assert_array_equal(cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE), image)
    require(cv2.resize(image, (32, 32)).shape == (32, 32), "Image resize failed")
    fingerprint = imagehash.phash(Image.fromarray(image))
    require(
        fingerprint == imagehash.phash(Image.fromarray(image.copy())),
        "Identical images produced different perceptual hashes",
    )

    video = directory / "synthetic.mp4"
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for the synthetic video probe")
    subprocess.run(  # noqa: S603 -- fixed arguments and a fresh temporary output
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x64:rate=8:duration=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(video),
        ],
        check=True,
    )
    # Exercise the project's actual FFmpeg/ffprobe path, including -fps_mode.
    # OpenCV-only sampling can pass with an FFmpeg too old for project decoding.
    config = VideoConfig(n_frames=4, height=64, width=64)
    decoded_video = decode_video(video, config)
    repeated_video = decode_video(video, config)
    require(decoded_video.frames.shape == (4, 64, 64, 3), "Unexpected video shape")
    require(decoded_video.frame_indices == (0, 5, 10, 15), "Unexpected frame indices")
    np.testing.assert_array_equal(decoded_video.frames, repeated_video.frames)
    return {
        "audio_samples": len(decoded),
        "mel_shape": list(mel.shape),
        "phash": str(fingerprint),
        "sampled_frames": len(decoded_video.frames),
        "project_video_decode": "PASS",
    }


def science_checks(directory: Path) -> dict[str, object]:
    """Apply group isolation to toy entities; do not construct project splits."""
    groups = np.repeat(np.arange(10), 2)
    frame = pd.DataFrame({"group": groups, "label": np.tile([0, 1], 10)})
    folds = list(GroupKFold(n_splits=5).split(frame, frame.label, groups=groups))
    for train, valid in folds:
        require(not set(groups[train]) & set(groups[valid]), "Group leakage detected")
    require(yaml.safe_load("seed: 42") == {"seed": 42}, "YAML parsing failed")
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt  # noqa: PLC0415 -- select Agg first

    figure, axis = plt.subplots()
    axis.plot([0, 1, 2], [0, 1, 0])
    figure.savefig(directory / "smoke.png")
    plt.close(figure)
    require((directory / "smoke.png").stat().st_size > 0, "Empty rendered plot")
    return {
        "group_disjoint_folds": len(folds),
        "plot": "PASS",
        "ipykernel": ipykernel.__version__,
        "scipy": scipy.__version__,
    }


def ml_checks(device: torch.device) -> dict[str, object]:
    """Check binary pairing, forward passes, AMP and gradients without updates."""
    torch.manual_seed(42)
    torch.set_num_threads(2)
    array = np.arange(16, dtype=np.float32).reshape(4, 4)
    tensor = torch.from_numpy(array).to(device)
    np.testing.assert_allclose((tensor @ tensor.T).cpu().numpy(), array @ array.T)
    boxes = torch.tensor(
        [[0, 0, 10, 10], [1, 1, 9, 9]], device=device, dtype=torch.float32
    )
    keep = torchvision.ops.nms(boxes, torch.tensor([0.9, 0.8], device=device), 0.5)
    require(keep.tolist() == [0], "torchvision NMS binary check failed")

    # Frame scores remain continuous; no face requirement or probability clipping.
    frame_scores = torch.sigmoid(torch.linspace(-1, 1, 4, device=device))
    torch.testing.assert_close(frame_scores.mean(), torch.tensor(0.5, device=device))

    # Two windows (4 then 1 samples) check the incomplete accumulation tail.
    inputs = torch.linspace(-1, 1, 5, device=device)
    targets = 0.5 * inputs + 0.1
    for start in (0, 4):
        x, y = inputs[start : start + 4], targets[start : start + 4]
        full = torch.tensor(0.3, device=device, requires_grad=True)
        ((full * x - y) ** 2).mean().backward()
        accumulated = full.detach().clone().requires_grad_()
        for offset in range(0, len(x), 2):
            xb, yb = x[offset : offset + 2], y[offset : offset + 2]
            (((accumulated * xb - yb) ** 2).sum() / len(x)).backward()
        torch.testing.assert_close(accumulated.grad, full.grad)

    with torch.inference_mode():
        vision = timm.create_model("resnet18", pretrained=False).to(device).eval()
        output = vision(torch.zeros(1, 3, 64, 64, device=device))
        require(output.shape == (1, 1000), "Unexpected random vision output shape")
        require(torch.isfinite(output).all(), "Nonfinite random vision output")
        config = Wav2Vec2Config(
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            conv_dim=(8, 8),
            conv_stride=(4, 4),
            conv_kernel=(8, 4),
            num_conv_pos_embedding_groups=2,
            num_conv_pos_embeddings=16,
            feat_extract_norm="layer",
        )
        audio = Wav2Vec2Model(config).to(device).eval()
        encoded = audio(torch.zeros(1, 1600, device=device)).last_hidden_state
        require(encoded.shape[-1] == 16, "Unexpected random audio output shape")
        require(torch.isfinite(encoded).all(), "Nonfinite random audio output")

    amp_status = "SKIPPED: CPU selected"
    gpu: dict[str, object] = {}
    if device.type == "cuda":
        optimizer_parameter = torch.nn.Parameter(torch.ones(4, 4, device=device))
        optimizer = torch.optim.SGD([optimizer_parameter], lr=0.1)
        scaler = torch.amp.GradScaler("cuda")
        with torch.amp.autocast("cuda", dtype=torch.float16):
            loss = (optimizer_parameter @ (tensor / 16)).square().mean()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if optimizer_parameter.grad is None:
            raise RuntimeError("AMP backward did not produce a gradient")
        require(
            torch.isfinite(optimizer_parameter.grad).all(), "Nonfinite AMP gradient"
        )
        # No optimizer step: this checks the runtime, not a training recipe.
        torch.cuda.synchronize()
        amp_status = "PASS: autocast, GradScaler, backward, unscale"
        properties = torch.cuda.get_device_properties(device)
        gpu = {
            "name": properties.name,
            "total_bytes": properties.total_memory,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
    return {
        "device": str(device),
        "torch_cuda_build": torch.version.cuda,
        "nms": "PASS",
        "frame_average": "PASS",
        "accumulation_full_and_tail": "PASS",
        "amp": amp_status,
        "random_vision_shape": list(output.shape),
        "random_audio_shape": list(encoded.shape),
        "gpu": gpu,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this process")
    media_tools = {}
    for name in ("ffmpeg", "ffprobe"):
        executable = shutil.which(name)
        if executable is None:
            raise RuntimeError(f"{name} is required for project media decoding")
        version = subprocess.run(  # noqa: S603 -- fixed media executables from PATH
            [executable, "-version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.splitlines()[0]
        media_tools[name] = {"path": executable, "version": version}
    distributions = (
        "torch",
        "torchvision",
        "timm",
        "transformers",
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "soundfile",
        "librosa",
        "opencv-python-headless",
        "pillow",
        "imagehash",
        "pyyaml",
        "tqdm",
        "matplotlib",
        "ipykernel",
        "pypdf",
    )
    report: dict[str, object] = {
        "recorded_at": datetime.now(UTC).isoformat(),
        "scope": "synthetic environment checks; no weights/data downloads or training",
        "versions": {name: importlib.metadata.version(name) for name in distributions},
        "media_tools": media_tools,
    }
    with tempfile.TemporaryDirectory(prefix="aigc-environment-") as temporary:
        directory = Path(temporary)
        report["media"] = media_checks(directory)
        report["science"] = science_checks(directory)
        report["ml"] = ml_checks(torch.device(args.device))
    report["status"] = "PASS"
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
