# -*- coding: utf-8 -*-
"""M3：验证 ONNX Runtime 路径能否复现 transformers 参考分数。
预处理严格复刻 CLIPImageProcessor：shortest_edge=440 等比缩放 → 中心裁剪 384 →
1/255 归一化 → CLIP mean/std 标准化。结果与 communityforensics_vit/scores.csv 逐张比较。
用法：
    python onnx_validate.py [--variant fp32|int8]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import bootstrap, MODELS_CACHE, UPGRADE_ROOT  # noqa: E402

bootstrap()

DATASET = Path(r"D:\刑侦查竞赛\数据\案例1\案例1")
M3 = UPGRADE_ROOT / "experiments" / "m3_detector"
LOG = UPGRADE_ROOT / "experiments" / "logs" / "m3_onnx_validate.log"
SNAPSHOT = ("D:/刑科技项目升级2.0/experiments/models_cache/hub/"
            "models--buildborderless--CommunityForensics-DeepfakeDet-ViT/snapshots/"
            "ac6ee457bea904a373065754107451793b56db00")

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
SHORTEST_EDGE = 440
CROP = 384


def log(message: str) -> None:
    print(f"[onnx] {message}", flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(f"{message}\n")


def preprocess(image):
    from PIL import Image

    width, height = image.size
    if width <= height:
        new_width = SHORTEST_EDGE
        new_height = int(SHORTEST_EDGE * height / width)
    else:
        new_height = SHORTEST_EDGE
        new_width = int(SHORTEST_EDGE * width / height)
    resized = image.resize((new_width, new_height), Image.BICUBIC)
    top = (new_height - CROP) // 2
    left = (new_width - CROP) // 2
    cropped = resized.crop((left, top, left + CROP, top + CROP))
    array = np.asarray(cropped, dtype=np.float32) / 255.0
    array = (array - MEAN) / STD
    return np.transpose(array, (2, 0, 1))[None, ...].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("fp32", "int8"), default="fp32")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()
    tag = args.tag or f"onnx_{args.variant}"

    import onnxruntime as ort
    from PIL import Image

    model_file = "model.onnx" if args.variant == "fp32" else "model_int8.onnx"
    model_path = Path(SNAPSHOT) / "onnx" / model_file
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    log(f"{tag}: 模型={model_file} 输入={input_name}{session.get_inputs()[0].shape} "
        f"输出={output_name}{session.get_outputs()[0].shape}")

    rows = []
    for case_dir in sorted(DATASET.glob("案例*")):
        for image_path in sorted(case_dir.glob("*.jpg")):
            image = Image.open(image_path).convert("RGB")
            logits = session.run([output_name], {input_name: preprocess(image)})[0]
            score = float(1.0 / (1.0 + np.exp(-float(np.ravel(logits)[0]))))
            rows.append({"case": case_dir.name, "file": image_path.name,
                         "y": 1 if image_path.name.startswith("05") else 0,
                         "score": score})

    out_dir = M3 / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "scores.csv", "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "file", "y", "score"])
        writer.writeheader()
        writer.writerows(rows)

    reference = {}
    with open(M3 / "communityforensics_vit" / "scores.csv", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            reference[(row["case"], row["file"])] = float(row["score"])

    paired = [(row["score"], reference[(row["case"], row["file"])]) for row in rows
              if (row["case"], row["file"]) in reference]
    onnx_scores = np.array([item[0] for item in paired])
    torch_scores = np.array([item[1] for item in paired])
    correlation = float(np.corrcoef(onnx_scores, torch_scores)[0, 1])

    from fusion_eval import roc_auc
    y = np.array([row["y"] for row in rows])
    scores = np.array([row["score"] for row in rows])
    summary = {
        "variant": args.variant, "tag": tag, "n": len(rows),
        "auc": round(roc_auc(y, scores), 6),
        "reference_auc": 0.897813,
        "pearson_vs_reference": round(correlation, 6),
        "max_abs_diff": round(float(np.abs(onnx_scores - torch_scores).max()), 8),
        "mean_abs_diff": round(float(np.abs(onnx_scores - torch_scores).mean()), 10),
    }
    (out_dir / "onnx_validation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"{tag}: AUC={summary['auc']} 与参考相关={summary['pearson_vs_reference']} "
        f"最大绝对差={summary['max_abs_diff']}")


if __name__ == "__main__":
    main()
