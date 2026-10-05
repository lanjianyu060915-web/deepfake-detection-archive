r"""M2 前置探针：定位身份检索「检不到人脸」的真实原因，并量化可恢复比例。

对 20 个 SWAN-DF 案例查询图，用冻结 worker 相同的 buffalo_l 检测器，
在多种输入尺度 / 预处理下统计检出结果。

只读冻结资产；模型取自工作区 packaging_v0.3/resources/models/buffalo_l。

用法：
    D:\python\.venv\Scripts\python.exe probe_face_recovery.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

SWAN = Path(r"D:\刑侦查竞赛\数据\案例1\案例1")
MODEL_ROOT = Path(r"D:\刑科技项目升级2.0\workspace\fsts\packaging_v0.3\resources")
OUT = Path(r"D:\刑科技项目升级2.0\experiments\m2_preprocess")

FROZEN_DET_SIZE = (1280, 1280)


def letterbox(image: np.ndarray, side: int = 1280) -> np.ndarray:
    """保持长宽比缩放到 side x side 画布，空白补黑，供检测使用。"""

    height, width = image.shape[:2]
    scale = side / max(height, width)
    resized = cv2.resize(
        image, (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    canvas = np.zeros((side, side, 3), np.uint8)
    top = (side - resized.shape[0]) // 2
    left = (side - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return canvas


def build_app(det_size: tuple[int, int]):
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name="buffalo_l",
        root=str(MODEL_ROOT),
        allowed_modules=["detection"],
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=det_size)
    return app


VARIANTS = {
    "frozen_1280": dict(kind="raw", det_size=FROZEN_DET_SIZE),
    "raw_640": dict(kind="raw", det_size=(640, 640)),
    "raw_1920": dict(kind="raw", det_size=(1920, 1920)),
    "letterbox_1280": dict(kind="letterbox", det_size=FROZEN_DET_SIZE),
}


def detect(app, image: np.ndarray) -> list[dict]:
    return [
        {
            "bbox": [round(float(value), 1) for value in face.bbox],
            "det_score": round(float(face.det_score), 4),
        }
        for face in app.get(image)
    ]


def main() -> int:
    cases = [f"案例{index:02d}" for index in range(1, 21)]
    rows: list[dict] = []
    for name, spec in VARIANTS.items():
        app = build_app(spec["det_size"])
        print(f"=== {name} (det_size={spec['det_size']}, kind={spec['kind']}) ===", flush=True)
        found = 0
        for case in cases:
            image = cv2.imread(str(SWAN / case / "05_换脸后图片.jpg"))
            height, width = image.shape[:2]
            prepared = (
                letterbox(image, spec["det_size"][0]) if spec["kind"] == "letterbox" else image
            )
            faces = detect(app, prepared)
            if spec["kind"] == "letterbox":
                scale = spec["det_size"][0] / max(height, width)
                pad_top = (spec["det_size"][0] - int(round(height * scale))) // 2
                pad_left = (spec["det_size"][0] - int(round(width * scale))) // 2
                for face in faces:
                    face["bbox"] = [
                        round((face["bbox"][0] - pad_left) / scale, 1),
                        round((face["bbox"][1] - pad_top) / scale, 1),
                        round((face["bbox"][2] - pad_left) / scale, 1),
                        round((face["bbox"][3] - pad_top) / scale, 1),
                    ]
            if faces:
                found += 1
            sizes = [
                f"{round(face['bbox'][2] - face['bbox'][0])}x{round(face['bbox'][3] - face['bbox'][1])}"
                f"@{face['det_score']}"
                for face in faces
            ]
            rows.append({"variant": name, "case": case, "faces": len(faces), "detail": sizes})
            print(f"  {case}: {len(faces)} 张 {sizes}", flush=True)
        print(f"  -> 检出 {found}/{len(cases)}", flush=True)
    (OUT / "probe_face_recovery.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n汇总")
    for name in VARIANTS:
        hits = [row for row in rows if row["variant"] == name and row["faces"]]
        print(f"  {name:16} {len(hits):2}/20")
    return 0


if __name__ == "__main__":
    sys.exit(main())
