# -*- coding: utf-8 -*-
"""把检不到人脸的查询图拼成一张联络表，便于直观判断失败原因。"""

import cv2
import numpy as np
from pathlib import Path

SWAN = Path(r"D:\刑侦查竞赛\数据\案例1\案例1")
OUT = Path(r"D:\刑科技项目升级2.0\experiments\m2_preprocess")
OUT.mkdir(parents=True, exist_ok=True)

FAILED = [1, 2, 3, 4, 5, 6, 7, 17, 18, 19]
THUMB = (216, 384)


def thumbnail(case: str) -> np.ndarray:
    image = cv2.imread(str(SWAN / case / "05_换脸后图片.jpg"))
    image = cv2.resize(image, THUMB, interpolation=cv2.INTER_AREA)
    cv2.putText(
        image, case, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA
    )
    return image


rows = []
for start in range(0, len(FAILED), 5):
    chunk = FAILED[start : start + 5]
    while len(chunk) < 5:
        chunk.append(None)
    row = [
        thumbnail(f"案例{index:02d}") if index else np.zeros((THUMB[1], THUMB[0], 3), np.uint8)
        for index in chunk
    ]
    rows.append(np.hstack(row))
montage = np.vstack(rows)
target = OUT / "failed_queries_montage.png"
cv2.imwrite(str(target), montage)
print("wrote", target, montage.shape)
