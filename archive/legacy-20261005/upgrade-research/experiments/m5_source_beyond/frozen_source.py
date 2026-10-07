# -*- coding: utf-8 -*-
r"""复刻冻结的底图检索流程，并把“库侧”计算（人脸掩膜、全局特征、SIFT 描述子）做成可复用缓存。

分数口径与 D:\刑科技项目升级2.0\workspace\fsts\modules\source_search 完全一致，未做任何修改：
    全局分 = 0.30*hsv + 0.25*gray + 0.15*edge + 0.30*border
    短名单 = 全部候选按全局分降序（同分按路径小写）后的前 min(max(4k, 20), N) 个
    局部分 = 0.6*内点率 + 0.4*min(内点数/40, 1)；有效匹配不足 4 时 = min(good/40, 0.15)
    最终分 = 0.45*全局分 + 0.55*局部分
    最终排序键 = (-final, -global, path.lower())

缓存只避免对同一张库图重复做人脸掩膜 / 全局特征 / SIFT，不改变任何数值。
复刻正确性由 verify_replica.py 与冻结 EXE 的实际输出逐案例比对。
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"D:\刑科技项目升级2.0")
SOURCE_PKG = ROOT / "workspace" / "fsts" / "modules" / "source_search"
if str(SOURCE_PKG) not in sys.path:
    sys.path.insert(0, str(SOURCE_PKG))

from source_search.face_mask import FaceMasker  # noqa: E402
from source_search.global_features import (  # noqa: E402
    extract_global,
    global_similarity_with_components,
)
from source_search.image_io import read_image  # noqa: E402

GLOBAL_WEIGHT = 0.45
LOCAL_WEIGHT = 0.55
LOWE_RATIO = 0.75
RANSAC_THRESHOLD = 5.0
SIFT_NFEATURES = 2500

MASKER = FaceMasker()
SIFT = cv2.SIFT_create(nfeatures=SIFT_NFEATURES)

LIBRARIES = ROOT / "experiments" / "m5_source_beyond" / "libraries"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def library_fingerprint(paths: list[Path]) -> str:
    payload = "\n".join(str(Path(p).resolve()) for p in paths)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class FrozenLibrary:
    """库侧特征（掩膜 + 全局特征）一次算好，供任意多个查询复用。"""

    def __init__(self, paths: list[Path], features: dict[str, np.ndarray], name: str = ""):
        self.paths = [Path(p) for p in paths]
        self.name = name
        self.features = features
        self._mask_cache: dict[int, np.ndarray] = {}
        self._sift_cache: dict[int, tuple[list, np.ndarray]] = {}
        self.mask_cache_limit = 48
        self.sift_cache_limit = 48

    # ---------- 构造 / 持久化 ----------
    @staticmethod
    def extract(paths: list[Path], progress_every: int = 100) -> dict[str, np.ndarray]:
        hist, thumb, edge, border, border_valid = [], [], [], [], []
        for index, path in enumerate(paths, 1):
            image = read_image(Path(path))
            if image is None:
                raise ValueError(f"cannot decode {path}")
            mask = MASKER.mask(image.shape, MASKER.detect(image))
            feats = extract_global(image, mask)
            hist.append(feats["hist"])
            thumb.append(feats["thumb"])
            edge.append(feats["edge"])
            border.append(feats["border"])
            border_valid.append(feats["border_valid"])
            if progress_every and index % progress_every == 0:
                print(f"    库特征 {index}/{len(paths)}", flush=True)
        return {
            "hist": np.stack(hist).astype(np.float32),
            "thumb": np.stack(thumb).astype(np.float32),
            "edge": np.stack(edge).astype(np.float32),
            "border": np.stack(border).astype(np.float32),
            "border_valid": np.stack(border_valid).astype(bool),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            fingerprint=np.array([library_fingerprint(self.paths)]),
            paths=np.array([str(p) for p in self.paths]),
            **self.features,
        )

    @staticmethod
    def load(path: Path, name: str = "") -> "FrozenLibrary":
        data = np.load(path, allow_pickle=False)
        paths = [Path(p) for p in data["paths"].tolist()]
        features = {key: data[key] for key in ("hist", "thumb", "edge", "border", "border_valid")}
        return FrozenLibrary(paths, features, name=name)

    # ---------- 库侧按需计算 ----------
    def _evict(self, cache: dict, limit: int) -> None:
        while len(cache) > limit:
            cache.pop(next(iter(cache)))

    def mask_of(self, index: int) -> np.ndarray:
        if index not in self._mask_cache:
            image = read_image(self.paths[index])
            if image is None:
                raise ValueError(f"cannot decode {self.paths[index]}")
            self._mask_cache[index] = MASKER.mask(image.shape, MASKER.detect(image))
            self._evict(self._mask_cache, self.mask_cache_limit)
        return self._mask_cache[index]

    def sift_of(self, index: int) -> tuple[list, np.ndarray]:
        if index not in self._sift_cache:
            image = read_image(self.paths[index])
            if image is None:
                raise ValueError(f"cannot decode {self.paths[index]}")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = SIFT.detectAndCompute(gray, self.mask_of(index))
            self._sift_cache[index] = (keypoints or [], descriptors)
            self._evict(self._sift_cache, self.sift_cache_limit)
        return self._sift_cache[index]

    # ---------- 查询 ----------
    def global_scores(self, query_features: dict[str, np.ndarray]) -> np.ndarray:
        scores = np.empty(len(self.paths), dtype=np.float64)
        for index in range(len(self.paths)):
            item = {
                "hist": self.features["hist"][index],
                "thumb": self.features["thumb"][index],
                "edge": self.features["edge"][index],
                "border": self.features["border"][index],
                "border_valid": self.features["border_valid"][index],
            }
            score, components = global_similarity_with_components(query_features, item)
            scores[index] = score if not components else score
        return scores


def read_query(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    image = read_image(Path(path))
    if image is None:
        raise ValueError(f"cannot decode query {path}")
    mask = MASKER.mask(image.shape, MASKER.detect(image))
    return image, mask, extract_global(image, mask)


def _local_match(query_image, query_mask, query_signature, candidate_signature):
    """与 LocalMatcher.match 同一步骤：SIFT 描述子已由调用方给出。"""
    keypoints_a, descriptors_a = query_signature
    keypoints_b, descriptors_b = candidate_signature
    if (
        descriptors_a is None
        or descriptors_b is None
        or len(descriptors_a) < 2
        or len(descriptors_b) < 2
    ):
        return {"score": 0.0, "good_count": 0, "inlier_count": 0, "ransac_attempted": False}

    raw = cv2.BFMatcher(cv2.NORM_L2).knnMatch(descriptors_a, descriptors_b, k=2)
    good = [
        match
        for pair in raw
        if len(pair) == 2
        for match, other in [pair]
        if match.distance < LOWE_RATIO * other.distance
    ]
    if len(good) < 4:
        return {
            "score": min(len(good) / 40.0, 0.15),
            "good_count": len(good),
            "inlier_count": 0,
            "ransac_attempted": False,
        }

    src = np.float32([keypoints_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([keypoints_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    try:
        _homography, inliers = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THRESHOLD)
    except cv2.error:
        inliers = None
    if inliers is None:
        count = 0
    else:
        count = int(inliers.ravel().astype(int).sum())
    inlier_ratio = count / max(len(good), 1)
    score = float(np.clip(0.6 * inlier_ratio + 0.4 * min(count / 40.0, 1.0), 0.0, 1.0))
    return {
        "score": score,
        "good_count": len(good),
        "inlier_count": count,
        "ransac_attempted": True,
    }


def search(
    library: FrozenLibrary,
    query_path: Path,
    *,
    top_k: int = 5,
    query_image=None,
    query_mask=None,
    query_features=None,
    detail_limit: int | None = None,
) -> dict:
    """返回与冻结模块同口径的候选排序，并附带库内全局分序列。"""
    if query_features is None:
        query_image, query_mask, query_features = read_query(query_path)
    scores = library.global_scores(query_features)

    order = sorted(range(len(library.paths)), key=lambda i: (-scores[i], str(library.paths[i]).lower()))
    shortlist_size = min(max(top_k * 4, 20), len(order))
    shortlist = order[:shortlist_size]

    if query_image is None:
        query_image = read_image(Path(query_path))
    if query_mask is None:
        query_mask = MASKER.mask(query_image.shape, MASKER.detect(query_image))
    gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
    query_signature = SIFT.detectAndCompute(gray, query_mask)
    query_signature = (query_signature[0] or [], query_signature[1])

    candidates = []
    for index in shortlist:
        local = _local_match(query_image, query_mask, query_signature, library.sift_of(index))
        candidates.append(
            {
                "index": int(index),
                "path": str(library.paths[index]),
                "name": library.paths[index].name,
                "global_score": float(scores[index]),
                "local_score": float(local["score"]),
                "final_score": float(
                    GLOBAL_WEIGHT * float(scores[index]) + LOCAL_WEIGHT * float(local["score"])
                ),
                "good_match_count": int(local["good_count"]),
                "ransac_inlier_count": int(local["inlier_count"]),
                "ransac_attempted": bool(local["ransac_attempted"]),
            }
        )
    candidates.sort(key=lambda c: (-c["final_score"], -c["global_score"], c["path"].lower()))

    result = {
        "query": str(query_path),
        "library_size": len(library.paths),
        "shortlist_size": shortlist_size,
        "candidates": candidates[: max(top_k, 5)],
        "global_scores": scores,
        "global_order": order,
    }
    return result

def hashes_of(paths: list[Path], progress_every: int = 0) -> dict[int, str]:
    """按内容哈希标注库内同源重复（案例(3) 的原始图片存在跨案例重复）。"""
    digests: dict[int, str] = {}
    for index, path in enumerate(paths):
        digests[index] = sha256_of(Path(path))
        if progress_every and (index + 1) % progress_every == 0:
            print(f"    库哈希 {index + 1}/{len(paths)}", flush=True)
    return digests
