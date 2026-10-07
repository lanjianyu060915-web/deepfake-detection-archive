from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


class PreprocessingError(RuntimeError):
    """可向中文界面直接展示的预处理错误。"""


@dataclass(frozen=True)
class PreprocessConfig:
    sample_rate_fps: float = 1.0
    max_frames: int = 30
    face_size: tuple[int, int] = (224, 224)
    face_expand_ratio: float = 0.25

    def __post_init__(self) -> None:
        if self.sample_rate_fps <= 0:
            raise ValueError("抽帧频率必须大于0。")
        if self.max_frames <= 0:
            raise ValueError("单个视频最大抽帧数必须大于0。")
        if self.face_size[0] <= 0 or self.face_size[1] <= 0:
            raise ValueError("人脸输出尺寸必须大于0。")
        if self.face_expand_ratio < 0:
            raise ValueError("人脸框扩展比例不能小于0。")


@dataclass(frozen=True)
class VideoInfo:
    duration_seconds: float
    fps: float
    width: int
    height: int
    total_frames: int


@dataclass
class PreprocessResult:
    source_path: str
    filename: str
    label: str
    file_type: str = "未知"
    sha256: str = ""
    video_info: VideoInfo | None = None
    extracted_frames: int = 0
    face_frames: int = 0
    no_face_frames: int = 0
    frame_paths: list[str] = field(default_factory=list)
    face_paths: list[str] = field(default_factory=list)
    sample_output_dir: str = ""
    elapsed_seconds: float = 0.0
    error_message: str = ""

    def to_summary_dict(self) -> dict[str, Any]:
        info = self.video_info
        return {
            "文件名": self.filename,
            "标签或样本类别": self.label,
            "原始文件路径": self.source_path,
            "文件类型": self.file_type,
            "SHA-256": self.sha256,
            "视频时长": round(info.duration_seconds, 3) if info else None,
            "FPS": round(info.fps, 3) if info else None,
            "分辨率": f"{info.width}x{info.height}" if info else None,
            "总帧数": info.total_frames if info else None,
            "实际抽帧数": self.extracted_frames,
            "检测到人脸的帧数": self.face_frames,
            "无人脸帧数": self.no_face_frames,
            "输出文件路径": {
                "样本目录": self.sample_output_dir,
                "原始帧": self.frame_paths,
                "人脸裁剪": self.face_paths,
            },
            "处理耗时": round(self.elapsed_seconds, 3),
            "错误信息": self.error_message,
        }


def calculate_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _read_image(path: Path) -> np.ndarray | None:
    """使用 imdecode 支持 Windows 中文路径。"""
    try:
        raw = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if raw.size == 0:
        return None
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".jpg"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise PreprocessingError(f"无法编码输出图像：{path.name}")
    try:
        encoded.tofile(path)
    except OSError as exc:
        raise PreprocessingError(f"无法保存输出图像：{path}") from exc


def read_image(path: Path) -> np.ndarray | None:
    """公开的中文路径兼容图片读取接口，保持与正式预处理实现一致。"""
    return _read_image(path)


def write_image(path: Path, image: np.ndarray) -> None:
    """公开的中文路径兼容图片写入接口。"""
    _write_image(path, image)


def _safe_name(value: str) -> str:
    name = re.sub(r"[^0-9A-Za-z._\-\u4e00-\u9fff]+", "_", value).strip("._")
    return name or "sample"


def expand_face_box(
    box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    expand_ratio: float,
) -> tuple[int, int, int, int]:
    x, y, width, height = box
    pad_x = int(round(width * expand_ratio))
    pad_y = int(round(height * expand_ratio))
    return (
        max(0, x - pad_x),
        max(0, y - pad_y),
        min(image_width, x + width + pad_x),
        min(image_height, y + height + pad_y),
    )


class OpenCVFaceDetector:
    def __init__(self, cascade_path: Path | None = None) -> None:
        default_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.cascade_path = cascade_path or default_path
        self.classifier = cv2.CascadeClassifier(str(self.cascade_path))
        if self.classifier.empty():
            raise PreprocessingError(f"OpenCV人脸检测器加载失败：{self.cascade_path}")

    def detect_largest(self, image: np.ndarray) -> tuple[int, int, int, int] | None:
        faces = self._detect(image)
        if len(faces) > 0:
            return self.select_largest(faces)

        # 数据帧可能已经紧贴人脸边缘；补边后再次检测，再映射回原图。
        image_height, image_width = image.shape[:2]
        pad_x = max(16, image_width // 2)
        pad_y = max(16, image_height // 2)
        padded = cv2.copyMakeBorder(
            image,
            pad_y,
            pad_y,
            pad_x,
            pad_x,
            cv2.BORDER_REPLICATE,
        )
        padded_faces = self._detect(padded)
        if len(padded_faces) == 0:
            return None
        x, y, width, height = self.select_largest(padded_faces)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(image_width, x + width - pad_x)
        y2 = min(image_height, y + height - pad_y)
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2 - x1, y2 - y1

    def _detect(self, image: np.ndarray) -> Any:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        return self.classifier.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

    @staticmethod
    def select_largest(faces: Any) -> tuple[int, int, int, int]:
        x, y, width, height = max(faces, key=lambda box: int(box[2]) * int(box[3]))
        return int(x), int(y), int(width), int(height)


def extract_largest_face(
    image: np.ndarray,
    face_detector: OpenCVFaceDetector,
    config: PreprocessConfig,
) -> np.ndarray | None:
    """使用统一预处理配置裁剪并缩放最大人脸。"""
    box = face_detector.detect_largest(image)
    if box is None:
        return None
    x1, y1, x2, y2 = expand_face_box(
        box,
        image.shape[1],
        image.shape[0],
        config.face_expand_ratio,
    )
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, config.face_size, interpolation=cv2.INTER_AREA)


def load_face_crop(
    source_path: Path,
    config: PreprocessConfig | None = None,
    face_detector: OpenCVFaceDetector | None = None,
) -> np.ndarray:
    """读取图片并按现有预处理规则返回最大人脸裁剪。"""
    source_path = source_path.resolve()
    if not source_path.exists():
        raise PreprocessingError(f"输入文件不存在：{source_path}")
    if not source_path.is_file():
        raise PreprocessingError(f"输入路径不是文件：{source_path}")
    if source_path.suffix.lower() not in IMAGE_EXTENSIONS:
        supported = ", ".join(sorted(IMAGE_EXTENSIONS))
        raise PreprocessingError(
            f"不支持的图片格式：{source_path.suffix.lower() or '无扩展名'}；支持：{supported}"
        )

    image = _read_image(source_path)
    if image is None:
        raise PreprocessingError(f"无法读取图片，文件可能已损坏：{source_path.name}")

    actual_config = config or PreprocessConfig()
    actual_detector = face_detector or OpenCVFaceDetector()
    face = extract_largest_face(image, actual_detector, actual_config)
    if face is None:
        raise PreprocessingError(f"图片中未检测到人脸：{source_path.name}")
    return face


class MediaPreprocessor:
    def __init__(
        self,
        project_root: Path,
        output_root: Path,
        config: PreprocessConfig | None = None,
        face_detector: OpenCVFaceDetector | None = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.output_root = output_root.resolve()
        self.config = config or PreprocessConfig()
        self.face_detector = face_detector or OpenCVFaceDetector()

    def process_file(self, source_path: Path, label: str = "未标注") -> PreprocessResult:
        started = time.perf_counter()
        source_path = source_path.resolve()
        suffix = source_path.suffix.lower()
        file_type = "图片" if suffix in IMAGE_EXTENSIONS else "视频" if suffix in VIDEO_EXTENSIONS else "未知"
        result = PreprocessResult(
            source_path=self._display_path(source_path),
            filename=source_path.name,
            label=label,
            file_type=file_type,
        )

        try:
            if not source_path.exists():
                raise PreprocessingError(f"输入文件不存在：{source_path}")
            if not source_path.is_file():
                raise PreprocessingError(f"输入路径不是文件：{source_path}")

            result.sha256 = calculate_sha256(source_path)
            if file_type == "未知":
                supported = ", ".join(sorted(IMAGE_EXTENSIONS | VIDEO_EXTENSIONS))
                raise PreprocessingError(f"不支持的文件格式：{suffix or '无扩展名'}；支持：{supported}")

            sample_dir = self.output_root / f"{_safe_name(source_path.stem)}_{result.sha256[:8]}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            result.sample_output_dir = self._display_path(sample_dir)

            if file_type == "图片":
                self._process_image(source_path, sample_dir, result)
            else:
                self._process_video(source_path, sample_dir, result)
        except PreprocessingError as exc:
            result.error_message = str(exc)
        except Exception as exc:  # 防止单个坏样本导致批处理退出
            result.error_message = f"媒体预处理失败：{type(exc).__name__}：{exc}"
        finally:
            result.elapsed_seconds = time.perf_counter() - started
        return result

    def _process_image(self, source_path: Path, sample_dir: Path, result: PreprocessResult) -> None:
        image = _read_image(source_path)
        if image is None:
            raise PreprocessingError(f"无法读取图片，文件可能已损坏：{source_path.name}")

        frame_path = sample_dir / "frames" / "frame_000000.jpg"
        _write_image(frame_path, image)
        result.frame_paths.append(self._display_path(frame_path))
        result.extracted_frames = 1

        face = self._extract_largest_face(image)
        if face is None:
            result.no_face_frames = 1
            raise PreprocessingError(f"图片中未检测到人脸：{source_path.name}")

        face_path = sample_dir / "faces" / "face_000000.jpg"
        _write_image(face_path, face)
        result.face_paths.append(self._display_path(face_path))
        result.face_frames = 1

    def _process_video(self, source_path: Path, sample_dir: Path, result: PreprocessResult) -> None:
        capture = cv2.VideoCapture(str(source_path))
        if not capture.isOpened():
            capture.release()
            raise PreprocessingError(f"无法打开视频，文件可能已损坏或编码不受支持：{source_path.name}")

        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            total_frames = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            width = max(0, int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
            height = max(0, int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            duration = total_frames / fps if fps > 0 else 0.0
            result.video_info = VideoInfo(duration, fps, width, height, total_frames)

            if fps <= 0 or width <= 0 or height <= 0:
                raise PreprocessingError(f"无法读取有效的视频元数据：{source_path.name}")

            targets = self._sample_targets(total_frames, fps)
            if targets:
                for output_index, frame_index in enumerate(targets):
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        continue
                    self._save_video_frame(frame, frame_index, output_index, sample_dir, result)
            else:
                self._sample_unknown_length_video(capture, fps, sample_dir, result)
        finally:
            capture.release()

        if result.extracted_frames == 0:
            raise PreprocessingError(f"视频中没有可读取的画面：{source_path.name}")
        if result.face_frames == 0:
            raise PreprocessingError(f"视频抽取帧中未检测到人脸：{source_path.name}")

    def _sample_targets(self, total_frames: int, fps: float) -> list[int]:
        if total_frames <= 0:
            return []
        interval = max(1, int(round(fps / self.config.sample_rate_fps)))
        return list(range(0, total_frames, interval))[: self.config.max_frames]

    def _sample_unknown_length_video(
        self,
        capture: cv2.VideoCapture,
        fps: float,
        sample_dir: Path,
        result: PreprocessResult,
    ) -> None:
        interval = max(1, int(round(fps / self.config.sample_rate_fps)))
        frame_index = 0
        while result.extracted_frames < self.config.max_frames:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if frame_index % interval == 0:
                self._save_video_frame(
                    frame,
                    frame_index,
                    result.extracted_frames,
                    sample_dir,
                    result,
                )
            frame_index += 1

    def _save_video_frame(
        self,
        frame: np.ndarray,
        source_frame_index: int,
        output_index: int,
        sample_dir: Path,
        result: PreprocessResult,
    ) -> None:
        frame_path = sample_dir / "frames" / f"frame_{output_index:04d}_idx{source_frame_index:06d}.jpg"
        _write_image(frame_path, frame)
        result.frame_paths.append(self._display_path(frame_path))
        result.extracted_frames += 1

        face = self._extract_largest_face(frame)
        if face is None:
            result.no_face_frames += 1
            return

        face_path = sample_dir / "faces" / f"face_{output_index:04d}_idx{source_frame_index:06d}.jpg"
        _write_image(face_path, face)
        result.face_paths.append(self._display_path(face_path))
        result.face_frames += 1

    def _extract_largest_face(self, image: np.ndarray) -> np.ndarray | None:
        return extract_largest_face(image, self.face_detector, self.config)

    def expand_box(
        self,
        box: tuple[int, int, int, int],
        image_width: int,
        image_height: int,
    ) -> tuple[int, int, int, int]:
        return expand_face_box(
            box,
            image_width,
            image_height,
            self.config.face_expand_ratio,
        )

    def _display_path(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.project_root).as_posix()
        except ValueError:
            return str(path.resolve())
