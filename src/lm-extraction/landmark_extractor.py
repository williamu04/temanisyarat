"""
landmark_extractor.py
=====================
Reusable MediaPipe landmark extractor (Pose + Face + Hand).
Hanya menyimpan & menggambar subset landmark yang relevan:
  - Pose : 9 titik upper-body  (POSE_UPPER_IDX)
  - Face : 46 titik kunci      (FACE_KEY_IDX)
  - Hand : 21 titik + skeleton (HAND_CONNECTIONS)

Usage (single video):
    from landmark_extractor import LandmarkExtractor
    ext = LandmarkExtractor()
    result = ext.process_video("input.mp4", out_video_path="output.mp4")
    # result.pose  : [T, 9,  4]
    # result.face  : [T, 46, 3]
    # result.hands : [T, 2, 21, 3]

Usage (batch):
    results = LandmarkExtractor().process_folder(
        "videos/",
        out_video_dir="videos_out/",
        out_npy_dir="npy_out/",
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}

# ---------------------------------------------------------------------------
# Subset definitions
# ---------------------------------------------------------------------------

# Upper-body pose landmarks (dari 33 total)
POSE_UPPER_IDX = [0, 11, 12, 13, 14, 15, 16, 23, 24]

# Key facial landmarks (dari 478 total)
FACE_KEY_IDX = [
    # bibir
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    # mata kiri
    33, 160, 158, 133, 153, 144,
    # mata kanan
    362, 385, 387, 263, 373, 380,
    # alis kiri
    70, 63, 105, 66, 107,
    # alis kanan
    336, 296, 334, 293, 300,
]

# Hand skeleton connections (semua 21 landmark dipakai, koneksi digambar)
HAND_CONNECTIONS = [
    (0, 1),  (1, 2),  (2, 3),   (3, 4),    # ibu jari
    (0, 5),  (5, 6),  (6, 7),   (7, 8),    # telunjuk
    (0, 9),  (9, 10), (10, 11), (11, 12),  # jari tengah
    (0, 13), (13, 14),(14, 15), (15, 16),  # jari manis
    (0, 17), (17, 18),(18, 19), (19, 20),  # kelingking
    (5, 9),  (9, 13), (13, 17),            # telapak
]

# Full landmark counts (MediaPipe spec)
N_POSE_FULL = 33
N_FACE_FULL = 478
N_HAND      = 21


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------
@dataclass
class ExtractorConfig:
    pose_model_path: str = "tasks/pose_landmarker_heavy.task"
    face_model_path: str = "tasks/face_landmarker.task"
    hand_model_path: str = "tasks/hand_landmarker.task"

    # Pose
    num_poses: int = 1
    pose_detection_conf: float = 0.5
    pose_presence_conf: float  = 0.5
    pose_tracking_conf: float  = 0.5

    # Face
    num_faces: int = 1
    face_detection_conf: float = 0.1
    face_presence_conf: float  = 0.1 
    face_tracking_conf: float  = 0.1

    # Hand
    num_hands: int = 2
    hand_detection_conf: float = 0.5
    hand_presence_conf: float  = 0.5
    hand_tracking_conf: float  = 0.5

    # Drawing toggles
    draw_pose:  bool = True
    draw_face:  bool = True
    draw_hands: bool = True

    # Drawing style
    pose_color: tuple[int, int, int] = (0, 255, 0)   # BGR green
    face_color: tuple[int, int, int] = (255, 0, 0)   # BGR blue
    hand_color: tuple[int, int, int] = (0, 0, 255)   # BGR red
    pose_radius: int = 4
    face_radius: int = 2
    hand_radius: int = 3
    hand_line_thickness: int = 1

    fourcc: str = "mp4v"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class ExtractionResult:
    """
    pose  : [T, len(POSE_UPPER_IDX), 4]   (x, y, z, visibility)
    face  : [T, len(FACE_KEY_IDX),   3]   (x, y, z)
    hands : [T, 2, N_HAND,           3]   (hand_idx, landmark, xyz)
    """
    video_path:   str
    total_frames: int
    fps:          float
    pose:  np.ndarray = field(default_factory=lambda: np.array([]))
    face:  np.ndarray = field(default_factory=lambda: np.array([]))
    hands: np.ndarray = field(default_factory=lambda: np.array([]))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _landmarks_to_array(
    list_lm: list,
    extra_fields: Optional[list[str]] = None,
) -> np.ndarray:
    """Convert MediaPipe landmark list → float32 array [N, 3+extras]."""
    cols = ["x", "y", "z"] + (extra_fields or [])
    arr  = np.full((len(list_lm), len(cols)), np.nan, dtype=np.float32)
    for i, lm in enumerate(list_lm):
        for j, name in enumerate(cols):
            arr[i, j] = getattr(lm, name, np.nan)
    return arr


def _subset(arr: np.ndarray, indices: list[int]) -> np.ndarray:
    """Select rows by index list; out-of-range indices → NaN row."""
    out = np.full((len(indices), arr.shape[1]), np.nan, dtype=np.float32)
    for out_i, src_i in enumerate(indices):
        if src_i < len(arr):
            out[out_i] = arr[src_i]
    return out


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _px(lm, w: int, h: int) -> tuple[int, int]:
    return int(lm.x * w), int(lm.y * h)


def _valid(x: int, y: int, w: int, h: int) -> bool:
    return 0 <= x < w and 0 <= y < h


def _draw_pose_subset(
    frame: np.ndarray,
    lm_list: list,
    indices: list[int],
    color: tuple[int, int, int],
    radius: int,
) -> None:
    """Gambar titik-titik pose subset saja."""
    h, w = frame.shape[:2]
    for idx in indices:
        if idx >= len(lm_list):
            continue
        x, y = _px(lm_list[idx], w, h)
        if _valid(x, y, w, h):
            cv2.circle(frame, (x, y), radius, color, -1)


def _draw_face_subset(
    frame: np.ndarray,
    lm_list: list,
    indices: list[int],
    color: tuple[int, int, int],
    radius: int,
) -> None:
    """Gambar titik-titik face key points saja."""
    h, w = frame.shape[:2]
    for idx in indices:
        if idx >= len(lm_list):
            continue
        x, y = _px(lm_list[idx], w, h)
        if _valid(x, y, w, h):
            cv2.circle(frame, (x, y), radius, color, -1)


def _draw_hand_skeleton(
    frame: np.ndarray,
    lm_list: list,
    connections: list[tuple[int, int]],
    color: tuple[int, int, int],
    radius: int,
    thickness: int,
) -> None:
    """Gambar semua titik tangan + garis skeleton sesuai HAND_CONNECTIONS."""
    h, w = frame.shape[:2]

    # Kumpulkan semua posisi pixel yang valid
    pts: dict[int, tuple[int, int]] = {}
    for i, lm in enumerate(lm_list):
        x, y = _px(lm, w, h)
        if _valid(x, y, w, h):
            pts[i] = (x, y)
            cv2.circle(frame, (x, y), radius, color, -1)

    # Gambar koneksi/tulang
    for a, b in connections:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], color, thickness)


# ---------------------------------------------------------------------------
# Main extractor class
# ---------------------------------------------------------------------------

class LandmarkExtractor:
    """
    MediaPipe Tasks extractor (Pose / Face / Hand) — VIDEO mode.

    Pose  → simpan & gambar hanya POSE_UPPER_IDX  (9 titik)
    Face  → simpan & gambar hanya FACE_KEY_IDX    (46 titik)
    Hand  → simpan semua 21 titik, gambar dengan HAND_CONNECTIONS
    """

    def __init__(self, config: Optional[ExtractorConfig] = None) -> None:
        self.cfg = config or ExtractorConfig()
        self._validate_models()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_video(
        self,
        video_path: str | Path,
        out_video_path: Optional[str | Path] = None,
        out_npy_path: Optional[str | Path] = None,
    ) -> ExtractionResult:
        """
        Proses satu video, kembalikan ExtractionResult.

        Parameters
        ----------
        video_path     : Path video input.
        out_video_path : Jika diisi, tulis video teranotasi ke sini.
        out_npy_path   : Jika diisi, simpan .npz (pose, face, hands).
        """
        video_path = Path(video_path)
        logger.info("Processing: %s", video_path)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer: Optional[cv2.VideoWriter] = None
        if out_video_path is not None:
            Path(out_video_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter.fourcc(*self.cfg.fourcc)
            writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

        pose_seq, face_seq, hand_seq = [], [], []
        n_pose_out = len(POSE_UPPER_IDX)
        n_face_out = len(FACE_KEY_IDX)

        BaseOpts = mp.tasks.BaseOptions
        RunMode  = mp.tasks.vision.RunningMode

        pose_opts = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=BaseOpts(model_asset_path=self.cfg.pose_model_path),
            running_mode=RunMode.VIDEO,
            num_poses=self.cfg.num_poses,
            min_pose_detection_confidence=self.cfg.pose_detection_conf,
            min_pose_presence_confidence=self.cfg.pose_presence_conf,
            min_tracking_confidence=self.cfg.pose_tracking_conf,
        )
        face_opts = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=BaseOpts(model_asset_path=self.cfg.face_model_path),
            running_mode=RunMode.VIDEO,
            num_faces=self.cfg.num_faces,
            min_face_detection_confidence=self.cfg.face_detection_conf,
            min_face_presence_confidence=self.cfg.face_presence_conf,
            min_tracking_confidence=self.cfg.face_tracking_conf,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        hand_opts = mp.tasks.vision.HandLandmarkerOptions(
            base_options=BaseOpts(model_asset_path=self.cfg.hand_model_path),
            running_mode=RunMode.VIDEO,
            num_hands=self.cfg.num_hands,
            min_hand_detection_confidence=self.cfg.hand_detection_conf,
            min_hand_presence_confidence=self.cfg.hand_presence_conf,
            min_tracking_confidence=self.cfg.hand_tracking_conf,
        )

        frame_idx = 0
        try:
            with (
                mp.tasks.vision.PoseLandmarker.create_from_options(pose_opts) as pose_lm,
                mp.tasks.vision.FaceLandmarker.create_from_options(face_opts) as face_lm,
                mp.tasks.vision.HandLandmarker.create_from_options(hand_opts) as hand_lm,
            ):
                while True:
                    ok, bgr = cap.read()
                    if not ok:
                        break

                    rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    ts_ms  = int(frame_idx * 1000.0 / fps)

                    pose_res = pose_lm.detect_for_video(mp_img, ts_ms)
                    face_res = face_lm.detect_for_video(mp_img, ts_ms)
                    hand_res = hand_lm.detect_for_video(mp_img, ts_ms)

                    # ---- Pose → subset upper-body ----
                    if pose_res.pose_landmarks:
                        full     = _landmarks_to_array(
                            pose_res.pose_landmarks[0], extra_fields=["visibility"]
                        )
                        pose_arr = _subset(full, POSE_UPPER_IDX)   # [9, 4]
                    else:
                        pose_arr = np.full((n_pose_out, 4), np.nan, dtype=np.float32)
                    pose_seq.append(pose_arr)

                    # ---- Face → subset key points ----
                    if face_res.face_landmarks:
                        full     = _landmarks_to_array(face_res.face_landmarks[0])
                        face_arr = _subset(full, FACE_KEY_IDX)     # [46, 3]
                    else:
                        face_arr = np.full((n_face_out, 3), np.nan, dtype=np.float32)
                    face_seq.append(face_arr)

                    # ---- Hands → semua 21 titik ----
                    frame_hands = np.full((2, N_HAND, 3), np.nan, dtype=np.float32)
                    if hand_res.hand_landmarks:
                        for hi, lm_list in enumerate(hand_res.hand_landmarks[:2]):
                            ha = _landmarks_to_array(lm_list)
                            n  = min(len(ha), N_HAND)
                            frame_hands[hi, :n] = ha[:n]
                    hand_seq.append(frame_hands)

                    # ---- Draw ----
                    if writer is not None:
                        if self.cfg.draw_pose and pose_res.pose_landmarks:
                            _draw_pose_subset(
                                bgr, pose_res.pose_landmarks[0],
                                POSE_UPPER_IDX,
                                self.cfg.pose_color, self.cfg.pose_radius,
                            )
                        if self.cfg.draw_face and face_res.face_landmarks:
                            _draw_face_subset(
                                bgr, face_res.face_landmarks[0],
                                FACE_KEY_IDX,
                                self.cfg.face_color, self.cfg.face_radius,
                            )
                        if self.cfg.draw_hands and hand_res.hand_landmarks:
                            for lm_list in hand_res.hand_landmarks:
                                _draw_hand_skeleton(
                                    bgr, lm_list,
                                    HAND_CONNECTIONS,
                                    self.cfg.hand_color,
                                    self.cfg.hand_radius,
                                    self.cfg.hand_line_thickness,
                                )
                        writer.write(bgr)

                    frame_idx += 1
                    if frame_idx % 100 == 0:
                        logger.debug("  frame %d", frame_idx)

        finally:
            cap.release()
            if writer is not None:
                writer.release()

        result = ExtractionResult(
            video_path=str(video_path),
            total_frames=frame_idx,
            fps=fps,
            pose=np.stack(pose_seq)  if pose_seq  else np.empty((0, n_pose_out, 4)),
            face=np.stack(face_seq)  if face_seq  else np.empty((0, n_face_out, 3)),
            hands=np.stack(hand_seq) if hand_seq  else np.empty((0, 2, N_HAND, 3)),
        )

        if out_npy_path is not None:
            self.save_npy(result, out_npy_path)

        logger.info(
            "Done: %s | frames=%d | pose=%s face=%s hands=%s",
            video_path.name, frame_idx,
            result.pose.shape, result.face.shape, result.hands.shape,
        )
        return result

    def process_folder(
        self,
        folder_path: str | Path,
        out_video_dir: Optional[str | Path] = None,
        out_npy_dir: Optional[str | Path] = None,
        recursive: bool = False,
    ) -> dict[str, ExtractionResult]:
        """
        Batch-process semua video dalam folder.

        Returns
        -------
        dict  filename → ExtractionResult
        """
        folder_path = Path(folder_path)
        if not folder_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {folder_path}")

        pattern = "**/*" if recursive else "*"
        videos  = sorted(
            p for p in folder_path.glob(pattern)
            if p.suffix.lower() in VIDEO_EXTENSIONS
        )

        if not videos:
            logger.warning("No video files found in: %s", folder_path)
            return {}

        logger.info("Found %d video(s) in %s", len(videos), folder_path)

        if out_video_dir:
            Path(out_video_dir).mkdir(parents=True, exist_ok=True)
        if out_npy_dir:
            Path(out_npy_dir).mkdir(parents=True, exist_ok=True)

        results: dict[str, ExtractionResult] = {}
        for idx, vid in enumerate(videos, 1):
            logger.info("[%d/%d] %s", idx, len(videos), vid.name)
            ovp = Path(out_video_dir) / (vid.stem + "_landmarks" + vid.suffix) if out_video_dir else None
            onp = Path(out_npy_dir)   / (vid.stem + ".npz")                    if out_npy_dir  else None
            try:
                results[vid.name] = self.process_video(vid, ovp, onp)
            except Exception as exc:  # noqa: BLE001
                logger.error("  FAILED %s: %s", vid.name, exc)

        logger.info("Batch complete: %d/%d succeeded", len(results), len(videos))
        return results

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    @staticmethod
    def save_npy(result: ExtractionResult, path: str | Path) -> None:
        """Simpan ExtractionResult ke .npz terkompresi."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(path), pose=result.pose, face=result.face, hands=result.hands)
        logger.info("Saved: %s", path)

    @staticmethod
    def load_npy(path: str | Path) -> dict[str, np.ndarray]:
        """Load .npz → dict dengan key 'pose', 'face', 'hands'."""
        data = np.load(str(path))
        return {k: data[k] for k in ("pose", "face", "hands")}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _validate_models(self) -> None:
        missing = [
            str(Path(getattr(self.cfg, attr)))
            for attr in ("pose_model_path", "face_model_path", "hand_model_path")
            if not Path(getattr(self.cfg, attr)).exists()
        ]
        if missing:
            raise FileNotFoundError("Model file(s) not found:\n  " + "\n  ".join(missing))
