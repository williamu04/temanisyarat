"""
run_extraction.py
=================
CLI runner — single video atau batch dari folder.

Single video:
    python run_extraction.py single input.mp4
    python run_extraction.py single input.mp4 --out-video out.mp4 --out-npy out.npz

Batch folder:
    python run_extraction.py batch videos/
    python run_extraction.py batch videos/ --out-video-dir videos_out/ --out-npy-dir npy_out/
    python run_extraction.py batch videos/ --recursive

Custom model paths:
    python run_extraction.py single input.mp4 \\
        --pose-model path/pose.task \\ --face-model path/face.task \\
        --hand-model path/hand.task
"""

from __future__ import annotations

import argparse
import logging
import sys

from landmark_extractor import ExtractorConfig, LandmarkExtractor

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared model arguments (reused by both sub-commands)
# ---------------------------------------------------------------------------
def _add_model_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("Model paths")
    g.add_argument("--pose-model", default="tasks/pose_landmarker_heavy.task")
    g.add_argument("--face-model", default="tasks/face_landmarker.task")
    g.add_argument("--hand-model", default="tasks/hand_landmarker.task")


def _add_confidence_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("Confidence thresholds")
    g.add_argument("--pose-conf", type=float, default=0.5)
    g.add_argument("--face-conf", type=float, default=0.3)
    g.add_argument("--hand-conf", type=float, default=0.5)


def _build_config(args: argparse.Namespace) -> ExtractorConfig:
    return ExtractorConfig(
        pose_model_path=args.pose_model,
        face_model_path=args.face_model,
        hand_model_path=args.hand_model,
        pose_detection_conf=args.pose_conf,
        pose_presence_conf=args.pose_conf,
        pose_tracking_conf=args.pose_conf,
        face_detection_conf=args.face_conf,
        face_presence_conf=args.face_conf,
        face_tracking_conf=args.face_conf,
        hand_detection_conf=args.hand_conf,
        hand_presence_conf=args.hand_conf,
        hand_tracking_conf=args.hand_conf,
    )


# ---------------------------------------------------------------------------
# Sub-command: single
# ---------------------------------------------------------------------------
def cmd_single(args: argparse.Namespace) -> None:
    cfg = _build_config(args)
    extractor = LandmarkExtractor(cfg)
    result = extractor.process_video(
        args.input,
        out_video_path=args.out_video,
        out_npy_path=args.out_npy,
    )
    print(
        f"\nResult summary"
        f"\n  video : {result.video_path}"
        f"\n  frames: {result.total_frames}"
        f"\n  fps   : {result.fps:.2f}"
        f"\n  pose  : {result.pose.shape}"
        f"\n  face  : {result.face.shape}"
        f"\n  hands : {result.hands.shape}"
    )


# ---------------------------------------------------------------------------
# Sub-command: batch
# ---------------------------------------------------------------------------
def cmd_batch(args: argparse.Namespace) -> None:
    cfg = _build_config(args)
    extractor = LandmarkExtractor(cfg)
    results = extractor.process_folder(
        args.folder,
        out_video_dir=args.out_video_dir,
        out_npy_dir=args.out_npy_dir,
        recursive=args.recursive,
    )

    print(f"\nBatch summary ({len(results)} video(s) processed):")
    for name, r in results.items():
        print(
            f"  {name:40s} | frames={r.total_frames:5d} | "
            f"pose={r.pose.shape} face={r.face.shape} hands={r.hands.shape}"
        )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MediaPipe landmark extractor — single video or batch folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- single ----
    p_single = sub.add_parser("single", help="Process one video")
    p_single.add_argument("input", help="Input video path")
    p_single.add_argument("--out-video", default=None, help="Annotated video output path")
    p_single.add_argument("--out-npy", default=None, help=".npz output path")
    _add_model_args(p_single)
    _add_confidence_args(p_single)
    p_single.set_defaults(func=cmd_single)

    # ---- batch ----
    p_batch = sub.add_parser("batch", help="Process all videos in a folder")
    p_batch.add_argument("folder", help="Input folder path")
    p_batch.add_argument("--out-video-dir", default=None, help="Dir for annotated videos")
    p_batch.add_argument("--out-npy-dir", default=None, help="Dir for .npz files")
    p_batch.add_argument(
        "--recursive", action="store_true", help="Scan sub-directories too"
    )
    _add_model_args(p_batch)
    _add_confidence_args(p_batch)
    p_batch.set_defaults(func=cmd_batch)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except (FileNotFoundError, NotADirectoryError) as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted.")
        sys.exit(0)


if __name__ == "__main__":
    main()
