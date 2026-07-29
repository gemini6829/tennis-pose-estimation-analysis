from pathlib import Path
import argparse
import csv
import sys

import cv2
import mediapipe as mp
import pandas as pd
from tqdm import tqdm


MANIFEST_PATH = Path("data/metadata/video_manifest.csv")
VIDEO_DIR = Path("data/raw_videos/organized")

OUTPUT_ROOT = Path("results/mediapipe")
KEYPOINTS_DIR = OUTPUT_ROOT / "keypoints"
OVERLAY_DIR = OUTPUT_ROOT / "overlay_videos"
LOGS_DIR = OUTPUT_ROOT / "logs"

MEDIAPIPE_KEYPOINT_NAMES = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]


def read_manifest(manifest_path: Path) -> list[dict]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError("Manifest is empty.")

    return rows


def ensure_output_dirs() -> None:
    KEYPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def make_video_writer(output_path: Path, fps: float, width: int, height: int):
    """
    Creates a video writer for overlay videos.

    Some OpenCV installations cannot write MP4 files reliably, especially on macOS.
    This function tries MP4 first, then falls back to AVI.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fps <= 0:
        fps = 30.0

    if width <= 0 or height <= 0:
        raise RuntimeError(
            f"Invalid video dimensions for writer: width={width}, height={height}"
        )

    candidates = [
        (output_path.with_suffix(".mp4"), "mp4v"),
        (output_path.with_suffix(".mp4"), "avc1"),
        (output_path.with_suffix(".avi"), "MJPG"),
        (output_path.with_suffix(".avi"), "XVID"),
    ]

    for candidate_path, codec in candidates:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(candidate_path), fourcc, fps, (width, height))

        if writer.isOpened():
            print(f"Using codec {codec} for overlay video: {candidate_path}")
            return writer, candidate_path

        writer.release()

    raise RuntimeError(
        f"Could not create output video writer for {output_path}. "
        f"Tried codecs: {[codec for _, codec in candidates]}"
    )


def process_video(row: dict, pose, drawing_utils, pose_connections, draw_overlays: bool = True) -> tuple[list[dict], dict]:
    video_id = row["video_id"]
    filename = row["filename"]
    input_path = VIDEO_DIR / filename

    summary = {
        "video_id": video_id,
        "filename": filename,
        "status": "started",
        "frames_total_reported": 0,
        "frames_processed": 0,
        "frames_with_pose": 0,
        "fps": "",
        "width": "",
        "height": "",
        "overlay_path": "",
        "error": "",
    }

    keypoint_rows = []

    if not input_path.exists():
        summary["status"] = "error"
        summary["error"] = f"File not found: {input_path}"
        return keypoint_rows, summary

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        summary["status"] = "error"
        summary["error"] = f"Could not open video. Possible codec issue: {input_path}"
        return keypoint_rows, summary

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30.0

    summary["frames_total_reported"] = frames_total
    summary["fps"] = round(float(fps), 3)
    summary["width"] = width
    summary["height"] = height

    writer = None
    overlay_path = OVERLAY_DIR / f"{video_id}_{Path(filename).stem}_mediapipe_overlay.mp4"

    if draw_overlays:
        writer, actual_overlay_path = make_video_writer(overlay_path, fps, width, height)
        summary["overlay_path"] = str(actual_overlay_path)

    frame_idx = 0
    frames_with_pose = 0

    progress_total = frames_total if frames_total > 0 else None
    pbar_label = f"{video_id} {filename}"

    with tqdm(total=progress_total, desc=pbar_label, unit="frame") as pbar:
        while True:
            success, frame_bgr = cap.read()

            if not success:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False

            results = pose.process(frame_rgb)

            frame_rgb.flags.writeable = True

            if results.pose_landmarks:
                frames_with_pose += 1

                if draw_overlays:
                    drawing_utils.draw_landmarks(
                        frame_bgr,
                        results.pose_landmarks,
                        pose_connections,
                    )

                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    keypoint_name = (
                        MEDIAPIPE_KEYPOINT_NAMES[idx]
                        if idx < len(MEDIAPIPE_KEYPOINT_NAMES)
                        else f"keypoint_{idx}"
                    )

                    keypoint_rows.append(
                        {
                            "video_id": video_id,
                            "filename": filename,
                            "model": "mediapipe",
                            "stroke": row["stroke"],
                            "view": row["view"],
                            "condition": row["condition"],
                            "player_id": row["player_id"],
                            "rep": row["rep"],
                            "frame": frame_idx,
                            "time_sec": frame_idx / fps,
                            "keypoint": keypoint_name,
                            "x": landmark.x,
                            "y": landmark.y,
                            "z": landmark.z,
                            "score": landmark.visibility,
                            "image_width": width,
                            "image_height": height,
                        }
                    )

            if writer is not None:
                writer.write(frame_bgr)

            frame_idx += 1
            pbar.update(1)

    cap.release()

    if writer is not None:
        writer.release()

    summary["frames_processed"] = frame_idx
    summary["frames_with_pose"] = frames_with_pose
    summary["status"] = "ok"

    return keypoint_rows, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of videos to process. Use this for testing first.",
    )
    parser.add_argument(
        "--no_overlays",
        action="store_true",
        help="Skip writing overlay videos. Useful if you only want keypoints.",
    )
    parser.add_argument(
        "--min_detection_confidence",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--min_tracking_confidence",
        type=float,
        default=0.5,
    )
    args = parser.parse_args()

    ensure_output_dirs()

    manifest_rows = read_manifest(MANIFEST_PATH)

    if args.limit is not None:
        manifest_rows = manifest_rows[: args.limit]

    mp_pose = mp.solutions.pose
    drawing_utils = mp.solutions.drawing_utils

    all_keypoint_rows = []
    processing_summaries = []

    print(f"Processing {len(manifest_rows)} video(s) with MediaPipe Pose.")

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    ) as pose:
        for row in manifest_rows:
            keypoint_rows, summary = process_video(
                row=row,
                pose=pose,
                drawing_utils=drawing_utils,
                pose_connections=mp_pose.POSE_CONNECTIONS,
                draw_overlays=not args.no_overlays,
            )

            all_keypoint_rows.extend(keypoint_rows)
            processing_summaries.append(summary)

    keypoints_path = KEYPOINTS_DIR / "mediapipe_keypoints.csv"
    summary_path = LOGS_DIR / "processing_summary.csv"

    keypoints_df = pd.DataFrame(all_keypoint_rows)
    summary_df = pd.DataFrame(processing_summaries)

    keypoints_df.to_csv(keypoints_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\nMediaPipe processing complete.")
    print(f"Saved keypoints to: {keypoints_path}")
    print(f"Saved processing summary to: {summary_path}")
    print(f"Saved overlay videos to: {OVERLAY_DIR}")

    error_rows = summary_df[summary_df["status"] != "ok"]
    if not error_rows.empty:
        print("\nSome videos had errors:")
        print(error_rows[["video_id", "filename", "error"]].to_string(index=False))
        sys.exit(1)

    no_pose_rows = summary_df[summary_df["frames_with_pose"] == 0]
    if not no_pose_rows.empty:
        print("\nWarning: Some videos had zero detected pose frames:")
        print(no_pose_rows[["video_id", "filename"]].to_string(index=False))


if __name__ == "__main__":
    main()
