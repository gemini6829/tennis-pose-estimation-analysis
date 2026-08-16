from pathlib import Path
import argparse
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


KEYPOINTS_PATH = Path("results/mediapipe/keypoints/mediapipe_keypoints.csv")
PROCESSING_SUMMARY_PATH = Path("results/mediapipe/logs/processing_summary.csv")

OUTPUT_DIR = Path("results/mediapipe/features")
FIGURE_DIR = Path("results/mediapipe/figures/feature_plots")


FEATURE_COLUMNS = [
    "shoulder_line_angle",
    "hip_line_angle",
    "shoulder_hip_separation_proxy",
    "left_knee_angle",
    "right_knee_angle",
    "left_elbow_angle",
    "right_elbow_angle",
    "body_center_x",
    "body_center_y",
]


def angle_between_points(a, b, c):
    """
    Computes angle ABC in degrees.
    Points are np arrays: [x, y].
    """
    if a is None or b is None or c is None:
        return np.nan

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return np.nan

    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)

    return float(np.degrees(np.arccos(cosine)))


def line_angle(p1, p2):
    """
    Computes angle of the line from p1 to p2 in degrees.
    """
    if p1 is None or p2 is None:
        return np.nan

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    return float(np.degrees(np.arctan2(dy, dx)))


def angle_difference(angle_a, angle_b):
    """
    Normalizes angle difference to [-180, 180].
    """
    if np.isnan(angle_a) or np.isnan(angle_b):
        return np.nan

    diff = angle_a - angle_b
    diff = (diff + 180) % 360 - 180

    return float(diff)


def get_point(frame_df, keypoint_name, threshold):
    """
    Returns normalized [x, y] point if keypoint confidence is high enough.
    Otherwise returns None.
    """
    row = frame_df[frame_df["keypoint"] == keypoint_name]

    if row.empty:
        return None

    row = row.iloc[0]

    if row["score"] < threshold:
        return None

    return np.array([row["x"], row["y"]], dtype=float)


def get_min_score(frame_df, keypoint_names):
    scores = []

    for keypoint_name in keypoint_names:
        row = frame_df[frame_df["keypoint"] == keypoint_name]

        if row.empty:
            return np.nan

        scores.append(float(row.iloc[0]["score"]))

    return min(scores)


def compute_features_for_frame(frame_df, threshold):
    ls = get_point(frame_df, "left_shoulder", threshold)
    rs = get_point(frame_df, "right_shoulder", threshold)
    le = get_point(frame_df, "left_elbow", threshold)
    re = get_point(frame_df, "right_elbow", threshold)
    lw = get_point(frame_df, "left_wrist", threshold)
    rw = get_point(frame_df, "right_wrist", threshold)

    lh = get_point(frame_df, "left_hip", threshold)
    rh = get_point(frame_df, "right_hip", threshold)
    lk = get_point(frame_df, "left_knee", threshold)
    rk = get_point(frame_df, "right_knee", threshold)
    la = get_point(frame_df, "left_ankle", threshold)
    ra = get_point(frame_df, "right_ankle", threshold)

    shoulder_angle = line_angle(ls, rs)
    hip_angle = line_angle(lh, rh)
    shoulder_hip_separation = angle_difference(shoulder_angle, hip_angle)

    left_knee_angle = angle_between_points(lh, lk, la)
    right_knee_angle = angle_between_points(rh, rk, ra)

    left_elbow_angle = angle_between_points(ls, le, lw)
    right_elbow_angle = angle_between_points(rs, re, rw)

    torso_points = [p for p in [ls, rs, lh, rh] if p is not None]

    if len(torso_points) == 4:
        body_center = np.mean(torso_points, axis=0)
        body_center_x = float(body_center[0])
        body_center_y = float(body_center[1])
    else:
        body_center_x = np.nan
        body_center_y = np.nan

    confidence_mins = {
        "shoulder_line_angle_confidence_min": get_min_score(
            frame_df, ["left_shoulder", "right_shoulder"]
        ),
        "hip_line_angle_confidence_min": get_min_score(
            frame_df, ["left_hip", "right_hip"]
        ),
        "shoulder_hip_separation_proxy_confidence_min": get_min_score(
            frame_df,
            ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
        ),
        "left_knee_angle_confidence_min": get_min_score(
            frame_df, ["left_hip", "left_knee", "left_ankle"]
        ),
        "right_knee_angle_confidence_min": get_min_score(
            frame_df, ["right_hip", "right_knee", "right_ankle"]
        ),
        "left_elbow_angle_confidence_min": get_min_score(
            frame_df, ["left_shoulder", "left_elbow", "left_wrist"]
        ),
        "right_elbow_angle_confidence_min": get_min_score(
            frame_df, ["right_shoulder", "right_elbow", "right_wrist"]
        ),
    }

    features = {
        "shoulder_line_angle": shoulder_angle,
        "hip_line_angle": hip_angle,
        "shoulder_hip_separation_proxy": shoulder_hip_separation,
        "left_knee_angle": left_knee_angle,
        "right_knee_angle": right_knee_angle,
        "left_elbow_angle": left_elbow_angle,
        "right_elbow_angle": right_elbow_angle,
        "body_center_x": body_center_x,
        "body_center_y": body_center_y,
    }

    features.update(confidence_mins)

    return features


def compute_features_by_frame(keypoints, threshold):
    group_cols = [
        "video_id",
        "filename",
        "stroke",
        "view",
        "condition",
        "player_id",
        "rep",
        "frame",
        "time_sec",
    ]

    rows = []

    grouped = keypoints.groupby(group_cols, sort=False)

    for group_values, frame_df in grouped:
        base = dict(zip(group_cols, group_values))
        feature_values = compute_features_for_frame(frame_df, threshold)

        row = {}
        row.update(base)
        row.update(feature_values)

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_features_by_video(features_by_frame, processing_summary):
    processing_lookup = processing_summary.set_index("video_id")

    rows = []

    group_cols = [
        "video_id",
        "filename",
        "stroke",
        "view",
        "condition",
        "player_id",
        "rep",
    ]

    for group_values, video_df in features_by_frame.groupby(group_cols, sort=False):
        base = dict(zip(group_cols, group_values))
        video_id = base["video_id"]

        if video_id in processing_lookup.index:
            frames_processed = int(processing_lookup.loc[video_id, "frames_processed"])
        else:
            frames_processed = len(video_df)

        pose_frames = len(video_df)

        for feature in FEATURE_COLUMNS:
            series = video_df[feature]
            available = series.dropna()

            row = {}
            row.update(base)

            row.update(
                {
                    "feature": feature,
                    "frames_processed": frames_processed,
                    "pose_frames": pose_frames,
                    "available_frames": int(available.shape[0]),
                    "availability_rate": (
                        available.shape[0] / frames_processed
                        if frames_processed > 0
                        else np.nan
                    ),
                    "mean": available.mean() if not available.empty else np.nan,
                    "std": available.std() if not available.empty else np.nan,
                    "min": available.min() if not available.empty else np.nan,
                    "max": available.max() if not available.empty else np.nan,
                }
            )

            rows.append(row)

    return pd.DataFrame(rows)


def summarize_features_by_group(feature_summary_by_video):
    summary = (
        feature_summary_by_video.groupby(["condition", "stroke", "view", "feature"])
        .agg(
            availability_mean=("availability_rate", "mean"),
            availability_std=("availability_rate", "std"),
            value_mean=("mean", "mean"),
            value_std=("mean", "std"),
            clips=("video_id", "count"),
        )
        .reset_index()
    )

    return summary


def sanitize_filename(text):
    text = Path(str(text)).stem
    text = re.sub(r"[^A-Za-z0-9_\\-]+", "_", text)
    return text


def select_videos_for_plots(features_by_frame, max_plot_videos, plot_all):
    unique_videos = (
        features_by_frame[
            ["video_id", "filename", "stroke", "view", "condition"]
        ]
        .drop_duplicates()
        .sort_values(["condition", "stroke", "view", "video_id"])
    )

    if plot_all:
        return unique_videos["video_id"].tolist()

    # Representative set: one clip per condition/stroke/view, capped by max_plot_videos.
    representative = (
        unique_videos.groupby(["condition", "stroke", "view"], sort=False)
        .head(1)
        .head(max_plot_videos)
    )

    return representative["video_id"].tolist()


def make_feature_plots(features_by_frame, max_plot_videos=9, plot_all=False):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plot_features = [
        "shoulder_line_angle",
        "hip_line_angle",
        "shoulder_hip_separation_proxy",
        "left_knee_angle",
        "right_knee_angle",
        "left_elbow_angle",
        "right_elbow_angle",
    ]

    selected_video_ids = select_videos_for_plots(
        features_by_frame,
        max_plot_videos=max_plot_videos,
        plot_all=plot_all,
    )

    for video_id in selected_video_ids:
        video_df = features_by_frame[features_by_frame["video_id"] == video_id].copy()

        if video_df.empty:
            continue

        video_df = video_df.sort_values("time_sec")
        filename = video_df["filename"].iloc[0]
        stroke = video_df["stroke"].iloc[0]
        view = video_df["view"].iloc[0]
        condition = video_df["condition"].iloc[0]

        for feature in plot_features:
            plt.figure(figsize=(10, 5))
            plt.plot(video_df["time_sec"], video_df[feature])
            plt.title(f"{feature} over time: {video_id} ({stroke}, {view}, {condition})")
            plt.xlabel("Time (seconds)")
            plt.ylabel(feature)
            plt.grid(True)
            plt.tight_layout()

            output_name = f"{video_id}_{sanitize_filename(filename)}_{feature}.png"
            output_path = FIGURE_DIR / output_name

            plt.savefig(output_path, dpi=200)
            plt.close()


def print_summary(feature_summary_by_video, feature_summary_by_group):
    print("\nFeature availability by feature")
    print("--------------------------------")

    by_feature = (
        feature_summary_by_video.groupby("feature")["availability_rate"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )

    print(by_feature.to_string(index=False))

    print("\nClean clips: feature availability")
    print("---------------------------------")

    clean = feature_summary_by_video[feature_summary_by_video["condition"] == "clean"]

    clean_summary = (
        clean.groupby("feature")["availability_rate"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )

    print(clean_summary.to_string(index=False))

    print("\nClean clips: torso/lower-body feature values by stroke and view")
    print("---------------------------------------------------------------")

    selected_features = [
        "shoulder_hip_separation_proxy",
        "left_knee_angle",
        "right_knee_angle",
    ]

    display = feature_summary_by_group[
        (feature_summary_by_group["condition"] == "clean")
        & (feature_summary_by_group["feature"].isin(selected_features))
    ].copy()

    display = display.sort_values(["feature", "stroke", "view"])

    print(display.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Minimum MediaPipe score required to use a landmark in a feature.",
    )
    parser.add_argument(
        "--max_plot_videos",
        type=int,
        default=9,
        help="Maximum number of representative videos to plot unless --plot_all is used.",
    )
    parser.add_argument(
        "--plot_all",
        action="store_true",
        help="Generate feature plots for every video.",
    )

    args = parser.parse_args()

    if not KEYPOINTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing keypoints file: {KEYPOINTS_PATH}\n"
            "Run src/run_mediapipe.py first."
        )

    if not PROCESSING_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Missing processing summary: {PROCESSING_SUMMARY_PATH}\n"
            "Run src/run_mediapipe.py first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading MediaPipe keypoints...")
    keypoints = pd.read_csv(KEYPOINTS_PATH, dtype={"rep": str})
    processing_summary = pd.read_csv(PROCESSING_SUMMARY_PATH)

    print("Computing tennis features by frame...")
    features_by_frame = compute_features_by_frame(
        keypoints=keypoints,
        threshold=args.threshold,
    )

    print("Summarizing features by video...")
    feature_summary_by_video = summarize_features_by_video(
        features_by_frame=features_by_frame,
        processing_summary=processing_summary,
    )

    print("Summarizing features by condition/stroke/view...")
    feature_summary_by_group = summarize_features_by_group(
        feature_summary_by_video=feature_summary_by_video,
    )

    features_by_frame_path = OUTPUT_DIR / "tennis_features_by_frame.csv"
    summary_by_video_path = OUTPUT_DIR / "tennis_feature_summary_by_video.csv"
    summary_by_group_path = OUTPUT_DIR / "tennis_feature_summary_by_group.csv"

    features_by_frame.to_csv(features_by_frame_path, index=False)
    feature_summary_by_video.to_csv(summary_by_video_path, index=False)
    feature_summary_by_group.to_csv(summary_by_group_path, index=False)

    print("Creating feature plots...")
    make_feature_plots(
        features_by_frame=features_by_frame,
        max_plot_videos=args.max_plot_videos,
        plot_all=args.plot_all,
    )

    print_summary(
        feature_summary_by_video=feature_summary_by_video,
        feature_summary_by_group=feature_summary_by_group,
    )

    print("\nSaved feature files:")
    print(f"  {features_by_frame_path}")
    print(f"  {summary_by_video_path}")
    print(f"  {summary_by_group_path}")

    print("\nSaved feature plots to:")
    print(f"  {FIGURE_DIR}")


if __name__ == "__main__":
    main()
