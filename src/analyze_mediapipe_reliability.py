from pathlib import Path
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


MANIFEST_PATH = Path("data/metadata/video_manifest.csv")
KEYPOINTS_PATH = Path("results/mediapipe/keypoints/mediapipe_keypoints.csv")
PROCESSING_SUMMARY_PATH = Path("results/mediapipe/logs/processing_summary.csv")

OUTPUT_DIR = Path("results/mediapipe/analysis")
FIGURE_DIR = Path("results/mediapipe/figures")


IMPORTANT_KEYPOINTS = [
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


BODY_REGIONS = {
    "shoulders": ["left_shoulder", "right_shoulder"],
    "elbows": ["left_elbow", "right_elbow"],
    "wrists": ["left_wrist", "right_wrist"],
    "hips": ["left_hip", "right_hip"],
    "knees": ["left_knee", "right_knee"],
    "ankles": ["left_ankle", "right_ankle"],
}


FEATURE_REQUIREMENTS = {
    "left_elbow_angle": ["left_shoulder", "left_elbow", "left_wrist"],
    "right_elbow_angle": ["right_shoulder", "right_elbow", "right_wrist"],
    "left_knee_angle": ["left_hip", "left_knee", "left_ankle"],
    "right_knee_angle": ["right_hip", "right_knee", "right_ankle"],
    "shoulder_line_angle": ["left_shoulder", "right_shoulder"],
    "hip_line_angle": ["left_hip", "right_hip"],
    "shoulder_hip_separation_proxy": [
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
    ],
}


def load_inputs():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing manifest: {MANIFEST_PATH}")

    if not KEYPOINTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing MediaPipe keypoints CSV: {KEYPOINTS_PATH}\n"
            "Run src/run_mediapipe.py first."
        )

    if not PROCESSING_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Missing MediaPipe processing summary: {PROCESSING_SUMMARY_PATH}\n"
            "Run src/run_mediapipe.py first."
        )

    manifest = pd.read_csv(MANIFEST_PATH, dtype={"rep": str})
    keypoints = pd.read_csv(KEYPOINTS_PATH, dtype={"rep": str})
    processing = pd.read_csv(PROCESSING_SUMMARY_PATH)

    return manifest, keypoints, processing


def prepare_video_metadata(manifest, processing):
    metadata = manifest.merge(
        processing[
            [
                "video_id",
                "filename",
                "status",
                "frames_processed",
                "frames_with_pose",
                "fps",
                "width",
                "height",
            ]
        ],
        on=["video_id", "filename"],
        how="left",
    )

    if metadata["frames_processed"].isna().any():
        missing = metadata[metadata["frames_processed"].isna()][["video_id", "filename"]]
        raise ValueError(
            "Some manifest videos are missing from processing_summary.csv:\n"
            + missing.to_string(index=False)
        )

    metadata["frames_processed"] = metadata["frames_processed"].astype(int)
    metadata["frames_with_pose"] = metadata["frames_with_pose"].astype(int)

    return metadata


def compute_keypoint_detection_rates(keypoints, video_metadata, threshold):
    keypoints = keypoints.copy()
    keypoints["detected"] = keypoints["score"] >= threshold

    detected_counts = (
        keypoints[keypoints["detected"]]
        .groupby(["video_id", "keypoint"])
        .size()
        .reset_index(name="detected_frames")
    )

    all_keypoints = sorted(keypoints["keypoint"].unique())

    full_index = (
        video_metadata[["video_id"]]
        .assign(_key=1)
        .merge(pd.DataFrame({"keypoint": all_keypoints, "_key": 1}), on="_key")
        .drop(columns="_key")
    )

    rates = full_index.merge(
        detected_counts,
        on=["video_id", "keypoint"],
        how="left",
    )

    rates["detected_frames"] = rates["detected_frames"].fillna(0).astype(int)

    rates = rates.merge(
        video_metadata[
            [
                "video_id",
                "filename",
                "stroke",
                "view",
                "condition",
                "player_id",
                "rep",
                "frames_processed",
                "frames_with_pose",
            ]
        ],
        on="video_id",
        how="left",
    )

    rates["detection_rate"] = rates["detected_frames"] / rates["frames_processed"]

    column_order = [
        "video_id",
        "filename",
        "stroke",
        "view",
        "condition",
        "player_id",
        "rep",
        "keypoint",
        "detected_frames",
        "frames_processed",
        "frames_with_pose",
        "detection_rate",
    ]

    return rates[column_order]


def compute_body_region_reliability(keypoint_rates):
    rows = []

    for region, region_keypoints in BODY_REGIONS.items():
        region_df = keypoint_rates[keypoint_rates["keypoint"].isin(region_keypoints)]

        region_by_video = (
            region_df.groupby(
                [
                    "video_id",
                    "filename",
                    "stroke",
                    "view",
                    "condition",
                    "player_id",
                    "rep",
                ],
                as_index=False,
            )["detection_rate"]
            .mean()
        )

        region_by_video["region"] = region
        rows.append(region_by_video)

    combined = pd.concat(rows, ignore_index=True)

    return combined[
        [
            "video_id",
            "filename",
            "stroke",
            "view",
            "condition",
            "player_id",
            "rep",
            "region",
            "detection_rate",
        ]
    ]


def compute_feature_availability(keypoints, video_metadata, threshold):
    scores = keypoints.pivot_table(
        index=["video_id", "frame"],
        columns="keypoint",
        values="score",
        aggfunc="max",
    )

    feature_rows = []

    for _, video in video_metadata.iterrows():
        video_id = video["video_id"]
        frames_processed = int(video["frames_processed"])

        if video_id in scores.index.get_level_values("video_id"):
            video_scores = scores.loc[video_id]
        else:
            video_scores = pd.DataFrame()

        for feature_name, required_keypoints in FEATURE_REQUIREMENTS.items():
            missing_columns = [
                keypoint
                for keypoint in required_keypoints
                if keypoint not in video_scores.columns
            ]

            if missing_columns or video_scores.empty:
                available_frames = 0
            else:
                available = video_scores[required_keypoints].ge(threshold).all(axis=1)
                available_frames = int(available.sum())

            availability_rate = (
                available_frames / frames_processed if frames_processed > 0 else np.nan
            )

            feature_rows.append(
                {
                    "video_id": video_id,
                    "filename": video["filename"],
                    "stroke": video["stroke"],
                    "view": video["view"],
                    "condition": video["condition"],
                    "player_id": video["player_id"],
                    "rep": video["rep"],
                    "feature": feature_name,
                    "required_keypoints": "|".join(required_keypoints),
                    "available_frames": available_frames,
                    "frames_processed": frames_processed,
                    "availability_rate": availability_rate,
                }
            )

    return pd.DataFrame(feature_rows)


def summarize_by_group(df, value_col, group_cols):
    summary = (
        df.groupby(group_cols)[value_col]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )

    summary = summary.rename(
        columns={
            "mean": f"{value_col}_mean",
            "std": f"{value_col}_std",
            "min": f"{value_col}_min",
            "max": f"{value_col}_max",
            "count": "n",
        }
    )

    return summary


def compute_overall_reliability_by_video(keypoint_rates):
    important = keypoint_rates[keypoint_rates["keypoint"].isin(IMPORTANT_KEYPOINTS)]

    overall = (
        important.groupby(
            [
                "video_id",
                "filename",
                "stroke",
                "view",
                "condition",
                "player_id",
                "rep",
            ],
            as_index=False,
        )["detection_rate"]
        .mean()
        .rename(columns={"detection_rate": "important_keypoint_detection_mean"})
    )

    return overall


def compute_clean_vs_fast_summary(overall_by_video):
    grouped = (
        overall_by_video.groupby(["stroke", "view", "condition"])[
            "important_keypoint_detection_mean"
        ]
        .mean()
        .reset_index()
    )

    pivot = grouped.pivot_table(
        index=["stroke", "view"],
        columns="condition",
        values="important_keypoint_detection_mean",
    ).reset_index()

    if "clean" not in pivot.columns:
        pivot["clean"] = np.nan

    if "fast" not in pivot.columns:
        pivot["fast"] = np.nan

    pivot["fast_minus_clean"] = pivot["fast"] - pivot["clean"]

    return pivot


def make_simple_bar_chart(
    df,
    x_col,
    y_col,
    title,
    xlabel,
    ylabel,
    output_path,
    rotate_xticks=False,
    xtick_fontsize=10,
):
    if df.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.bar(df[x_col], df[y_col])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(0, 1.05)

    if rotate_xticks:
        plt.xticks(rotation=30, ha="right", fontsize=xtick_fontsize)
    else:
        plt.xticks(fontsize=xtick_fontsize)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def make_figures(overall_by_video, region_by_video, feature_by_video):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    clean_overall = overall_by_video[overall_by_video["condition"] == "clean"]

    by_stroke = (
        clean_overall.groupby("stroke")["important_keypoint_detection_mean"]
        .mean()
        .reset_index()
    )

    make_simple_bar_chart(
        by_stroke,
        x_col="stroke",
        y_col="important_keypoint_detection_mean",
        title="MediaPipe Detection Reliability by Stroke (Clean Clips)",
        xlabel="Stroke",
        ylabel="Mean Detection Rate",
        output_path=FIGURE_DIR / "mediapipe_detection_by_stroke_clean.png",
    )

    by_view = (
        clean_overall.groupby("view")["important_keypoint_detection_mean"]
        .mean()
        .reset_index()
    )

    make_simple_bar_chart(
        by_view,
        x_col="view",
        y_col="important_keypoint_detection_mean",
        title="MediaPipe Detection Reliability by View (Clean Clips)",
        xlabel="Camera View",
        ylabel="Mean Detection Rate",
        output_path=FIGURE_DIR / "mediapipe_detection_by_view_clean.png",
    )

    by_condition = (
        overall_by_video.groupby("condition")["important_keypoint_detection_mean"]
        .mean()
        .reset_index()
    )

    make_simple_bar_chart(
        by_condition,
        x_col="condition",
        y_col="important_keypoint_detection_mean",
        title="MediaPipe Detection Reliability: Clean vs Fast",
        xlabel="Condition",
        ylabel="Mean Detection Rate",
        output_path=FIGURE_DIR / "mediapipe_detection_clean_vs_fast.png",
    )

    clean_region = region_by_video[region_by_video["condition"] == "clean"]

    by_region = (
        clean_region.groupby("region")["detection_rate"]
        .mean()
        .reset_index()
        .sort_values("detection_rate", ascending=False)
    )

    make_simple_bar_chart(
        by_region,
        x_col="region",
        y_col="detection_rate",
        title="MediaPipe Reliability by Body Region (Clean Clips)",
        xlabel="Body Region",
        ylabel="Mean Detection Rate",
        output_path=FIGURE_DIR / "mediapipe_detection_by_body_region_clean.png",
    )

    clean_features = feature_by_video[feature_by_video["condition"] == "clean"]

    by_feature = (
        clean_features.groupby("feature")["availability_rate"]
        .mean()
        .reset_index()
        .sort_values("availability_rate", ascending=False)
    )

    make_simple_bar_chart(
        by_feature,
        x_col="feature",
        y_col="availability_rate",
        title="MediaPipe Feature Availability (Clean Clips)",
        xlabel="Feature",
        ylabel="Availability Rate",
        output_path=FIGURE_DIR / "mediapipe_feature_availability_clean.png",
        rotate_xticks=True,
        xtick_fontsize=8,
    )


def print_console_summary(overall_by_video, region_by_video, feature_by_video, threshold):
    print("\nAnalysis summary")
    print("----------------")
    print(f"Confidence threshold: {threshold}")

    print("\nOverall mean detection by condition:")
    condition_summary = (
        overall_by_video.groupby("condition")["important_keypoint_detection_mean"]
        .mean()
        .reset_index()
    )
    print(condition_summary.to_string(index=False))

    print("\nOverall mean detection by stroke, clean clips:")
    clean = overall_by_video[overall_by_video["condition"] == "clean"]
    stroke_summary = (
        clean.groupby("stroke")["important_keypoint_detection_mean"]
        .mean()
        .reset_index()
    )
    print(stroke_summary.to_string(index=False))

    print("\nOverall mean detection by view, clean clips:")
    view_summary = (
        clean.groupby("view")["important_keypoint_detection_mean"]
        .mean()
        .reset_index()
    )
    print(view_summary.to_string(index=False))

    print("\nBody region reliability, clean clips:")
    clean_region = region_by_video[region_by_video["condition"] == "clean"]
    region_summary = (
        clean_region.groupby("region")["detection_rate"]
        .mean()
        .reset_index()
        .sort_values("detection_rate", ascending=False)
    )
    print(region_summary.to_string(index=False))

    print("\nFeature availability, clean clips:")
    clean_features = feature_by_video[feature_by_video["condition"] == "clean"]
    feature_summary = (
        clean_features.groupby("feature")["availability_rate"]
        .mean()
        .reset_index()
        .sort_values("availability_rate", ascending=False)
    )
    print(feature_summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Minimum MediaPipe visibility score required to count a keypoint as detected.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    manifest, keypoints, processing = load_inputs()
    video_metadata = prepare_video_metadata(manifest, processing)

    print("Computing MediaPipe reliability metrics...")

    keypoint_rates = compute_keypoint_detection_rates(
        keypoints=keypoints,
        video_metadata=video_metadata,
        threshold=args.threshold,
    )

    important_keypoint_rates = keypoint_rates[
        keypoint_rates["keypoint"].isin(IMPORTANT_KEYPOINTS)
    ].copy()

    region_by_video = compute_body_region_reliability(keypoint_rates)

    feature_by_video = compute_feature_availability(
        keypoints=keypoints,
        video_metadata=video_metadata,
        threshold=args.threshold,
    )

    overall_by_video = compute_overall_reliability_by_video(keypoint_rates)

    keypoint_summary = summarize_by_group(
        important_keypoint_rates,
        value_col="detection_rate",
        group_cols=["condition", "stroke", "view", "keypoint"],
    )

    region_summary = summarize_by_group(
        region_by_video,
        value_col="detection_rate",
        group_cols=["condition", "stroke", "view", "region"],
    )

    feature_summary = summarize_by_group(
        feature_by_video,
        value_col="availability_rate",
        group_cols=["condition", "stroke", "view", "feature"],
    )

    overall_summary = summarize_by_group(
        overall_by_video,
        value_col="important_keypoint_detection_mean",
        group_cols=["condition", "stroke", "view"],
    )

    clean_vs_fast = compute_clean_vs_fast_summary(overall_by_video)

    keypoint_rates.to_csv(OUTPUT_DIR / "keypoint_detection_rates_by_video.csv", index=False)
    important_keypoint_rates.to_csv(
        OUTPUT_DIR / "important_keypoint_detection_rates_by_video.csv",
        index=False,
    )
    keypoint_summary.to_csv(
        OUTPUT_DIR / "important_keypoint_detection_summary.csv",
        index=False,
    )

    region_by_video.to_csv(OUTPUT_DIR / "body_region_reliability_by_video.csv", index=False)
    region_summary.to_csv(OUTPUT_DIR / "body_region_reliability_summary.csv", index=False)

    feature_by_video.to_csv(OUTPUT_DIR / "feature_availability_by_video.csv", index=False)
    feature_summary.to_csv(OUTPUT_DIR / "feature_availability_summary.csv", index=False)

    overall_by_video.to_csv(OUTPUT_DIR / "overall_reliability_by_video.csv", index=False)
    overall_summary.to_csv(OUTPUT_DIR / "overall_reliability_summary.csv", index=False)

    clean_vs_fast.to_csv(OUTPUT_DIR / "clean_vs_fast_summary.csv", index=False)

    make_figures(
        overall_by_video=overall_by_video,
        region_by_video=region_by_video,
        feature_by_video=feature_by_video,
    )

    print_console_summary(
        overall_by_video=overall_by_video,
        region_by_video=region_by_video,
        feature_by_video=feature_by_video,
        threshold=args.threshold,
    )

    print("\nSaved analysis files to:")
    print(f"  {OUTPUT_DIR}")

    print("\nSaved figures to:")
    print(f"  {FIGURE_DIR}")


if __name__ == "__main__":
    main()
