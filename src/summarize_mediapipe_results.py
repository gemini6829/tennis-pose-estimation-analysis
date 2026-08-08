from pathlib import Path

import pandas as pd


ANALYSIS_DIR = Path("results/mediapipe/analysis")
LOGS_DIR = Path("results/mediapipe/logs")
REPORT_DIR = Path("report")

OVERALL_BY_VIDEO_PATH = ANALYSIS_DIR / "overall_reliability_by_video.csv"
BODY_REGION_BY_VIDEO_PATH = ANALYSIS_DIR / "body_region_reliability_by_video.csv"
FEATURE_BY_VIDEO_PATH = ANALYSIS_DIR / "feature_availability_by_video.csv"
CLEAN_VS_FAST_PATH = ANALYSIS_DIR / "clean_vs_fast_summary.csv"
PROCESSING_SUMMARY_PATH = LOGS_DIR / "processing_summary.csv"

OUTPUT_PATH = REPORT_DIR / "mediapipe_initial_findings.md"


def fmt(value):
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def markdown_table(df, columns=None, rename=None, max_rows=None):
    if columns is not None:
        df = df[columns].copy()
    else:
        df = df.copy()

    if max_rows is not None:
        df = df.head(max_rows)

    if rename:
        df = df.rename(columns=rename)

    headers = list(df.columns)

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) for col in headers) + " |")

    return "\n".join(lines)


def load_required_csv(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}\n"
            "Run the MediaPipe analysis step before this summary step."
        )
    return pd.read_csv(path, dtype={"rep": str})


def add_section(lines, title):
    lines.append("")
    lines.append(f"## {title}")
    lines.append("")


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    overall = load_required_csv(OVERALL_BY_VIDEO_PATH)
    body_regions = load_required_csv(BODY_REGION_BY_VIDEO_PATH)
    features = load_required_csv(FEATURE_BY_VIDEO_PATH)
    clean_vs_fast = load_required_csv(CLEAN_VS_FAST_PATH)
    processing = load_required_csv(PROCESSING_SUMMARY_PATH)

    overall = overall.merge(
        processing[["video_id", "overlay_path"]],
        on="video_id",
        how="left",
    )

    lines = []

    lines.append("# MediaPipe Initial Findings")
    lines.append("")
    lines.append("This file summarizes the first MediaPipe reliability analysis for the tennis pose estimation dataset.")
    lines.append("")
    lines.append("The metric used here is the average detection rate across important body keypoints:")
    lines.append("")
    lines.append("- shoulders")
    lines.append("- elbows")
    lines.append("- wrists")
    lines.append("- hips")
    lines.append("- knees")
    lines.append("- ankles")
    lines.append("")
    lines.append("A value closer to 1.000 means the keypoints were detected confidently in more frames.")

    # Dataset overview
    add_section(lines, "Dataset Overview")

    dataset_overview = (
        overall.groupby(["condition", "stroke", "view"])
        .size()
        .reset_index(name="clip_count")
        .sort_values(["condition", "stroke", "view"])
    )

    lines.append(markdown_table(dataset_overview))

    # Overall by condition
    add_section(lines, "Overall Reliability by Condition")

    condition_summary = (
        overall.groupby("condition")["important_keypoint_detection_mean"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_detection",
                "std": "std_detection",
                "min": "min_detection",
                "max": "max_detection",
                "count": "clips",
            }
        )
        .sort_values("mean_detection", ascending=False)
    )

    lines.append(markdown_table(condition_summary))

    # Clean by stroke
    clean = overall[overall["condition"] == "clean"].copy()

    add_section(lines, "Clean Clips: Reliability by Stroke")

    clean_by_stroke = (
        clean.groupby("stroke")["important_keypoint_detection_mean"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_detection",
                "std": "std_detection",
                "min": "min_detection",
                "max": "max_detection",
                "count": "clips",
            }
        )
        .sort_values("mean_detection", ascending=False)
    )

    lines.append(markdown_table(clean_by_stroke))

    # Clean by view
    add_section(lines, "Clean Clips: Reliability by Camera View")

    clean_by_view = (
        clean.groupby("view")["important_keypoint_detection_mean"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_detection",
                "std": "std_detection",
                "min": "min_detection",
                "max": "max_detection",
                "count": "clips",
            }
        )
        .sort_values("mean_detection", ascending=False)
    )

    lines.append(markdown_table(clean_by_view))

    # Clean stroke/view combos
    add_section(lines, "Clean Clips: Stroke/View Combinations")

    clean_combo = (
        clean.groupby(["stroke", "view"])["important_keypoint_detection_mean"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_detection",
                "std": "std_detection",
                "min": "min_detection",
                "max": "max_detection",
                "count": "clips",
            }
        )
        .sort_values("mean_detection", ascending=False)
    )

    lines.append(markdown_table(clean_combo))

    # Clean vs fast
    add_section(lines, "Clean vs Fast Comparison")

    clean_vs_fast_display = clean_vs_fast.copy()

    for col in ["clean", "fast", "fast_minus_clean"]:
        if col not in clean_vs_fast_display.columns:
            clean_vs_fast_display[col] = pd.NA

    clean_vs_fast_display = clean_vs_fast_display.sort_values(
        "fast_minus_clean",
        ascending=True,
        na_position="last",
    )

    lines.append(
        "Negative values in `fast_minus_clean` mean the fast clip had lower reliability than the clean average."
    )
    lines.append("")
    lines.append(markdown_table(clean_vs_fast_display))

    # Body regions
    add_section(lines, "Clean Clips: Body Region Reliability")

    clean_regions = body_regions[body_regions["condition"] == "clean"].copy()

    region_summary = (
        clean_regions.groupby("region")["detection_rate"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_detection",
                "std": "std_detection",
                "min": "min_detection",
                "max": "max_detection",
                "count": "measurements",
            }
        )
        .sort_values("mean_detection", ascending=False)
    )

    lines.append(markdown_table(region_summary))

    # Feature availability
    add_section(lines, "Clean Clips: Feature Availability")

    clean_features = features[features["condition"] == "clean"].copy()

    feature_summary = (
        clean_features.groupby("feature")["availability_rate"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_availability",
                "std": "std_availability",
                "min": "min_availability",
                "max": "max_availability",
                "count": "measurements",
            }
        )
        .sort_values("mean_availability", ascending=False)
    )

    lines.append(markdown_table(feature_summary))

    # Best clips
    add_section(lines, "Strongest Individual Clips")

    strongest = overall.sort_values(
        "important_keypoint_detection_mean",
        ascending=False,
    ).head(8)

    lines.append(markdown_table(
        strongest,
        columns=[
            "video_id",
            "filename",
            "stroke",
            "view",
            "condition",
            "important_keypoint_detection_mean",
        ],
        rename={"important_keypoint_detection_mean": "mean_detection"},
    ))

    # Weakest clips
    add_section(lines, "Weakest Individual Clips")

    weakest = overall.sort_values(
        "important_keypoint_detection_mean",
        ascending=True,
    ).head(8)

    lines.append(markdown_table(
        weakest,
        columns=[
            "video_id",
            "filename",
            "stroke",
            "view",
            "condition",
            "important_keypoint_detection_mean",
        ],
        rename={"important_keypoint_detection_mean": "mean_detection"},
    ))

    # Manual inspection list
    add_section(lines, "Overlay Videos to Manually Inspect")

    lines.append("Inspect these videos to check whether the numeric reliability scores match the visual overlay quality.")
    lines.append("Prioritize the weakest clips, a few strongest clips, and representative fast clips.")
    lines.append("")

    inspection_candidates = pd.concat(
        [
            weakest.head(5),
            strongest.head(3),
            overall[overall["condition"] == "fast"]
            .sort_values("important_keypoint_detection_mean", ascending=True)
            .head(5),
        ],
        ignore_index=True,
    )

    inspection_candidates = inspection_candidates.drop_duplicates("video_id")

    lines.append(markdown_table(
        inspection_candidates,
        columns=[
            "video_id",
            "filename",
            "stroke",
            "view",
            "condition",
            "important_keypoint_detection_mean",
            "overlay_path",
        ],
        rename={"important_keypoint_detection_mean": "mean_detection"},
    ))

    # Initial interpretation template
    add_section(lines, "Notes for Interpretation")

    lines.append("Fill this section in after watching the selected overlay videos.")
    lines.append("")
    lines.append("Questions to answer:")
    lines.append("")
    lines.append("1. Which stroke appears most reliable?")
    lines.append("2. Which camera view appears most reliable?")
    lines.append("3. Does fast motion visibly reduce tracking quality?")
    lines.append("4. Are wrists less stable than shoulders, hips, or knees?")
    lines.append("5. Which features seem trustworthy enough for a tennis stroke analyzer?")
    lines.append("6. Which features should be avoided when confidence is low?")
    lines.append("")
    lines.append("Possible application to the tennis stroke analyzer:")
    lines.append("")
    lines.append("- Recommend the most reliable recording angle.")
    lines.append("- Add a confidence check before giving feedback.")
    lines.append("- Avoid wrist-path feedback when wrist tracking is unstable.")
    lines.append("- Use lower-body feedback only when ankles and knees remain visible.")
    lines.append("- Treat serve analysis more cautiously if fast-motion clips are less reliable.")

    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved MediaPipe findings summary to: {OUTPUT_PATH}")

    print("\nTop clean stroke/view combinations:")
    print(clean_combo.head(9).to_string(index=False))

    print("\nClean vs fast comparison:")
    print(clean_vs_fast_display.to_string(index=False))

    print("\nWeakest clips to inspect:")
    print(
        weakest[
            [
                "video_id",
                "filename",
                "stroke",
                "view",
                "condition",
                "important_keypoint_detection_mean",
                "overlay_path",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
