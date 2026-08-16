from pathlib import Path
import shutil


SOURCE_FIGURE_DIR = Path("results/mediapipe/figures")
SOURCE_FEATURE_DIR = Path("results/mediapipe/figures/feature_plots")
REPORT_FIGURE_DIR = Path("report/figures")


def copy_exact(source_path: Path, output_name: str):
    if not source_path.exists():
        print(f"Missing: {source_path}")
        return False

    destination = REPORT_FIGURE_DIR / output_name
    shutil.copy2(source_path, destination)
    print(f"Copied: {destination}")
    return True


def copy_first_match(pattern: str, output_name: str):
    matches = sorted(SOURCE_FEATURE_DIR.glob(pattern))

    if not matches:
        print(f"No match found for pattern: {pattern}")
        return False

    source_path = matches[0]
    destination = REPORT_FIGURE_DIR / output_name
    shutil.copy2(source_path, destination)
    print(f"Copied: {source_path} -> {destination}")
    return True


def main():
    REPORT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("Copying selected MediaPipe reliability figures...\n")

    copy_exact(
        SOURCE_FIGURE_DIR / "mediapipe_detection_by_body_region_clean.png",
        "mediapipe_body_region_reliability.png",
    )

    copy_exact(
        SOURCE_FIGURE_DIR / "mediapipe_feature_availability_clean.png",
        "mediapipe_feature_availability.png",
    )

    copy_exact(
        SOURCE_FIGURE_DIR / "mediapipe_detection_by_view_clean.png",
        "mediapipe_detection_by_view.png",
    )

    copy_exact(
        SOURCE_FIGURE_DIR / "mediapipe_detection_clean_vs_fast.png",
        "mediapipe_clean_vs_fast.png",
    )

    print("\nCopying selected tennis feature plots...\n")

    copy_first_match(
        "TPEA_0001_*_shoulder_hip_separation_proxy.png",
        "feature_forehand_side_shoulder_hip_separation.png",
    )

    copy_first_match(
        "TPEA_0001_*_right_knee_angle.png",
        "feature_forehand_side_right_knee_angle.png",
    )

    copy_first_match(
        "TPEA_0019_*_shoulder_line_angle.png",
        "feature_serve_side_shoulder_line_angle.png",
    )

    copy_first_match(
        "TPEA_0019_*_right_knee_angle.png",
        "feature_serve_side_right_knee_angle.png",
    )

    print("\nDone. Selected figures are in:")
    print(REPORT_FIGURE_DIR)


if __name__ == "__main__":
    main()
