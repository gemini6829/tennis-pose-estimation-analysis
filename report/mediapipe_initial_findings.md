# MediaPipe Initial Findings

This file summarizes the first MediaPipe reliability analysis for the tennis pose estimation dataset.

The metric used here is the average detection rate across important body keypoints:

- shoulders
- elbows
- wrists
- hips
- knees
- ankles

A value closer to 1.000 means the keypoints were detected confidently in more frames.

## Dataset Overview

| condition | stroke | view | clip_count |
| --- | --- | --- | --- |
| clean | backhand | back | 3 |
| clean | backhand | diag | 3 |
| clean | backhand | side | 3 |
| clean | forehand | back | 3 |
| clean | forehand | diag | 3 |
| clean | forehand | side | 3 |
| clean | serve | back | 3 |
| clean | serve | diag | 3 |
| clean | serve | side | 3 |
| fast | backhand | back | 1 |
| fast | backhand | diag | 1 |
| fast | backhand | side | 1 |
| fast | forehand | back | 1 |
| fast | forehand | diag | 1 |
| fast | forehand | side | 1 |
| fast | serve | back | 1 |
| fast | serve | diag | 1 |
| fast | serve | side | 1 |

## Overall Reliability by Condition

| condition | mean_detection | std_detection | min_detection | max_detection | clips |
| --- | --- | --- | --- | --- | --- |
| fast | 0.895 | 0.056 | 0.829 | 0.966 | 9 |
| clean | 0.892 | 0.054 | 0.822 | 0.974 | 27 |

## Clean Clips: Reliability by Stroke

| stroke | mean_detection | std_detection | min_detection | max_detection | clips |
| --- | --- | --- | --- | --- | --- |
| serve | 0.917 | 0.046 | 0.851 | 0.974 | 9 |
| backhand | 0.884 | 0.058 | 0.822 | 0.972 | 9 |
| forehand | 0.875 | 0.052 | 0.831 | 0.974 | 9 |

## Clean Clips: Reliability by Camera View

| view | mean_detection | std_detection | min_detection | max_detection | clips |
| --- | --- | --- | --- | --- | --- |
| diag | 0.938 | 0.026 | 0.910 | 0.974 | 9 |
| side | 0.888 | 0.062 | 0.831 | 0.974 | 9 |
| back | 0.849 | 0.017 | 0.822 | 0.877 | 9 |

## Clean Clips: Stroke/View Combinations

| stroke | view | mean_detection | std_detection | min_detection | max_detection | clips |
| --- | --- | --- | --- | --- | --- | --- |
| serve | side | 0.970 | 0.004 | 0.966 | 0.974 | 3 |
| backhand | diag | 0.958 | 0.018 | 0.938 | 0.972 | 3 |
| forehand | diag | 0.941 | 0.030 | 0.915 | 0.974 | 3 |
| serve | diag | 0.916 | 0.006 | 0.910 | 0.922 | 3 |
| serve | back | 0.865 | 0.013 | 0.851 | 0.877 | 3 |
| backhand | side | 0.853 | 0.020 | 0.832 | 0.871 | 3 |
| forehand | back | 0.843 | 0.011 | 0.831 | 0.852 | 3 |
| forehand | side | 0.842 | 0.011 | 0.831 | 0.852 | 3 |
| backhand | back | 0.840 | 0.017 | 0.822 | 0.856 | 3 |

## Clean vs Fast Comparison

Negative values in `fast_minus_clean` mean the fast clip had lower reliability than the clean average.

| stroke | view | clean | fast | fast_minus_clean |
| --- | --- | --- | --- | --- |
| forehand | back | 0.843 | 0.829 | -0.014 |
| forehand | side | 0.842 | 0.837 | -0.005 |
| backhand | diag | 0.958 | 0.953 | -0.004 |
| serve | side | 0.970 | 0.966 | -0.004 |
| backhand | back | 0.840 | 0.838 | -0.002 |
| backhand | side | 0.853 | 0.857 | 0.004 |
| serve | diag | 0.916 | 0.924 | 0.008 |
| forehand | diag | 0.941 | 0.951 | 0.010 |
| serve | back | 0.865 | 0.900 | 0.035 |

## Clean Clips: Body Region Reliability

| region | mean_detection | std_detection | min_detection | max_detection | measurements |
| --- | --- | --- | --- | --- | --- |
| shoulders | 1.000 | 0.000 | 1.000 | 1.000 | 27 |
| hips | 1.000 | 0.000 | 1.000 | 1.000 | 27 |
| ankles | 0.992 | 0.018 | 0.920 | 1.000 | 27 |
| knees | 0.948 | 0.072 | 0.774 | 1.000 | 27 |
| elbows | 0.766 | 0.099 | 0.608 | 0.932 | 27 |
| wrists | 0.645 | 0.249 | 0.195 | 0.966 | 27 |

## Clean Clips: Feature Availability

| feature | mean_availability | std_availability | min_availability | max_availability | measurements |
| --- | --- | --- | --- | --- | --- |
| hip_line_angle | 1.000 | 0.000 | 1.000 | 1.000 | 27 |
| shoulder_hip_separation_proxy | 1.000 | 0.000 | 1.000 | 1.000 | 27 |
| shoulder_line_angle | 1.000 | 0.000 | 1.000 | 1.000 | 27 |
| right_knee_angle | 0.959 | 0.095 | 0.571 | 1.000 | 27 |
| left_knee_angle | 0.938 | 0.131 | 0.547 | 1.000 | 27 |
| right_elbow_angle | 0.637 | 0.326 | 0.145 | 1.000 | 27 |
| left_elbow_angle | 0.576 | 0.308 | 0.151 | 1.000 | 27 |

## Strongest Individual Clips

| video_id | filename | stroke | view | condition | mean_detection |
| --- | --- | --- | --- | --- | --- |
| TPEA_0021 | serve_side_clean_p1_03.MOV | serve | side | clean | 0.974 |
| TPEA_0009 | forehand_diag_clean_p1_03.MOV | forehand | diag | clean | 0.974 |
| TPEA_0016 | backhand_diag_clean_p1_01.MOV | backhand | diag | clean | 0.972 |
| TPEA_0019 | serve_side_clean_p1_01.MOV | serve | side | clean | 0.970 |
| TPEA_0034 | serve_side_fast_p1_01.MOV | serve | side | fast | 0.966 |
| TPEA_0020 | serve_side_clean_p1_02.MOV | serve | side | clean | 0.966 |
| TPEA_0017 | backhand_diag_clean_p1_02.MOV | backhand | diag | clean | 0.964 |
| TPEA_0033 | backhand_diag_fast_p1_01.MOV | backhand | diag | fast | 0.953 |

## Weakest Individual Clips

| video_id | filename | stroke | view | condition | mean_detection |
| --- | --- | --- | --- | --- | --- |
| TPEA_0015 | backhand_back_clean_p1_03.MOV | backhand | back | clean | 0.822 |
| TPEA_0029 | forehand_back_fast_p1_01.MOV | forehand | back | fast | 0.829 |
| TPEA_0004 | forehand_back_clean_p1_01.MOV | forehand | back | clean | 0.831 |
| TPEA_0003 | forehand_side_clean_p1_03.MOV | forehand | side | clean | 0.831 |
| TPEA_0012 | backhand_side_clean_p1_03.MOV | backhand | side | clean | 0.832 |
| TPEA_0028 | forehand_side_fast_p1_01.MOV | forehand | side | fast | 0.837 |
| TPEA_0032 | backhand_back_fast_p1_01.MOV | backhand | back | fast | 0.838 |
| TPEA_0014 | backhand_back_clean_p1_02.MOV | backhand | back | clean | 0.842 |

## Overlay Videos to Manually Inspect

Inspect these videos to check whether the numeric reliability scores match the visual overlay quality.
Prioritize the weakest clips, a few strongest clips, and representative fast clips.

| video_id | filename | stroke | view | condition | mean_detection | overlay_path |
| --- | --- | --- | --- | --- | --- | --- |
| TPEA_0015 | backhand_back_clean_p1_03.MOV | backhand | back | clean | 0.822 | results/mediapipe/overlay_videos/TPEA_0015_backhand_back_clean_p1_03_mediapipe_overlay.mp4 |
| TPEA_0029 | forehand_back_fast_p1_01.MOV | forehand | back | fast | 0.829 | results/mediapipe/overlay_videos/TPEA_0029_forehand_back_fast_p1_01_mediapipe_overlay.mp4 |
| TPEA_0004 | forehand_back_clean_p1_01.MOV | forehand | back | clean | 0.831 | results/mediapipe/overlay_videos/TPEA_0004_forehand_back_clean_p1_01_mediapipe_overlay.mp4 |
| TPEA_0003 | forehand_side_clean_p1_03.MOV | forehand | side | clean | 0.831 | results/mediapipe/overlay_videos/TPEA_0003_forehand_side_clean_p1_03_mediapipe_overlay.mp4 |
| TPEA_0012 | backhand_side_clean_p1_03.MOV | backhand | side | clean | 0.832 | results/mediapipe/overlay_videos/TPEA_0012_backhand_side_clean_p1_03_mediapipe_overlay.mp4 |
| TPEA_0021 | serve_side_clean_p1_03.MOV | serve | side | clean | 0.974 | results/mediapipe/overlay_videos/TPEA_0021_serve_side_clean_p1_03_mediapipe_overlay.mp4 |
| TPEA_0009 | forehand_diag_clean_p1_03.MOV | forehand | diag | clean | 0.974 | results/mediapipe/overlay_videos/TPEA_0009_forehand_diag_clean_p1_03_mediapipe_overlay.mp4 |
| TPEA_0016 | backhand_diag_clean_p1_01.MOV | backhand | diag | clean | 0.972 | results/mediapipe/overlay_videos/TPEA_0016_backhand_diag_clean_p1_01_mediapipe_overlay.mp4 |
| TPEA_0028 | forehand_side_fast_p1_01.MOV | forehand | side | fast | 0.837 | results/mediapipe/overlay_videos/TPEA_0028_forehand_side_fast_p1_01_mediapipe_overlay.mp4 |
| TPEA_0032 | backhand_back_fast_p1_01.MOV | backhand | back | fast | 0.838 | results/mediapipe/overlay_videos/TPEA_0032_backhand_back_fast_p1_01_mediapipe_overlay.mp4 |
| TPEA_0031 | backhand_side_fast_p1_01.MOV | backhand | side | fast | 0.857 | results/mediapipe/overlay_videos/TPEA_0031_backhand_side_fast_p1_01_mediapipe_overlay.mp4 |
| TPEA_0035 | serve_back_fast_p1_01.MOV | serve | back | fast | 0.900 | results/mediapipe/overlay_videos/TPEA_0035_serve_back_fast_p1_01_mediapipe_overlay.mp4 |

## Notes for Interpretation

Fill this section in after watching the selected overlay videos.

Questions to answer:

1. Which stroke appears most reliable?
2. Which camera view appears most reliable?
3. Does fast motion visibly reduce tracking quality?
4. Are wrists less stable than shoulders, hips, or knees?
5. Which features seem trustworthy enough for a tennis stroke analyzer?
6. Which features should be avoided when confidence is low?

Possible application to the tennis stroke analyzer:

- Recommend the most reliable recording angle.
- Add a confidence check before giving feedback.
- Avoid wrist-path feedback when wrist tracking is unstable.
- Use lower-body feedback only when ankles and knees remain visible.
- Treat serve analysis more cautiously if fast-motion clips are less reliable.