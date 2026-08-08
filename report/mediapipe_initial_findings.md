# MediaPipe Initial Findings

The metric used is the average detection rate across important body keypoints:

- shoulders
- elbows
- wrists
- hips
- knees
- ankles

Values closer to 1 have higher confidence.

## Dataset Overview

- Strokes: forehand, backhand, serve
- Views: side, back, diagonal
- 3 clean and 1 fast clip for each stroke-view combination (36 clips total)

## MediaPipe Reliability Results

MediaPipe outputs a visibility value for each body landmark in each video frame. A keypoint was counted as **detected** in a frame if its value was at least 0.5. The detection rate was calculated using `detected frames / total frames processed`.

### Overall Reliability

| Condition | Mean Detection | Std. Dev. | Min | Max | Clips |
|---|---|---|---|---|---|
| Clean | 0.892 | 0.054 | 0.822 | 0.974 | 27 |
| Fast | 0.895 | 0.056 | 0.829 | 0.966 | 9 |

For both clean and fast clips, the detections rates were above 0.82. That means for most clips the model could reliably detect body landmarks, making it usable for analysis. The fast clips performed about the same as the clean clips, suggesting that moderate increases in stroke speed did not significantly reduce tracking reliability in this dataset.

### Reliability by Stroke (Clean Only)

| Stroke | Mean Detection | Std. Dev. | Min | Max |
|---|---|---|---|---|
| Serve | 0.917 | 0.046 | 0.851 | 0.974 |
| Backhand | 0.884 | 0.058 | 0.822 | 0.972 |
| Forehand | 0.875 | 0.052 | 0.831 | 0.974 |

Serves had the highest average detection rate, while forehands had the lowest, though all three strokes performed well overall.

### Reliability by Camera View

| Camera View | Mean Detection | Std. Dev. | Min | Max |
|---|---|---|---|---|
| Diagonal | 0.938 | 0.026 | 0.910 | 0.974 |
| Side | 0.888 | 0.062 | 0.831 | 0.974 |
| Back | 0.849 | 0.017 | 0.822 | 0.877 |

Diagonal view had the highest average detection rate, followed by side view and back view. Even the lowest average, back view, remained above 0.84, so all three camera views are usable under controlled recording conditions.

### Reliability by Stroke/View Combination

| Stroke | Camera View | Mean Detection | Std. Dev. | Min | Max |
|---|---|---|---|---|---|
| Serve | Side | 0.970 | 0.004 | 0.966 | 0.974 |
| Backhand | Diagonal | 0.958 | 0.018 | 0.938 | 0.972 |
| Forehand | Diagonal | 0.941 | 0.030 | 0.915 | 0.974 |
| Serve | Diagonal | 0.916 | 0.006 | 0.910 | 0.922 |
| Serve | Back | 0.865 | 0.013 | 0.851 | 0.877 |
| Backhand | Side | 0.853 | 0.020 | 0.832 | 0.871 |
| Forehand | Back | 0.843 | 0.011 | 0.831 | 0.852 |
| Forehand | Side | 0.842 | 0.011 | 0.831 | 0.852 |
| Backhand | Back | 0.840 | 0.017 | 0.822 | 0.856 |

The strongest combinations were serve-side, backhand-diagonal, and forehand-diagonal. The lowest-scoring combinations were still above 0.82, showing that MediaPipe produced usable full-body pose estimates across all stroke-view combinations.

---

### Body Region Reliability

| Body Region | Mean Detection | Std. Dev. | Min | Max |
|---|---|---|---|---|
| Shoulders | 1.000 | 0.000 | 1.000 | 1.000 |
| Hips | 1.000 | 0.000 | 1.000 | 1.000 |
| Ankles | 0.992 | 0.018 | 0.920 | 1.000 |
| Knees | 0.948 | 0.072 | 0.774 | 1.000 |
| Elbows | 0.766 | 0.099 | 0.608 | 0.932 |
| Wrists | 0.645 | 0.249 | 0.195 | 0.966 |

The detection rate for each body region was calculated by averaging the left and right landmarks (for example, the shoulders region includes the left and right shoulders). Shoulders and hips were detected perfectly in the clean clips, while ankles and knees were also highly reliable. Elbows and wrists had lower confidence-based detection rates, likely because arm landmarks are more affected by fast movement, body overlap, and camera angle.

### Feature Availability

| Feature | Mean Availability | Std. Dev. | Min | Max |
|---|---|---|---|---|
| Hip Line Angle | 1.000 | 0.000 | 1.000 | 1.000 |
| Shoulder-Hip Separation Proxy | 1.000 | 0.000 | 1.000 | 1.000 |
| Shoulder Line Angle | 1.000 | 0.000 | 1.000 | 1.000 |
| Right Knee Angle | 0.959 | 0.095 | 0.571 | 1.000 |
| Left Knee Angle | 0.938 | 0.131 | 0.547 | 1.000 |
| Right Elbow Angle | 0.637 | 0.326 | 0.145 | 1.000 |
| Left Elbow Angle | 0.576 | 0.308 | 0.151 | 1.000 |

Analysis often requires looking at relationships between different body landmarks. A feature was counted as available only when all required keypoints had confidence scores above the threshold (for example, the right elbow angle requires the right shoulder, right elbow, and right wrist to be detected at once). Shoulder and hip-based features were available in every clean clip frame, knee-angle features were mostly available, and elbow-angle features were less reliable because they depend on wrist and elbow tracking.

---

### Clean vs. Fast Comparison by Stroke/View Combination

This table compares each fast clip to the clean average for the same stroke and camera view. Negative values mean the fast clip had lower reliability than the clean average; positive values mean it had higher reliability. The differences were small overall, suggesting that fast motion did not create major tracking failures in this controlled dataset.

| Stroke | Camera View | Clean Avg. | Fast Clip | Difference |
|---|---|---|---|---|
| Forehand | Back | 0.843 | 0.829 | -0.014 |
| Forehand | Side | 0.842 | 0.837 | -0.005 |
| Backhand | Diagonal | 0.958 | 0.953 | -0.004 |
| Serve | Side | 0.970 | 0.966 | -0.004 |
| Backhand | Back | 0.840 | 0.838 | -0.002 |
| Backhand | Side | 0.853 | 0.857 | 0.004 |
| Serve | Diagonal | 0.916 | 0.924 | 0.008 |
| Forehand | Diagonal | 0.941 | 0.951 | 0.010 |
| Serve | Back | 0.865 | 0.900 | 0.035 |

## Conclusion

- MediaPipe performed well on the controlled tennis video dataset. The overall detection scores suggest that MediaPipe could reliably detect the player’s body in most frames, and there were no major full-body tracking failures.
- Fast movement did not significantly reduce detection reliability in this dataset. The fast clips performed about the same as the clean clips, suggesting that moderate increases in stroke speed did not cause major pose-tracking issues when the player remained visible and the camera was stable.
- Camera angle affected reliability, but all tested angles were usable. Diagonal-view clips had the highest mean detection score, followed by side view and back view.
- Stroke type did not create major reliability differences. Serves had the highest average detection score among clean clips, followed by backhands and forehands. All three strokes were tracked well overall.
- MediaPipe was strongest on central and lower-body landmarks. Shoulders and hips were detected perfectly in the clean clips, while ankles and knees also had high detection rates.
- Wrists and elbows were the least reliable body regions. Wrist detection had the lowest mean detection score, and elbow detection was also lower. This likely reflects tennis-specific movement patterns where the arms and wrists can be blocked by the torso, overlap with other limbs, or move quickly through the frame.
- Feature availability followed the same pattern as body-region reliability. Shoulder and hip-based features were available in every clean clip frame, knee-angle features were mostly available, and elbow-angle features were less consistently available.
- The most reliable first-stage features are shoulder line angle, hip line angle, shoulder-hip separation proxy, and knee angle. Arm-based features, such as elbow angle and wrist path, should be treated more cautiously. They may still be useful, but only when the required shoulder, elbow, and wrist landmarks remain confidently detected.
- Manual inspection supported the automatic results. The skeleton overlays looked visually strong across clips, with no major failures across stroke type, camera angle, or speed condition.

### Limitations

- This was a controlled case study using one primary player. The results show that MediaPipe can work well under clear recording conditions, but they do not prove that the model will perform equally well for all players, environments, lighting conditions, or camera setups.

## Possible Applications to the Tennis Stroke Analyzer

- Add a recording-quality check before generating feedback. Since MediaPipe performed well when the full body was visible and the camera was stable, the analyzer should first check whether the player is fully in frame and whether enough important keypoints are detected.
- Use recording-angle recommendations. Since diagonal and side views performed strongly, the analyzer could recommend these views for form analysis. Back view may still be useful, but it may be better suited for general body-position or footwork feedback rather than detailed arm-position feedback.

- Use confidence thresholds before calculating tennis features. For example, elbow angle should only be calculated when the shoulder, elbow, and wrist all have confidence scores above the chosen threshold.
- Give feedback based on feature reliability. Instead of always returning every possible analysis, the system could decide which feedback is trustworthy for each clip. For example, one clip might receive shoulder-turn and knee-bend feedback, while another might skip wrist-path feedback because wrist tracking was unstable.

- Add warnings when occlusion affects tracking. If the wrist or elbow disappears behind the body during contact or follow-through, the analyzer should lower confidence in arm-based feedback or explain that the clip is not ideal for arm analysis.
