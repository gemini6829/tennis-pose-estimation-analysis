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

The initial MediaPipe analysis suggests that pose estimation is a promising foundation for tennis stroke analysis under controlled recording conditions. Across the 36 self-recorded clips, MediaPipe produced consistently strong full-body pose tracking. The overall detection rates were high for both clean clips (`0.892`) and fast clips (`0.895`), indicating that moderate increases in stroke speed did not significantly reduce tracking reliability in this dataset.

The results also show that recording angle and body region matter. Diagonal-view clips had the highest average detection rate among clean clips (`0.938`), followed by side view (`0.888`) and back view (`0.849`). However, all three angles still produced usable results, which suggests that a tennis stroke analyzer may not need to rely on one perfect camera angle as long as the player remains fully visible and the video is clearly framed.

The body-region analysis revealed the most important limitation. MediaPipe tracked central and lower-body landmarks extremely well: shoulders and hips were detected with perfect reliability in the clean clips, ankles were highly reliable, and knees were also strong. In contrast, elbows and wrists were less reliable, with wrists showing the lowest mean detection rate (`0.645`). This difference is likely related to tennis-specific movement and camera angle. Depending on the view and stroke phase, the hitting arm, elbow, or wrist can be partially obstructed by the torso, the non-hitting arm, or the follow-through position. Back-view and side-view clips can especially create moments where the wrist is hidden behind the body or overlaps with another limb.

This type of obstruction can affect the confidence score because the model has less visual evidence for the exact landmark location. Even if the skeleton overlay still appears generally correct, the wrist or elbow may receive a lower confidence score or become less stable across frames. This is important for tennis analysis because arm-based features, such as wrist path and elbow angle, depend on accurately locating the shoulder, elbow, and wrist at the same time.

As a result, not all pose-based tennis features should be treated equally. Torso rotation, shoulder alignment, hip alignment, and knee-bend features appear to be more reliable first-stage features, while wrist-path and elbow-angle feedback should be used more cautiously. *The analyzer should only generate detailed arm feedback when the relevant wrist and elbow landmarks remain confidently detected across the important parts of the stroke.*

Manual overlay inspection supported the overall reliability of the model. The skeletons appeared visually strong across the clips, with no major failures across stroke type, angle, or speed condition. However, the lower confidence scores for wrists and elbows show why visual inspection alone is not enough. A pose overlay can look generally correct while still being less reliable for precise arm-position measurements.

The main takeaway is that MediaPipe can support a tennis stroke analyzer, but the analyzer should be confidence-aware. *Instead of assuming that every pose measurement is trustworthy, the system should check whether the required keypoints are reliable before generating feedback.* Features based on shoulders, hips, and knees can be prioritized first, while arm-based feedback should only be shown when wrist and elbow confidence remain high enough.

Because this dataset uses one primary player and controlled recording conditions, these findings should be interpreted as a case study rather than a universal benchmark. Future analysis should test additional players, more difficult recording conditions, and other pose estimation models such as MoveNet or YOLO Pose. Still, these initial results provide a strong technical basis for building a more reliable tennis stroke analyzer: one that evaluates the quality of its own pose data before giving coaching feedback.

Possible application to the tennis stroke analyzer:

- Recommend the most reliable recording angle.
- Add a confidence check before giving feedback.
- Avoid wrist-path feedback when wrist tracking is unstable.
- Use lower-body feedback only when ankles and knees remain visible.
- Treat serve analysis more cautiously if fast-motion clips are less reliable.
