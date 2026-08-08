# Tennis Pose Estimation Analysis

This project evaluates whether pose estimation models can reliably support tennis stroke analysis in real-world practice videos.

## Main Question

Can general-purpose pose estimation models extract tennis-relevant body mechanics from forehand, backhand, and serve videos reliably enough to support automated stroke feedback?

## Project Steps

- Run pose estimation models on tennis practice clips
- Generate skeleton overlay videos
- Export pose keypoints and confidence scores
- Compute tennis-specific features such as elbow angle, knee bend, and shoulder-hip separation
- Compare reliability across strokes and camera angles
- Identify failure cases such as motion blur, occlusion, and partial body visibility
- Apply findings to improve a tennis stroke analyzer

## Models

Initial model:
- MediaPipe Pose

Planned comparison model:
- MoveNet

Possible future model:
- YOLO Pose

## Dataset

The dataset consists of short tennis practice clips recorded from different camera angles.
Initial dataset: 36 self-recorded tennis clips. Includes forehands, backhands, and serves, from side, back, and diagonal views. Both slow and fast swing speeds.

## Initial MediaPipe Findings

| | |
|:---:|:---:|
| Clean clips | Mean detection rate: 0.892 |
| Fast clips | Mean detection rate: 0.895 |
| Best stroke, clean clips | Serve: 0.917 |
| Best view, clean clips | Diagonal: 0.938 |
| Most reliable regions | Shoulders, hips, ankles, knees |
| Least reliable regions | Wrists, elbows |

Fast-motion clips performed about the same as clean clips, suggesting that moderate increases in stroke speed did not significantly reduce MediaPipe reliability under controlled recording conditions.

The strongest body regions were shoulders and hips, both with a mean detection rate of 1.000. Wrists had the lowest detection rate at 0.645, which means arm-based feedback should use confidence checks before being shown to users.

### Main Takeaway

MediaPipe appears suitable for extracting basic tennis movement features from controlled practice videos, especially torso rotation and lower body features. However, wrist and elbow-based feedback should be treated more cautiously because those landmarks were less reliable by confidence score.

For the full MediaPipe analysis, see:

[`report/mediapipe_initial_findings.md`](report/mediapipe_initial_findings.md)
