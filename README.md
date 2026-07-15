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
