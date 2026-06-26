# DRIVERDAPP — Edge Detection & Explainability Simulation

This repository contains the **AI detection and explainability module** of the
**DRIVERDAPP** framework — an integrated system for secure, explainable, and
deployable driver-distraction monitoring in intelligent transportation systems.

DRIVERDAPP unifies three pillars:

1. **Real-time edge detection** — in-cabin RGB frames are classified locally on an
   NVIDIA Jetson Nano edge device, triggering in-vehicle audio alerts for unsafe
   driver activities.
2. **Explainable AI (XAI)** — Gradient-weighted Class Activation Mapping
   (Grad-CAM) produces visual explanations of distraction-related predictions.
3. **Trustworthy event logging** — confirmed distraction events are recorded
   immutably via Solidity smart contracts on a permissioned Hyperledger Besu
   (QBFT) consortium blockchain, while raw visual data is retained off-chain to
   preserve privacy.

> **Scope of this repo.** This codebase covers **pillars 1 and 2** — the
> on-device inference model and the Grad-CAM explainability pipeline. The
> blockchain / smart-contract components live in a separate repository.

---

## Contents

| File                                                 | Description                                                 |
| ---------------------------------------------------- | ----------------------------------------------------------- |
| [`classification_model.pt`](classification_model.pt) | The trained driver-state classification model (best model). |
| [`live_yolo.py`](live_yolo.py)                       | Real-time webcam inference with on-screen FPS and latency.  |
| [`grad_cam.py`](grad_cam.py)                         | Batch Grad-CAM heatmap generation for a folder of images.   |

---

## The model

`classification_model.pt` is an **Ultralytics YOLO image-classification model** that maps a single
in-cabin frame to one of **ten driver-behavior states**.

| Property            | Value                                                                       |
| ------------------- | --------------------------------------------------------------------------- |
| Task                | Image classification (`classify`)                                           |
| Backbone            | YOLO classification model (Ultralytics)                                     |
| Classes             | 10 (`c0`–`c9`)                                                              |
| Training image size | `640`                                                                       |
| Approx. file size   | ~3 MB (lightweight, suitable for Jetson Nano edge inference)                |
| Training datasets   | State Farm Distracted Driver Detection + American University in Cairo (AUC) |

### Class labels

The model stores classes as `c0`–`c9`, following the State Farm Distracted Driver
convention:

| Label | Driver state                 |
| ----- | ---------------------------- |
| `c0`  | Safe driving                 |
| `c1`  | Texting — right hand         |
| `c2`  | Talking on the phone — right |
| `c3`  | Texting — left hand          |
| `c4`  | Talking on the phone — left  |
| `c5`  | Operating the radio          |
| `c6`  | Drinking                     |
| `c7`  | Reaching behind              |
| `c8`  | Hair and makeup              |
| `c9`  | Talking to passenger         |

States `c1`–`c9` are treated as distraction/unsafe activities.

---

## Requirements

- Python 3.9–3.11 (Ultralytics does not yet officially support 3.14)
- A working webcam (for `live_yolo.py`)

Install dependencies:

```bash
pip install ultralytics opencv-python torch numpy
```

`ultralytics` pulls in a compatible PyTorch build. For Jetson Nano / GPU
deployment, install the platform-specific PyTorch wheel instead.

---

## Usage

### 1. Real-time inference — `live_yolo.py`

Runs the classifier on a live webcam feed and overlays per-frame **FPS** and
**latency (ms)**, mirroring the edge-inference behavior on the target device.

```bash
python live_yolo.py
```

- Press **`q`** to quit.
- Inference runs on CPU by default (`device="cpu"` in
  [`live_yolo.py`](live_yolo.py)). On a CUDA / Jetson device, change this to
  `device=0` for GPU acceleration.
- Capture resolution is set to 640×480; adjust `CAP_PROP_FRAME_WIDTH/HEIGHT`
  as needed.

### 2. Grad-CAM explainability — `grad_cam.py`

Generates Grad-CAM heatmaps over a folder of images, overlaying the activation
map and the predicted class/confidence on each image.

Before running, edit the paths near the top of [`grad_cam.py`](grad_cam.py):

```python
IMAGE_DIR  = "./PATH_TO_DATASET_IMAGES"   # folder of input images
OUTPUT_DIR = "./output/"                  # heatmaps are written here
```

Then run:

```bash
python grad_cam.py
```

For each image the script:

1. Runs a forward pass and reads the top class + softmax confidence.
2. Hooks the **last convolutional layer**, backpropagates the target class, and
   computes the Grad-CAM activation map.
3. Overlays a JET-colormap heatmap on the original image and annotates the
   predicted label, saving the result to `OUTPUT_DIR`.

---

## Datasets

- **State Farm Distracted Driver Detection** — Kaggle competition dataset (10 classes, `c0`–`c9`):
  <https://www.kaggle.com/c/state-farm-distracted-driver-detection>
- **American University in Cairo (AUC) Distracted Driver** dataset (Distracted Driver V2):
  <https://heshameraqi.github.io/distraction_detection>

---

## Citation

If you use this code or model, please cite the DRIVERDAPP paper:

```
<TBA>
```
