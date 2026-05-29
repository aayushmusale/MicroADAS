# MicroADAS: Edge-Optimized Autonomous Hazard Perception

**Architected by Aayush Musale**

### 🚀 Overview

MicroADAS is a highly compressed, 11-class computer vision pipeline designed for autonomous hardware operating under extreme memory and compute constraints. Standard deep learning vision models require massive GPU processing and memory footprints (20MB+), making them impossible to deploy directly onto embedded microcontrollers.

This project solves the challenge of **Safety-Critical Edge Deployment**. By applying aggressive INT8 post-training quantization to a custom YOLOv8 Nano architecture, MicroADAS achieves real-time hazard detection (1.6ms inference) within a strict **<2.0 MB hardware storage limit**.

### 🛑 The Engineering Challenge

Autonomous agents require real-time processing to avoid collisions. However, deploying state-of-the-art vision models onto embedded hardware creates severe bottlenecks:

1. **Memory Limits:** Standard PyTorch models (`.pt`) use 32-bit floating-point math, instantly exceeding edge storage capabilities.
2. **Compute Latency:** Uncompressed matrix multiplication causes severe frame-drops on embedded CPUs, rendering the bot blind during movement.
3. **Dependency Conflicts:** Bridging PyTorch networks to Edge-compatible formats frequently triggers severe environment conflicts across ONNX and TensorFlow graph structures.

### 💡 The Solution Architecture

* **Algorithm Scaling:** Utilized a parameter-efficient YOLOv8 Nano backbone, scaling input resolution to `320x320` to drastically reduce overall Floating Point Operations (FLOPs).
* **Post-Training Quantization (PTQ):** Engineered a pipeline to safely round FP32 mathematical weights into INT8 integers. This compressed the physical payload by over 75% while preserving critical detection capabilities.
* **Toolchain Isolation:** Managed the complex export pipeline using strict Python virtual environments to bridge PyTorch, `sng4onnx`, and legacy `tf_keras` engines without graph corruption.

### 📊 Performance & Hardware Metrics

The model was trained on a custom dataset of **13,000+ images** across **11 dynamic classes** (Vehicles, Pedestrians, Traffic Light States).

* **Hardware Footprint:** < 2.0 MB (INT8 TFLite)
* **Inference Latency:** 1.6ms per frame *(Benchmarked on RTX 3050 Mobile)*
* **Total Pipeline Speed:** 2.7ms *(Includes Pre/Post-processing NMS)*
* **Vehicle Detection (mAP50):** 78.9%
* **Truck Detection (mAP50):** 72.9%
* **Traffic Signal State (Red):** 70.0%

### ⚠️ Limitations & Future Work

* **Class Imbalance Penalty:** The model exhibits high accuracy on dominant classes (Cars: 12,664 instances), but struggles with rare edge-cases (Yellow Left-Turn Signals: 4 instances). Future iterations require data augmentation (SMOTE/Mosaic) to balance minority classes.
* **Spatial Precision vs. Quantization:** The overall strict `mAP50-95` score sits at 0.293. This is a deliberate, acceptable trade-off. 8-bit quantization destroys the pixel-perfect boundary precision required for a high 50-95 score, but preserves the localized object center (mAP50), which is all that is strictly necessary for emergency braking triggers.

---

### 🚀 Quick Start & Environment Setup

*Note: To avoid dependency conflicts and GPU overwrite errors, please follow this strict two-step installation process.*

**1. Clone the Repository & Create Virtual Environment**

```bash
git clone https://github.com/yourusername/MicroADAS.git
cd MicroADAS
py -3.10 -m venv .venv
.\.venv\Scripts\activate

```

**2. Secure Hardware Acceleration (PyTorch + CUDA 12.1)**
*Run this first to prevent pip from defaulting to the CPU-only version.*

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

```

**3. Install Isolated Pipeline Dependencies**
*This installs the strictly pinned versions of TensorFlow and ONNX required to bypass legacy Keras conversion crashes.*

```bash
pip install -r requirements.txt

```

**4. Execute the Training & Export Pipeline**

```bash
python train_project.py

```

---

### 💻 Tech Stack

* **Deep Learning Frameworks:** PyTorch, Ultralytics YOLOv8, TensorFlow Lite
* **Optimization Pipeline:** ONNX, `onnx2tf`, `ai-edge-litert`
* **Data Processing:** OpenCV, NumPy
* **Environment:** Python 3.10 `venv`, CUDA 12.1
