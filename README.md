# Lightweight Deep Learning Research Project

> **Comparative Study of EfficientNet-B4 and MobileNet-V3 for Image Classification**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Overview

This research project benchmarks two state-of-the-art lightweight convolutional neural network architectures —  **EfficientNet-B4** and **MobileNet-V3-Large** — on an image classification task using transfer learning. The study evaluates each model across multiple dimensions: classification accuracy, model size, computational cost (FLOPs), and inference speed, to determine the optimal trade-off between accuracy and efficiency for resource-constrained deployment.

---

## 🎯 Objectives

1. Evaluate and compare EfficientNet-B4 and MobileNet-V3-Large on a standard image classification benchmark.
2. Analyse the accuracy–efficiency trade-off for each architecture under identical training conditions.
3. Measure real-world inference latency suitable for edge and mobile deployment scenarios.
4. Provide reproducible, well-documented experiment notebooks for each model.

---

## 🏗️ Models

### EfficientNet-B4
EfficientNet scales network width, depth, and input resolution jointly via a compound coefficient. B4 is the fourth variant in the family and delivers strong accuracy with a significantly smaller parameter count than traditional deep networks.

| Property | Value |
|---|---|
| Parameters | ~19 M |
| Input Resolution | 380 × 380 |
| Top-1 ImageNet Accuracy (pretrained) | ~83.0 % |

### MobileNet-V3-Large
MobileNet-V3 combines depthwise-separable convolutions with hard-swish activations and a Neural Architecture Search (NAS)-optimised layout. The "Large" variant targets accuracy-focused deployments while remaining highly efficient.

| Property | Value |
|---|---|
| Parameters | ~5.4 M |
| Input Resolution | 224 × 224 |
| Top-1 ImageNet Accuracy (pretrained) | ~75.2 % |

---

## 📂 Project Structure

```
Lightweight_Deeplearning_Research_Project/
│
├── efficientnet-b4.ipynb       # Full training & evaluation notebook for EfficientNet-B4
├── mobilenet-v3.ipynb          # Full training & evaluation notebook for MobileNet-V3-Large
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation (this file)
```

---

## 🗂️ Dataset

Both models are fine-tuned on the **CIFAR-10** dataset, which consists of:

- **60,000** colour images (32 × 32) across **10 classes**
- **50,000** training images / **10,000** test images
- Classes: *airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck*

> Images are resized to the native input resolution of each model during preprocessing.

---

## 🔬 Methodology

| Step | Details |
|---|---|
| **Pre-training** | ImageNet-pretrained weights (via `torchvision.models`) |
| **Fine-tuning strategy** | Feature extraction → gradual unfreezing of later blocks |
| **Optimiser** | Adam (`lr = 1e-3`, weight decay `1e-4`) |
| **LR Scheduler** | CosineAnnealingLR |
| **Loss** | Cross-Entropy |
| **Epochs** | 30 (with early stopping, patience = 5) |
| **Batch size** | 64 |
| **Data augmentation** | Random horizontal flip, random crop, colour jitter, normalisation |
| **Hardware** | NVIDIA GPU (CUDA) / Apple MPS / CPU fallback |

---

## 📊 Results

| Metric | EfficientNet-B4 | MobileNet-V3-Large |
|---|---|---|
| **Test Accuracy** | 94.8 % | 93.1 % |
| **Test Loss** | 0.172 | 0.213 |
| **Parameters** | 19.3 M | 5.4 M |
| **Model Size (MB)** | 73.8 | 21.4 |
| **Avg. Inference Time (ms/img)** | 12.4 | 4.7 |
| **Training Time (min / epoch)** | ~3.2 | ~1.8 |

> *Results obtained on CIFAR-10 test set. Inference time measured on a single CPU core (batch size = 1).*

### Key Findings

- **EfficientNet-B4** achieves ~1.7 percentage points higher accuracy but requires ~3.4× more parameters and ~2.6× longer inference time.
- **MobileNet-V3-Large** is the better choice for latency-sensitive or memory-constrained deployments, offering competitive accuracy at a fraction of the computational cost.
- Both models converge within 25 epochs when using transfer learning with gradual unfreezing.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/Kings-man-6969/Lightweight_Deeplearning_Research_Project.git
cd Lightweight_Deeplearning_Research_Project

# Install dependencies
pip install -r requirements.txt
```

### Running the Notebooks

```bash
jupyter notebook
```

Open either `efficientnet-b4.ipynb` or `mobilenet-v3.ipynb` and run all cells sequentially.

---

## 📦 Requirements

See [`requirements.txt`](requirements.txt) for the full dependency list. Core libraries:

- `torch` ≥ 2.0
- `torchvision` ≥ 0.15
- `numpy`, `matplotlib`, `scikit-learn`
- `jupyter`, `tqdm`

---

## 🔮 Future Work

- [ ] Extend evaluation to CIFAR-100 and Tiny-ImageNet datasets
- [ ] Explore quantisation-aware training (QAT) and post-training static quantisation
- [ ] Benchmark on actual edge hardware (Raspberry Pi 4, Jetson Nano)
- [ ] Integrate Grad-CAM visualisations to explain model predictions
- [ ] Compare with additional architectures: MobileNet-V2, ShuffleNet-V2, EfficientNet-B0

---

## 👨‍💻 Author

**Prashant Mishra**  
B.Tech – Computer Science (AI & ML)  
2023 Batch

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).