# Lightweight Deep Learning Research Project

> **Comparative Study of EfficientNet-B4 and MobileNet-V3-Small for Plant Disease Classification**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Overview

This research project benchmarks two state-of-the-art lightweight convolutional neural network architectures — **EfficientNet-B4** and **MobileNet-V3-Small** — on a plant disease classification task using transfer learning with TensorFlow/Keras. The study evaluates each model across multiple dimensions: classification accuracy, model size, and inference speed, to determine the optimal trade-off between accuracy and efficiency for resource-constrained deployment.

---

## 🎯 Objectives

1. Evaluate and compare EfficientNet-B4 and MobileNet-V3-Small on the PlantVillage disease classification benchmark.
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
| Input Resolution | 224 × 224 |
| Top-1 ImageNet Accuracy (pretrained) | ~83.0 % |

### MobileNet-V3-Small
MobileNet-V3 combines depthwise-separable convolutions with hard-swish activations and a Neural Architecture Search (NAS)-optimised layout. The "Small" variant targets ultra-lightweight deployments with minimal computational cost.

| Property | Value |
|---|---|
| Parameters | ~2.5 M |
| Input Resolution | 224 × 224 |
| Top-1 ImageNet Accuracy (pretrained) | ~67.7 % |

---

## 📂 Project Structure

```
Lightweight_Deeplearning_Research_Project/
│
├── efficientnet-b4.ipynb       # Full training & evaluation notebook for EfficientNet-B4
├── mobilenet-v3.ipynb          # Full training & evaluation notebook for MobileNet-V3-Small
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation (this file)
```

---

## 🗂️ Dataset

Both models are fine-tuned on the **PlantVillage** dataset, which consists of:

- **~87,000** colour images across **38 plant/disease classes**
- **80 % training** / **20 % validation** split (stratified, seed = 42)
- Images resized to **224 × 224** for both models
- Source: [`emmarex/plantdisease`](https://www.kaggle.com/datasets/emmarex/plantdisease) on Kaggle

---

## 🔬 Methodology

| Step | Details |
|---|---|
| **Framework** | TensorFlow / Keras (`tensorflow.keras`) |
| **Pre-training** | ImageNet-pretrained weights (via `tensorflow.keras.applications`) |
| **Fine-tuning strategy** | Feature extraction with frozen base model |
| **Custom head** | GlobalAveragePooling2D → BatchNorm → Dense(128, ReLU) → Dropout(0.3) → Dense(NUM_CLASSES, Softmax) |
| **Optimiser** | Adam (`lr = 1e-4`) |
| **Loss** | Sparse Categorical Cross-Entropy |
| **Epochs** | 10 |
| **Batch size** | 16 |
| **Mixed precision** | `mixed_float16` (GPU speed boost) |
| **Data pipeline** | `cache()` → `shuffle(1000)` → `prefetch(AUTOTUNE)` |
| **Data augmentation** | Random horizontal flip, random rotation (±10 %), random zoom (±20 %) |
| **Hardware** | Kaggle GPU (NVIDIA) with CUDA |

---

## 📊 Results

> *Results on the PlantVillage validation set (20 % split). Run notebooks on Kaggle or a CUDA-enabled GPU to reproduce.*

### Key Findings

- **EfficientNet-B4** delivers higher classification accuracy at the cost of greater memory and compute requirements (~19 M parameters).
- **MobileNet-V3-Small** is the better choice for latency-sensitive or memory-constrained deployments, offering competitive accuracy with only ~2.5 M parameters.
- Both models converge within 10 epochs when using ImageNet-pretrained weights with a frozen base.

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

The notebooks are designed to run on **Kaggle** with the PlantVillage dataset pre-loaded at `/kaggle/input/datasets/emmarex/plantdisease/PlantVillage`. To run locally, update the `DATA_DIR` / `DATA_DIR_PV` path in each notebook's dataset cell to point to your local copy of the dataset.

```bash
jupyter notebook
```

Open either `efficientnet-b4.ipynb` or `mobilenet-v3.ipynb` and run all cells sequentially.

---

## 📦 Requirements

See [`requirements.txt`](requirements.txt) for the full dependency list. Core libraries:

- `tensorflow` ≥ 2.12
- `numpy`, `matplotlib`, `scikit-learn`
- `seaborn`, `pandas`
- `jupyter`

---

## 🔮 Future Work

- [ ] Extend evaluation to additional plant disease datasets
- [ ] Explore quantisation-aware training (QAT) and post-training static quantisation
- [ ] Benchmark on actual edge hardware (Raspberry Pi 4, Jetson Nano)
- [ ] Integrate Grad-CAM visualisations to explain model predictions
- [ ] Compare with additional architectures: MobileNet-V2, EfficientNet-B0, ShuffleNet-V2
- [ ] Add fine-tuning phase (gradual unfreezing of later blocks)

---

## 👨‍💻 Author

**Prashant Mishra**  
B.Tech – Computer Science (AI & ML)  
2023 Batch

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).