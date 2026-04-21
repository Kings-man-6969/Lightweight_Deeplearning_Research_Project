# Lightweight Deep Learning Research Project

A comparative research study of lightweight convolutional neural network architectures for efficient image classification. This project explores and benchmarks **EfficientNet-B4** and **MobileNet-V3**, two state-of-the-art models designed to achieve high accuracy with significantly reduced computational cost.

---

## 📌 Overview

Modern deep learning models often demand enormous compute resources, making deployment on edge devices or resource-constrained environments challenging. This project investigates lightweight architectures that strike an optimal balance between model accuracy, parameter count, and inference speed.

**Research Questions:**
- How do EfficientNet-B4 and MobileNet-V3 compare in accuracy vs. efficiency trade-offs?
- What are the practical differences in training time, model size, and inference latency?
- Which architecture is better suited for edge/mobile deployment scenarios?

---

## 🧠 Models

### EfficientNet-B4
EfficientNet is a family of models that uniformly scales network width, depth, and resolution using a compound scaling method. EfficientNet-B4 is the fourth variant in the family, offering an excellent accuracy-efficiency balance.

- **Parameters:** ~19M
- **Input Resolution:** 380×380
- **Key Feature:** Compound scaling with MBConv blocks and Squeeze-and-Excitation

### MobileNet-V3
MobileNet-V3 is optimized for mobile/edge inference using Neural Architecture Search (NAS), depthwise separable convolutions, and hard-swish activations.

- **Parameters:** ~5.4M (Large variant)
- **Input Resolution:** 224×224
- **Key Feature:** NAS-tuned architecture with SE modules and h-swish activations

---

## 📁 Repository Structure

```
Lightweight_Deeplearning_Research_Project/
├── efficientnet-b4.xpynb      # EfficientNet-B4 experiments notebook
├── mobilenet-v3.xpynb         # MobileNet-V3 experiments notebook
└── README.md
```

---

## ⚙️ Setup & Requirements

### Prerequisites
- Python 3.8+
- PyTorch ≥ 1.12 or TensorFlow ≥ 2.9
- CUDA (optional, for GPU acceleration)

### Installation

```bash
# Clone the repository
git clone https://github.com/Kings-man-6969/Lightweight_Deeplearning_Research_Project.git
cd Lightweight_Deeplearning_Research_Project

# Install dependencies
pip install torch torchvision timm tensorflow tensorflow-hub
pip install numpy pandas matplotlib seaborn scikit-learn
pip install jupyter
```

---

## 🚀 Usage

Open the experiment notebooks to explore training, evaluation, and analysis:

```bash
jupyter notebook efficientnet-b4.xpynb
jupyter notebook mobilenet-v3.xpynb
```

Each notebook covers:
1. Dataset preparation and preprocessing
2. Model loading and configuration
3. Training with learning rate scheduling
4. Evaluation on test set
5. Performance metrics and visualizations

---

## 📊 Results

| Model           | Top-1 Accuracy | Parameters | Model Size | Inference Time (CPU) |
|-----------------|---------------|------------|------------|----------------------|
| EfficientNet-B4 | —             | ~19M       | ~74 MB     | —                    |
| MobileNet-V3-L  | —             | ~5.4M      | ~22 MB     | —                    |

> Results will be updated upon completion of experiments.

---

## 📚 References

- Tan, M., & Le, Q. V. (2019). [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946). *ICML 2019*.
- Howard, A., et al. (2019). [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244). *ICCV 2019*.

---

## 📄 License

This project is for academic and research purposes.
