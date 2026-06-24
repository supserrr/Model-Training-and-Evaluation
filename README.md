# Pneumonia Detection in Pediatric Chest X-Rays
### A Comparative Study of Classical Machine Learning and Deep Learning

Summative project for *Introduction to Machine Learning* — African Leadership University.

This project asks two questions: (1) can a model reliably flag pneumonia from pediatric
chest radiographs, and (2) is a lightweight, CPU-friendly **classical** pipeline a viable
low-compute alternative to **deep learning** in settings without GPUs? A single reproducible
pipeline puts both families of method on an identical footing — same data, same
train/validation/test split, same evaluation protocol — across **eight experiments**.

## Dataset
Pediatric chest X-ray dataset (Kermany et al., 2018) — ~5,856 anterior–posterior radiographs
labelled NORMAL / PNEUMONIA, collected at Guangzhou Women and Children's Medical Center.
Open-source (Creative Commons); **not** a built-in library dataset.

- Kaggle: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

The data is **not** committed to this repo. Download it with the Kaggle API (see below).

## Approaches compared
**Classical track (scikit-learn):** raw-pixel and HOG features → Logistic Regression, SVM (RBF), Random Forest.
**Deep-learning track (TensorFlow/Keras):** CNN from scratch (Sequential API), regularised CNN,
MobileNetV2 transfer learning frozen + fine-tuned (Functional API), plus a learning-rate sweep.
Input pipeline built with the `tf.data` API.

## Results (test set, sorted by F1)
| #  | Model | Accuracy | F1 | ROC-AUC |
|----|-------|----------|------|---------|
| E4 | CNN from scratch | 0.961 | **0.973** | **0.992** |
| E7 | MobileNetV2 (fine-tuned) | 0.952 | 0.967 | 0.990 |
| E2 | SVM + HOG | 0.951 | 0.966 | 0.987 |
| E3 | Random Forest + HOG | 0.940 | 0.958 | 0.980 |
| E6 | MobileNetV2 (frozen) | 0.937 | 0.956 | 0.983 |
| E1 | Logistic Regression (pixels) | 0.932 | 0.952 | 0.974 |
| E5 | Regularised CNN | 0.848 | 0.886 | 0.943 |

**Key finding:** the three leading models — a CNN trained from scratch (E4), fine-tuned
MobileNetV2 (E7), and a tuned SVM on HOG features (E2) — sit within ~0.007 F1 of one another.
The classical SVM reaches that level in about 90 seconds on a CPU, with no GPU, which makes the
classical pipeline a credible low-resource triage option. The regularised CNN (E5) trained
unstably and finished the weakest deep model — an instructive failure analysed in the report.

## Repository structure
```
pneumonia-xray-classification/
├── pneumonia_xray_classification.ipynb   # full, reproducible pipeline, executed (run top-to-bottom)
├── Pneumonia_Report.pdf                  # written report (IEEE references, figures)
├── requirements.txt                      # pinned dependency versions
└── README.md
```

## How to run
```bash
# 1. install dependencies (Python 3.12 recommended)
pip install -r requirements.txt

# 2. set up the Kaggle API (place kaggle.json in ~/.kaggle/)
#    https://www.kaggle.com/docs/api
mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# 3. launch the notebook (Jupyter or Google Colab) and run all cells
jupyter notebook pneumonia_xray_classification.ipynb
```
Random seeds are fixed for reproducibility. The notebook downloads the dataset, builds a fresh
stratified 80/10/10 split, and logs every experiment to a shared results table. On Apple Silicon,
install the optional `tensorflow-metal` plugin for GPU acceleration.

## Report & demo
- **Report:** `Pneumonia_Report.pdf`
- **Demo video:** https://drive.google.com/drive/folders/17L7Wuw9Q4uCeVoKbq9Euvr6LR915iTfD?usp=share_link
- **GitHub repository:** https://github.com/supserrr/Model-Training-and-Evaluation
