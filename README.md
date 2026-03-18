# VISTA: Visual Intelligence for Satellite Threat Analysis

<img src="images/Vista-Highres.png" width="450">


**VISTA** was developed as the final project of the Data Science & AI Bootcamp at Le Wagon Barcelona by a [team of 3](#-contributors), completed in 2 weeks in March 2026. 

This is a computer vision-powered building damage assessment from satellite imagery. Two independent pipelines were developed and benchmarked classifying post-disaster damage at pixel level: semantic segmentation (ResNet+U-Net) and object detection (YOLO).

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#️-dataset)
- [Model Architectures](#-model-architectures)
- [Pipeline](#️-pipeline)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Project Structure](#️-project-structure)
- [Contributors](#-contributors)
- [References](#-references)

---

## 🔭 Overview

VISTA leverages satellite imagery to automatically assess building damage following natural disasters. The system processes pre- and post-disaster image pairs and classifies each building pixel into one of three damage categories:

| Class | Label | Description |
|-------|-------|-------------|
| 0 | Background | Non-building pixels |
| 1 | No damage | Building present, no visible damage |
| 2 | Damaged | Minor or major structural damage |
| 3 | Destroyed | Complete or near-complete destruction |

Two independent approaches were developed and compared:

- **U-Net segmentation pipeline** — two-stage semantic segmentation using ResNet-34 for localization and damage classification, based on the xView2 challenge first-place solution - [Link](https://github.com/DIUx-xView/xView2_first_place)
- **YOLO-based pipeline** — object detection approach using YOLOv26 for building localization and damage classification, based on a challenge solution shared in Kaggle - [Link](https://www.kaggle.com/code/satvikjain09/ce712-yolov8)

---

## 🗃️ Dataset

The project uses the **xBD dataset** from the [xView2 Building Damage Assessment Challenge](https://xview2.org/).

| Split | Images | Disaster Types |
|-------|--------|---------------|
| Train | ~18,000 pre/post pairs | Flooding, wildfire, tornado, hurricane, volcano, tsunami |
| Tier3 | ~2,000 pre/post pairs | Additional training data |
| Test | ~933 pre/post pairs | Held-out evaluation set |

**Image specs:** 1024×1024 px, RGB, satellite view  
**Annotation format:** GeoJSON polygons with per-building damage labels  
**Original damage classes:** no-damage, minor-damage, major-damage, destroyed, un-classified  
**VISTA classes:** minor-damage and major-damage are merged into a single "damaged" class to reduce label noise and improve precision

**Download:** [xView2 Challenge Dataset](https://xview2.org/dataset)

---

## 🧠 Model Architectures

### Pipeline 1 — U-Net Segmentation (Primary)

**Stage 1 — Localization**

| Component | Details |
|-----------|---------|
| Model | ResNet-34 + U-Net decoder |
| Task | Binary building segmentation |
| Input | Pre-disaster image (3 channels) |
| Output | Binary building mask |
| Training epochs | 60 |
| Input resolution | 736×736 |
| Metric | Dice (best: ~0.85) |

**Stage 2 — Damage Classification**

| Component | Details |
|-----------|---------|
| Model | ResNet-34 + U-Net decoder |
| Task | Per-pixel damage classification (3 classes) |
| Input | Pre + post disaster images concatenated (6 channels) |
| Output | 4-channel mask (background + 3 damage classes) |
| Training epochs | 30 |
| Input resolution | 608×608 |
| Loss | ComboLoss (Dice + 12× Focal), channel weights [0.05, 0.2, 0.8, 0.5] |
| Checkpoint criterion | Macro precision (insurance use case — minimise false positives) |

### Pipeline 2 — YOLO Object Detection

| Component | Details |
|-----------|---------|
| Model | YOLOv26n (nano) |
| Framework | Ultralytics |
| Task | Building detection + damage classification |
| Input | Post-disaster image (single frame) |
| Output | Bounding boxes with damage class label |
| Training epochs | Up to 10,000 (early stopping) |
| Batch size | 16 |
| Input resolution | 640×640 |
| Train/Val split | 80/20 |
| Annotation format | YOLO bounding box (converted from GeoJSON polygons) |
| Class imbalance | Minority-class oversampling with pixel-level augmentation |

---

## ⚙️ Pipeline

### Pipeline 1 — ResNet

```
Pre-disaster image ──┐
                     ├──► Res34_Unet_Loc ──► Building mask
Post-disaster image ─┘          │
                                ▼
Pre + Post    ──►    Res34_Unet_Double ──► Damage classification
                                │
                                ▼
                         Final visualization
```

### Pipeline 2 — YOLO

```
Post-disaster image ──► YOLOv26n ──► Bounding boxes with damage class
                                           │
                                           ▼
                              Damage heatmap visualization
```

---

## 💻 Installation

```bash
# Clone the repository
git clone https://github.com/martimlvg/vista-damage-assessment.git
cd vista-damage-assessment

# Install dependencies
pip install opencv-python-headless torch torchvision scikit-image scikit-learn \
            scipy pandas tqdm pretrainedmodels matplotlib shapely
```

**Requirements:**
- Python 3.8+
- PyTorch 1.12+
- CUDA-capable GPU (recommended: 16GB+ VRAM)
- ~50GB disk space for dataset + weights

---

## 📖 Usage

### Pipeline 1 — U-Net Segmentation (ResNet-34)

#### 1. Prepare masks
```python
# Run create_masks.py once to generate ground truth masks from JSON labels
# (skip if masks already exist)
```

#### 2. Train localization model
```bash
# In notebook: run Cell 4 (Train Localization)
# Or via script:
python train34_loc.py 0   # seed=0
```

#### 3. Predict localization on validation split
```bash
# Required before training classification
# In notebook: run Cell 5
```

#### 4. Train classification model
```bash
# In notebook: run Cell 6 (Train Classification)
# Or via script:
python train34_cls.py 0   # seed=0
```

#### 5. Run inference and build visualization output
```bash
# In notebook: run Cells 8, 9, 10
# Output: visualization/{image}_localization_prediction.png
#         visualization/{image}_damage_prediction.png
```

### Pipeline 2 — YOLO (YOLOv26)

#### 1. Configure dataset path
```python
# In notebook cell 8 (ce712-yolov8_log.ipynb):
DATASET_BASE = "xbd-dataset/xbd/"
OUTPUT_DIR   = "visual-aid/yolo_dataset2"
```

#### 2. Convert annotations to YOLO format
```python
# In notebook: run Steps 2–4
# Parses GeoJSON polygons → YOLO bounding boxes
# Maps 5 damage classes → 3 classes
# Applies minority-class oversampling with pixel-level augmentation
```

#### 3. Train YOLOv26
```python
# In notebook: run Step 6
# Model: YOLOv26n, 640×640, batch=16, up to 10,000 epochs
```

#### 4. Evaluate and visualize
```python
# In notebook: run Steps 7–9
# Outputs: bounding box predictions, confusion matrix, damage heatmaps
```

---

### 📓 Notebooks

| Notebook | Pipeline | Description |
|----------|----------|-------------|
| `xview2_pipeline_res34_3_classes.ipynb` | U-Net | ResNet-34 loc + cls, 3 damage classes |
| `xview2_pipeline_yolo_3_classes.ipynb` | YOLO | YOLOv26 training, evaluation and damage heatmaps |

---

## 📊 Results

> Results are reported on the validation split (10% holdout, `random_state=seed`).  
> U-Net checkpoint selection criterion: **macro precision** (optimised for insurance use case — minimise false positives).

### U-Net Pipeline — ResNet-34

| Metric | Value |
|--------|-------|
| Val Score (0.3×Dice + 0.7×F1) | 0.76 |
| Dice (localization) | 0.85 |
| Macro F1 | 0.72 |
| F1 — No damage | 0.92 |
| F1 — Damaged | 0.55 |
| F1 — Destroyed | 0.79 |
| Macro Precision | 0.83 |
| Precision — No damage | 0.87 |
| Precision — Damaged | 0.80 |
| Precision — Destroyed | 0.81 |

**Sample predictions — ResNet-34 pipeline:**

![Sample 1](images/resnet_demo_1.png) 

![Sample 2](images/resnet_demo_2.png) 

---

### YOLO Pipeline — YOLOv26n

| Metric | Value |
|--------|-------|
| mAP@0.5 | 0.41 |
| mAP@0.5:0.95 | 0.19 |
| Global Precision | 0.62 |
| Global Recall | 0.37 |
| Global F1 | 0.46 |

**Sample predictions — YOLO pipeline:**

![Sample](images/yolo_demo.png) 

---

## 🗂️ Project Structure

```
vista-damage-assessment/
├── resnet/
│   ├── xView2_first_place_3_classes/
│   │   ├── zoo/
│   │   │   ├── models.py          # ResNet-34 U-Net variants
│   │   │   ├── senet.py
│   │   │   └── dpn.py
│   │   ├── train34_loc.py         # Localization training (ResNet-34)
│   │   ├── train34_cls.py         # Classification training (ResNet-34, 3 classes)
│   │   ├── tune34_cls.py          # Fine-tuning (ResNet-34)
│   │   ├── create_masks.py        # Generate ground truth masks from JSON labels
│   │   ├── utils.py               # Augmentations, metrics, preprocessing
│   │   ├── losses.py              # ComboLoss, DiceLoss, FocalLoss
│   │   ├── adamw.py               # AdamW optimizer
│   │   ├── predict34_loc.py
│   │   ├── predict34cls.py
│   │   ├── create_submission.py
│   │   └── weights/               # Saved model checkpoints
│   └── xview2_pipeline_res34_3_classes.ipynb
├── yolo/
│   ├── xview2_pipeline_yolo_3_classes.ipynb     # YOLOv26 training & evaluation notebook
│   └── runs/detect/               # Training outputs, weights, metrics
├── images/                        # README screenshots and sample predictions
└── README.md
```

---

## 👥 Contributors

| Name | GitHub |
|------|--------|
| Edison Kruger | [@EKRUGER-BCN](https://github.com/EKRUGER-BCN) |
| Ildebrando Jesus | [@ijesusjr](https://github.com/ijesusjr) |
| Martim Gomes | [@martimlvg](https://github.com/martimlvg) |

---

## 📚 References

- **xView2 Dataset & Challenge:** [xview2.org](https://xview2.org/)  
  *Ritwik Gupta et al. — "xBD: A Dataset for Assessing Building Damage from Satellite Imagery" (2019)*

- **xView2 First-Place Solution:**  
  [github.com/DIUx-xView/xView2_first_place](https://github.com/DIUx-xView/xView2_first_place)

- **xView2 YOLO v.8 Solution:**  
  [https://www.kaggle.com/code/satvikjain09/ce712-yolov8](https://www.kaggle.com/code/satvikjain09/ce712-yolov8)

- **ResNet:** He et al. — "Deep Residual Learning for Image Recognition" (CVPR 2016)

- **U-Net:** Ronneberger et al. — "U-Net: Convolutional Networks for Biomedical Image Segmentation" (MICCAI 2015)

- **YOLOv26:** Ultralytics — [github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

- **Research paper:** Yap et al. — "Building Detection from Satellite Images using Deep Learning"

- **ComboLoss:** Taghanaki et al. — "Combo Loss: Handling Input and Output Imbalance in Multi-Organ Segmentation" (2019)
