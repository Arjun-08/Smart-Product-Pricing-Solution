

> A multimodal machine learning pipeline that predicts e-commerce product prices using catalog text and product images.

---

## Problem Statement

Determining the optimal price of a product is a critical task for e-commerce platforms. Product pricing depends on multiple factors including textual descriptions, specifications, quantity, and visual appearance. The objective of this project is to build a **multimodal machine learning model** that predicts product prices using:

- **Catalog content (text)**
- **Product images**

---

## Evaluation Metric

The model is evaluated using **Symmetric Mean Absolute Percentage Error (SMAPE)**:

$$SMAPE = \frac{1}{n}\sum \frac{|y_{pred}-y_{true}|}{(|y_{true}|+|y_{pred}|)/2}$$

- Bounded between **0% and 200%**
- Lower SMAPE indicates better performance
- Robust to scale differences and price outliers

---

## Dataset Description

The dataset contains structured product metadata and image links.

### Columns

| Column Name | Description |
|--------------------|-------------|
| `sample_id` | Unique product identifier |
| `catalog_content` | Title + description + Item Pack Quantity (IPQ) |
| `image_link` | Public URL of the product image |
| `price` | Target variable (train only) |

### Files

| File | Description |
|------|-------------|
| `train.csv` | 75,000 labeled samples |
| `test.csv` | 75,000 unlabeled samples |
| `sample_test_out.csv` | Submission format reference |

---

## Output Format

Predictions must be submitted as:

    sample_id,price
    100001,349.99
    100002,1299.50

- All prices must be **positive floating-point values**.

---

## Methodology

### Overview

We implement a **late-fusion multimodal pipeline** that independently extracts:

- Semantic signals from text
- Visual signals from images

These representations are fused and learned by a **LightGBM regressor** for final price prediction.

---

## Data Preprocessing

### Text Processing

- Lowercasing
- Whitespace normalization
- Removal of noisy formatting tokens

### Image Processing

- Images downloaded using provided utility (`src/utils.py`)
- Parallel downloading with retry handling
- Cached locally
- Missing images replaced with **zero embeddings** (no external lookup used)

---

## Feature Engineering

### Text Features

Extracted using TF-IDF:
```python
TfidfVectorizer(
    ngram_range=(1,2),
    min_df=3,
    max_features=150000
)
```

Captures:
- Brand names
- Quantities (IPQ)
- Material/type descriptors
- Pricing keywords (e.g., "premium", "pack of 12")

### Image Features

Extracted using pretrained **EfficientNet-B0** as a frozen encoder:

- `weights = "imagenet"`
- `include_top = False`
- `pooling = "avg"`
- Input size: **224 x 224**
- Output: **1280-dimensional embedding**

No fine-tuning performed to comply with compute and parameter constraints.

---

## Feature Fusion

Text (sparse TF-IDF) and image (dense embeddings) are concatenated:
```python
scipy.sparse.hstack([text_features, image_features])
```

This produces a unified multimodal feature space.

---

## Model Training

### Regressor

**LightGBM (LGBMRegressor)** — chosen for:

- Strong tabular learning capability
- Ability to mix sparse + dense features
- Fast training on large datasets
- Stable optimization under SMAPE evaluation

### Objective

Model optimized using **MAE loss**, which empirically correlates well with SMAPE minimization.

### Hyperparameters
```python
{
    "objective": "mae",
    "n_estimators": 5000,
    "learning_rate": 0.03,
    "num_leaves": 255,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.2,
    "reg_lambda": 0.4,
    "random_state": 42
}
```

### Training Setup

- 85% Train / 15% Validation split
- Early stopping: 200 rounds
- Predictions clipped to >= 0.01 (SMAPE stability)

---

## Model Architecture
```
[catalog_content]
        |
TF-IDF Vectorizer
        |
Sparse Text Features \
                      --> Feature Concatenation --> LightGBM --> Price
                     /
  1280-D Embedding  /
        |
EfficientNet-B0 (frozen)
        |
  [product image]
```

---

## Results

| Metric | Validation |
| ------ | ---------- |
| MAE    | **11.46**  |
| SMAPE  | **52.63%** |

Public leaderboard SMAPE: **~50.25%**

---

## Key Findings

- Multimodal fusion significantly outperforms text-only baselines.
- Visual embeddings help capture:
  - Perceived product quality
  - Size/packaging cues
  - Category differentiation
- LightGBM effectively models nonlinear relationships between modalities.
- Output clipping improves SMAPE robustness.

---

## Insights & Future Improvements

Potential enhancements:

- Log-price regression to address skewed distributions
- CLIP / Vision-Language embeddings
- CNN fine-tuning with small learning rate
- Image availability indicator feature (`has_image`)
- Category-aware models or stacking ensembles

---

## Implementation Details

| Component   | Value                              |
| ----------- | ---------------------------------- |
| Language    | Python 3.12                        |
| Environment | Google Colab                       |
| Frameworks  | TensorFlow 2.17.1, LightGBM 4.5.0 |
| Hardware    | GPU-accelerated feature extraction |

---

## Project Paths

| Resource          | Path                                               |
| ----------------- | -------------------------------------------------- |
| Dataset           | `/content/drive/MyDrive/ARJUN/Exp/exp-28/dataset/` |
| Image Cache       | `/content/image_cache/`                            |
| Submission Output | `test_out.csv`                                     |

---

## Dependencies
```
numpy==2.1.3
pandas==2.2.3
scipy==1.13.1
scikit-learn==1.5.2
lightgbm==4.5.0
tensorflow==2.17.1
Pillow==10.4.0
tqdm>=4.67
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Reproducibility Steps

**1. Place dataset inside:**
```
dataset/
├── train.csv
├── test.csv
```

**2. Download images:**
```python
from src.utils import download_images
download_images("dataset/train.csv", "images/train")
download_images("dataset/test.csv", "images/test")
```

**3. Train model and generate predictions.**

**4. Export submission:**
```
test_out.csv
```

---

## Licensing and Compliance

- Code License: **MIT**
- TensorFlow: **Apache 2.0**
- LightGBM: **MIT**
- Model size: **<< 8B parameters**
- External data: **None used**
- No external price lookup performed (competition compliant)

This project strictly follows the challenge rule prohibiting:

- Web scraping
- External APIs
- Manual lookup
- Any external pricing source

---

## Summary

This solution demonstrates an efficient and scalable **multimodal regression pipeline** that combines NLP and computer vision features for price prediction while remaining lightweight, reproducible, and competition-compliant.

The approach balances:

- Accuracy
- Computational efficiency
- Interpretability
- Rule compliance
