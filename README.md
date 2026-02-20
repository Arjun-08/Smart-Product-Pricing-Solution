
## Problem Statement

Determining the optimal price of a product is a critical task for e-commerce platforms. Product pricing depends on multiple factors including textual descriptions, specifications, quantity, and visual appearance.

The objective of this project is to build a **multimodal machine learning model** that predicts the price of a product using both:

-  Catalog content (text)
-  Product images

### Evaluation Metric

The model is evaluated using **Symmetric Mean Absolute Percentage Error (SMAPE)**.  
Lower SMAPE values indicate better performance.

---

## Dataset Description

The dataset consists of structured CSV files containing product metadata and images.

### Columns

| Column Name        | Description |
|--------------------|-------------|
| `sample_id`        | Unique identifier for each product |
| `catalog_content` | Product title, description, and Item Pack Quantity (IPQ) |
| `image_link`       | Public URL of the product image |
| `price`            | Target variable (training data only) |

### Files

- `train.csv` — 75,000 labeled products  
- `test.csv` — 75,000 unlabeled products  
- `sample_test_out.csv` — Submission format example  

### Output Format: In csv sample_id,price
All predicted prices must be positive floating-point values.

## Methodology

# Methodology

- **Overview**
  - A late-fusion multimodal learning pipeline combines textual and visual features.
  - A single LightGBM regressor learns from the fused feature space to predict prices efficiently.

- **Data Preprocessing**
  - **Text Processing**
    - Lowercasing
    - Whitespace normalization
  - **Image Processing**
    - Parallel downloading with retry logic
    - Cached locally at `/content/image_cache/`
    - Missing or invalid images replaced with zero-vector embeddings

- **Feature Engineering**
  - **Text Features**
    - Extracted using TF-IDF:
      - `TfidfVectorizer(`
        - `ngram_range=(1, 2),`
        - `min_df=3,`
        - `max_features=150_000`
      - `)`
    - Captures unigrams and bigrams representing brands, attributes, and quantities.
  - **Image Features**
    - Pretrained EfficientNet-B0 used as a frozen feature extractor:
      - `include_top=False`
      - `pooling='avg'`
      - `weights='imagenet'`
    - Image size: 224 × 224
    - Output: 1280-dimensional embedding
    - Batch size: 64

- **Feature Fusion**
  - Text TF-IDF (sparse) and image embeddings (dense) concatenated using:
    - `scipy.sparse.hstack`

- **Model Training**
  - **Regressor**
    - LightGBM (LGBMRegressor)
  - **Objective**
    - Mean Absolute Error (MAE)
  - **Hyperparameters**
    - `{`
      - `"objective": "mae",`
      - `"n_estimators": 5000,`
      - `"learning_rate": 0.03,`
      - `"num_leaves": 255,`
      - `"subsample": 0.8,`
      - `"colsample_bytree": 0.8,`
      - `"reg_alpha": 0.2,`
      - `"reg_lambda": 0.4,`
      - `"random_state": 42`
    - `}`
  - **Training Setup**
    - 85% training / 15% validation split
    - Early stopping after 200 rounds
    - Predictions clipped to >= 0.01

- **Model Architecture**
  - `[text: catalog_content]`
    - normalize
    - TF-IDF (1–2 grams, 150k features)
  - `[image: image_link]`
    - download & cache
    - EfficientNet-B0 (frozen, avg pooling)
    - 1280-D embedding
  - Concatenate
  - LightGBM
  - price

- **Results**
  - **Metric**
    - Validation
    - Test
  - **MAE**
    - 11.4555
    - —
  - **SMAPE (%)**
    - 52.63
    - 50.25

- **Key Findings**
  - Multimodal model outperforms text-only baselines
  - Image embeddings add strong visual cues
  - MAE objective provides stable SMAPE optimization

- **Insights and Discussion**
  - **Observations**
    - TF-IDF captures brand and quantity signals effectively
    - EfficientNet embeddings improve visual understanding
    - LightGBM balances speed, accuracy, and interpretability
    - Output clipping stabilizes SMAPE
  - **Potential Improvements**
    - Log-price regression for skewed targets
    - TF-IDF ensemble vocabularies
    - Add has_image feature
    - Fine-tune CNN backbone
    - Explore CLIP or Vision Transformers

- **Implementation Details**
  - **Language**
    - Python 3.12
  - **Environment**
    - Google Colab
  - **Frameworks**
    - TensorFlow 2.17.1
    - LightGBM 4.5.0
    - scikit-learn 1.5.2

- **Paths**
  - Dataset: `/content/drive/MyDrive/ARJUN/Exp/exp-28/dataset/`
  - Cache: `/content/image_cache/`
  - Output: `test_out.csv`

- **Dependencies**
  - `numpy==2.1.3`
  - `pandas==2.2.3`
  - `scipy==1.13.1`
  - `scikit-learn==1.5.2`
  - `lightgbm==4.5.0`
  - `tensorflow==2.17.1`
  - `Pillow==10.4.0`
  - `tqdm>=4.67`

- **Licensing and Compliance**
  - Model & Code License: MIT
  - TensorFlow License: Apache 2.0
  - LightGBM License: MIT
  - Parameter Count: Under 8 billion
  - External Data: None used

- **Conclusion**
  - This project demonstrates a robust and efficient multimodal regression framework for product price prediction.
  - By combining TF-IDF text embeddings and EfficientNet-B0 image features within a LightGBM regressor, the model achieves approximately 50% SMAPE on unseen test data while maintaining:
    - Lightweight architecture
    - Strong generalization
    - High interpretability
    - Full reproducibility

