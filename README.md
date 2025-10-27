Table of Contents

Problem Statement

Dataset Description

Methodology

Overview

Data Preprocessing

Feature Engineering

Model Training

Model Architecture

Results

Insights and Discussion

Implementation Details

Licensing and Compliance

Conclusion

Problem Statement

Determining the optimal price of a product is a critical task for e-commerce platforms. Product pricing depends on multiple factors including textual descriptions, specifications, quantity, and visual appearance.

The goal of this challenge is to build a machine learning model that predicts the price of a product using both its catalog content (text) and product image.

The evaluation metric for this task is Symmetric Mean Absolute Percentage Error (SMAPE), which measures the relative difference between predicted and actual prices. Lower SMAPE values indicate better performance.

Dataset Description

The dataset consists of training and test files with the following structure:

Column	Description
sample_id	Unique identifier for each product
catalog_content	Text field containing product title, description, and Item Pack Quantity (IPQ)
image_link	Public URL of the product image
price	Target variable (available only in training data)

Files:

train.csv — 75,000 labeled products

test.csv — 75,000 unlabeled products

sample_test_out.csv — Sample output format

Output Format:

sample_id,price


The final submission file must contain all test IDs and corresponding predicted float prices (positive values only).

Methodology
Overview

A multimodal learning pipeline was designed combining text features and image embeddings.
A single LightGBM regressor learns from the concatenated feature set to efficiently and accurately predict prices.

Data Preprocessing

Text Cleaning: Lowercasing and whitespace normalization.

Image Downloading: Parallel downloading with retries, cached locally at /content/image_cache/.

Missing Handling: Invalid URLs or unreadable images replaced with zero-vector embeddings.

Feature Engineering
Text Features

Extracted using:

TfidfVectorizer(ngram_range=(1,2), min_df=3, max_features=150_000)


Captures unigrams and bigrams representing product names, attributes, and quantities.

Image Features

Pretrained EfficientNet-B0 used as a frozen feature extractor:

include_top=False, pooling='avg', weights='imagenet'


Each image generates a 1280-dimensional embedding.

Images resized to 224×224 and processed in batches of 64.

Feature Fusion

Text TF-IDF (sparse) and image embeddings (dense) concatenated using:

scipy.sparse.hstack


The resulting multimodal feature space is used as input to the LightGBM regressor.

Model Training

Regressor: LightGBM (LGBMRegressor)
Objective: MAE (robust and well aligned with SMAPE)

Parameters:

{
  'objective': 'mae',
  'n_estimators': 5000,
  'learning_rate': 0.03,
  'num_leaves': 255,
  'subsample': 0.8,
  'colsample_bytree': 0.8,
  'reg_alpha': 0.2,
  'reg_lambda': 0.4,
  'random_state': 42
}


Training Setup:

85% training and 15% validation split

Early stopping after 200 rounds

Predictions clipped to positive values (>= 0.01)

Model Architecture
[text: catalog_content] → normalize → TF-IDF (1–2 grams, 150k features)
                                               \
                                                +→ concatenate → LightGBM → price
                                               /
[image: image_link] → download/cache → EfficientNet-B0 (avg-pool, frozen) → 1280-D embedding


This late-fusion design efficiently combines linguistic and visual signals without requiring end-to-end neural training.

Results
Metric	Validation	Test
MAE	11.4555	—
SMAPE (%)	52.63	50.25

Key Findings:

The hybrid model significantly outperformed text-only baselines.

Image embeddings improved performance even with a frozen backbone.

MAE objective provided smoother convergence and more stable SMAPE reduction.

Insights and Discussion

Observations:

TF-IDF features capture brand and quantity signals effectively.

EfficientNet embeddings improve visual understanding (e.g., size, packaging).

LightGBM fusion balances interpretability, speed, and generalization.

Clipping and normalization stabilize the metric.

Potential Improvements:

Apply log-price regression to handle skewed distributions.

Ensemble models over multiple TF-IDF vocabularies.

Add an explicit has_image indicator or perform light CNN fine-tuning.

Explore CLIP or Vision Transformer embeddings for stronger visual semantics.

Implementation Details

Language: Python 3.12

Frameworks: TensorFlow 2.17.1, LightGBM 4.5.0, scikit-learn 1.5.2

Environment: Google Colab

Paths:

Dataset: /content/drive/MyDrive/ARJUN/Exp/exp-28/dataset/

Cache: /content/image_cache/

Output: test_out.csv

Key Dependencies:

numpy==2.1.3
pandas==2.2.3
scipy==1.13.1
scikit-learn==1.5.2
lightgbm==4.5.0
tensorflow==2.17.1
Pillow==10.4.0
tqdm>=4.67

Licensing and Compliance

Model & Code License: MIT

Framework Licenses: TensorFlow (Apache-2.0), LightGBM (MIT)

Parameter Limit: Under 8 billion total parameters

External Data: None used — strictly trained on the provided dataset only

Conclusion

This project demonstrates a robust and efficient multimodal regression framework for product price prediction.
By combining TF-IDF text embeddings and EfficientNet-B0 image features within a LightGBM model, we achieved a SMAPE of approximately 50% on unseen test data — all while maintaining lightweight architecture, interpretability, and reproducibility.

Back to top
