# Smart-Product-Pricing-Solution

🏷️ Problem Statement

Determining the optimal price of a product is a critical task for e-commerce platforms. Product pricing depends on multiple factors including textual descriptions, specifications, quantity, and visual appearance.

The goal of this challenge is to build a machine learning model that predicts the price of a product using both its catalog content (text) and product image.

The evaluation metric for this task is Symmetric Mean Absolute Percentage Error (SMAPE), which measures the relative difference between predicted and actual prices. Lower SMAPE values indicate better performance.

📦 Dataset Description

The dataset consists of training and test files with the following structure:

Column	Description
sample_id	Unique identifier for each product
catalog_content	Text field containing product title, description, and Item Pack Quantity (IPQ)
image_link	Public URL of the product image
price	Target variable (available only in training data)
Files

train.csv — 75,000 labeled products

test.csv — 75,000 unlabeled products

sample_test_out.csv — Sample output format

Output Format

Your final submission file must have:

sample_id,price


with all test IDs and predicted float prices (positive values only).

⚙️ Methodology
1. Overview

We designed a multimodal learning pipeline combining text features and image embeddings.
A single LightGBM regressor learns from the concatenated feature set to predict prices efficiently and accurately.

2. Data Preprocessing

Text Cleaning: lowercasing, whitespace normalization.

Image Downloading: parallel download with retries, caching locally to /content/image_cache/.

Missing Handling: invalid URLs or unreadable images replaced with zero-vector embeddings.

3. Feature Engineering
📝 Text Features

Extracted with TfidfVectorizer(ngram_range=(1,2), min_df=3, max_features=150_000)

Captures unigrams and bigrams to represent product names, attributes, and quantities.

🖼️ Image Features

Pretrained EfficientNet-B0 used as frozen feature extractor (include_top=False, pooling='avg', weights='imagenet').

Each image produces a 1280-dimensional embedding.

Images resized to 224×224 and processed in batches of 64.

🔗 Feature Fusion

Text TF-IDF (sparse) and image embeddings (dense) concatenated using scipy.sparse.hstack.

Resulting multimodal feature space used for LightGBM regression.

4. Model Training

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

85% training, 15% validation split

Early stopping after 200 rounds

Predictions clipped to positive values (>= 0.01)

Model Architecture
[text: catalog_content] → normalize → TF-IDF (1–2 grams, 150k features)
                                               \
                                                +→ concatenate → LightGBM → price
                                               /
[image: image_link] → download/cache → EfficientNet-B0 (avg-pool, frozen) → 1280-D embedding


This late-fusion design allows efficient combination of linguistic and visual signals without requiring end-to-end neural training.

Results
Metric	Validation	Test
MAE	11.4555	—
SMAPE (%)	52.63	50.25

Key Findings:

The hybrid model significantly outperformed text-only baselines.

Image embeddings improved performance even with a frozen backbone.

MAE objective offered smoother convergence and stable SMAPE reduction.

Insights & Discussion

TF-IDF text features capture brand and quantity signals effectively.

EfficientNet embeddings enhance visual understanding (e.g., size, packaging).

LightGBM fusion balances interpretability, speed, and generalization.

Clipping & normalization improved metric stability.

Potential Improvements:

Try log-price regression to handle skewed distributions.

Ensemble models over different TF-IDF vocabularies.

Add explicit has_image indicator or small CNN fine-tuning.

Explore CLIP/Vision Transformer embeddings for stronger visual semantics.

Implementation Details

Language: Python 3.12

Frameworks: TensorFlow 2.17.1, LightGBM 4.5.0, scikit-learn 1.5.2

Environment: Google Colab

Paths:

Dataset: /content/drive/MyDrive/ARJUN/Exp/exp-28/dataset/

Cache: /content/image_cache/

Output: test_out.csv

Key Dependencies
numpy==2.1.3
pandas==2.2.3
scipy==1.13.1
scikit-learn==1.5.2
lightgbm==4.5.0
tensorflow==2.17.1
Pillow==10.4.0
tqdm>=4.67

#Licensing & Compliance

Model & Code License: MIT

Frameworks: TensorFlow (Apache-2.0), LightGBM (MIT)

Parameter Limit: Under 8 billion total parameters

External Data: None used — strictly trained on provided dataset only

🏁 Conclusion

This project demonstrates a simple yet powerful multimodal regression framework for price prediction.
By combining TF-IDF text embeddings and EfficientNet-B0 image features within a LightGBM model, we achieved a SMAPE of ~50% on unseen test data — all while staying lightweight, interpretable, and reproducible.
