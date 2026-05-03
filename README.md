# Bank Marketing Subscription Prediction

A supervised machine learning project comparing KNN and Deep Neural Network 
classifiers to predict customer subscription to term deposits, using the UCI 
Bank Marketing dataset.

## Overview

This project investigates how effectively classification algorithms can predict 
customer subscription to a term deposit, and how these predictions can optimise 
bank marketing efficiency.

**Research Question:** How effectively can classification algorithms predict 
customer subscription to a term deposit, and how can these predictions optimise 
marketing efficiency?

## Dataset

- **Source:** UCI Machine Learning Repository — Bank Marketing Dataset 
  (Moro et al., 2014)
- **Size:** 41,188 records, 21 features
- **Target variable:** Whether a customer subscribed to a term deposit (yes/no)
- **Class imbalance:** No: 36,548 | Yes: 4,640

## Methodology

1. **EDA** — class distribution analysis, subscription rates by month and 
   occupation, economic indicator exploration
2. **Preprocessing** — one-hot encoding (11 categorical columns → 53 features), 
   stratified 80/20 train-test split, StandardScaler on training data
3. **Imbalance handling** — SMOTE applied to training set 
   (balanced to 29,238 per class)
4. **Model 1: KNN** — GridSearchCV hyperparameter tuning (k = {2,3,4,5}), 
   F1-score optimised, best k = 2
5. **Model 2: Keras Neural Network** — 4-layer sequential model 
   (32→16→8→1 neurons), ReLU activations, Sigmoid output, 
   Adam optimiser, 30 epochs, class weights {0:1, 1:4}

## Results

| Model | Accuracy | Minority Class F1 |
|-------|----------|-------------------|
| KNN (k=2) | 87.5% | 0.44 |
| Neural Network | 89% | 0.00 |

**Key finding:** High overall accuracy is misleading in imbalanced datasets. 
KNN with SMOTE showed improved minority class detection; the neural network 
overfit to the majority class despite class weighting.

## Business Insight

Models based purely on accuracy would miss the majority of potential subscribers. 
Recommended next steps include focal loss, cost-sensitive learning, and SMOTE 
within cross-validation pipelines to improve minority class detection and 
campaign ROI.

## Tools and Libraries

- Python 3
- scikit-learn (KNN, GridSearchCV, SMOTE, StandardScaler, OneHotEncoder)
- TensorFlow / Keras (Neural Network)
- imbalanced-learn (SMOTE)
- pandas, numpy
- matplotlib, seaborn

## Files

- `bank_marketing_classification.ipynb` — full analysis notebook
- `requirements.txt` — Python dependencies

## How to Run

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download the dataset from the 
   [UCI Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing)
4. Open the notebook in Jupyter or Google Colab
5. Run all cells sequentially

## Author

Amarachukwu Iku
