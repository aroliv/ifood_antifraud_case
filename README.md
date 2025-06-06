# iFood Fraud Detection Case Study

## 🌐 Overview

This repository contains the code and presentation slides for the iFood Fraud Detection technical assessment. The goal is to develop a machine learning model capable of identifying fraudulent credit card transactions, enhancing the financial security of the iFood platform.

With the growing volume and diversity of transactions, fraud detection has become essential to protect users and preserve trust. This project uses historical labeled data to train and evaluate models for fraud classification.

---

## 📂 Contents

- **Notebook**: Contains the entire data science pipeline, including data exploration, preprocessing with PySpark, model training using scikit-learn and ensemble methods, and evaluation.
- **Presentation (PDF)**: Summarizes the business challenge, data-driven approach, results, and final recommendations.

---

## 🗂 Dataset

- **Source**: Kaggle - *Fraud Detection Dataset*
- **Files**:
  - `fraudTrain.csv`: Labeled dataset used for training and tuning the model.
  - `fraudTest.csv`: Dataset used exclusively for final model evaluation.

The target variable is `is_fraud`, where `1` indicates a fraudulent transaction and `0` indicates a legitimate one.

---

## 🔧 Tools and Libraries

- **Languages & Engines**: Python, PySpark
- **Main Libraries**:
  - Data Handling: `pandas`, `pyspark.sql`, `numpy`
  - Machine Learning: `scikit-learn`, `xgboost`, `lightgbm`, `imbalanced-learn`
  - Evaluation: `shap`, `statsmodels`, `sklearn.metrics`
  - Visualization: `matplotlib`, `seaborn`, `plotly`

---

## 🔍 Workflow

1. **Data Exploration**  
   Understand the structure, distributions, and potential data quality issues using PySpark and visualization tools.

2. **Data Preprocessing**  
   Perform data cleaning, feature engineering, encoding, and normalization. Handle class imbalance using SMOTE and undersampling techniques.

3. **Modeling**  
   Train various classification models (e.g., Random Forest, XGBoost, LightGBM) using the training dataset. Tune hyperparameters via cross-validation.

4. **Evaluation**  
   Use the test dataset to evaluate model performance with metrics suited for imbalanced classification:
   - ROC AUC
   - F1 Score
   - Precision / Recall
   - Confusion Matrix

5. **Interpretation**  
   Leverage SHAP values and feature importance to interpret model behavior and understand which features influence predictions.

6. **Reporting**  
   Deliver key findings and actionable recommendations based on model results and transaction behavior analysis.

---

## 📊 Results and Insights

The final model demonstrates strong performance in detecting fraudulent activity with an emphasis on high recall and balanced precision. Feature interpretation revealed important behavioral patterns useful for real-time fraud prevention.

---

## 📎 Presentation

A slide deck (PDF) summarizes the end-to-end process, including:
- Business problem
- Data insights
- Modeling approach
- Evaluation results
- Strategic recommendations

---

## 🔮 Future Work

- Explore real-time fraud detection using streaming data (e.g., Spark Structured Streaming).
- Incorporate time-based features and session-level behavior patterns.
- Test deep learning models for sequence-based fraud detection.
- Continuously monitor and retrain the model with new data to adapt to evolving fraud tactics.

---

## 👤 Contributor

**Andressa Ribeiro de Oliveira**
