# MSDS 600: Week 5 DS Automation Assignment
## Overview
This repository contains the complete implementation for the **Week 5 DS Automation Assignment**,

1. **AutoML Model Selection with PyCaret**: Trained and evaluated 14 machine learning algorithms on prepared churn data (`cleaned_churn_data.csv`).
2. **Metric Justification (AUC)**: Evaluated models using ROC-AUC (Area Under the Curve) rather than default accuracy to overcome the class imbalance (~26.6% churn rate). Top model: **Logistic Regression** (AUC = 0.8400), closely followed by **Gradient Boosting** (AUC = 0.8366).
3. **Saved Pipeline**: Serialized the complete model preprocessing and inference pipeline to disk (`pycaret_churn_model.pkl`).
4. **Production Python Module (`predict_churn.py`)**: Standalone module and CLI tool for predicting churn on new customer records.
5. **Testing & Validation**: Evaluated on `new_churn_data.csv` and `new_unmodified_churn_data.csv`, achieving **100% accuracy (5/5)** matching the ground truth `[1, 0, 0, 1, 0]`.

---

## All Additional Challenges Completed

| # | Challenge | Implementation Details |
|---|---|---|
| **1** | **Churn Probability & Percentile Ranking** | Computes the probability of churn (0.0 to 1.0) and uses `scipy.stats.percentileofscore` against the historical training distribution (`train_churn_probabilities.csv`) to provide the exact risk percentile (e.g. 99.8th percentile). |
| **2** | **Comparison with Other AutoML Packages (H2O)** | Integrated **H2O AutoML** within the notebook; compared top Stacked Ensembles (AUC = 0.8419) against PyCaret with a detailed architectural trade-off analysis. |
| **3** | **Object-Oriented Class Design** | Built the `ChurnPredictor` class in `predict_churn.py` encapsulating model loading, preprocessing, scoring, and percentile estimation. |
| **4** | **Interactive / CLI User Input** | Supports command-line file arguments (`sys.argv[1]`) as well as interactive console prompt (`input()`) if no file argument is passed. |
| **5** | **Automatic Preprocessing of Raw Unmodified Data** | Implemented automated preprocessing in `predict_churn.py` that handles `TotalCharges` numeric coercion, `charge_per_tenure` feature engineering, and categorical encoding for both prepared data and raw unmodified data (`new_unmodified_churn_data.csv`). |



