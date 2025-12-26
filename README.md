# Diabetes Prediction System (Multi-Stage Machine Learning)

> An end-to-end machine learning project that predicts **diabetes risk**, **diabetes presence**, and **diabetes type** using a **multi-stage modeling pipeline**, robust preprocessing, and hyperparameter optimization.

 **Live App (Streamlit)**: *https://diabetes-prediction-system-6321.streamlit.app/*

---

##  Problem Statement
Early and accurate diabetes prediction is critical for preventive healthcare.  
This project addresses the problem in **three stages**:

1. **Estimate diabetes risk score** (Regression)
2. **Predict whether a patient has diabetes** (Binary Classification)
3. **Identify the type of diabetes** (Multi-Class Classification)

Each stage is modeled separately while sharing information through carefully designed pipelines.

---

##  Dataset
- **Source**: CSV file (loaded from Google Drive)
- **File**: `diabetes_data.csv`
- **Features include**:
  - Demographic attributes
  - Clinical indicators
  - Lifestyle-related variables

Different target variables are derived for each modeling stage.

---

##  Modeling Approach

###  Stage 1: Diabetes Risk Score (Regression)
- Model: **Gradient Boosting Regressor**
- Output: Continuous diabetes risk score

###  Stage 2: Diabetes Prediction (Binary Classification)
- Models evaluated:
  - Logistic Regression
  - Random Forest
  - Decision Tree
  - XGBoost
- Best-performing model selected using cross-validation

###  Stage 3: Diabetes Type Prediction (Multi-Class)
- Uses:
  - Original features
  - Out-of-Fold (OOF) predictions from previous stages
- Prevents data leakage and improves generalization

---

##  Machine Learning Pipeline
- Modular preprocessing pipeline built using `scikit-learn`
- Handles:
  - Missing value treatment
  - Categorical encoding
  - Feature scaling
- Fully reproducible and deployment-ready

---

##  Hyperparameter Optimization
- **Optuna** used for tuning all major models
- Separate optimization studies for:
  - Risk score regression
  - Binary classification
  - Multi-class classification
- Includes Optuna visualization and analysis

---

##  Evaluation Metrics
- **Regression**:
  - Mean Squared Error (MSE)
  - R² Score
- **Classification**:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- Feature importance analysis for interpretability

---

##  Deployment
The final trained pipelines are integrated into a **Streamlit web application** that allows users to:
- Input patient data
- Receive predictions for:
  - Diabetes risk
  - Diabetes presence
  - Diabetes type

 **Streamlit App**: *https://diabetes-prediction-system-6321.streamlit.app/*

---

##  Tech Stack
- Python
- Pandas, NumPy
- scikit-learn
- XGBoost
- Optuna
- Streamlit

---




