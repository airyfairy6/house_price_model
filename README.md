<p align="center">
  <img src="assets/banner.png" width="100%">
</p>

# Predicting House Prices — Regression Modeling & Feature Selection

This project builds and compares several regression models to predict **median house prices (MEDV)**  
using a structured housing dataset sourced from the IBM Developer Skills Network.  

The focus is on **model performance**, **feature selection**, and **explainability** — key skills for  
Model Developer and ML Engineering roles.

---

## 📁 Data Source

Dataset URL:
https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ST0151EN-SkillsNetwork/labs/boston_housing.csv


This dataset contains **506 samples**, **14 predictive features**, and 1 target variable (**MEDV** – median home value  
in $1,000s). It is a variant of the classic **Boston Housing dataset** used in regression modeling.

---

## 📊 Features

| Column | Description |
|--------|-------------|
| CRIM | Per capita crime rate by town |
| ZN | Land zoned for large lots |
| INDUS | Proportion of industrial acres |
| CHAS | Charles River dummy variable |
| NOX | Nitric oxide concentration |
| RM | Average number of rooms |
| AGE | Proportion of older buildings |
| DIS | Distance to employment centers |
| RAD | Highway accessibility index |
| TAX | Property-tax rate |
| PTRATIO | Pupil–teacher ratio |
| B | Demographic index |
| LSTAT | Lower-status population % |
| **MEDV** | *Target* — Median home value |

---

## 🎯 Project Objectives

1. Build regression models with increasing complexity:
   - **Linear Regression**
   - **Ridge Regression** (L2 regularization)
   - **Lasso Regression** (L1 regularization)
   - **Random Forest Regressor** (non-linear ensemble)

2. Perform **feature selection** via:
   - Correlation analysis
   - Lasso shrinkage
   - Random Forest feature importance

3. Evaluate models using:
   - **RMSE**
   - **MAE**
   - **R² (coefficient of determination)**

4. Save the final selected model as a reusable artifact (`.pkl`).

---

## 🧠 Modeling Approach

### 1️⃣ Exploratory Data Analysis
- Distribution analysis  
- Correlation heatmap  
- Numeric summary statistics

### 2️⃣ Train/Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



