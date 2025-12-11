import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# -----------------------------
# 1. Load Dataset
# -----------------------------
DATA_PATH = "data/boston_housing.csv"   # Adjust path if needed

print(f"Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)


# -----------------------------
# 2. Prepare Features & Target
# -----------------------------
X = df.drop(columns=["MEDV"])
y = df["MEDV"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -----------------------------
# 3. Helper Function for Evaluation
# -----------------------------
def evaluate_model(name, model):
    """Return evaluation metrics for a fitted model."""
    y_pred_test = model.predict(X_test)

    return {
        "Model": name,
        "Test R2": r2_score(y_test, y_pred_test),
        "Test RMSE": mean_squared_error(y_test, y_pred_test, squared=False),
        "Test MAE": mean_absolute_error(y_test, y_pred_test)
    }


results = []
models = {}   # store trained model objects


# -----------------------------
# 4. Linear Regression
# -----------------------------
print("\nTraining Linear Regression...")
lin = LinearRegression()
lin.fit(X_train, y_train)
results.append(evaluate_model("Linear Regression", lin))
models["Linear Regression"] = lin


# -----------------------------
# 5. Ridge Regression
# -----------------------------
print("Training Ridge Regression (GridSearch)...")
ridge_params = {"alpha": [0.1, 1, 10, 100]}
ridge = Ridge()

ridge_cv = GridSearchCV(
    ridge, ridge_params, cv=5,
    scoring="neg_root_mean_squared_error"
)
ridge_cv.fit(X_train, y_train)

results.append(evaluate_model("Ridge Regression", ridge_cv.best_estimator_))
models["Ridge Regression"] = ridge_cv.best_estimator_


# -----------------------------
# 6. Lasso Regression
# -----------------------------
print("Training Lasso Regression (GridSearch)...")
lasso_params = {"alpha": [0.001, 0.01, 0.1, 1]}
lasso = Lasso(max_iter=10000)

lasso_cv = GridSearchCV(
    lasso, lasso_params, cv=5,
    scoring="neg_root_mean_squared_error"
)
lasso_cv.fit(X_train, y_train)

results.append(evaluate_model("Lasso Regression", lasso_cv.best_estimator_))
models["Lasso Regression"] = lasso_cv.best_estimator_


# -----------------------------
# 7. Random Forest Regressor
# -----------------------------
print("Training Random Forest Regressor...")
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

results.append(evaluate_model("Random Forest", rf))
models["Random Forest"] = rf


# -----------------------------
# 8. Compare Models
# -----------------------------
results_df = pd.DataFrame(results)
print("\nModel Comparison:")
print(results_df)


# -----------------------------
# 9. Select Best Model
# -----------------------------
best_model_name = results_df.sort_values("Test RMSE").iloc[0]["Model"]
best_model = models[best_model_name]

print(f"\nBest model selected: {best_model_name}")


# -----------------------------
# 10. Save the Model
# -----------------------------
os.makedirs("models", exist_ok=True)
MODEL_PATH = "models/house_price_model.pkl"

joblib.dump(best_model, MODEL_PATH)
print(f"Saved best model to: {MODEL_PATH}")
