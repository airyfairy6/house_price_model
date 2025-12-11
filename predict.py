import joblib
import numpy as np
import pandas as pd
import argparse


# -----------------------------
# Load trained model
# -----------------------------
MODEL_PATH = "models/house_price_model.pkl"

print(f"Loading model from: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)


# -----------------------------
# Column order required by the model
# -----------------------------
FEATURES = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"
]


# -----------------------------
# Predict function
# -----------------------------
def predict_from_args(values):
    """
    Accepts a list of 13 numerical feature values.
    Returns a predicted house price (MEDV).
    """
    arr = np.array(values).reshape(1, -1)
    pred = model.predict(arr)[0]
    return pred


# -----------------------------
# CLI (Command-Line Interface)
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict house price using trained Random Forest model"
    )

    parser.add_argument(
        "--values",
        nargs=13,
        type=float,
        required=True,
        metavar="V",
        help="13 numerical feature values in correct order"
    )

    args = parser.parse_args()
    values = args.values

    print("\nInput features (in order):")
    for f, v in zip(FEATURES, values):
        print(f"{f}: {v}")

    prediction = predict_from_args(values)

    print("\nPredicted MEDV (house price in $1000's):")
    print(f"➡ {prediction:.2f}")
