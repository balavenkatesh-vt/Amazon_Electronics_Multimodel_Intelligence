import pandas as pd
import numpy as np
import os
import joblib

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor

def train_baseline_xgboost():
    # 1. Load Data Splits & Preprocessor
    train_x_path = "data/splits/X_train.csv"
    test_x_path = "data/splits/X_test.csv"
    train_y_path = "data/splits/y_train.csv"
    test_y_path = "data/splits/y_test.csv"
    preprocessor_path = "models/preprocessor.pkl"

    if not os.path.exists(train_x_path) or not os.path.exists(preprocessor_path):
        print("❌ Data splits or Preprocessor not found! Complete Phase 3 first.")
        return

    print("📁 Loading Data Splits and Preprocessor...")
    X_train = pd.read_csv(train_x_path)
    X_test = pd.read_csv(test_x_path)
    y_train = pd.read_csv(train_y_path).values.ravel()
    y_test = pd.read_csv(test_y_path).values.ravel()

    preprocessor = joblib.load(preprocessor_path)

    # 2. Transform Features
    print("🧹 Transforming X_train and X_test using fitted preprocessor...")
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # 3. Model Definition & Training
    print("🚀 Training Baseline XGBoost Regressor Model...")
    model = XGBRegressor(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_transformed, y_train)

    # 4. Model Predictions & Evaluation
    print("Evaluating Model Performance on Unseen Test Data...")
    y_pred = model.predict(X_test_transformed)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n=================== BASELINE MODEL PERFORMANCE ===================")
    print(f"Root Mean Squared Error (RMSE) : ₹{rmse:.2f}")
    print(f"Mean Absolute Error (MAE)      : ₹{mae:.2f}")
    print(f"R2 Score (Accuracy Metric)    : {r2:.4f} ({r2*100:.2f}%)")

    # 5. Save Trained Model
    os.makedirs("models", exist_ok=True)
    model_save_path = "models/xgboost_baseline_model.pkl"
    joblib.dump(model, model_save_path)
    print(f"\nBaseline Model successfully saved to: {model_save_path}")

if __name__ == "__main__":
    train_baseline_xgboost()