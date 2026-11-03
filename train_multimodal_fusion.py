import pandas as pd
import numpy as np
import os
import joblib

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor

def train_multimodal_fusion():
    print("📁 Loading Data Splits, Preprocessor, and Embeddings...")
    
    X_train = pd.read_csv("data/splits/X_train.csv")
    X_test = pd.read_csv("data/splits/X_test.csv")
    y_train = pd.read_csv("data/splits/y_train.csv").values.ravel()
    y_test = pd.read_csv("data/splits/y_test.csv").values.ravel()

    # 1. Transform Tabular Features
    preprocessor = joblib.load("models/preprocessor.pkl")
    X_train_tab = preprocessor.transform(X_train)
    X_test_tab = preprocessor.transform(X_test)

    if hasattr(X_train_tab, "toarray"):
        X_train_tab = X_train_tab.toarray()
    if hasattr(X_test_tab, "toarray"):
        X_test_tab = X_test_tab.toarray()

    # 2. Load Embeddings
    train_bert = np.load("data/embeddings/train_bert.npy")
    test_bert = np.load("data/embeddings/test_bert.npy")
    train_resnet = np.load("data/embeddings/train_resnet.npy")
    test_resnet = np.load("data/embeddings/test_resnet.npy")

    # 3. Multimodal Late Fusion Concatenation
    print("🔗 Concatenating Tabular + BERT + ResNet Features...")
    X_train_fused = np.hstack([X_train_tab, train_bert, train_resnet])
    X_test_fused = np.hstack([X_test_tab, test_bert, test_resnet])

    # 4. Train Multimodal Model
    print("🚀 Retraining Multimodal XGBoost Model on Uncapped Real Prices...")
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_fused, y_train)

    # 5. Evaluate
    y_pred = model.predict(X_test_fused)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n=================== UNCAPPED MODEL PERFORMANCE ===================")
    print(f"📉 Root Mean Squared Error (RMSE) : ₹{rmse:.2f}")
    print(f"📉 Mean Absolute Error (MAE)      : ₹{mae:.2f}")
    print(f"🎯 R2 Score (Accuracy Metric)    : {r2:.4f} ({r2*100:.2f}%)")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/multimodal_fusion_model.pkl")
    print("\n💾 Uncapped Model successfully saved to: models/multimodal_fusion_model.pkl")

if __name__ == "__main__":
    train_multimodal_fusion()