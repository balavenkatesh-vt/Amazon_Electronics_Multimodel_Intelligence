import pandas as pd
import numpy as np
import os
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

def fit_and_save_preprocessor():
    train_path = "data/splits/X_train.csv"
    test_path = "data/splits/X_test.csv"
    
    if not os.path.exists(train_path):
        print("❌ Train split files not found! Run train_test_split_step.py first.")
        return

    print("📁 Loading Train and Test Splits...")
    X_train = pd.read_csv(train_path)
    X_test = pd.read_csv(test_path)

    text_features = 'name'
    numerical_features = ['ratings', 'no_of_ratings', 'actual_price', 'discount_percentage']

    # 1. Define ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(max_features=500, stop_words='english'), text_features),
            ('num', StandardScaler(), numerical_features)
        ],
        remainder='drop' # Image column handles separately in Vision pipeline
    )

    # 2. Fit ONLY on X_train (Prevents Data Leakage)
    print("🧹 Fitting Preprocessor on Training Data ONLY...")
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    print("=================== TRANSFORMATION SUMMARY ===================")
    print(f"Transformed X_train Matrix Shape : {X_train_transformed.shape}")
    print(f"Transformed X_test Matrix Shape  : {X_test_transformed.shape}")

    # 3. Save preprocessor artifact for Streamlit inference
    os.makedirs("models", exist_ok=True)
    preprocessor_path = "models/preprocessor.pkl"
    joblib.dump(preprocessor, preprocessor_path)
    print(f"\n💾 Preprocessor Pipeline saved to: {preprocessor_path}")

if __name__ == "__main__":
    fit_and_save_preprocessor()