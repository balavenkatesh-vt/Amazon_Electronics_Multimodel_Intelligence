import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def prepare_splits_and_preprocessor():
    data_path = "data/cleaned_multimodal_data.csv"
    if not os.path.exists(data_path):
        print("❌ Cleaned dataset not found!")
        return

    print("📁 Loading Cleaned Dataset...")
    df = pd.read_csv(data_path)

    feature_cols = ['name', 'custom_sub_category', 'ratings', 'no_of_ratings', 'actual_price', 'image']
    target_col = 'discount_price'

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    os.makedirs("data/splits", exist_ok=True)
    X_train.to_csv("data/splits/X_train.csv", index=False)
    X_test.to_csv("data/splits/X_test.csv", index=False)
    y_train.to_csv("data/splits/y_train.csv", index=False)
    y_test.to_csv("data/splits/y_test.csv", index=False)

    # Preprocessor with OneHotEncoder for custom_sub_category
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(max_features=500, stop_words='english'), 'name'),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['custom_sub_category']),
            ('num', StandardScaler(), ['ratings', 'no_of_ratings', 'actual_price'])
        ]
    )

    print("⚙️ Fitting Preprocessor on X_train with Custom Sub-Categories...")
    preprocessor.fit(X_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(preprocessor, "models/preprocessor.pkl")
    print("💾 Preprocessor saved to 'models/preprocessor.pkl'")

if __name__ == "__main__":
    prepare_splits_and_preprocessor()