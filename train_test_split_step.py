import pandas as pd
import os
from sklearn.model_selection import train_test_split

def perform_train_test_split():
    data_path = "data/cleaned_multimodal_data.csv"
    if not os.path.exists(data_path):
        print("❌ Cleaned dataset not found!")
        return

    df = pd.read_csv(data_path)
    
    # Selecting useful features only (Dropped zero-variance main_category & sub_category)
    X = df[['name', 'ratings', 'no_of_ratings', 'actual_price', 'discount_percentage', 'image']]
    y = df['discount_price_capped']

    # 80-20 Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save splits to prevent Data Leakage
    os.makedirs("data/splits", exist_ok=True)
    X_train.to_csv("data/splits/X_train.csv", index=False)
    X_test.to_csv("data/splits/X_test.csv", index=False)
    y_train.to_csv("data/splits/y_train.csv", index=False)
    y_test.to_csv("data/splits/y_test.csv", index=False)

    print("=================== TRAIN-TEST SPLIT SUMMARY ===================")
    print(f"Total Dataset Shape : {df.shape}")
    print(f"X_train (80% Train) : {X_train.shape}")
    print(f"X_test  (20% Test)  : {X_test.shape}")
    print(f"y_train Shape       : {y_train.shape}")
    print(f"y_test Shape        : {y_test.shape}")
    print("\n✅ Cleaned feature splits successfully saved in 'data/splits/' directory!")

if __name__ == "__main__":
    perform_train_test_split()