import pandas as pd
import numpy as np
import os

def clean_and_preprocess_data():
    data_path = "data/electronics_product.csv"
    csv_files = [f for f in os.listdir("data") if f.endswith('.csv')]
    if csv_files:
        data_path = os.path.join("data", csv_files[0])

    print(f"📁 Loading dataset from: {data_path}...")
    df = pd.read_csv(data_path)

    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

    def clean_currency(val):
        if pd.isna(val):
            return np.nan
        val_str = str(val).replace('₹', '').replace(',', '').strip()
        try:
            return float(val_str)
        except ValueError:
            return np.nan

    def clean_number(val):
        if pd.isna(val):
            return np.nan
        val_str = str(val).replace(',', '').strip()
        try:
            return float(val_str)
        except ValueError:
            return np.nan

    print("🧹 Cleaning Currency and Numerical columns...")
    df['discount_price'] = df['discount_price'].apply(clean_currency)
    df['actual_price'] = df['actual_price'].apply(clean_currency)
    df['no_of_ratings'] = df['no_of_ratings'].apply(clean_number)
    df['ratings'] = pd.to_numeric(df['ratings'], errors='coerce')

    # 💡 1. Rule-Based Custom Sub-Category Extraction
    print("🏷️ Extracting Custom Sub-Categories from Product Titles...")
    def extract_sub_category(name):
        name = str(name).lower()
        if any(k in name for k in ['phone', 'mobile', '5g', 'redmi', 'oneplus', 'iphone', 'samsung', 'realme', 'pro', 'pro max']):
            return 'Smartphones'
        elif any(k in name for k in ['earbuds', 'headphone', 'earphone', 'bluetooth', 'bullets', 'wireless ear', 'speaker', 'audio', 'soundbar']):
            return 'Audio'
        elif any(k in name for k in ['watch', 'smartwatch', 'band', 'tracker']):
            return 'Wearables'
        elif any(k in name for k in ['tv', 'television', 'led tv', 'android tv', 'monitor']):
            return 'TV'
        else:
            return 'Accessories'

    df['custom_sub_category'] = df['name'].apply(extract_sub_category)
    print("📌 Created Categories Distribution:")
    print(df['custom_sub_category'].value_counts())

    # 🩹 2. Category-Wise Median Imputation
    print("🩹 Imputing Missing Values using Custom Category Medians...")
    df['ratings'] = df['ratings'].fillna(df['ratings'].median())
    df['no_of_ratings'] = df['no_of_ratings'].fillna(df['no_of_ratings'].median())
    
    # Custom Category wise Median Imputation
    df['actual_price'] = df.groupby('custom_sub_category')['actual_price'].transform(lambda x: x.fillna(x.median()))
    df['actual_price'] = df['actual_price'].fillna(df['actual_price'].median())
    df['discount_price'] = df['discount_price'].fillna(df['actual_price'])

    df = df.dropna(subset=['discount_price', 'actual_price'])

    output_path = "data/cleaned_multimodal_data.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✨ Cleaned dataset with Custom Categories saved to: {output_path}")

if __name__ == "__main__":
    clean_and_preprocess_data()