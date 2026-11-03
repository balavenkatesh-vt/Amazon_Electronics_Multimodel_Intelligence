import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_advanced_eda():
    data_path = "data/cleaned_multimodal_data.csv"
    if not os.path.exists(data_path):
        print("Cleaned dataset not found! Run eda_preprocessing.py first.")
        return

    df = pd.read_csv(data_path)
    print("zCleaned Dataset Loaded Successfully for Advanced EDA!\n")

    # 1. Summary Statistics
    print("=================== NUMERICAL SUMMARY ===================")
    print(df[['ratings', 'no_of_ratings', 'actual_price', 'discount_price', 'discount_price_capped', 'discount_percentage']].describe())

    # 2. Categorical Distribution
    print("\n=================== TOP 5 SUB-CATEGORIES ===================")
    print(df['sub_category'].value_counts().head(5))

    # Save EDA Plots Folder
    os.makedirs("eda_plots", exist_ok=True)

    # Plot 1: Target Variable Distribution (Discount Price vs Capped Price)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(df['discount_price'], kde=True, color='red', bins=30)
    plt.title("Original Discount Price (Skewed)")

    plt.subplot(1, 2, 2)
    sns.histplot(df['discount_price_capped'], kde=True, color='green', bins=30)
    plt.title("Capped Discount Price (Winsorized)")
    plt.tight_layout()
    plt.savefig("eda_plots/target_distribution.png")
    plt.close()

    # Plot 2: Correlation Heatmap
    plt.figure(figsize=(8, 6))
    numeric_df = df[['ratings', 'no_of_ratings', 'actual_price', 'discount_price_capped', 'discount_percentage']]
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("eda_plots/correlation_heatmap.png")
    plt.close()

    print("\n✅ Advanced EDA Plots saved successfully in 'eda_plots/' directory!")

if __name__ == "__main__":
    run_advanced_eda()