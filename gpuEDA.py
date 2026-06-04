# %%
import pandas as pd
import matplotlib.pyplot as plt

## 1. Loading the dataset (GPU_benchmarks_v7) 
## by Alanjo https://www.kaggle.com/datasets/alanjo/gpu-benchmarks

print("////////////////////////////////////////////////////////")
datasetname = 'GPU Benchmarks'
print(f"--- LOADING DATA SET: {datasetname} ---")
df = pd.read_csv("C:/Users/luis_/Desktop/GPU_benchmarks_v7.csv")
print(f"--- Shape of the dataset: {df.shape} ---")
print(f"--- Dataset size: {df.shape[0]} GPUs ---")
print("////////////////////////////////////////////////////////")
print(f"--- Dataset column names: {df.columns.array} ---")


# %%
print("///////////////////////////////////////////////////////")

print("--- DATA CLEANING ---")
print("--- Missing values audit ---")

# isna().sum() counts the number of missing values in each row
missing_data = df.isna().sum()
print(missing_data)

# Calculate the percentage of missing data
total_rows = len(df)
print("\n--- Percentage Missing ---")
print((missing_data / total_rows * 100).round(2).astype(str) + '%')
print("///////////////////////////////////////////////////////")


# %% 
print("///////////////////////////////////////////////////////")
# Now we drop any row that is missing a 'price' or 'TDP'
df_clean = df.dropna(subset=['price', 'TDP'])

print(f"Cleaned dataset size: {df_clean.shape[0]} GPUs")

# Verify the damage is gone
print("\n--- Missing Values Audit (Post-Cleanup) ---")
print(df_clean[['price', 'G3Dmark', 'TDP']].isna().sum())


# --- Outlier Detection ---
print("--- Top 5 Most Expensive GPUs ---")
# .sort_values() orders the data. ascending=False puts the highest numbers at the top.
most_expensive = df_clean.sort_values(by='price', ascending=False).head(5)

# We only print the columns we care about right now
print(most_expensive[['gpuName', 'price', 'G3Dmark', 'category']])

print("\n--- Top 5 Highest Performing GPUs ---")
best_performers = df_clean.sort_values(by='G3Dmark', ascending=False).head(5)
print(best_performers[['gpuName', 'price', 'G3Dmark', 'category']])



# %%

## 2. Understading the pricing and performance
print("--- Hardware and performance ---")
## Filtering to just a few useful numeric columns:

print(df_clean[['price', 'G3Dmark', 'TDP']].describe())
print(f"--- We see that the price mean ({df_clean['price'].mean().round(2)}) is very far from the median ({df_clean['price'].median()})) ---")


# %%
# 3. Visualizing Performance vs. Price
plt.figure(figsize=(10, 6))
plt.scatter(df_clean['price'], df_clean['G3Dmark'], alpha=0.5, color='black')

plt.title('GPU Performance (G3Dmark) vs. Price')
plt.xlabel('Performance Score (G3Dmark)')
plt.ylabel('Price ($)')
plt.grid(True, linestyle='--', alpha=0.6)

plt.show()