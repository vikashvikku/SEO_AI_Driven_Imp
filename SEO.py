import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import os

# Number of samples per formula
num_samples = 20000

# List of formulas
formulas = [
    "MDP", "Q-Learning", "Bellman Optimality", "Temporal Difference", "Policy Gradient",
    "CTR", "PageRank", "DCG", "NDCG", "Cosine Similarity",
    "TF-IDF", "RankBrain", "Softmax Keyword Selection", "Multi-Armed Bandit", "Deep Q-Learning",
    "Logistic Regression", "F-Score", "Keyword Probability", "Reinforcement Learning Loss", "Hybrid Model 1",
    "Hybrid Model 2", "Hybrid Model 3", "Hybrid Model 4", "Hybrid Model 5", "Hybrid Model 6",
    "Hybrid Model 7", "Hybrid Model 8", "Hybrid Model 9", "Hybrid Model 10"
]

# Load dataset if available
dataset_path = "formula_dataset.csv"
use_dataset = os.path.exists(dataset_path)

df = None
if use_dataset:
    df = pd.read_csv(dataset_path)
    print("✅ Dataset loaded successfully.")
    # Ensure all values are numeric except for the 'Formula' column
    df.iloc[:, :-1] = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')
    df.dropna(inplace=True)  # Drop any rows with NaN values
else:
    print("⚠️ No dataset found. Using synthetic data instead.")

# Generate synthetic data
def generate_data(formula_name):
    np.random.seed(42)
    X = np.random.rand(num_samples, 5).astype(np.float32)  # 5 random features
    
    if formula_name == "MDP":
        y = np.max(X, axis=1) + 0.9 * np.sum(X, axis=1)
    elif formula_name == "Q-Learning":
        y = X[:, 0] + 0.1 * (X[:, 1] + np.max(X, axis=1))
    else:  # Default for other formulas
        y = np.random.rand(num_samples) * np.sum(X, axis=1) / (np.var(X, axis=1) + 1e-6)
    
    return X, y.astype(np.float32)

# Train model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = Sequential([
        Input(shape=(5,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    
    y_pred = model.predict(X_test).flatten()
    return mean_squared_error(y_test, y_pred)

# Run training for each formula sequentially
results = {}

for formula_name in formulas:
    if use_dataset and formula_name in df["Formula"].unique():
        formula_data = df[df["Formula"] == formula_name]
        X = formula_data.iloc[:, :-2].values.astype(np.float32)  # Select Feature_1 to Feature_5
        y = formula_data.iloc[:, -2].values.astype(np.float32)  # Select Target column
        print(f"📊 Using dataset for {formula_name} ({len(X)} samples)")
    else:
        X, y = generate_data(formula_name)
        print(f"🧪 Using synthetic data for {formula_name} ({len(X)} samples)")
    
    if X.shape[0] != y.shape[0]:
        print(f"Skipping {formula_name} due to shape mismatch: X={X.shape}, y={y.shape}")
        continue
    
    mse = train_model(X, y)
    results[formula_name] = mse
    print(f"{formula_name}: MSE = {mse:.4f}")

if results:
    best_formula = min(results, key=results.get)
    print(f"\n🏆 Best Formula: {best_formula} with MSE = {results[best_formula]:.4f}")
else:
    print("⚠️ No formulas processed successfully. Check dataset and column mappings!")
