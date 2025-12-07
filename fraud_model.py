
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

print("📌 Generating synthetic fraud dataset...")

# Generate synthetic dataset
np.random.seed(42)
num_samples = 5000

data = pd.DataFrame({
    "Time": np.random.uniform(0, 100000, num_samples),
    "Amount": np.random.uniform(0, 2000, num_samples),

    # Random PCA-like numeric features (like V1-V28)
    **{f"V{i}": np.random.normal(0, 1, num_samples) for i in range(1, 29)}
})

# Create labels: 0 = normal, 1 = fraud
data["Class"] = np.random.choice([0, 1], size=num_samples, p=[0.97, 0.03])

print("Total samples:", len(data))
print("Fraud cases:", data['Class'].sum())

# Split dataset
X = data.drop("Class", axis=1)
y = data["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale only Time and Amount
scaler = StandardScaler()
X_train[["Time", "Amount"]] = scaler.fit_transform(X_train[["Time", "Amount"]])
X_test[["Time", "Amount"]] = scaler.transform(X_test[["Time", "Amount"]])

print("🚀 Training model...")

model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)

print("✅ Model training complete!")
print("Saving model files...")

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("🎉 model.pkl and scaler.pkl saved successfully!")
