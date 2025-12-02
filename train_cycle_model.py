import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, confusion_matrix
import numpy as np
import joblib

# -----------------------------
# 1️⃣ Load and clean data
# -----------------------------
df = pd.read_csv("Menstural_cyclelength.csv")

# Normalize column names
df.columns = df.columns.str.strip().str.lower()

# Drop invalid ages
df = df[df['age'] >= 9]

# Drop rows where target is missing
df = df.dropna(subset=['cycle_length'])

# -----------------------------
# 2️⃣ Encode categorical data
# -----------------------------
if 'conception_cycle' in df.columns:
    le = LabelEncoder()
    df['conception_cycle'] = le.fit_transform(df['conception_cycle'].astype(str))

# -----------------------------
# 3️⃣ Select features and target
# -----------------------------
feature_cols = ['age', 'cycle_number', 'conception_cycle']
X = df[feature_cols]
y = df['cycle_length']

# -----------------------------
# 4️⃣ Handle missing values
# -----------------------------
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# -----------------------------
# 5️⃣ Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# -----------------------------
# 6️⃣ Train Linear Regression model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# 7️⃣ Evaluate model
# -----------------------------
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\n📊 Model Evaluation Results:")
print(f"R² Score: {r2:.3f}")
print(f"Mean Absolute Error: {mae:.2f} days")

# -----------------------------
# 8️⃣ Confusion Matrix (Categorical Evaluation)
# -----------------------------
# Convert cycle lengths into discrete classes:
# Short (<25 days), Normal (25–35 days), Long (>35 days)
def categorize_cycle_length(length):
    if length < 25:
        return "Short"
    elif 25 <= length <= 35:
        return "Normal"
    else:
        return "Long"

y_true_cat = [categorize_cycle_length(v) for v in y_test]
y_pred_cat = [categorize_cycle_length(v) for v in y_pred]

labels = ["Short", "Normal", "Long"]
cm = confusion_matrix(y_true_cat, y_pred_cat, labels=labels)

print("\n📈 Confusion Matrix (Cycle Length Categories):")
print(pd.DataFrame(cm, index=[f"Actual {l}" for l in labels],
                        columns=[f"Predicted {l}" for l in labels]))

# -----------------------------
# 9️⃣ Save model and imputer
# -----------------------------
joblib.dump(model, "cycle_model.pkl")
joblib.dump(imputer, "cycle_imputer.pkl")

print("\n✅ Model trained and saved successfully as 'cycle_model.pkl' and 'cycle_imputer.pkl'.")
