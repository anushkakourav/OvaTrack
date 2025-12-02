# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# -----------------------------
# 1️⃣ Load dataset
# -----------------------------
data = pd.read_csv("PCOS_data[1].csv")

# Clean column names (remove spaces)
data.columns = data.columns.str.strip()

# -----------------------------
# 2️⃣ Select required features
# -----------------------------
selected_features = [
    'Age (yrs)',
    'Weight (Kg)',
    'Height(Cm)',
    'Cycle length(days)',
    'hair growth(Y/N)',
    'Pimples(Y/N)',
    'Hair loss(Y/N)',
    'Reg.Exercise(Y/N)',
    'Fast food (Y/N)'
]

target_column = 'PCOS (Y/N)'

# Keep only selected columns and target
data = data[selected_features + [target_column]]

# Drop rows with missing values
data.dropna(inplace=True)

# -----------------------------
# 3️⃣ Convert categorical Y/N columns to numeric (0 and 1)
# -----------------------------
for col in ['hair growth(Y/N)', 'Pimples(Y/N)', 'Hair loss(Y/N)', 'Reg.Exercise(Y/N)', 'Fast food (Y/N)']:
    data[col] = data[col].replace({'Y': 1, 'N': 0})

# -----------------------------
# 4️⃣ Calculate BMI manually
# -----------------------------
data['BMI'] = round(data['Weight (Kg)'] / ((data['Height(Cm)'] / 100) ** 2), 2)

# Reorder columns to include BMI
final_features = [
    'Age (yrs)',
    'Weight (Kg)',
    'Height(Cm)',
    'BMI',
    'Cycle length(days)',
    'hair growth(Y/N)',
    'Pimples(Y/N)',
    'Hair loss(Y/N)',
    'Reg.Exercise(Y/N)',
    'Fast food (Y/N)'
]

# -----------------------------
# 5️⃣ Define X and y
# -----------------------------
X = data[final_features]
y = data[target_column]

# Filter out unrealistic ages (<9 years)
X = X[X['Age (yrs)'] >= 9]
y = y[X.index]

# -----------------------------
# 6️⃣ Split data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 7️⃣ Scale data
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 8️⃣ Train model
# -----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# -----------------------------
# 9️⃣ Evaluate model
# -----------------------------
y_pred = model.predict(X_test_scaled)

# Accuracy and classification report
print("\n📊 Model Evaluation Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# 🔟 Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=["Actual No PCOS", "Actual PCOS"],
                        columns=["Predicted No PCOS", "Predicted PCOS"])

print("\n📈 Confusion Matrix:")
print(cm_df)

# -----------------------------
# 11️⃣ Save model and scaler
# -----------------------------
joblib.dump(model, "RandomForest_PCOS.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n✅ Model training complete. Files saved as 'RandomForest_PCOS.pkl' and 'scaler.pkl'.")
