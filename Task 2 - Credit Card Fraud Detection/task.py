import pandas as pd
import numpy as np
# Machine Learning, Preprocessing, & Evaluation Tools
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Load your pre-separated files
train_df = pd.read_csv("fraudTrain.csv")
test_df = pd.read_csv("fraudTest.csv")
# Drop any rows where 'is_fraud' or critical features are blank (NaN)
train_df = train_df.dropna(subset=['is_fraud', 'amt', 'city_pop', 'category', 'gender'])
test_df = test_df.dropna(subset=['is_fraud', 'amt', 'city_pop', 'category', 'gender'])
print(f"📦 Cleaned Train dataset: {train_df.shape[0]} rows")
print(f"📦 Cleaned Test dataset: {test_df.shape[0]} rows")
# Select the most critical columns
features = ['amt', 'city_pop', 'category', 'gender']
X_train = train_df[features].copy()
y_train = train_df['is_fraud']
X_test = test_df[features].copy()
y_test = test_df['is_fraud']
# Convert text values like 'gender' or 'category' into clean numbers
le = LabelEncoder()
for col in ['category', 'gender']:
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])

# Scale features so large amounts don't overwhelm the AI weights
scaler = StandardScaler()
X_train[['amt', 'city_pop']] = scaler.fit_transform(X_train[['amt', 'city_pop']])
X_test[['amt', 'city_pop']] = scaler.transform(X_test[['amt', 'city_pop']])
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n📊 --- FRAUD MODEL EVALUATION REPORT ---")
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

print("\nDetailed Performance Matrix:")
print(classification_report(y_test, y_pred))

print("\n🧱 --- CONFUSION MATRIX LABELS ---")
cm = confusion_matrix(y_test, y_pred)
print(f"True Legitimate (Predicted Normal, Was Normal): {cm[0][0]}")
print(f"False Alarms (Predicted Fraud, Was Normal): {cm[0][1]}")
print(f"Missed Frauds (Predicted Normal, Was Fraud) 🛑: {cm[1][0]}")
print(f"Caught Frauds (Predicted Fraud, Was Fraud): {cm[1][1]}")
