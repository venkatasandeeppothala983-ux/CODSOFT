import pandas as pd
import numpy as np
# Machine Learning, Preprocessing, & Evaluation Tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

file_path = "Churn_Modelling.csv"
try:
    df = pd.read_csv(file_path)
    print(f"🎉 Dataset loaded with {df.shape[0]} customers and {df.shape[1]} columns.")
except FileNotFoundError:
    print(f"❌ Error: '{file_path}' not found. Please upload it or check the file name.")

# Preview the customer information columns
print("\n--- CUSTOMER DATA SNEAK PEEK ---")
print(df.head(3))


# Drop columns that have no impact on behavior (RowNumber, CustomerId, Surname)
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, errors='ignore')

# Convert categorical columns (Geography, Gender) into numbers
le = LabelEncoder()
for col in ['Geography', 'Gender']:
    if col in df.columns:
        df[col] = le.fit_transform(df[col])

# Make sure there are no missing NaN rows in our dataset
df = df.dropna()

# 'Exited' is the target column (1 = Customer left, 0 = Customer stayed)
X = df.drop('Exited', axis=1)
y = df['Exited']

# Split into 80% training and 20% validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale numeric features (CreditScore, Age, Balance, etc.)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"🎉 Data split completed!")
print(f"📦 Training records: {X_train.shape[0]}")
print(f"📦 Testing records: {X_test.shape[0]}")

print("\n🧠 Initializing Random Forest Ensemble Classifier...")
# We use class_weight='balanced' here from the start to prevent the accuracy paradox!
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

print("⏳ Training model to identify retention risk patterns...")
model.fit(X_train, y_train)
print("🎉 Model training successfully completed!")
y_pred = model.predict(X_test)

print("\n📊 --- CHURN MODEL EVALUATION REPORT ---")
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

print("\nDetailed Performance Matrix:")
print(classification_report(y_test, y_pred))

print("\n🧱 --- CONFUSION MATRIX ---")
cm = confusion_matrix(y_test, y_pred)
print(f"Loyal Customers Correctly Identified: {cm[0][0]}")
print(f"False Alarms (Predicted to Leave, But Stayed): {cm[0][1]}")
print(f"Missed Risks (Predicted to Stay, But Left) 🛑: {cm[1][0]}")
print(f"At-Risk Customers Correctly Caught: {cm[1][1]}")
