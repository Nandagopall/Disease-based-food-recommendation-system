from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from joblib import dump
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data
data = pd.read_csv("datas.csv")

# Define the feature columns and target column
feature_cols = [
    "Age",
    "BMI",
    "Level_of_Hemoglobin",
    "Sex",
    "Smoking",
    "Physical_activity",
    "Genetic_Pedigree_Coefficient",
]
target_col = "Blood_Pressure_Abnormality"

# Split the data into features and target
X = data[feature_cols]
y = data[target_col]

# Handle missing values
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

# Data preprocessing
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# Create the SVM model
svm_model = SVC()

# Train the model
svm_model.fit(X_train, y_train)

# Predict on the test set
y_pred = svm_model.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save the model, scaler, and imputer
dump((svm_model, scaler), "pressure_model.joblib")
print("Saved")
