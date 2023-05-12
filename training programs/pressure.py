from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from joblib import dump
import pandas as pd
from sklearn.model_selection import train_test_split

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

# Data preprocessing
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# Create the XGBoost model
xgb_model = xgb.XGBClassifier()

# Train the model
xgb_model.fit(X_train, y_train)

# Save the model and scaler
dump((xgb_model, scaler), "pressure_model.joblib")
print("Saved")
