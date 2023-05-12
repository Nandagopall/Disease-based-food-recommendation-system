import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from joblib import dump

# Load the dataset
data = pd.read_csv("diabetes.csv")

# Define the feature columns and target column
feature_cols = [
    "Glucose",
    "BloodPressure",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
target_col = "Outcome"

# Split the data into features and target
X = data[feature_cols]
y = data[target_col]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the preprocessing pipeline
preprocessor = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler())

# Preprocess the training data
X_train_preprocessed = preprocessor.fit_transform(X_train)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train_preprocessed, y_train)

# Save the model and preprocessing pipeline
dump((preprocessor, model), "diabetes_model.joblib")
print("Saved")
