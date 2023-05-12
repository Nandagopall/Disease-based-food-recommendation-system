import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from joblib import dump

# Load the dataset
data = pd.read_csv("pred_food.csv")

# Define the feature columns and target column
feature_cols = [
    "Sodium Content",
    "Potassium Content",
    "Magnesium Content",
    "Calcium Content",
    "Fiber Content",
]
target_col = "Suitable for Blood Pressure"

# Split the data into features and target
X = data[feature_cols]
y = data[target_col]

# Create and fit the imputer
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save the trained model and imputer
dump((clf, imputer), "prefood_model.joblib")
