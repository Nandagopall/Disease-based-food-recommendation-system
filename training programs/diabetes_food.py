import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from joblib import dump

# Load the dataset
data = pd.read_csv("pred_food.csv")

# Define the feature columns and target column
feature_cols = ["Glycemic Index", "Calories", "Carbohydrates", "Protein", "Fat"]
target_col = "Suitable for Diabetes"

# Split the data into features and target
X = data[feature_cols]
y = data[target_col]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create a pipeline with SimpleImputer and RandomForestClassifier
pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="mean")),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
    ]
)

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Save the pipeline including both the imputer and classifier
dump(pipeline, "diafood_model.joblib")
print("Saved")
