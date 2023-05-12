import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from joblib import dump

# Load the dataset
data = pd.read_csv("pred_food.csv")

# Define the feature columns and target columns
feature_cols = [
    "Glycemic Index",
    "Calories",
    "Carbohydrates",
    "Protein",
    "Fat",
    "Sodium Content",
    "Potassium Content",
    "Magnesium Content",
    "Calcium Content",
    "Fiber Content",
]
target_cols = ["Suitable for Diabetes", "Suitable for Blood Pressure"]

# Split the data into features and targets
X = data[feature_cols]
y = data[target_cols]

# Convert the target columns to numerical values using label encoding
label_encoders = {}
y_encoded = pd.DataFrame()

for col in target_cols:
    label_encoder = LabelEncoder()
    y_encoded[col] = label_encoder.fit_transform(y[col].astype(str))
    label_encoders[col] = label_encoder
    dump(label_encoder, f"{col}_label_encoder.joblib")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Perform mean imputation on the training and testing data
imputer = SimpleImputer(strategy="mean")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Train and save a Random Forest classifier for each target column
classifiers = {}

for col in target_cols:
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train[col])
    classifiers[col] = clf
    dump(clf, f"{col}_model.joblib")
