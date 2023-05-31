import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
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

# Preprocess the training and testing data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Train the SVM model
svm_model = SVC()
svm_model.fit(X_train_preprocessed, y_train)

# Predict labels for the test data
y_pred = svm_model.predict(X_test_preprocessed)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save the model and preprocessing pipeline
dump((preprocessor, svm_model), "diabetes_model.joblib")
print("Saved")
