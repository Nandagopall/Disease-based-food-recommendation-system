import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from joblib import load


diabetes_model = load("Suitable for Diabetes_model.joblib")
diabetes_label_encoder = load("Suitable for Diabetes_label_encoder.joblib")
pressure_model = load("Suitable for Blood Pressure_model.joblib")
pressure_label_encoder = load("Suitable for Blood Pressure_label_encoder.joblib")


def diapress_food(
    Glycemic_Index,
    Calories,
    Carbohydrates,
    Protein,
    Fat,
    Sodium_Content,
    Potassium_Content,
    Magnesium_Content,
    Calcium_Content,
    Fiber_Content,
):
    data = pd.read_csv("pred_food.csv")
    diabetes_model = load("Suitable for Diabetes_model.joblib")
    diabetes_label_encoder = load("Suitable for Diabetes_label_encoder.joblib")
    pressure_model = load("Suitable for Blood Pressure_model.joblib")
    pressure_label_encoder = load("Suitable for Blood Pressure_label_encoder.joblib")
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

    food_features = [
        Glycemic_Index,
        Calories,
        Carbohydrates,
        Protein,
        Fat,
        Sodium_Content,
        Potassium_Content,
        Magnesium_Content,
        Calcium_Content,
        Fiber_Content,
    ]

    food_data = pd.DataFrame([food_features], columns=feature_cols)
    imputer = SimpleImputer(strategy="mean")
    imputer.fit(data[feature_cols])
    food_data_imputed = imputer.transform(food_data)

    diabetes_prediction = diabetes_model.predict(food_data_imputed)
    diabetes_prediction_label = diabetes_label_encoder.inverse_transform(
        diabetes_prediction
    )[0]

    pressure_prediction = pressure_model.predict(food_data_imputed)
    pressure_prediction_label = pressure_label_encoder.inverse_transform(
        pressure_prediction
    )[0]
    good_food = False
    if diabetes_prediction_label and pressure_prediction_label:
        good_food = True

    return good_food


def pressure_food(
    Sodium_Content,
    Potassium_Content,
    Magnesium_Content,
    Calcium_Content,
    Fiber_Content,
):
    clf, imputer = load("prefood_model.joblib")

    food_features = [
        Sodium_Content,
        Potassium_Content,
        Magnesium_Content,
        Calcium_Content,
        Fiber_Content,
    ]

    food_data = imputer.transform([food_features])

    food_suitability_pred = clf.predict(food_data)
    good_food = False

    if food_suitability_pred:
        good_food = True
    return good_food


def diabetes_food(Glycemic_Index, Calories, Carbohydrates, Protein, Fat):
    food_features = [Glycemic_Index, Calories, Carbohydrates, Protein, Fat]

    loaded_model = load("diafood_model.joblib")

    imputer = loaded_model["imputer"]
    food_data = imputer.transform([food_features])

    food_suitability_pred = loaded_model.predict(food_data)
    good_food = False

    if food_suitability_pred:
        good_food = True

    return good_food
