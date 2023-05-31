import pandas as pd
import random
import food_data
from joblib import load
import diapress_food


def generate_meal_plan(user_data):
    glucose = int(user_data.get("glucose"))
    bp = int(user_data.get("bloodPressure"))
    insulin = int(user_data.get("insulin"))
    age = int(user_data.get("age"))
    sex = str(user_data.get("sex"))
    height = float(user_data.get("height"))
    weight = float(user_data.get("weight"))
    hemogoblin = float(user_data.get("hemoglobinLevel"))
    smoking = str(user_data.get("smoking"))
    physical_activity = str(user_data.get("physicalActivity"))
    dpf = user_data.get("diabetesPedigreeFunction")
    bpf = user_data.get("geneticPedigree")
    sexf = user_data.get("sex")
    smokef = user_data.get("smoking")
    print(user_data)

    height_t = height / 100
    bmi = weight / (height_t**2)
    if dpf.lower() == "yes":
        dpf = 0.7
    else:
        dpf = 0

    if physical_activity.lower() == "extra active":
        pa = 42000
    elif physical_activity.lower() == "very active":
        pa = 30000
    elif physical_activity.lower() == "moderately active":
        pa = 20000
    elif physical_activity.lower() == "lightly active":
        pa = 10000
    else:
        pa = 1000

    if bpf.lower() == "yes":
        bpf = 0.7
    else:
        bpf = 0

    if sexf.lower() == "male":
        sex = 1
    else:
        sex = 0
    if smokef.lower() == "yes":
        smoking = 1
    else:
        smoking = 0

    preprocessor, model = load("diabetes_model.joblib")

    new_data = pd.DataFrame(
        {
            "Glucose": [glucose],
            "BloodPressure": [bp],
            "Insulin": [insulin],
            "BMI": [bmi],
            "DiabetesPedigreeFunction": [dpf],
            "Age": [age],
        }
    )
    new_data_pre = preprocessor.transform(new_data)

    prediction = model.predict(new_data_pre)

    if prediction == 1:
        has_diabetes = True
    else:
        has_diabetes = False

    xgb_model, scaler = load("pressure_model.joblib")

    new_data = pd.DataFrame(
        {
            "Age": [age],
            "BMI": [bmi],
            "Level_of_Hemoglobin": [hemogoblin],
            "Sex": [sex],
            "Smoking": [smoking],
            "Physical_activity": [pa],
            "Genetic_Pedigree_Coefficient": [bpf],
        }
    )

    new_data_scaled = scaler.transform(new_data)

    prediction = xgb_model.predict(new_data_scaled)

    if prediction == 1:
        has_pressure = True
    else:
        has_pressure = False

    def calculate_calorie_intake(age, gender, height, weight, activity_level):
        if gender == "male":
            bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
        elif gender == "female":
            bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
        else:
            raise ValueError("Invalid gender. Choose 'male' or 'female'.")

        activity_factors = {
            "sedentary": 1.2,
            "lightly active": 1.375,
            "moderately active": 1.55,
            "very active": 1.725,
            "extra active": 1.9,
        }

        calorie_intake = bmr * activity_factors[activity_level]
        calorie_intake = calorie_intake / 3
        carb_max = (60 / 100) * calorie_intake
        protein_max = (20 / 100) * calorie_intake
        fat_max = (40 / 100) * calorie_intake
        max = {
            "calorie": calorie_intake,
            "carb": carb_max,
            "protein": protein_max,
            "fat": fat_max,
        }
        return max

    max = {}
    max = calculate_calorie_intake(age, sexf, height, weight, physical_activity)

    def meal(meal_type, has_diabetes, has_pressure, max):
        bad_food = True
        while bad_food:
            if has_diabetes and has_pressure:
                if meal_type == "breakfast":
                    random_food = random.choice(food_data.dbbreakfast())
                elif meal_type == "lunch":
                    random_food = random.choice(food_data.dblunch())
                else:
                    random_food = random.choice(food_data.dbdinner())
                GI = int(random_food["glycemic_index"])
                name = str(random_food["name"])
                calories = int(random_food["calories"])
                carbohydrates = int(random_food["carbohydrates"])
                protein = int(random_food["protein"])
                fat = int(random_food["fat"])
                sodium = int(random_food["sodium"])
                potassium = int(random_food["potassium"])
                magnesium = int(random_food["magnesium"])
                calcium = int(random_food["calcium"])
                fiber = int(random_food["fiber"])

                good_food = diapress_food.diapress_food(
                    GI,
                    calories,
                    carbohydrates,
                    protein,
                    fat,
                    sodium,
                    potassium,
                    magnesium,
                    calcium,
                    fiber,
                )
                if (
                    good_food
                    and calories < max["calorie"]
                    and carbohydrates < max["carb"]
                    and protein < max["protein"]
                    and fat < max["fat"]
                ):
                    bad_food = False
                    return random_food

            elif has_diabetes:
                if meal_type == "breakfast":
                    random_food = random.choice(food_data.dbreakfast())
                elif meal_type == "lunch":
                    random_food = random.choice(food_data.dblunch())
                else:
                    random_food = random.choice(food_data.ddinner())

                good_food = diapress_food.diabetes_food(
                    int(random_food["glycemic_index"]),
                    int(random_food["calories"]),
                    int(random_food["carbohydrates"]),
                    int(random_food["protein"]),
                    int(random_food["fat"]),
                )
                if (
                    good_food
                    and int(random_food["calories"]) < max["calorie"]
                    and int(random_food["carbohydrates"]) < max["carb"]
                    and int(random_food["protein"]) < max["protein"]
                    and int(random_food["fat"]) < max["fat"]
                ):
                    bad_food = False
                    return random_food

            elif has_pressure:
                if meal_type == "breakfast":
                    random_food = random.choice(food_data.bbreakfast())
                elif meal_type == "lunch":
                    random_food = random.choice(food_data.blunch())
                else:
                    random_food = random.choice(food_data.bdinner())
                sodium = int(random_food["sodium"])
                potassium = int(random_food["potassium"])
                magnesium = int(random_food["magnesium"])
                calcium = int(random_food["calcium"])
                fiber = float(random_food["fiber"])

                good_food = diapress_food.pressure_food(
                    sodium,
                    potassium,
                    magnesium,
                    calcium,
                    fiber,
                )
                if good_food:
                    bad_food = False
                    return random_food

            else:
                if meal_type == "breakfast":
                    random_food = random.choice(food_data.breakfast())
                elif meal_type == "lunch":
                    random_food = random.choice(food_data.lunch())
                else:
                    random_food = random.choice(food_data.dinner())
                calories = random_food["calories"]
                carbohydrates = random_food["carbohydrates"]
                protein = random_food["protein"]
                fat = random_food["fat"]
                if (
                    calories < max["calorie"]
                    and carbohydrates < max["carb"]
                    and protein < max["protein"]
                    and fat < max["fat"]
                ):
                    bad_food = False
                    return random_food

    if has_pressure and has_diabetes == False:
        meal_plan = ""
        meal_plan += "You may have High Blood Pressure\n\n"
        for day in range(1, 8):
            meal_plan += f"Day {day}\n"
            breakfast = meal("breakfast", has_diabetes, has_pressure, max)
            lunch = meal("lunch", has_diabetes, has_pressure, max)
            dinner = meal("dinner", has_diabetes, has_pressure, max)

            meal_plan += f"\tBreakfast : {breakfast['name']}\n"
            meal_plan += f"\tLunch : {lunch['name']}\n"
            meal_plan += f"\tDinner : {dinner['name']}\n\n"
    else:
        meal_plan = ""
        if has_diabetes and has_pressure:
            meal_plan += "You may have Diabetes and High Blood Pressure\n\n"
        elif has_diabetes:
            meal_plan += "You may have Diabetes\n\n"
        else:
            meal_plan += "You don't have diabetes and High blood pressure\n\n"
        for day in range(1, 8):
            meal_plan += f"Day {day}\n"
            breakfast = meal("breakfast", has_diabetes, has_pressure, max)
            lunch = meal("lunch", has_diabetes, has_pressure, max)
            dinner = meal("dinner", has_diabetes, has_pressure, max)

            meal_plan += f"\tBreakfast : {breakfast['name']} :- "
            meal_plan += f"\tCalorie : {breakfast['calories']}, "
            meal_plan += f"\tCarb : {breakfast['carbohydrates']}, "
            meal_plan += f"\tProtein : {breakfast['protein']}, "
            meal_plan += f"\tFat : {breakfast['fat']}\n"

            meal_plan += f"\tLunch : {lunch['name']} :- "
            meal_plan += f"\tCalorie : {lunch['calories']}, "
            meal_plan += f"\tCarb : {lunch['carbohydrates']}, "
            meal_plan += f"\tProtein : {lunch['protein']}, "
            meal_plan += f"\tFat : {lunch['fat']}\n"

            meal_plan += f"\tDinner : {dinner['name']} :- "
            meal_plan += f"\tCalorie : {dinner['calories']}, "
            meal_plan += f"\tCarb : {dinner['carbohydrates']} , "
            meal_plan += f"\tProtein : {dinner['protein']}, "
            meal_plan += f"\tFat : {dinner['fat']}\n\n"
    print(meal_plan)

    return meal_plan


# user_data = {
#     "glucose": 120,
#     "bloodPressure": 80,
#     "insulin": 120,
#     "age": 40,
#     "sex": "female",
#     "height": 165,
#     "weight": 60,
#     "geneticPedigree": "yes",
#     "hemoglobinLevel": 13.5,
#     "smoking": "no",
#     "physicalActivity": "very active",
#     "diabetesPedigreeFunction": "yes",
# }

# meal_plan = generate_meal_plan(user_data)
