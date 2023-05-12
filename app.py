from flask import Flask, request, jsonify, render_template
import product

app = Flask(__name__, static_folder="static", template_folder="templates")


@app.route("/process_input", methods=["POST"])
def process_input():
    user_data = request.json

    meal_plan = product.generate_meal_plan(user_data)

    return jsonify(meal_plan=meal_plan)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/result", methods=["GET"])
def result():
    meal_plan = request.args.get("meal_plan")
    print(meal_plan)

    return render_template("result.html", meal_plan=meal_plan)


if __name__ == "__main__":
    app.run(debug=True)
