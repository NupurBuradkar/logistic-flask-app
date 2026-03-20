from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load trained model
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

history = []

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        hours = float(data["hours"])
        attendance = float(data["attendance"])
        marks = float(data["marks"])

        if hours < 0 or attendance < 0 or marks < 0:
            return jsonify({"error": "No negative values"}), 400

        input_df = pd.DataFrame(
            [[hours, attendance, marks]],
            columns=["hours", "attendance", "marks"]
        )

        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        result = {
            "prediction": int(pred),
            "probability": round(prob, 2)
        }

        history.append(result)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/history")
def get_history():
    return jsonify(history)


if __name__ == "__main__":
    app.run(debug=True)