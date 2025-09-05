# app.py - Flask API for IPL Score Predictor
from flask import Flask, request, jsonify
from flask_cors import CORS   # ✅ Added
import joblib 
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)  # ✅ Allow all domains (React frontend can connect)

# ----------------------------
# Load Model + Encoder + Features
# ----------------------------
model = joblib.load("model.pkl")
encoder = joblib.load("encoder.pkl")
feature_cols = joblib.load("features.pkl")

# ----------------------------
# Home Route
# ----------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Welcome to the IPL Score Predictor API!",
        "usage": "Send a POST request to /predict with JSON data to get predictions."
    })

# ----------------------------
# Prediction Route
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Extract input
        bat_team = data["bat_team"]
        bowl_team = data["bowl_team"]
        venue = data["venue"]
        runs = data["runs"]
        wickets = data["wickets"]
        overs = data["overs"]

        # ----------------------------
        # Feature Engineering (same as train.py)
        # ----------------------------
        balls_bowled = int(overs) * 6 + int(round((overs % 1) * 10, 0))
        balls_left = 120 - balls_bowled
        wickets_left = 10 - wickets
        current_run_rate = runs / overs if overs > 0 else 0

        # Prepare dataframe
        input_df = pd.DataFrame({
            "cumulative_runs": [runs],
            "cumulative_wickets": [wickets],
            "balls_bowled": [balls_bowled],
            "balls_left": [balls_left],
            "wickets_left": [wickets_left],
            "current_run_rate": [current_run_rate],
            "bat_team": [bat_team],
            "bowl_team": [bowl_team],
            "venue": [venue]
        })

        # Encode categorical features
        encoded = encoder.transform(input_df[["bat_team", "bowl_team", "venue"]])
        encoded_df = pd.DataFrame(
            encoded,
            columns=encoder.get_feature_names_out(["bat_team", "bowl_team", "venue"])
        )

        # Combine with numeric features
        final_df = pd.concat(
            [input_df.drop(columns=["bat_team", "bowl_team", "venue"]), encoded_df],
            axis=1
        )

        # Align with feature_cols
        final_df = final_df.reindex(columns=feature_cols, fill_value=0)

        # Predict
        prediction = model.predict(final_df)[0]

        return jsonify({"predicted_score": round(prediction, 2)})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
