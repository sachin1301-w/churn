from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import pickle
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "customer_churn_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "encoder.pkl"), "rb") as f:
    encoders = pickle.load(f)

@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame([data])

    # Drop target if user sends it
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])

    # Apply encoders
    for col, encoder in encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].apply(
                lambda x: x if x in encoder.classes_ else encoder.classes_[0]
            )
            df[col] = encoder.transform(df[col])

    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])

    return jsonify({
        "churn": prediction,
        "probability": round(probability, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
