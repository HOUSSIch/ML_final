from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

MODEL_PATH = "model/model_dermatologie.pkl"
FEATURES_PATH = "model/features.pkl"

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)

class_names = {
    1: "Psoriasis",
    2: "Seboreic Dermatitis",
    3: "Lichen Planus",
    4: "Pityriasis Rosea",
    5: "Chronic Dermatitis",
    6: "Pityriasis Rubra Pilaris"
}

def pretty_label(name):
    return str(name).replace("_", " ").replace("-", " ").title()

@app.route("/")
def home():
    return render_template(
        "index.html",
        features=features,
        pretty_label=pretty_label,
        prediction=None
    )

@app.route("/predict", methods=["POST"])
def predict():
    values = []
    for feature in features:
        values.append(float(request.form.get(feature, 0)))

    data = pd.DataFrame([values], columns=features)
    prediction = model.predict(data)[0]

    try:
        result = class_names.get(int(prediction), str(prediction))
    except Exception:
        result = str(prediction)

    probabilities = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(data)[0]
            classes = model.classes_
            probabilities = []
            for cls, p in zip(classes, proba):
                try:
                    label = class_names.get(int(cls), str(cls))
                except Exception:
                    label = str(cls)
                probabilities.append({"label": label, "value": round(float(p) * 100, 2)})
            probabilities = sorted(probabilities, key=lambda x: x["value"], reverse=True)
        except Exception:
            probabilities = None

    return render_template(
        "index.html",
        features=features,
        pretty_label=pretty_label,
        prediction=result,
        probabilities=probabilities
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
