import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # textutil (needed to unpickle the model)
import joblib
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
model = joblib.load(os.path.join(os.path.dirname(__file__), "model.pkl"))


def predict(text):
    p = float(model.predict_proba([text])[0, 1])
    return {"label": "Real disaster" if p >= 0.5 else "Not a disaster", "disaster": p >= 0.5, "probability": round(p, 4)}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    text = (request.get_json(silent=True) or {}).get("text", "").strip()
    if not text:
        return jsonify(error="Please enter some tweet text."), 400
    return jsonify(predict(text))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
