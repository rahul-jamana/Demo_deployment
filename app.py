from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

# Load your saved model
model = pickle.load(open("house.pkl", "rb"))

app = Flask(__name__, template_folder="html_file")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    # Extract inputs in correct order
    features = [
        int(data["input1"]),
        int(data["input2"]),
        int(data["input3"]),
        int(data["input4"])
    ]
    prediction = model.predict([features])[0]
    return jsonify({"prediction": str(prediction)})

if __name__ == "__main__":
    app.run(debug=True)
