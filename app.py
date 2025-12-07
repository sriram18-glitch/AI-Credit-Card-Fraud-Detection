
from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            time = float(request.form["time"])
            amount = float(request.form["amount"])

            pca_values = [float(request.form[f"v{i}"]) for i in range(1, 29)]

            data = {
                "Time": [time],
                "Amount": [amount],
            }

            for i in range(1, 29):
                data[f"V{i}"] = [pca_values[i - 1]]

            df = pd.DataFrame(data)

            df[["Time", "Amount"]] = scaler.transform(df[["Time", "Amount"]])

            result = model.predict(df)[0]

            if result == 1:
                prediction = "⚠️ FRAUD DETECTED!"
            else:
                prediction = "✅ Transaction is Normal"

        except:
            prediction = "❌ Invalid input, check values!"

    return render_template("index.html", result=prediction)

if __name__ == "__main__":
    app.run(debug=True)
