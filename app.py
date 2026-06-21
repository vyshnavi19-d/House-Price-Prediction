from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("house_model.pkl")
columns = joblib.load("columns.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    lotarea = float(request.form["lotarea"])
    yearbuilt = float(request.form["yearbuilt"])
    overallcond = float(request.form["overallcond"])
    totalbsmt = float(request.form["totalbsmt"])
    yearremod = float(request.form["yearremod"])

    sample = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)

    if "LotArea" in columns:
        sample["LotArea"] = lotarea
    if "YearBuilt" in columns:
        sample["YearBuilt"] = yearbuilt
    if "OverallCond" in columns:
        sample["OverallCond"] = overallcond
    if "TotalBsmtSF" in columns:
        sample["TotalBsmtSF"] = totalbsmt
    if "YearRemodAdd" in columns:
        sample["YearRemodAdd"] = yearremod

    prediction = model.predict(sample)[0]

    return render_template("index.html", prediction=round(prediction, 2))

if __name__ == "__main__":
    app.run(debug=True)