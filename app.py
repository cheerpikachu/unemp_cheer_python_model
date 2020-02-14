import datetime as dt
from flask import Flask, request
import time
import os
import pandas as pd

from ie_bike_model.model import predict, train_and_persist
from ie_bike_model.util import read_data, get_model_path

app = Flask(__name__)


@app.route("/")
def hello():
    name = request.args.get("name", "World")
    return "Hello, " + name + "!"


@app.route("/predict")
def get_predict():

    parameters = dict(request.args)
    hour_original = read_data()
    if "date" not in parameters:
        parameters["date"] = "2012-01-01T18:00:00"
    if "weathersit" not in parameters:
        parameters["weathersit"] = hour_original["weathersit"].median()
    if "temperature_C" not in parameters:
        parameters["temperature_C"] = 41.0 * hour_original["temp"].mean()
    if "feeling_temperature_C" not in parameters:
        parameters["feeling_temperature_C"] = 50.0 * hour_original["atemp"].mean()
    if "humidity" not in parameters:
        parameters["humidity"] = 100.0 * hour_original["hum"].mean()
    if "windspeed" not in parameters:
        parameters["windspeed"] = 67.0 * hour_original["windspeed"].mean()
    parameters["date"] = dt.datetime.fromisoformat(parameters["date"])
    parameters["weathersit"] = int(parameters["weathersit"])
    parameters["temperature_C"] = float(parameters["temperature_C"])
    parameters["feeling_temperature_C"] = float(parameters["feeling_temperature_C"])
    parameters["humidity"] = float(parameters["humidity"])
    parameters["windspeed"] = float(parameters["windspeed"])
    if "model" in parameters:
        model = str(parameters["model"])
        del parameters["model"]
    else:
        model = "xgboost"
    result = predict(parameters, model=model)
    return {"result": result}
