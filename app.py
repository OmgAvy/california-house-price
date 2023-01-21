from flask import (
    Flask,
    request,
    app,
    jsonify,
    url_for,
    render_template,
    redirect,
    flash,
    escape,
    session
    )

import pickle
import numpy as np
import pandas as pd


app=Flask(__name__)

## Loading Model
regmodel = pickle.load(open('regmodel.pkl', 'rb'))

## Loading scaler for transformation
scaler = pickle.load(open('scaler.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    new_data = np.array(list(data.values())).reshape(1,-1)
    scaled_data=scaler.transform(new_data)
    output = regmodel.predict(scaled_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict', methods=['POST'])
def predict():
    data = [ float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1, -1))
    print(final_input)
    output = regmodel.predict(final_input)[0]
    return render_template("home.html", prediction_text=f"The median value of the house is {output}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

