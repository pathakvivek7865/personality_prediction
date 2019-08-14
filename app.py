from __future__ import division
from math import sqrt
from flask import Flask, render_template, request, jsonify
from collections import Counter
from flask import Flask, request
from predict import Predictor
from model import Model

app = Flask(__name__)

M = Model()
predictor = Predictor()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json
    prediction =  predictor.predict([text])
    return jsonify(prediction)



if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True, debug=True)