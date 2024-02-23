from flask import Flask, request, jsonify
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))
app = Flask(__name__)


@app.route('/')
def home():
    return 'Hello World!'


@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve data from form
    cgpa = float(request.form.get('cgpa'))
    iq = float(request.form.get('iq'))
    profile_score = float(request.form.get('profile_score'))

    # Prepare input data for prediction
    input_query = np.array([[cgpa, iq, profile_score]])

    # Make prediction
    result = model.predict(input_query)[0]

    # Return prediction result
    return jsonify({'placement': str(result)})


if __name__ == '__main__':
    app.run(debug=True)
