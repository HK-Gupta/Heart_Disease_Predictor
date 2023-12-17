from flask import Flask, request, jsonify
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello Patient"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from the request and convert them to float
        age = float(request.form.get('age'))
        sex = float(request.form.get('sex'))
        cp = float(request.form.get('cp'))
        trtbps = float(request.form.get('trtbps'))
        chol = float(request.form.get('chol'))
        fbs = float(request.form.get('fbs'))
        restecg = float(request.form.get('restecg'))
        thalachh = float(request.form.get('thalachh'))
        exng = float(request.form.get('exng'))
        oldpeak = float(request.form.get('oldpeak'))
        slp = float(request.form.get('slp'))
        caa = float(request.form.get('caa'))
        thall = float(request.form.get('thall'))

        # Create a NumPy array with the input features
        input_query = np.array([[age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall]])

        # Make predictions using the loaded model
        result = model.predict(input_query)[0]

        # Return the result as JSON
        return jsonify({'output': str(result)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
