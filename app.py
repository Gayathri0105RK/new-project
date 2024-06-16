from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template('home.html')

@app.route('/predict', methods=["GET", "POST"])
def predict():
    if request.method == 'POST':
        # Collecting form data
        age = int(request.form.get('age'))
        anaemia = int(request.form.get('anaemia'))
        cpk = int(request.form.get('creatinine_phosphokinase'))
        diabetes = int(request.form.get('diabetes'))
        ejection_fraction = int(request.form.get('ejection_fraction'))
        high_blood_pressure = int(request.form.get('high_blood_pressure'))
        platelets = float(request.form.get('platelets'))
        sex = int(request.form.get('sex'))
        serum_creatinine = float(request.form.get('serum_creatinine'))
        serum_sodium = int(request.form.get('serum_sodium'))
        smoking = int(request.form.get('smoking'))
        time = int(request.form.get('time'))

        # Loading the model
        with open('linear_regression_model.pkl', 'rb') as file:
            model = pickle.load(file)

        # Creating features array
        features = np.array([[age, anaemia, cpk, diabetes, ejection_fraction, high_blood_pressure, 
                              platelets, sex, serum_creatinine, serum_sodium, smoking, time]])
        
        # Making prediction
        prediction = model.predict(features)[0]

        # Rendering result
        return render_template('result.html', prediction=prediction)
    
    else:
        return render_template("predict.html")

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5500)
