from flask import Flask, render_template, request
import joblib
# import scikit

app = Flask(__name__)

# Load the trained Random Forest model
model = joblib.load('D:\\Work\\4th sem project\\sarthakproject\\sarthakproject\\fronend\\best_rf_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    #get them values :]
    pregnancies = int(request.form['pregnancies'])
    glucose = int(request.form['glucose'])
    blood_pressure = int(request.form['blood_pressure'])
    skin_thickness = int(request.form['skin_thickness'])
    insulin = int(request.form['insulin'])
    bmi = float(request.form['bmi'])
    diabetes_pedigree_function = float(request.form['diabetes_pedigree_function'])
    age = int(request.form['age'])

    
    features = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]]

    outcome = model.predict(features)[0]

    return render_template('result.html', outcome=outcome)

if __name__ == '__main__':
    app.run(debug=True)