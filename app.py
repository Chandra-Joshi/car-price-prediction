
from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    name = request.form['name']
    company = request.form['company']
    year = int(request.form['year'])
    kms = int(request.form['kms_driven'])
    fuel = request.form['fuel_type']

    input_data = pd.DataFrame([[name, company, year, kms, fuel]],
        columns=['name','company','year','kms_driven','fuel_type'])

    prediction = model.predict(input_data)

    return render_template('index.html',
           prediction_text="Estimated Car Price: RS " + str(round(prediction[0],2)))

if __name__ == "__main__":
    app.run(debug=True)
