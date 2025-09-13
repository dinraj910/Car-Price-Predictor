from flask import Flask, render_template, request
from currency import CURRENCY_RATES
import joblib
import pandas as pd

app = Flask(__name__)

# Load the saved pipeline model
model = joblib.load("models/car_price_model.pkl")

@app.route('/', methods=['GET','POST'])
def home():
    if request.method == 'POST':
        make = request.form['make']
        model_name = request.form['model']  # maybe rename
        year = int(request.form['year'])
        mileage = float(request.form['mileage'])
        engine_size = float(request.form['engine_size'])
        fuel_type = request.form['fuel_type']
        transmission = request.form['transmission']

        car_age = 2025 - year  # or current year

        input_df = pd.DataFrame([{
            'Make': make,
            'Model': model_name,
            'Mileage': mileage,
            'Engine Size': engine_size,
            'Fuel Type': fuel_type,
            'Transmission': transmission,
            'Car_Age': car_age
        }])

        prediction = model.predict(input_df)[0]
        return render_template(
            "result.html",
            predicted_price=round(prediction, 2),
            input_data=input_df.to_dict(orient='records')[0],
            currency_rates=CURRENCY_RATES,
            default_currency='USD'
        )
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
