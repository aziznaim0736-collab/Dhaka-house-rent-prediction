from flask import Flask, render_template, request
import joblib, numpy as np, pandas as pd

app = Flask(__name__)

bundle = joblib.load('model.pkl')
model, columns, areas = bundle['model'], bundle['columns'], bundle['areas']


def predict_rent(area_name, area_sqft, bed, bath):
    row = pd.DataFrame(0, index=[0], columns=columns)
    row['Area'] = area_sqft
    row['Bed'] = bed
    row['Bath'] = bath
    col = f'Area_Name_{area_name}'
    if col in row.columns:
        row[col] = 1
    return float(np.expm1(model.predict(row)[0]))


@app.route('/', methods=['GET', 'POST'])
def home():
    result, error, form = None, None, {}
    if request.method == 'POST':
        form = request.form
        try:
            area_name = form['area_name']
            size = float(form['size'].replace(',', ''))
            bed = int(form['bed'])
            bath = int(form['bath'])
        except (KeyError, ValueError):
            error = 'Please enter valid numbers.'
        else:
            if area_name not in areas:
                error = 'Please select a valid area.'
            elif not 200 <= size <= 4000:
                error = 'Size must be between 200 and 4000 sqft.'
            elif not (1 <= bed <= 6 and 1 <= bath <= 7):
                error = 'Bedrooms must be 1-6 and bathrooms 1-7.'
            else:
                result = int(round(predict_rent(area_name, size, bed, bath) / 100) * 100)
    return render_template('index.html', areas=areas, result=result, error=error, form=form)


if __name__ == '__main__':
    app.run(debug=True)
