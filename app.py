
from fastapi import FastAPI
import pandas as pd
import joblib


model = joblib.load('best_model.pkl')


forecast_df = pd.read_csv(
    'future_8weeks_forecast.csv'
)


app = FastAPI(
    title="Sales Forecast API",
    description="Forecast next 8 weeks and single prediction API",
    version="1.0"
)


@app.get('/')

def home():

    return {
        'message': 'Sales Forecast API Running'
    }


@app.get('/forecast/{state}')

def get_forecast(state: str):

    result = forecast_df[
        forecast_df['state'].str.lower()
        == state.lower()
    ]

    if result.empty:

        return {
            'error': 'State not found'
        }

    output = result.to_dict(
        orient='records'
    )

    return {
        'state': state,
        'next_8_weeks_forecast': output
    }



@app.post('/predict')

def predict(

    lag_1: float,
    lag_7: float,
    lag_30: float,
    rolling_mean_4: float,
    rolling_std_4: float,
    month: int,
    quarter: int,
    weekofyear: int,
    category_encoded: int

):

    input_data = pd.DataFrame([{

        'lag_1': lag_1,
        'lag_7': lag_7,
        'lag_30': lag_30,
        'rolling_mean_4': rolling_mean_4,
        'rolling_std_4': rolling_std_4,
        'month': month,
        'quarter': quarter,
        'weekofyear': weekofyear,
        'category_encoded': category_encoded

    }])

    prediction = model.predict(input_data)[0]

    return {

        'predicted_sales': float(prediction)

    }
