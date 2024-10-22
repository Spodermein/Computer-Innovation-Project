from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from enum import Enum
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

# Configure CORS
origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your trained model
model = joblib.load('LogisticRegression.joblib')

# Define an Enum for cities
class CityEnum(str, Enum):
    Albury = "Albury"
    BadgerysCreek = "BadgerysCreek"
    Cobar = "Cobar"
    CoffsHarbour = "CoffsHarbour"
    Moree = "Moree"
    Newcastle = "Newcastle"
    NorahHead = "NorahHead"
    NorfolkIsland = "NorfolkIsland"
    Penrith = "Penrith"
    Richmond = "Richmond"
    Sydney = "Sydney"
    SydneyAirport = "SydneyAirport"
    WaggaWagga = "WaggaWagga"
    Williamtown = "Williamtown"
    Wollongong = "Wollongong"
    Canberra = "Canberra"
    Tuggeranong = "Tuggeranong"
    MountGinini = "MountGinini"
    Ballarat = "Ballarat"
    Bendigo = "Bendigo"
    Sale = "Sale"
    MelbourneAirport = "MelbourneAirport"
    Melbourne = "Melbourne"
    Mildura = "Mildura"
    Nhil = "Nhil"
    Portland = "Portland"
    Watsonia = "Watsonia"
    Dartmoor = "Dartmoor"
    Brisbane = "Brisbane"
    Cairns = "Cairns"
    GoldCoast = "GoldCoast"
    Townsville = "Townsville"
    Adelaide = "Adelaide"
    MountGambier = "MountGambier"
    Nuriootpa = "Nuriootpa"
    Woomera = "Woomera"
    Albany = "Albany"
    Witchcliffe = "Witchcliffe"
    PearceRAAF = "PearceRAAF"
    PerthAirport = "PerthAirport"
    Perth = "Perth"
    SalmonGums = "SalmonGums"
    Walpole = "Walpole"
    Hobart = "Hobart"
    Launceston = "Launceston"
    AliceSprings = "AliceSprings"
    Darwin = "Darwin"
    Katherine = "Katherine"
    Uluru = "Uluru"

# Define the input data structure
class WeatherInput(BaseModel):
    MinTemp: float
    MaxTemp: float
    Rainfall: float
    Evaporation: float
    Sunshine: float
    WindGustSpeed: float
    WindSpeed9am: float
    WindSpeed3pm: float
    Humidity9am: float
    Humidity3pm: float
    Pressure9am: float
    Pressure3pm: float
    Cloud9am: float
    Cloud3pm: float
    Temp9am: float
    Temp3pm: float
    WindGustDir: str
    WindDir9am: str
    WindDir3pm: str
    RainToday: str 
    Location: CityEnum

@app.post('/predict/')
async def predict(weather: WeatherInput):
    try:
        # Prepare input data as a single array
        input_data = np.array([[weather.MinTemp, weather.MaxTemp, weather.Rainfall, weather.Evaporation, 
                                 weather.Sunshine, weather.WindGustSpeed, weather.WindSpeed9am, 
                                 weather.WindSpeed3pm, weather.Humidity9am, weather.Humidity3pm, 
                                 weather.Pressure9am, weather.Pressure3pm, weather.Cloud9am, 
                                 weather.Cloud3pm, weather.Temp9am, weather.Temp3pm]])

        # Create a DataFrame for the input data
        input_df = pd.DataFrame(input_data, columns=[
            'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 
            'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 
            'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 
            'Cloud3pm', 'Temp9am', 'Temp3pm'
        ])

        # One-hot encode categorical features
        categorical_features = pd.DataFrame({
            'Location': [weather.Location],
            'WindGustDir': [weather.WindGustDir],
            'WindDir9am': [weather.WindDir9am],
            'WindDir3pm': [weather.WindDir3pm],
            'RainToday': [weather.RainToday]
        })

        # Perform one-hot encoding with drop_first=True to avoid the dummy variable trap
        encoded_features = pd.get_dummies(categorical_features, drop_first=True)

        # Concatenate the input data with one-hot encoded columns
        input_df = pd.concat([input_df, encoded_features], axis=1)

        # Ensure input_df matches the model's expected input shape and columns
        input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

        # Make a prediction
        prediction = model.predict(input_df)

        # Interpret the prediction output as "Rain" or "No Rain"
        result = 'Yes' if prediction[0] == 1 else 'No'

        return {
            "predicted_result": result,
            "input_features": weather.dict()
        }
    except Exception as e:
        return {"error": f"Error occurred: {str(e)}"}
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
