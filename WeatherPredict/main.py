from fastapi import FastAPI
from pydantic import BaseModel
import joblib


app = FastAPI()

# Define the input data structure
class WeatherInput(BaseModel):
    location: str
    min_temp: float
    max_temp: float
    rainfall: float
    evaporation: float
    sunshine: float
    wind_gust_dir: str
    wind_gust_speed: float
    wind_dir_9am: str
    wind_dir_3pm: str
    wind_speed_9am: float
    wind_speed_3pm: float
    humidity_9am: float
    humidity_3pm: float
    pressure_9am: float
    pressure_3pm: float
    cloud_9am: float
    cloud_3pm: float
    temp_9am: float
    temp_3pm: float
    rain_today: int  # 0 or 1

# Load your trained model (make sure to save it as a .py file)
model = joblib.load('backend/assignme.joblib')

@app.post('/predict/')
async def predict(weather: WeatherInput):
    # Prepare the input data for the model
    input_data = [[
        weather.location,
        weather.min_temp,
        weather.max_temp,
        weather.rainfall,
        weather.evaporation,
        weather.sunshine,
        weather.wind_gust_dir,
        weather.wind_gust_speed,
        weather.wind_dir_9am,
        weather.wind_dir_3pm,
        weather.wind_speed_9am,
        weather.wind_speed_3pm,
        weather.humidity_9am,
        weather.humidity_3pm,
        weather.pressure_9am,
        weather.pressure_3pm,
        weather.cloud_9am,
        weather.cloud_3pm,
        weather.temp_9am,
        weather.temp_3pm,
        weather.rain_today
    ]]

    # Make a prediction
    prediction = model.predict(input_data)

    return {"prediction": prediction[0]}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Weather Prediction API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
