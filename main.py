import pickle
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

# --- 1. Setup Open-Meteo Client ---
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# --- 2. Label Mapping ---
CROP_MAPPING = {0: 'Maize', 1: 'Potato', 2: 'Rice', 3: 'Sugarcane', 4: 'Tomato', 5: 'Wheat'}

# --- 3. Define Input Schema ---
# The user only sends location and soil pH now!
class LocationCropRequest(BaseModel):
    latitude: float
    longitude: float
    pH_Value: float

xgb_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global xgb_model
    with open("agrigate-model.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    yield
    xgb_model = None

app = FastAPI(lifespan=lifespan)

# --- 4. The Smart Endpoint ---
@app.post("/recommend-crop")
async def recommend_crop(data: LocationCropRequest):
    # Step A: Fetch Weather Data from Open-Meteo
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": data.latitude,
        "longitude": data.longitude,
        "current": ["temperature_2m", "relative_humidity_2m"],
        "daily": ["precipitation_sum"],
        "past_days": 90 # Fetch last 3 months of rain
    }
    
    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        
        # Extract Current Temp and Humidity
        current = response.Current()
        temp = current.Variables(0).Value()
        humidity = current.Variables(1).Value()
        
        # Extract Historical Rainfall and sum it up
        daily = response.Daily()
        rainfall_array = daily.Variables(0).ValuesAsNumpy()
        total_rainfall = rainfall_array.sum()
        
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Weather API Error: {str(e)}")

    # Step B: Prepare Data for XGBoost
    input_df = pd.DataFrame([{
        "Temperature": temp,
        "Humidity": humidity,
        "pH_Value": data.pH_Value,
        "Rainfall": total_rainfall
    }])
    
    # Step C: Predict and Map to String
    prediction_idx = int(xgb_model.predict(input_df)[0])
    recommended_crop = CROP_MAPPING.get(prediction_idx, "Unknown")
    
    return {
        "status": "success",
        "location": {"lat": data.latitude, "lon": data.longitude},
        "fetched_features": {
            "temperature": round(temp, 2),
            "humidity": round(humidity, 2),
            "total_rainfall_90d": round(total_rainfall, 2)
        },
        "recommendation": recommended_crop
    }
