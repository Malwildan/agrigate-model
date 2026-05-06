import pickle
import pandas as pd
import numpy as np # <-- Added NumPy to handle missing math
import openmeteo_requests
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os

# --- 1. Setup Open-Meteo Client ---
openmeteo = openmeteo_requests.Client()

# --- 2. Label Mapping ---
CROP_MAPPING = {0: 'Maize', 1: 'Potato', 2: 'Rice', 3: 'Sugarcane', 4: 'Tomato', 5: 'Wheat'}

class LocationCropRequest(BaseModel):
    latitude: float
    longitude: float
    pH_Value: float

xgb_model = None

# --- 3. Bypass GitHub: Download Model on Startup ---
MODEL_URL = "https://wpxikzshqksgyjckntrt.supabase.co/storage/v1/object/public/models/model.pkl" 

@asynccontextmanager
async def lifespan(app: FastAPI):
    global xgb_model
    temp_model_path = "temp_model.pkl"
    
    print("Downloading fresh model from cloud...")
    try:
        response = requests.get(MODEL_URL)
        response.raise_for_status() 
        
        with open(temp_model_path, "wb") as f:
            f.write(response.content)
            
        with open(temp_model_path, "rb") as f:
            xgb_model = pickle.load(f)
            
        print("Model downloaded and loaded successfully!")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load model from URL: {e}")
        
    yield 
    
    xgb_model = None
    if os.path.exists(temp_model_path):
        os.remove(temp_model_path)

app = FastAPI(lifespan=lifespan)

# --- 4. The Smart Endpoint ---
@app.post("/recommend-crop")
def recommend_crop(data: LocationCropRequest):
    
    if xgb_model is None:
         raise HTTPException(status_code=500, detail="The model failed to load on startup. Check the logs.")
         
    # Fetch Weather Data from Open-Meteo
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": data.latitude,
        "longitude": data.longitude,
        "current": ["temperature_2m", "relative_humidity_2m"],
        "daily": ["precipitation_sum"],
        "past_days": 90,
        "timezone": "auto"
    }
    
    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        
        # Extract Current Temp and Humidity (Safely fallback to averages if NaN)
        current = response.Current()
        raw_temp = current.Variables(0).Value()
        raw_hum = current.Variables(1).Value()
        
        temp = float(raw_temp) if not np.isnan(raw_temp) else 25.0
        humidity = float(raw_hum) if not np.isnan(raw_hum) else 70.0
        
        # Extract Historical Rainfall (Safely sum it, ignoring NaNs)
        daily = response.Daily()
        rainfall_array = daily.Variables(0).ValuesAsNumpy()
        total_rainfall = float(np.nansum(rainfall_array)) # <-- THE FIX IS HERE
        
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Weather API Error: {str(e)}")

    # Prepare Data for XGBoost
    try:
        input_df = pd.DataFrame([{
            "Temperature": temp,
            "Humidity": humidity,
            "pH_Value": data.pH_Value,
            "Rainfall": total_rainfall
        }])
        
        # Predict and Map to String
        prediction_idx = int(xgb_model.predict(input_df)[0])
        recommended_crop = CROP_MAPPING.get(prediction_idx, "Unknown")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model Prediction Error: {str(e)}")
    
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
