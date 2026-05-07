# AgriGate Crop Recommendation API

FastAPI service for crop recommendation based on:
- Geographic location (latitude, longitude)
- Soil pH value
- Weather features fetched from Open-Meteo (temperature, humidity, and 90-day rainfall)

The app loads an XGBoost model from Supabase Storage at startup, then exposes a prediction endpoint.

## Features

- FastAPI REST endpoint: `POST /recommend-crop`
- Automatic weather data retrieval from Open-Meteo
- NaN-safe weather preprocessing with NumPy fallbacks
- XGBoost inference with mapped crop labels

## Tech Stack

- FastAPI
- Uvicorn
- XGBoost
- Pandas, NumPy
- Open-Meteo API (`openmeteo-requests`)

## Project Structure

- `main.py` - API app, model loading, weather fetch, and prediction logic
- `requirements.txt` - Python dependencies

## Requirements

- Python 3.10+ recommended
- Internet access (required to download model and fetch weather data)

## Installation

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Locally

```bash
uvicorn main:app --reload
```

Server will run by default at:
- http://127.0.0.1:8000

Interactive API docs:
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

## Production Deployment

This API is deployed on Railway:
- Base URL: https://agrigate-model-production.up.railway.app
- Swagger UI: https://agrigate-model-production.up.railway.app/docs
- ReDoc: https://agrigate-model-production.up.railway.app/redoc

You can call the production endpoint directly:

```bash
curl -X POST "https://agrigate-model-production.up.railway.app/recommend-crop" \
  -H "Content-Type: application/json" \
  -d "{\"latitude\":-6.2,\"longitude\":106.8,\"pH_Value\":6.5}"
```

## API Endpoint

### POST /recommend-crop

Request body:

```json
{
  "latitude": -6.2,
  "longitude": 106.8,
  "pH_Value": 6.5
}
```

Successful response example:

```json
{
  "status": "success",
  "location": {
    "lat": -6.2,
    "lon": 106.8
  },
  "fetched_features": {
    "temperature": 29.1,
    "humidity": 73.4,
    "total_rainfall_90d": 412.7
  },
  "recommendation": "Rice"
}
```

## Quick Test

### cURL

```bash
curl -X POST "http://127.0.0.1:8000/recommend-crop" \
  -H "Content-Type: application/json" \
  -d "{\"latitude\":-6.2,\"longitude\":106.8,\"pH_Value\":6.5}"
```

### PowerShell

```powershell
$body = @{
  latitude  = -6.2
  longitude = 106.8
  pH_Value  = 6.5
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/recommend-crop" -Method POST -ContentType "application/json" -Body $body
```

## Notes

- The model file is downloaded on app startup from a public Supabase Storage URL configured in `main.py`.
- If model download fails, the endpoint returns HTTP 500 with startup error details.
- If weather API fails, the endpoint returns HTTP 502.
