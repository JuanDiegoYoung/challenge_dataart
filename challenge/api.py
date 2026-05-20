from pathlib import Path

import fastapi
import pandas as pd

from challenge.model import DelayModel


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "data.csv"
VALID_OPERATORS = {
    "Aerolineas Argentinas",
    "Aeromexico",
    "Air Canada",
    "Air France",
    "Alitalia",
    "American Airlines",
    "Austral",
    "Avianca",
    "British Airways",
    "Copa Air",
    "Delta Air",
    "Gol Trans",
    "Grupo LATAM",
    "Iberia",
    "JetSmart SPA",
    "K.L.M.",
    "Lacsa",
    "Latin American Wings",
    "Oceanair Linhas Aereas",
    "Plus Ultra Lineas Aereas",
    "Qantas Airways",
    "Sky Airline",
    "United Airlines",
}
VALID_FLIGHT_TYPES = {"I", "N"}
VALID_MONTHS = set(range(1, 13))


def _load_model() -> DelayModel:
    data = pd.read_csv(DATA_PATH)
    model = DelayModel()
    features, target = model.preprocess(data=data, target_column="delay")
    model.fit(features=features, target=target)
    return model


model = _load_model()

app = fastapi.FastAPI()

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(payload: dict) -> dict:
    flights = payload.get("flights")

    if not isinstance(flights, list) or not flights:
        raise fastapi.HTTPException(status_code=400, detail="Invalid flights payload")

    for flight in flights:
        if not _is_valid_flight(flight):
            raise fastapi.HTTPException(status_code=400, detail="Invalid flight data")

    features = model.preprocess(data=pd.DataFrame(flights))
    return {"predict": model.predict(features=features)}


def _is_valid_flight(flight: dict) -> bool:
    if not isinstance(flight, dict):
        return False

    try:
        month = int(flight.get("MES"))
    except (TypeError, ValueError):
        return False

    return (
        flight.get("OPERA") in VALID_OPERATORS
        and flight.get("TIPOVUELO") in VALID_FLIGHT_TYPES
        and month in VALID_MONTHS
    )