# main.py (Updated for Cached Float Data)
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import pandas as pd

# --- 1. INITIALIZE APP & LOAD ENV ---
load_dotenv()
app = FastAPI(title="Argo Float Data API", version="2.0.0")

# --- 2. CORS MIDDLEWARE ---
origins = ["http://localhost:9002", "http://localhost:9003"]  # Update if your frontend origin is different
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- 3. DATABASE CONNECTION ---
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set in the .env file")
engine = create_engine(DATABASE_URL)
print("FastAPI connected to the database (CockroachDB).")

# --- 4. PYDANTIC MODELS ---
class MeasurementPoint(BaseModel):
    date: str
    value: float | None
    latitude: float
    longitude: float

class ArgoFloat(BaseModel):
    id: str
    latitude: float
    longitude: float
    lastReported: str
    temperature: List[MeasurementPoint]
    salinity: List[MeasurementPoint]
    pressure: List[MeasurementPoint]

class FloatLocation(BaseModel):
    id: str
    latitude: float
    longitude: float
    lastReported: str

# --- 5. GLOBAL CACHE ---
float_cache: list[dict] = []  # Cache for pre-fetched float locations

# --- 6. FUNCTION TO LOAD FLOATS INTO CACHE ---
def load_float_cache():
    """Fetch latest floats from DB and store in global cache"""
    global float_cache
    try:
        print("🌊 Loading float cache at startup...")

        query = text("""
            WITH latest_measurements AS (
                SELECT *,
                       ROW_NUMBER() OVER (PARTITION BY platform_number ORDER BY time DESC) AS rn
                FROM argo_measurements
                WHERE DATE_TRUNC('month', time) = (
                    SELECT DATE_TRUNC('month', MAX(time))
                    FROM argo_measurements
                )
            )
            SELECT platform_number AS id, latitude, longitude, time AS "lastReported"
            FROM latest_measurements
            WHERE rn = 1;
        """)

        with engine.connect() as connection:
            df = pd.read_sql(query, connection)

        if df.empty:
            print("❌ No float data found in database")
            float_cache = []
            return

        # Convert timestamps to ISO format
        df['lastReported'] = df['lastReported'].apply(lambda x: x.isoformat())
        float_cache = df.to_dict(orient='records')

        print(f"✅ Float cache loaded with {len(float_cache)} floats")

    except Exception as e:
        print(f"❌ Failed to load float cache: {str(e)}")
        float_cache = []

# --- 7. STARTUP EVENT ---
@app.on_event("startup")
async def on_startup():
    """Load float cache when the app starts"""
    load_float_cache()

# --- 8. API ENDPOINTS ---
@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "message": "Backend is running"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Argo Float Data API", "version": "2.0.0"}

@app.get("/test-db")
async def test_database():
    """Test database connection with a simple query"""
    try:
        print("🔍 Testing database connection...")
        query = text("SELECT COUNT(*) as count FROM argo_measurements LIMIT 1")
        with engine.connect() as connection:
            result = connection.execute(query)
            count = result.fetchone()[0]
        print(f"✅ Database test successful. Row count: {count}")
        return {"status": "success", "row_count": count}
    except Exception as e:
        print(f"❌ Database test failed: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.get("/test-floats")
async def test_floats():
    """Test getting a few float records"""
    try:
        print("🔍 Testing float data retrieval...")
        query = text("""
            SELECT 
                platform_number AS id,
                latitude,
                longitude,
                time AS "lastReported"
            FROM argo_measurements
            LIMIT 5;
        """)
        with engine.connect() as connection:
            df = pd.read_sql(query, connection)
        print(f"✅ Retrieved {len(df)} test records")
        df['lastReported'] = df['lastReported'].apply(lambda x: x.isoformat())
        result = df.to_dict(orient='records')
        return {"status": "success", "data": result}
    except Exception as e:
        print(f"❌ Test floats failed: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.get("/api/floats", response_model=List[FloatLocation])
async def get_float_locations():
    """Return pre-fetched float locations from cache"""
    global float_cache
    print(f"🎯 Returning {len(float_cache)} cached float locations")
    return float_cache

@app.get("/api/refresh-floats")
async def refresh_float_cache():
    """Manually refresh the float cache"""
    load_float_cache()
    return {"status": "success", "message": f"Float cache refreshed with {len(float_cache)} floats"}

@app.get("/api/float/{float_id}", response_model=ArgoFloat)
async def get_float_details(float_id: str):
    """Get all measurements for a single float"""
    query = text("""
        SELECT * FROM argo_measurements
        WHERE platform_number = :id
        ORDER BY time;
    """)
    with engine.connect() as connection:
        df = pd.read_sql(query, connection, params={'id': float_id})
    if df.empty:
        return {"error": "Float not found"}

    latest_data = df.iloc[-1]

    temp_points = [
        MeasurementPoint(date=row['time'].isoformat(), value=row.get('temp_adjusted'), latitude=row['latitude'], longitude=row['longitude'])
        for _, row in df.iterrows() if pd.notna(row.get('temp_adjusted'))
    ]
    salinity_points = [
        MeasurementPoint(date=row['time'].isoformat(), value=row.get('psal_adjusted'), latitude=row['latitude'], longitude=row['longitude'])
        for _, row in df.iterrows() if pd.notna(row.get('psal_adjusted'))
    ]
    pressure_points = [
        MeasurementPoint(date=row['time'].isoformat(), value=row.get('pres_adjusted'), latitude=row['latitude'], longitude=row['longitude'])
        for _, row in df.iterrows() if pd.notna(row.get('pres_adjusted'))
    ]

    return ArgoFloat(
        id=str(latest_data['platform_number']),
        latitude=latest_data['latitude'],
        longitude=latest_data['longitude'],
        lastReported=latest_data['time'].isoformat(),
        temperature=temp_points,
        salinity=salinity_points,
        pressure=pressure_points,
    )
