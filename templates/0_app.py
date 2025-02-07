from fastapi import FastAPI, Query
import psycopg2
from typing import List
from dotenv import load_dotenv
import os
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

POSTGIS_PASSWORD = os.getenv("POSTGIS_TEST1_PASSWORD")

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (or specify frontend URL)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to PostgreSQL
# psql -h localhost -p 5432 -U ggatej-pg postgis_test1
conn = psycopg2.connect(
    dbname="postgis_test1",
    user="ggatej-pg",
    password=POSTGIS_PASSWORD,
    host="localhost",
    port="5432"
)
cur = conn.cursor()

@app.get("/")
def home():
    return {"message": "FastAPI is running!"}


@app.get("/get_filenames/")
def get_filenames(min_lat: float, max_lat: float, min_lon: float, max_lon: float):
    """
    Retrieve filenames from dump1_cleaned where points fall within the selected bounding box.
    """
    query = """
        SELECT filename FROM dump1_cleaned
        WHERE ST_Within(
        ST_Transform(wkb_geometry, 4326), 
        ST_MakeEnvelope(%s, %s, %s, %s, 4326)
        );
    """
    cur.execute(query, (min_lon, min_lat, max_lon, max_lat))
    results = cur.fetchall()
    
    filenames = [row[0] for row in results]
    return {"selected_filenames": filenames}
