from fastapi import FastAPI, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from . import models, schemas, database
import secrets
import string
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json
import numpy as np
from fastapi.encoders import jsonable_encoder

SECRET_KEY = "1234567890"  # Change this to a secure secret key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30  # Token expiration time in minutes

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

models.Base.metadata.create_all(bind=database.engine)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None

def generate_reset_token(length: int = 32):
    chars = string.ascii_letters + string.digits
    return ''.join(secrets.choice(chars) for _ in range(length))

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/signup")
def signup(user: schemas.SignupModel, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = pwd_context.hash(user.password)
    new_user = models.User(
        fullname=user.fullname,
        email=user.email,
        hashed_password=hashed_password
    )
    access_token = create_access_token(data={"sub": new_user.email})
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    new_user.hashed_password = None
    return {"message": "User registered successfully", "access_token": access_token, "user": new_user}

@app.post("/login")
def login(user: schemas.LoginModel, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if not db_user or not pwd_context.verify(user.password, db_user.hashed_password):
        raise HTTPException(status_code=400, detail="Invalid email or password")

    # generate a token
    access_token = create_access_token(data={"sub": user.email})
    db_user.hashed_password = None
    
    return {"message": f"Welcome back, {db_user.fullname}!", "access_token": access_token, "user": db_user}


@app.post("/reset-password/request")
def reset_password_request(payload: schemas.PasswordResetRequest, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == payload.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Generate and log the token (in real world, save to DB or send via email)
    reset_token = generate_reset_token()
    token_expiry = datetime.utcnow() + timedelta(hours=1)
    user.reset_token = reset_token
    user.token_expiry = token_expiry
    db.commit()
    
    # For now, just return it in response (test only!)
    return {"message": "Reset token generated", "reset_token": reset_token}

@app.post("/reset-password")
def reset_password(payload: schemas.ResetPasswordRequest, db: Session = Depends(get_db)):
    # Check if the reset token exists and match it with the user in the database
    user = db.query(models.User).filter(models.User.reset_token == payload.reset_token).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="Invalid reset token")
    
    # Optionally check token expiration (if you added a timestamp)
    if datetime.utcnow() > user.token_expiry:
        raise HTTPException(status_code=400, detail="Reset token expired")

    # Hash the new password
    hashed_password = pwd_context.hash(payload.new_password)

    # Update the user's password and clear the reset token
    user.hashed_password = hashed_password
    user.reset_token = None  # Reset the token after use
    db.commit()

    return {"message": "Password reset successfully"}

@app.get("/crime-data")
def get_crime_data(
    crime_type: str = Query(None),
    start_date: str = Query(None),  # format: YYYY-MM-DD
    end_date: str = Query(None),
    district: int = Query(None),
    day_period: str = Query(None),  # "day" or "night"
    nrows: int = 1000,
):
    df = pd.read_csv("app/data.csv", nrows=nrows)
    # df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.where(pd.notnull(df), '')

    if crime_type:
        df = df[df["Primary Type"].str.lower() == crime_type.lower()]

    if start_date:
        df = df[df["Date"] >= start_date]

    if end_date:
        df = df[df["Date"] <= end_date]
    
    if district:
        df = df[df["District"] == district]

    if day_period in ["day", "night"]:
        df["hour"] = df["Date"].dt.hour
        if day_period == "day":
            df = df[(df["hour"] >= 6) & (df["hour"] < 18)]
    
    # Convert to list of dictionaries
    records = df.to_dict(orient="records")

    # Make sure it's safe for JSON response
    return {"data": jsonable_encoder(records)}

@app.get("/crime-count-by-date")
def crime_count_by_date(nrows: int = 1000):
    # Load CSV data (limit with nrows for performance)
    df = pd.read_csv("app/data.csv", nrows=nrows)

    df = df.where(pd.notnull(df), '')

    # Convert 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Drop rows with invalid or missing dates
    df = df.dropna(subset=['Date'])

    # Format just the date (YYYY-MM-DD)
    df['Day'] = df['Date'].dt.date

    # Group by Day and count rows
    result = df.groupby('Day').size().reset_index(name='count')

    # Convert to list of dicts for JSON response
    return {"data": result.to_dict(orient="records")}

@app.get("/dashboard")
def get_dashboard(nrows: int = 1000):
    df = pd.read_csv("app/data.csv", nrows=nrows)

    df = df.where(pd.notnull(df), '')

    # Basic cleanup
    df['Primary Type'] = df['Primary Type'].fillna("Unknown")
    df['District'] = df['District'].fillna("Unknown")
    df['Arrest'] = df['Arrest'].fillna(False)

    # Total incidents
    total_incidents = len(df)

    # Most common crime
    most_common_crime = df['Primary Type'].mode()[0]

    # District with highest crime
    highest_crime_district = df['District'].value_counts().idxmax()
    highest_crime_district_str = f"District {int(highest_crime_district)}" if pd.notna(highest_crime_district) and highest_crime_district != "Unknown" else "Unknown"

    # Crime solve rate = % of crimes with arrest == True
    crime_solve_rate = round((df['Arrest'] == True).sum() / total_incidents * 100, 2) if total_incidents else 0

    # Unique values
    unique_crime_types = sorted(df['Primary Type'].dropna().unique().tolist())
    

    districts = [
        { "number": 1, "name": "Central", "neighborhoods": "Loop, South Loop", "active": True },
        { "number": 2, "name": "Wentworth", "neighborhoods": "Bronzeville, Washington Park", "active": True },
        { "number": 3, "name": "Grand Crossing", "neighborhoods": "Greater Grand Crossing, Chatham", "active": True },
        { "number": 4, "name": "South Chicago", "neighborhoods": "South Chicago, East Side, South Deering", "active": True },
        { "number": 5, "name": "Calumet", "neighborhoods": "Pullman, Roseland", "active": True },
        { "number": 6, "name": "Gresham", "neighborhoods": "Auburn Gresham, West Englewood", "active": True },
        { "number": 7, "name": "Englewood", "neighborhoods": "Englewood, West Englewood", "active": True },
        { "number": 8, "name": "Chicago Lawn", "neighborhoods": "Chicago Lawn, Ashburn", "active": True },
        { "number": 9, "name": "Deering", "neighborhoods": "Bridgeport, McKinley Park", "active": True },
        { "number": 10, "name": "Ogden", "neighborhoods": "North Lawndale, Little Village", "active": True },
        { "number": 11, "name": "Harrison", "neighborhoods": "West Garfield Park, Humboldt Park", "active": True },
        { "number": 12, "name": "Near West", "neighborhoods": "Near West Side, West Loop", "active": True },
        { "number": 13, "name": "Discontinued", "neighborhoods": "-", "active": False },
        { "number": 14, "name": "Shakespeare", "neighborhoods": "Logan Square, Bucktown", "active": True },
        { "number": 15, "name": "Austin", "neighborhoods": "Austin", "active": True },
        { "number": 16, "name": "Jefferson Park", "neighborhoods": "Jefferson Park, Portage Park", "active": True },
        { "number": 17, "name": "Albany Park", "neighborhoods": "Albany Park, Irving Park", "active": True },
        { "number": 18, "name": "Near North", "neighborhoods": "River North, Gold Coast", "active": True },
        { "number": 19, "name": "Town Hall", "neighborhoods": "Lakeview, Lincoln Park", "active": True },
        { "number": 20, "name": "Lincoln", "neighborhoods": "Edgewater, Rogers Park (south)", "active": True },
        { "number": 21, "name": "Merged with District 1", "neighborhoods": "-", "active": False },
        { "number": 22, "name": "Morgan Park", "neighborhoods": "Morgan Park, Beverly", "active": True },
        { "number": 23, "name": "Merged with District 19", "neighborhoods": "-", "active": False },
        { "number": 24, "name": "Rogers Park", "neighborhoods": "Rogers Park, West Ridge", "active": True },
        { "number": 25, "name": "Grand Central", "neighborhoods": "Belmont Cragin, Montclare", "active": True },
    ]

    recent_incidents = df.sort_values(by="Date", ascending=False).head(10)
    recent_rows = recent_incidents.to_dict(orient="records")

    return {
        "total_incidents": total_incidents,
        "most_common_crime": most_common_crime,
        "highest_crime_district": highest_crime_district_str,
        "crime_solve_rate_percent": crime_solve_rate,
        "unique_crime_types": unique_crime_types,
        "unique_districts": districts,
        "recent_incidents": recent_rows
    }

@app.get("/heatmap")
def get_heatmap(
    crime_type: str = Query(None),
    start_date: str = Query(None),  # format: YYYY-MM-DD
    end_date: str = Query(None),
    district: int = Query(None),
    day_period: str = Query(None),  # "day" or "night"
    nrows: int = 1000,
):
    df = pd.read_csv("app/data.csv", nrows=nrows)

    # Convert empty strings to NaN
    df.replace("", np.nan, inplace=True)
    df.dropna(subset=["Latitude", "Longitude"], inplace=True)

    # Parse date column to datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Apply filters
    if crime_type:
        df = df[df["Primary Type"].str.lower() == crime_type.lower()]

    if start_date:
        start = pd.to_datetime(start_date)
        df = df[df["Date"] >= start]

    if end_date:
        end = pd.to_datetime(end_date)
        df = df[df["Date"] <= end]

    if district:
        df = df[df["District"] == district]

    if day_period in ["day", "night"]:
        df["hour"] = df["Date"].dt.hour
        if day_period == "day":
            df = df[(df["hour"] >= 6) & (df["hour"] < 18)]
        else:  # night
            df = df[(df["hour"] < 6) | (df["hour"] >= 18)]

    # Group by location and count incidents (intensity)
    grouped = df.groupby(['Latitude', 'Longitude']).size().reset_index(name='intensity')

    # Format to [lat, lng, intensity]
    heatmap_data = grouped[['Latitude', 'Longitude', 'intensity']].values.tolist()

    return {"data": heatmap_data}


def load_data():
    df = pd.read_csv("app/data.csv", nrows=100)  # Path to your CSV file
    df = df.where(pd.notnull(df), '')
    
    # Data preprocessing
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Month'] = df['Date'].dt.month 
    df['Hour'] = df['Date'].dt.hour
    df['Year'] = df['Date'].dt.year
    df['Season'] = df['Date'].dt.month % 12 // 3 + 1  # 1 = Winter, 2 = Spring, 3 = Summer, 4 = Fall
    df['Primary Type'] = df['Primary Type'].fillna("Unknown")
    return df

@app.get("/crime/bar-chart")
async def crime_bar_chart(month: int = Query(None, ge=1, le=12), year: int = Query(None, ge=2001, le=2025)):
    df = load_data()

    if year:
        df = df[df['Year'] == year]

    if month:
        df = df[df['Month'] == month]
    
    # Crime frequency by type
    crime_counts = df['Primary Type'].value_counts().reset_index()
    crime_counts.columns = ['Crime Type', 'Frequency']
    
    # Return the data in a format the frontend can use
    bar_chart_data = crime_counts.to_dict(orient='records')

    # convert the column names to camel case
    bar_chart_data = [{k.lower().replace(' ', '_'): v for k, v in row.items()} for row in bar_chart_data]
    
    return {
        "data": bar_chart_data
    }

@app.get("/crime/scatter-plot")
async def crime_scatter_plot():
    df = load_data()

    # Crime frequency by hour and crime type (Scatter plot data)
    scatter_data = df.groupby(['Season', 'Primary Type']).size().reset_index(name='Frequency')
    
    # Return the data in a format the frontend can use
    scatter_plot_data = scatter_data.to_dict(orient='records')

    # convert the column names to camel case
    scatter_plot_data = [{k.lower().replace(' ', '_'): v for k, v in row.items()} for row in scatter_plot_data]
    
    return {
        "data": scatter_plot_data
    }