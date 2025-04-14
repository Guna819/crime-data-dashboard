import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch

def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)

    # Fix the date
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    df = df.dropna(subset=["Date"])

    # Time features
    df["Hour"] = df["Date"].dt.hour
    df["Weekday"] = df["Date"].dt.weekday
    df["Month"] = df["Date"].dt.month

    # Features to use
    features = ["Hour", "Weekday", "Month", "District", "Beat", "Arrest", "Domestic"]
    df = df[features + ["Primary Type"]].dropna()

    # Encode categorical
    df["Arrest"] = df["Arrest"].astype(int)
    df["Domestic"] = df["Domestic"].astype(int)

    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["Primary Type"])

    X = df[features].values.astype(np.float32)
    y = df["label"].values.astype(np.int64)

    return train_test_split(X, y, test_size=0.2, random_state=42), label_encoder
