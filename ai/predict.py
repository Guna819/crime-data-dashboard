import torch
import torch.nn as nn
import pandas as pd
import joblib
import numpy as np

# Inverse label map
inv_label_map = {
    0: "ASSAULT",
    1: "BATTERY",
    2: "BURGLARY",
    3: "CRIM SEXUAL ASSAULT",
    4: "CRIMINAL DAMAGE",
    5: "CRIMINAL SEXUAL ASSAULT",
    6: "HOMICIDE",
    7: "ROBBERY",
    8: "THEFT"
}

# Features used during training
features = [
    "Beat", "District", "Ward", "Community Area",
    "X Coordinate", "Y Coordinate", "Year", "Latitude", "Longitude"
]

# Define model (same as train.py)
class CrimeClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CrimeClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Load scaler and model
scaler = joblib.load("scaler.pkl")
model = CrimeClassifier(input_size=len(features), num_classes=len(inv_label_map))
model.load_state_dict(torch.load("crime_model.pth"))
model.eval()

# Example input (replace with your own)
example = {
    "Beat": 1224,
    "District": 12,
    "Ward": 27,
    "Community Area": 28,
    "X Coordinate": 1171188,
    "Y Coordinate": 1901470,
    "Year": 2022,
    "Latitude": 41.885105137,
    "Longitude": -87.646823184
}

# Prepare input
X_input = pd.DataFrame([example])[features].astype(float)
X_scaled = scaler.transform(X_input)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# Predict
with torch.no_grad():
    output = model(X_tensor)
    predicted_class = torch.argmax(output, dim=1).item()

print("Predicted crime type:", inv_label_map[predicted_class])
