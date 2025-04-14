import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# Label mapping
label_map = {
    "ASSAULT": 0,
    "BATTERY": 1,
    "BURGLARY": 2,
    "CRIM SEXUAL ASSAULT": 3,
    "CRIMINAL DAMAGE": 4,
    "CRIMINAL SEXUAL ASSAULT": 5,
    "HOMICIDE": 6,
    "ROBBERY": 7,
    "THEFT": 8
}

# Load and preprocess data
df = pd.read_csv("data.csv")

df = df[df["Primary Type"].isin(label_map.keys())]
df["Label"] = df["Primary Type"].map(label_map)

features = [
    "Beat", "District", "Ward", "Community Area",
    "X Coordinate", "Y Coordinate", "Year", "Latitude", "Longitude"
]

X = df[features].fillna(0).astype(float)
y = df["Label"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
joblib.dump(scaler, "scaler.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)

# Define model
class CrimeClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CrimeClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

model = CrimeClassifier(input_size=X_train_tensor.shape[1], num_classes=len(label_map))

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train model
epochs = 20
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Save model
torch.save(model.state_dict(), "crime_model.pth")
print("Model and scaler saved.")