# train.py
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("../data/crime_data.csv")

# Filter rows with non-null values in necessary columns
df = df.dropna(subset=["Block", "District", "Date", "Primary Type"])
df = df[df["Primary Type"].isin(["ASSAULT", "BATTERY", "BURGLARY", "CRIM SEXUAL ASSAULT", "CRIMINAL DAMAGE", "CRIMINAL SEXUAL ASSAULT", "HOMICIDE", "ROBBERY", "THEFT"])]

# Feature engineering
df["Hour"] = pd.to_datetime(df["Date"]).dt.hour

# Label encoding
block_encoder = LabelEncoder()
district_encoder = LabelEncoder()
target_encoder = LabelEncoder()

block_encoded = block_encoder.fit_transform(df["Block"])
district_encoded = district_encoder.fit_transform(df["District"].astype(str))
target = target_encoder.fit_transform(df["Primary Type"])

features = torch.tensor(
    list(zip(block_encoded, district_encoded, df["Hour"])), dtype=torch.float32
)
targets = torch.tensor(target, dtype=torch.long)

# Dataset and DataLoader
class CrimeDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

dataset = CrimeDataset(features, targets)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Model
class CrimeModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CrimeModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

input_size = 3
hidden_size = 64
output_size = len(target_encoder.classes_)

model = CrimeModel(input_size, hidden_size, output_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Train loop
for epoch in range(10):
    for x_batch, y_batch in dataloader:
        optimizer.zero_grad()
        out = model(x_batch)
        loss = loss_fn(out, y_batch)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Save model and encoders
torch.save(model.state_dict(), "../models/crime_model.pth")
torch.save(block_encoder, "../models/block_encoder.pth")
torch.save(district_encoder, "../models/district_encoder.pth")
torch.save(target_encoder, "../models/target_encoder.pth")

joblib.dump(target_encoder, '../models/label_encoder.pkl')
joblib.dump(block_encoder, '../models/block_encoder.pkl')
joblib.dump(district_encoder, '../models/district_encoder.pkl')