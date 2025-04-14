import torch
import torch.nn as nn
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Load encoder
target_encoder = joblib.load('../models/label_encoder.pkl')
block_encoder = joblib.load('../models/block_encoder.pkl')
district_encoder = joblib.load('../models/district_encoder.pkl')
# Define the same model structure
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
model.load_state_dict(torch.load('../models/crime_model.pth'))
model.eval()

# Sample input for prediction
input_data = torch.tensor([[block_encoder.transform(['137XX S LEYDEN AVE'])[0], 0.95, 3.0]], dtype=torch.float32)
with torch.no_grad():
    output = model(input_data)
    predicted = torch.argmax(output, dim=1)

print('Predicted class:', target_encoder.inverse_transform(predicted.numpy())[0])

