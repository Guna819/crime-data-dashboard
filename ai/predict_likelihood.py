import torch
import numpy as np
import joblib
from train_likelihood import CrimeClassifier
from utils import load_and_preprocess

# Reload data and encoder
(_, _, _, _), label_encoder = load_and_preprocess("data.csv")
num_classes = len(label_encoder.classes_)
input_size = 7  # Hour, Weekday, Month, District, Beat, Arrest, Domestic

model = CrimeClassifier(input_size, num_classes)
model.load_state_dict(torch.load("crime_model_2.pth"))
model.eval()

def predict(input_data):
    """
    input_data: List or np.array with 7 features
    """
    x = torch.tensor(input_data, dtype=torch.float32)
    with torch.no_grad():
        output = model(x)
        prediction = torch.argmax(output, dim=1).item()
        return label_encoder.inverse_transform([prediction])[0]

# 🔍 Example input [Hour, Weekday, Month, District, Beat, Arrest, Domestic]
example = np.array([[15, 2, 5, 6, 631, 1, 0]])  # Adjust accordingly
result = predict(example)
print("Predicted Crime Type:", result)
