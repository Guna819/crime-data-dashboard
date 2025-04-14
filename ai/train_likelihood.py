import torch
import torch.nn as nn
import torch.optim as optim
from utils import load_and_preprocess

class CrimeClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CrimeClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Load data
(X_train, X_test, y_train, y_test), label_encoder = load_and_preprocess("data.csv")

X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)
X_test = torch.tensor(X_test)
y_test = torch.tensor(y_test)

model = CrimeClassifier(X_train.shape[1], len(label_encoder.classes_))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(10):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Save model and label encoder
torch.save(model.state_dict(), "crime_model_2.pth")
import joblib
joblib.dump(label_encoder, "label_encoder_2.pkl")
