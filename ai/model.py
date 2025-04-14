# model.py
import torch.nn as nn

class CrimeModel(nn.Module):
    def __init__(self, input_size):
        super(CrimeModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # Binary classification
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)
