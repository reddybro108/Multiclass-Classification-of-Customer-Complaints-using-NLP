# src/model.py
import torch
import torch.nn as nn

class ComplaintClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ComplaintClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
        )
        
    def forward(self, x):
        return self.network(x)
