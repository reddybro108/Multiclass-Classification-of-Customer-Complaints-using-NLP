# src/train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from preprocessing import Preprocessor
from model import ComplaintClassifier
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("data/complaints_10k.csv")
preprocessor = Preprocessor()

# Preprocess text
df['clean_text'] = df['complaint_text'].apply(preprocessor.clean_text)

# Feature extraction
X_glove = np.array([preprocessor.text_to_glove(text) for text in df['clean_text']])
X_tfidf = preprocessor.fit_transform_tfidf(df['clean_text'])
X = np.hstack([X_glove, X_tfidf])

# Encode labels
le = LabelEncoder()
y = le.fit_transform(df['category'])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# PyTorch dataset
train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                         torch.tensor(y_train, dtype=torch.long))
test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                        torch.tensor(y_test, dtype=torch.long))

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32)

# Model
input_dim = X.shape[1]
hidden_dim = 128
output_dim = len(le.classes_)
model = ComplaintClassifier(input_dim, hidden_dim, output_dim)

# Training
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 20

for epoch in range(epochs):
    model.train()
    running_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

# Save model and label encoder
torch.save(model.state_dict(), "model.pth")
import joblib
joblib.dump(le, "label_encoder.pkl")
