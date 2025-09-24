import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 50
LR = 0.00005
EMBEDDING_DIM = 256
NUM_HEADS = 8
FFN_HIDDEN = 512
NUM_LAYERS = 4
PATIENCE = 5

# Load data
path = "."  # The script and CSV files should be in the same directory
print("Looking for CSV files in:", path)

files = [f for f in os.listdir(path) if f.endswith(".csv")]
print("Files found in directory:", files)

dataframes = []
labels = []

for file in files:
    print(f"Loading file: {file}")
    df = pd.read_csv(os.path.join(path, file))
    df["label"] = file  # Assign filename as label
    dataframes.append(df)
    labels.append(file)

if not dataframes:
    print("Error: No valid CSV files were loaded. Check filenames and directory.")
    exit()

data = pd.concat(dataframes, ignore_index=True)

# Encode labels
label_encoder = LabelEncoder()
label_encoder.fit(labels)
data["label"] = label_encoder.transform(data["label"])

# Normalize features
features = data.drop(columns=["label"])
scaler = StandardScaler()
features = scaler.fit_transform(features)
labels = data["label"].values

# Train-test split
X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.3, stratify=labels, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Custom Dataset class
class MotorDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = MotorDataset(X_train, y_train)
val_dataset = MotorDataset(X_val, y_val)
test_dataset = MotorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Hyperparameters
BATCH_SIZE = 32
LR = 0.00005
EMBEDDING_DIM = 256
NUM_HEADS = 8
FFN_HIDDEN = 512
NUM_LAYERS = 4

# Transformer Model
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, EMBEDDING_DIM)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=EMBEDDING_DIM, nhead=NUM_HEADS, dim_feedforward=FFN_HIDDEN, batch_first=True),
            num_layers=NUM_LAYERS
        )
        self.fc = nn.Linear(EMBEDDING_DIM, num_classes)
    
def forward(self, x):
    x = self.embedding(x).unsqueeze(1)  # Add sequence dimension
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.fc(x)

# Model Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(label_encoder.classes_)
model = TransformerClassifier(input_dim=X_train.shape[1], num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training Loop with Early Stopping
best_val_loss = float('inf')
epochs_no_improve = 0
train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0
    y_preds, y_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            y_preds.extend(preds.cpu().numpy())
            y_true.extend(y_batch.cpu().numpy())
    
    val_loss /= len(val_loader)
    val_acc = accuracy_score(y_true, y_preds)
    
    train_losses.append(total_loss / len(train_loader))
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    
    print(f"Epoch {epoch+1}: Train Loss = {total_loss/len(train_loader):.4f}, Val Loss = {val_loss:.4f}, Val Accuracy = {val_acc:.4f}")
    
    # Save accuracy to file
    with open("eva.txt", "a") as f:
        f.write(f"Epoch {epoch+1}, Train Loss: {total_loss/len(train_loader):.4f}, "
                f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}\n")

    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print("Early stopping triggered.")
            break

# Evaluate on test set
model.eval()
y_test_preds, y_test_true = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        _, preds = torch.max(outputs, 1)
        y_test_preds.extend(preds.cpu().numpy())
        y_test_true.extend(y_batch.cpu().numpy())

test_acc = accuracy_score(y_test_true, y_test_preds)

# Save test accuracy
with open("test_accuracy.txt", "w") as f:
    f.write(f"Final Test Accuracy: {test_acc:.4f}\n")

# Save classification report
with open("classification_report.txt", "w") as f:
    f.write(classification_report(y_test_true, y_test_preds, target_names=label_encoder.classes_))

# Generate Confusion Matrix
conf_matrix = confusion_matrix(y_test_true, y_test_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_, 
            yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()
