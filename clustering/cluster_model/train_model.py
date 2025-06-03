import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# --- Config ---
BATCH_SIZE = 128
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3

# --- Device setup ---
device = torch.device('cuda')
print(f"Using device: {device}")

# --- Dataset class ---
class NPYDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.X = np.load(x_path)
        self.Y = np.load(y_path)
        
        self.Y = self.Y.flatten()
        # Normalize each input vector
        norms = np.linalg.norm(self.X, axis=1, keepdims=True)
        self.X = self.X / np.clip(norms, a_min=1e-10, a_max=None)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.Y[idx], dtype=torch.long)
        return x, y


class LinearModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

# --- Load datasets ---
train_dataset = NPYDataset('x_train.npy', 'y_train.npy')
val_dataset   = NPYDataset('x_val.npy', 'y_val.npy')

input_dim = train_dataset[0][0].shape[0]
num_classes = np.max(train_dataset.Y) + 1

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# --- Setup model, loss, optimizer ---
model = LinearModel(input_dim, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- Training loop with validation ---
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0

    for batch_X, batch_Y in train_loader:
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_X.size(0)

    avg_loss = running_loss / len(train_loader.dataset)
    
    # --- Validation ---
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for val_X, val_Y in val_loader:
            val_X, val_Y = val_X.to(device), val_Y.to(device)
            outputs = model(val_X)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == val_Y).sum().item()
            total += val_Y.size(0)

    val_acc = correct / total
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {avg_loss:.4f} - Val Acc: {val_acc:.4f}")
 

predicted_cluster_ids = []
# --- Validation ---
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for val_X, val_Y in val_loader:
        val_X, val_Y = val_X.to(device), val_Y.to(device)
        outputs = model(val_X)
        preds = torch.argmax(outputs, dim=1)
        for pred in preds:
            predicted_cluster_ids.append(pred)
        correct += (preds == val_Y).sum().item()
        total += val_Y.size(0)

val_acc = correct / total
print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {avg_loss:.4f} - Val Acc: {val_acc:.4f}")


predicted_cluster_ids = [t.cpu().numpy() for t in predicted_cluster_ids]
predicted_cluster_ids = np.array(predicted_cluster_ids)
np.save('predicted_cluster_ids_reduced', predicted_cluster_ids)
