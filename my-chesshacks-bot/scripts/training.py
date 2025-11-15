import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# Make sure you've run the PGN processing script first to generate these files
positions = np.load('positions.npy')
moves = np.load('moves.npy')
print(f"Loaded {len(positions)} positions")

X = torch.FloatTensor(positions).permute(0, 3, 1, 2)
y = torch.LongTensor(moves)

split = int(0.9 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ChessModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

best_acc = 0
for epoch in range(10):
    model.train()
    for i, (batch_X, batch_y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(batch_X), batch_y)
        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:
            print(f"Batch {i+1}, Loss: {loss.item():.4f}")
    
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            correct += (model(batch_X).argmax(1) == batch_y).sum().item()
    
    val_acc = 100 * correct / len(y_val)
    print(f"Epoch {epoch+1} - Val Acc: {val_acc:.2f}%")
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'chess_model.pth')

print(f"Best Val Acc: {best_acc:.2f}%")