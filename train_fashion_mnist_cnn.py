Python 3.13.2 (v3.13.2:4f8bb3947cf, Feb  4 2025, 11:51:10) [Clang 15.0.0 (clang-1500.3.9.4)] on darwin
Type "help", "copyright", "credits" or "license()" for more information.
#!/usr/bin/env python
# pip install torch torchvision torchaudio torchmetrics
import torch, torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy

BATCH = 128; EPOCHS = 5; LR = 1e-3; DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
... 
... transform = transforms.Compose([transforms.ToTensor()])
... train_ds = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
... test_ds  = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)
... train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
... test_dl  = DataLoader(test_ds,  batch_size=BATCH)
... 
... class CNN(nn.Module):
...     def __init__(self): 
...         super().__init__()
...         self.net = nn.Sequential(
...             nn.Conv2d(1,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
...             nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
...             nn.Flatten(), nn.Linear(64*7*7,128), nn.ReLU(), nn.Linear(128,10)
...         )
...     def forward(self,x): return self.net(x)
... 
... model = CNN().to(DEVICE)
... opt = torch.optim.Adam(model.parameters(), lr=LR)
... criterion = nn.CrossEntropyLoss()
... acc = Accuracy(task="multiclass", num_classes=10).to(DEVICE)
... 
... for epoch in range(EPOCHS):
...     model.train()
...     for xb,yb in train_dl:
...         xb,yb = xb.to(DEVICE), yb.to(DEVICE)
...         opt.zero_grad(); out = model(xb)
...         loss = criterion(out,yb); loss.backward(); opt.step()
...     model.eval(); acc.reset()
...     with torch.no_grad():
...         for xb,yb in test_dl:
...             xb,yb = xb.to(DEVICE), yb.to(DEVICE)
...             acc.update(model(xb), yb)
...     print(f"Epoch {epoch+1}: Test Acc = {acc.compute():.3f}")
... 
... torch.save(model.state_dict(), "fashion_cnn.pth")
... print("Model saved → fashion_cnn.pth")
