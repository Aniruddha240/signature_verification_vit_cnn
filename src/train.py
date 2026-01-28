import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T

from dataset import SignatureDataset
from model import SiameseNet

# Contrastive Loss
def contrastive_loss(e1, e2, label, margin=1.0):
    dist = F.pairwise_distance(e1, e2)
    loss = label * dist.pow(2) + (1 - label) * F.relu(margin - dist).pow(2)
    return loss.mean()

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

dataset = SignatureDataset(r"C:\Users\comp\signature_verification_vit_cnn\data\sign_data\train", transform)

loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

model = SiameseNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(2):
    total_loss = 0

    for i, (x1, x2, y) in enumerate(loader):
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)

        e1, e2 = model(x1, x2)
        loss = contrastive_loss(e1, e2, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 20 == 0:
            print(f"Batch {i} | Loss: {loss.item():.4f}")


        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f}")

import os
os.makedirs("models", exist_ok=True)

torch.save(model.state_dict(), "models/siamese_vit_cnn.pth")
print("✅ Model saved in models/siamese_vit_cnn.pth")
