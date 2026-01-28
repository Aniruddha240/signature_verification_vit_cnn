import os
import torch
import torchvision.transforms as T
from PIL import Image
from model import SiameseNet
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

# Load trained model
model = SiameseNet().to(device)
model.load_state_dict(torch.load("models/siamese_vit_cnn.pth", map_location=device))
model.eval()

threshold = 0.69   # Optimal from your experiments

y_true = []
y_score = []

root = r"C:\Users\comp\signature_verification_vit_cnn\data\sign_data\test"

persons = []
for folder in os.listdir(root):
    if not folder.endswith("_forg"):
        persons.append(folder)

print("Evaluating on persons:", len(persons))

with torch.no_grad():
    for person in persons:
        genuine_folder = os.path.join(root, person)
        forg_folder = os.path.join(root, person + "_forg")

        genuine_imgs = os.listdir(genuine_folder)
        forg_imgs = os.listdir(forg_folder)

        # Genuine-Genuine (label = 1)
        for i in range(len(genuine_imgs) - 1):
            img1 = os.path.join(genuine_folder, genuine_imgs[i])
            img2 = os.path.join(genuine_folder, genuine_imgs[i+1])

            x1 = transform(Image.open(img1).convert("RGB")).unsqueeze(0).to(device)
            x2 = transform(Image.open(img2).convert("RGB")).unsqueeze(0).to(device)

            e1, e2 = model(x1, x2)
            dist = F.pairwise_distance(e1, e2).item()

            y_true.append(1)
            y_score.append(dist)

        # Genuine-Forged (label = 0)
        for i in range(min(len(genuine_imgs), len(forg_imgs))):
            img1 = os.path.join(genuine_folder, genuine_imgs[i])
            img2 = os.path.join(forg_folder, forg_imgs[i])

            x1 = transform(Image.open(img1).convert("RGB")).unsqueeze(0).to(device)
            x2 = transform(Image.open(img2).convert("RGB")).unsqueeze(0).to(device)

            e1, e2 = model(x1, x2)
            dist = F.pairwise_distance(e1, e2).item()

            y_true.append(0)
            y_score.append(dist)

y_true = np.array(y_true)
y_score = np.array(y_score)

# Predictions using threshold
y_pred = (y_score < threshold).astype(int)

# Metrics
TP = np.sum((y_pred == 1) & (y_true == 1))
TN = np.sum((y_pred == 0) & (y_true == 0))
FP = np.sum((y_pred == 1) & (y_true == 0))
FN = np.sum((y_pred == 0) & (y_true == 1))

accuracy = (TP + TN) / (TP + TN + FP + FN)
far = FP / (FP + TN + 1e-8)   # False Accept Rate
frr = FN / (FN + TP + 1e-8)   # False Reject Rate

print("\n🔹 FINAL EVALUATION RESULTS 🔹")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("FAR (False Accept Rate):", round(far * 100, 2), "%")
print("FRR (False Reject Rate):", round(frr * 100, 2), "%")

# ROC + EER
fpr, tpr, thresholds = roc_curve(y_true, -y_score)  # negative because low distance = genuine
fnr = 1 - tpr

eer_idx = np.nanargmin(np.absolute((fnr - fpr)))
eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

print("EER (Equal Error Rate):", round(eer * 100, 2), "%")

# Plot ROC
plt.figure()
plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([0,1], [0,1], '--')
plt.xlabel("False Accept Rate")
plt.ylabel("True Accept Rate")
plt.title("ROC Curve - Signature Verification")
plt.legend()
plt.show()


best_threshold = thresholds[eer_idx]
print("Best threshold (at EER):", best_threshold)
