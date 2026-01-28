import torch.nn as nn
from torchvision import models
import timm
import torch

class HybridEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # CNN branch
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()

        # Freeze CNN
        for param in self.cnn.parameters():
            param.requires_grad = False

        # ViT branch (VERY HEAVY – freeze it)
        self.vit = timm.create_model("vit_tiny_patch16_224", pretrained=True)
        self.vit.head = nn.Identity()

        for param in self.vit.parameters():
            param.requires_grad = False

        # Trainable head only
        self.fc = nn.Sequential(
            nn.Linear(512 + 192, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        cnn_feat = self.cnn(x)
        vit_feat = self.vit(x)

        feat = torch.cat([cnn_feat, vit_feat], dim=1)
        emb = self.fc(feat)

        return emb


class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = HybridEncoder()

    def forward(self, x1, x2):
        e1 = self.encoder(x1)
        e2 = self.encoder(x2)
        return e1, e2
