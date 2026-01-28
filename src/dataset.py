import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import random

class SignatureDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_pairs=5000):
        self.root = root_dir
        self.transform = transform

        self.pairs = []   # (img1_path, img2_path, label)

        persons = []

        # Collect all persons (001, 002, ...)
        for folder in os.listdir(self.root):
            if not folder.endswith("_forg"):
                persons.append(folder)

        # Generate pairs
        for person in persons:
            genuine_folder = os.path.join(self.root, person)
            forg_folder = os.path.join(self.root, person + "_forg")

            genuine_imgs = os.listdir(genuine_folder)
            forg_imgs = os.listdir(forg_folder)

            # Genuine–Genuine pairs (positive)
            for i in range(len(genuine_imgs) - 1):
                img1 = os.path.join(genuine_folder, genuine_imgs[i])
                img2 = os.path.join(genuine_folder, genuine_imgs[i+1])
                self.pairs.append((img1, img2, 1))

            # Genuine–Forged pairs (negative)
            for i in range(min(len(genuine_imgs), len(forg_imgs))):
                img1 = os.path.join(genuine_folder, genuine_imgs[i])
                img2 = os.path.join(forg_folder, forg_imgs[i])
                self.pairs.append((img1, img2, 0))

        # Shuffle and limit pairs (VERY IMPORTANT for CPU)
        random.shuffle(self.pairs)
        self.pairs = self.pairs[:max_pairs]

        print("Total training pairs:", len(self.pairs))

    def __getitem__(self, idx):
        path1, path2, label = self.pairs[idx]

        img1 = Image.open(path1).convert("RGB")
        img2 = Image.open(path2).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.pairs)
