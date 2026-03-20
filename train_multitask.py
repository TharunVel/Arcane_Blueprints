import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from datetime import datetime

# ====================== AUGMENTATIONS ======================
aug = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.7),
    A.GaussianBlur(p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# ====================== PATHS ======================
BASE_DIR = "DR_Datasets"
IMAGE_DIR = os.path.join(BASE_DIR, "B. Disease Grading", "1. Original Images", "a. Training Set")
MASK_BASE = os.path.join(BASE_DIR, "A. Segmentation", "2. All Segmentation Groundtruths", "a. Training Set")
CSV_PATH  = os.path.join(BASE_DIR, "B. Disease Grading", "2. Groundtruths",
                         "a. IDRiD_Disease Grading_Training Labels.csv")

# ====================== ORDINAL LOSS ======================
def ordinal_loss(preds, targets):
    probs = torch.softmax(preds, dim=1)
    class_range = torch.arange(0, 5).float().to(preds.device)
    targets = targets.unsqueeze(1).float()
    loss = torch.sum(probs * (class_range - targets) ** 2, dim=1)
    return loss.mean()

# ====================== LESION BOX ======================
def get_combined_lesion_box(img_name):
    lesion_types = [
        ("1. Microaneurysms", "_MA.tif"),
        ("2. Haemorrhages", "_HE.tif"),
        ("3. Hard Exudates", "_EX.tif"),
        ("4. Soft Exudates", "_SE.tif"),
    ]
    all_x, all_y = [], []
    for folder, suffix in lesion_types:
        mask_path = os.path.join(MASK_BASE, folder, f"{img_name}{suffix}")
        if os.path.exists(mask_path):
            mask = np.array(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE))
            y_idx, x_idx = np.where(mask > 127)
            if len(x_idx) > 0:
                all_x.extend(x_idx)
                all_y.extend(y_idx)
    if not all_x:
        return torch.zeros(4, dtype=torch.float32), False  # no real lesion box
    
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    img_w, img_h = 2848, 2136
    cx = (x_min + x_max) / 2 / img_w
    cy = (y_min + y_max) / 2 / img_h
    w  = (x_max - x_min) / img_w
    h  = (y_max - y_min) / img_h
    return torch.tensor([cx, cy, w, h], dtype=torch.float32).clamp_(0.0, 1.0), True

# ====================== DATASET ======================
class IDRiDMultiTaskDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["Image name"].strip()
        img_path = os.path.join(IMAGE_DIR, f"{img_name}.jpg")

        image = cv2.imread(img_path)
        if image is None:
            print(f"Missing: {img_path}")
            return self.__getitem__((idx + 1) % len(self))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)["image"]

        label = int(row["Retinopathy grade"])
        box, has_box = get_combined_lesion_box(img_name)
        return image, torch.tensor(label), box, torch.tensor(has_box, dtype=torch.bool)

# ====================== MODEL ======================
class ArcaneMultiTaskNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.backbone.classifier = nn.Identity()
        self.cls_head = nn.Linear(1280, 5)
        self.box_head = nn.Linear(1280, 4)

    def forward(self, x):
        feat = self.backbone(x)
        return self.cls_head(feat), self.box_head(feat)

# ====================== MAIN ======================
if __name__ == "__main__":
    device = torch.device("cuda")
    df = pd.read_csv(CSV_PATH)
    df["Image name"] = df["Image name"].str.strip()
    print(f"Loaded {len(df)} images")

    df = df[df["Retinopathy grade"] != 1]
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df["Retinopathy grade"])

    train_ds = IDRiDMultiTaskDataset(train_df, aug)
    val_ds   = IDRiDMultiTaskDataset(val_df, aug)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    model = ArcaneMultiTaskNet().to(device)
    # 🔥 LOAD BEST MODEL
    model.load_state_dict(torch.load("best_model_qwk_0.8348_epoch20.pth", map_location=device, weights_only=True))
    print("✅ Loaded best model (QWK ~0.8348)")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)  # lower LR for fine-tuning
    
    # === NEW: Learning Rate Scheduler ===
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5, verbose=True)

    box_loss_fn = nn.SmoothL1Loss()
    best_qwk = 0.8348          # ← your current best (epoch 20)
    patience = 6
    no_improve = 0

    for epoch in range(21, 35):   # Continue from epoch 21 (best was epoch 20)
        model.train()
        total_loss = 0
        for images, labels, boxes, has_box in tqdm(train_loader, desc=f"Epoch {epoch}"):
            images  = images.to(device)
            labels  = labels.to(device)
            boxes   = boxes.to(device)
            has_box = has_box.to(device)          # bool mask: True for the ~54 images with real lesions

            cls_out, box_out = model(images)
            loss_cls = ordinal_loss(cls_out, labels)

            # ✅ Only compute box loss on images that actually have lesion annotations
            if has_box.any():
                loss_box = box_loss_fn(box_out[has_box], boxes[has_box])
            else:
                loss_box = torch.tensor(0.0, device=device)

            loss = loss_cls + 0.2 * loss_box

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        preds_all, labels_all = [], []
        with torch.no_grad():
            for images, labels, _, __ in val_loader:
                images = images.to(device)
                out, _ = model(images)
                preds_all.extend(out.argmax(dim=1).cpu().numpy())
                labels_all.extend(labels.numpy())

        qwk = cohen_kappa_score(labels_all, preds_all, weights="quadratic")
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | QWK: {qwk:.4f}")

        scheduler.step(qwk)

        if qwk > best_qwk:
            best_qwk = qwk
            torch.save(model.state_dict(), f"best_model_qwk_{qwk:.4f}_epoch{epoch}.pth")
            print("🔥 New best model saved!")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("⛔ Early stopping")
                break

    print("✅ Training finished! Final QWK:", best_qwk)