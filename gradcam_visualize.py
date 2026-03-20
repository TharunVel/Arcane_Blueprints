import torch
import torch.nn as nn
from torchvision import models
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import os
import json
import faiss

# ====================== MODEL DEFINITION ======================
class DRModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        self.backbone.classifier = nn.Identity()
        self.cls_head = nn.Linear(1280, num_classes)
        self.box_head = nn.Linear(1280, 4)

    def forward(self, x):
        feat = self.backbone(x)
        return self.cls_head(feat), self.box_head(feat)

# ====================== PATHS ======================
IMAGE_DIR  = r"DR_Datasets\B. Disease Grading\1. Original Images\a. Training Set"
MODEL_PATH = "best_model_v2_qwk_0.8257_epoch12.pth"   # v2: all 5 grades

INDEX_PATH     = "faiss_index.bin"
FILENAMES_PATH = "faiss_filenames.json"   # maps FAISS row → image filename
THUMBNAIL_DIR  = "thumbnails"

# ====================== LOAD MODEL ======================
device = torch.device("cpu")
model  = DRModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()
print(f"✅ Loaded: {MODEL_PATH}")

# ✅ GradCAM wrapper — model returns (cls_out, box_out); GradCAM needs a single tensor
class ClassificationWrapper(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, x):
        cls_out, _ = self.m(x)
        return cls_out

wrapped_model = ClassificationWrapper(model)
target_layer  = wrapped_model.m.backbone.features[5][2]
cam = GradCAM(model=wrapped_model, target_layers=[target_layer])

# ✅ ImageNet stats — must match training augmentation exactly
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(orig_rgb):
    """Resize to 224×224 + ImageNet normalise — identical to training pipeline."""
    H, W = orig_rgb.shape[:2]
    img_224  = cv2.resize(orig_rgb, (224, 224)).astype(np.float32) / 255.0
    img_norm = (img_224 - MEAN) / STD
    tensor   = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(device)
    rgb_display = orig_rgb.astype(np.float32) / 255.0  # full-res [0,1] for display
    return tensor, rgb_display, H, W

def get_embedding(orig_rgb):
    """Return the 1280-d backbone embedding for an image (used for FAISS indexing)."""
    img_224  = cv2.resize(orig_rgb, (224, 224)).astype(np.float32) / 255.0
    img_norm = (img_224 - MEAN) / STD
    tensor   = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.backbone(tensor)   # (1, 1280)
    return feat.cpu().numpy().astype("float32")

# ====================== BUILD / LOAD FAISS INDEX ======================
if not os.path.exists(INDEX_PATH) or not os.path.exists(FILENAMES_PATH):
    print("🔨 Building FAISS index (first run — may take a minute)...")
    embeddings   = []
    image_fnames = []

    for fname in sorted(os.listdir(IMAGE_DIR)):          # sorted → deterministic order
        if not fname.lower().endswith((".jpg", ".png")):
            continue
        img = cv2.imread(os.path.join(IMAGE_DIR, fname))
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        emb = get_embedding(rgb)                         # ✅ same preprocessing as training
        embeddings.append(emb[0])
        image_fnames.append(fname)

    embeddings = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)                       # cosine similarity via inner product

    index = faiss.IndexFlatIP(1280)
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)

    # ✅ Save filename list so we can map FAISS row → filename after reload
    with open(FILENAMES_PATH, "w") as f:
        json.dump(image_fnames, f)

    os.makedirs(THUMBNAIL_DIR, exist_ok=True)
    for i, fname in enumerate(image_fnames):
        img   = cv2.imread(os.path.join(IMAGE_DIR, fname))
        thumb = cv2.resize(img, (128, 128))
        cv2.imwrite(os.path.join(THUMBNAIL_DIR, f"{i}.jpg"), thumb)

    print(f"✅ FAISS index built — {len(image_fnames)} images indexed")
else:
    index = faiss.read_index(INDEX_PATH)
    with open(FILENAMES_PATH) as f:
        image_fnames = json.load(f)
    print(f"✅ FAISS index loaded — {len(image_fnames)} images")

# ====================== VISUALIZATION ======================
GRADE_LABELS = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative DR"}

def highlight_lesions(img_uint8, contours):
    """Semi-transparent yellow fill + bright green contour outline."""
    overlay = img_uint8.copy()
    cv2.fillPoly(overlay, contours, (255, 230, 0))
    img_uint8[:] = cv2.addWeighted(overlay, 0.30, img_uint8, 0.70, 0)
    cv2.drawContours(img_uint8, contours, -1, (0, 230, 80), 3)

def visualize(img_name):
    img_path = os.path.join(IMAGE_DIR, f"{img_name}.jpg")
    orig = cv2.imread(img_path)
    if orig is None:
        print(f"❌ Image not found: {img_path}")
        return

    orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

    # ✅ Correct preprocessing (resize + normalize)
    input_tensor, rgb_display, H, W = preprocess(orig_rgb)

    # Prediction
    with torch.no_grad():
        cls_out, _ = model(input_tensor)
        pred_class = cls_out.argmax(dim=1).item()
        confidence = torch.softmax(cls_out, dim=1)[0, pred_class].item()

    grade_str = GRADE_LABELS.get(pred_class, f"Grade {pred_class}")
    print(f"🧠 Predicted: {grade_str}  (Confidence: {confidence:.1%})")

    # GradCAM runs internally — only used to derive the lesion contour mask.
    # The heatmap itself is NOT shown to the user.
    cam_map_224  = cam(input_tensor=input_tensor,
                       targets=[ClassifierOutputTarget(pred_class)])[0]
    cam_map_full = cv2.resize(cam_map_224, (W, H))

    # Lesion contour highlight from attention map
    attn_norm  = cam_map_full / (cam_map_full.max() + 1e-8)
    attn_uint8 = np.uint8(attn_norm * 255)
    _, thresh  = cv2.threshold(attn_uint8, int(0.4 * 255), 255, cv2.THRESH_BINARY)
    kernel     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    thresh     = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    orig_highlighted = orig_rgb.copy()
    highlight_lesions(orig_highlighted, contours)

    # FAISS similar case retrieval
    emb = get_embedding(orig_rgb)
    faiss.normalize_L2(emb)
    _, I = index.search(emb, k=4)
    similar_indices = [idx for idx in I[0] if image_fnames[idx] != f"{img_name}.jpg"][:3]

    # Plot: original + lesion highlight | 3 similar cases (no heatmap shown)
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    axes[0].imshow(orig_highlighted)
    axes[0].set_title(f"{grade_str}  ·  {confidence:.1%}\n(Lesion regions highlighted)",
                      fontsize=12, fontweight='bold')
    axes[0].axis("off")

    for i, idx in enumerate(similar_indices):
        thumb_path = os.path.join(THUMBNAIL_DIR, f"{idx}.jpg")
        thumb = cv2.imread(thumb_path)
        if thumb is not None:
            thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
            axes[1 + i].imshow(thumb)
        axes[1 + i].set_title(f"Similar Case {i + 1}\n{image_fnames[idx]}", fontsize=11)
        axes[1 + i].axis("off")

    plt.suptitle(f"IDRiD  |  {img_name}  |  Predicted: {grade_str}", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"result_{img_name}.png", dpi=150, bbox_inches="tight")
    plt.show()

    cv2.imwrite(f"result_{img_name}.jpg", cv2.cvtColor(orig_highlighted, cv2.COLOR_RGB2BGR))
    print(f"✅ Saved: result_{img_name}.png  |  result_{img_name}.jpg")

# ====================== RUN ======================
if __name__ == "__main__":
    test_image = "IDRiD_125"   # ← CHANGE THIS TO ANY IMAGE NAME (without .jpg)
    visualize(test_image)