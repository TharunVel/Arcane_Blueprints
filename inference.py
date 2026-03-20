"""
inference.py — DR grading model inference module.
Loads the model once; exposes predict(pil_image) for use by app.py.
"""
import torch
import torch.nn as nn
from torchvision import models
import cv2
import numpy as np
import os
import json
import faiss
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ====================== MODEL ======================
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

class _ClassificationWrapper(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, x):
        cls_out, _ = self.m(x)
        return cls_out

GRADE_LABELS = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR",
}

MODEL_PATH = "best_model_v2_qwk_0.8257_epoch12.pth"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

device = torch.device("cpu")

# Load once at module import
_model = DRModel().to(device)
_model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
_model.eval()

_wrapped   = _ClassificationWrapper(_model)
_target_layer = _wrapped.m.backbone.features[5][2]
_cam = GradCAM(model=_wrapped, target_layers=[_target_layer])

# ====================== FAISS SETUP ======================
INDEX_PATH     = "faiss_index.bin"
FILENAMES_PATH = "faiss_filenames.json"
THUMBNAIL_DIR  = "thumbnails"

_faiss_index  = None
_faiss_fnames = []

if os.path.exists(INDEX_PATH) and os.path.exists(FILENAMES_PATH):
    import warnings
    # FAISS triggers a visible warning sometimes, silence it locally if needed
    _faiss_index = faiss.read_index(INDEX_PATH)
    with open(FILENAMES_PATH, "r") as f:
        _faiss_fnames = json.load(f)
    print(f"✅ FAISS index loaded in inference module — {len(_faiss_fnames)} images")


def _preprocess(rgb_np: np.ndarray):
    """rgb_np: HxWx3 uint8. Returns (input_tensor, rgb_display float32, H, W)."""
    H, W = rgb_np.shape[:2]
    img_224  = cv2.resize(rgb_np, (224, 224)).astype(np.float32) / 255.0
    img_norm = (img_224 - MEAN) / STD
    tensor   = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(device)
    return tensor, rgb_np.astype(np.float32) / 255.0, H, W


def _highlight_lesions(img_uint8: np.ndarray, contours) -> np.ndarray:
    """Draw semi-transparent yellow fill + green contour outline. Returns copy."""
    out     = img_uint8.copy()
    overlay = out.copy()
    cv2.fillPoly(overlay, contours, (255, 230, 0))
    out[:] = cv2.addWeighted(overlay, 0.30, out, 0.70, 0)
    cv2.drawContours(out, contours, -1, (0, 230, 80), 3)
    return out


def predict(pil_image: Image.Image) -> dict:
    """
    Run DR grading on a PIL image.

    Returns dict with keys:
        grade       (int 0-4)
        grade_str   (str)
        confidence  (float 0-1)
        result_image (PIL.Image — original + lesion highlight)
    """
    rgb_np = np.array(pil_image.convert("RGB"))
    input_tensor, rgb_display, H, W = _preprocess(rgb_np)

    with torch.no_grad():
        cls_out, _ = _model(input_tensor)
        grade      = cls_out.argmax(dim=1).item()
        confidence = torch.softmax(cls_out, dim=1)[0, grade].item()

    # GradCAM — internal use only, drives lesion contours
    cam_map_224  = _cam(input_tensor=input_tensor,
                        targets=[ClassifierOutputTarget(grade)])[0]
    cam_map_full = cv2.resize(cam_map_224, (W, H))

    attn_norm  = cam_map_full / (cam_map_full.max() + 1e-8)
    attn_uint8 = np.uint8(attn_norm * 255)
    _, thresh  = cv2.threshold(attn_uint8, int(0.4 * 255), 255, cv2.THRESH_BINARY)
    kernel     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    thresh     = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # ✅ KEY FIX: mask out the dark equipment border.
    # Any pixel in the ORIGINAL image that is near-black is part of the
    # ophthalmoscope vignette — never a lesion. Zero those regions out
    # in the attention threshold so contours only appear inside the retinal disc.
    gray_orig    = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)
    retinal_mask = (gray_orig > 30).astype(np.uint8) * 255   # bright = inside disc
    thresh       = cv2.bitwise_and(thresh, thresh, mask=retinal_mask)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    highlighted = _highlight_lesions(rgb_np.copy(), contours)

    # ── FAISS retrieval ──
    similar_cases = []
    if _faiss_index is not None:
        with torch.no_grad():
            feat = _model.backbone(input_tensor)
        emb = feat.cpu().numpy().astype("float32")
        faiss.normalize_L2(emb)
        _, I = _faiss_index.search(emb, k=3)
        for idx in I[0]:
            thumb_path = os.path.join(THUMBNAIL_DIR, f"{idx}.jpg")
            if os.path.exists(thumb_path):
                # Load via PIL, copy to avoid file lock issues
                img = Image.open(thumb_path).convert("RGB")
                img.load()
                similar_cases.append({
                    "image": img,
                    "filename": _faiss_fnames[idx]
                })

    return {
        "grade":        grade,
        "grade_str":    GRADE_LABELS.get(grade, f"Grade {grade}"),
        "confidence":   confidence,
        "result_image": Image.fromarray(highlighted),
        "similar_cases": similar_cases,
    }
