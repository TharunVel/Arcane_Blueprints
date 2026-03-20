"""
database.py — JSON-based patient registry.
All patient records persist across sessions in patients.json.
"""
import json
import os
import uuid
import shutil
from datetime import datetime
from PIL import Image

PATIENTS_FILE   = "patients.json"
IMAGES_DIR      = "patient_images"
RESULTS_DIR     = "patient_results"

os.makedirs(IMAGES_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def _load() -> list[dict]:
    if not os.path.exists(PATIENTS_FILE):
        return []
    with open(PATIENTS_FILE, "r") as f:
        return json.load(f)


def _save(records: list[dict]) -> None:
    with open(PATIENTS_FILE, "w") as f:
        json.dump(records, f, indent=2)


def add_patient(
    name: str,
    age: int,
    patient_id: str,
    grade: int,
    grade_str: str,
    confidence: float,
    original_image: Image.Image,
    result_image: Image.Image,
) -> dict:
    """Save a new patient record. Returns the saved record dict."""
    record_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    orig_path   = os.path.join(IMAGES_DIR,  f"{record_id}.jpg")
    result_path = os.path.join(RESULTS_DIR, f"{record_id}.jpg")

    original_image.convert("RGB").save(orig_path,   quality=95)
    result_image.convert("RGB").save(result_path,   quality=95)

    record = {
        "record_id":   record_id,
        "patient_id":  patient_id,
        "name":        name,
        "age":         age,
        "grade":       grade,
        "grade_str":   grade_str,
        "confidence":  round(confidence, 4),
        "date":        timestamp,
        "image_path":  orig_path,
        "result_path": result_path,
    }

    records = _load()
    records.append(record)
    _save(records)
    return record


def get_all_patients(sort_by: str = "severity_desc") -> list[dict]:
    """
    Retrieve all patients.
    sort_by options:
        'severity_desc'  — Proliferative first (grade 4 → 0)
        'severity_asc'   — No DR first (grade 0 → 4)
        'date_desc'      — Newest first
    """
    records = _load()
    if sort_by == "severity_desc":
        records.sort(key=lambda r: r["grade"], reverse=True)
    elif sort_by == "severity_asc":
        records.sort(key=lambda r: r["grade"])
    elif sort_by == "date_desc":
        records.sort(key=lambda r: r["date"], reverse=True)
    return records


def get_patient(record_id: str) -> dict | None:
    """Return a single patient record by record_id, or None if not found."""
    for r in _load():
        if r["record_id"] == record_id:
            return r
    return None


def delete_patient(record_id: str) -> bool:
    """Delete patient record and associated images. Returns True if found."""
    records = _load()
    filtered = [r for r in records if r["record_id"] != record_id]
    if len(filtered) == len(records):
        return False  # not found
    # Remove image files
    for r in records:
        if r["record_id"] == record_id:
            for path in (r.get("image_path"), r.get("result_path")):
                if path and os.path.exists(path):
                    os.remove(path)
    _save(filtered)
    return True
