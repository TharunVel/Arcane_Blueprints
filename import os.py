import os
import pandas as pd

# PATHS
IMAGE_DIR = "DR_Datasets/A. Segmentation/1. Original Images/a. Training Set"
CSV_PATH  = "DR_Datasets/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv"

# Load CSV
df = pd.read_csv(CSV_PATH)
df["Image name"] = df["Image name"].str.strip()

# Check images
print("Total images in folder:", len(os.listdir(IMAGE_DIR)))
print("Sample files:", os.listdir(IMAGE_DIR)[:5])

# Check CSV
print("CSV sample:", df["Image name"].head().tolist())

# Match check
available_images = set([f.replace(".jpg", "") for f in os.listdir(IMAGE_DIR)])
matches = df["Image name"].isin(available_images)

print("Total matches:", matches.sum())