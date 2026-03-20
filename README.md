# DR Grading System 

> **Screening Support – Non-Diagnostic.** This tool assists trained healthcare workers in triage screening only. It does NOT replace clinical examination by a qualified ophthalmologist.

An AI-powered triage assistant designed for rural eye camps. It analyses retinal (fundal) photographs, classifies Diabetic Retinopathy severity into 5 clinical stages, highlights lesion regions, and retrieves similar past cases — all in seconds.

Built using a lightweight **EfficientNet-B0** model (16MB), achieving a Quadratic Weighted Kappa (QWK) of **0.8257** on the IDRiD dataset.

---

## Setup & Installation

The project runs locally and does not require a GPU. 

### 1. Clone the repository
```bash
git clone https://github.com/TharunVel/Arcane_Blueprints.git
cd Arcane_Blueprints
```

### 2. Create a virtual environment
It is recommended to use Conda to manage dependencies.
```bash
conda create -n arcane_env python=3.10 -y
conda activate arcane_env
```

### 3. Install dependencies
Install the required packages using `pip`. The PyTorch installation specified here uses the CPU-only version to keep the download small.
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install streamlit opencv-python-headless numpy pillow
pip install pytorch-grad-cam faiss-cpu
```

---

## How to Run

Once installed, simply run the Streamlit application from the project root folder:
```bash
streamlit run app.py
```
The application will open automatically in your browser at `http://localhost:8501`.

---

## Usage Guide

1. **New Patient Analysis (Tab 1):**
   - Enter the Patient ID, Name, and Age.
   - Upload a fundal photograph (`.jpg` or `.png`).
     *(Built-in safety checks will automatically reject images that are completely blurry, dark, or not actual fundal photographs).*
   - Click **Run Analysis**.
   - Review the DR Grade, Confidence Score, Lesion Highlights, and Visually Similar past cases.

2. **Patient Registry (Tab 2):**
   - View a complete database of analysed patients.
   - Expand any row to see full historical results.
   - Sort by **Severity (High to Low)** to instantly prioritize patients needing urgent Referral (Grades 3 and 4).

---

