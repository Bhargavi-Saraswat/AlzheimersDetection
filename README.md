# Alzheimer's Disease Detection

A deep learning project that classifies brain MRI scans into four stages of Alzheimer's disease using transfer learning, with a Streamlit app for interactive predictions.

## Overview

This project fine-tunes **MobileNetV2** on the **OASIS** MRI dataset to classify scans into four categories of cognitive impairment. The trained model is served through a simple Streamlit web app that lets a user upload an MRI image and get an instant prediction.

## Model

- **Architecture:** MobileNetV2 (transfer learning, pretrained on ImageNet, fine-tuned on OASIS)
- **Dataset:** OASIS MRI dataset (sourced from Kaggle)
- **Classes:** 4-class classification (e.g., Non-Demented, Very Mild Demented, Mild Demented, Moderate Demented)
- **Accuracy:** ~75% on the held-out test set
- **Training:** Trained locally (no cloud/Colab) on a laptop with an NVIDIA RTX 2050 (4GB)

## Project Structure

```
AlzheimersDetection/
├── src/            # Training and preprocessing scripts
├── results/        # Saved metrics, plots, and/or model outputs
├── app.py          # Streamlit app for inference
├── requirements.txt
└── README.md
```

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Bhargavi-Saraswat/AlzheimersDetection.git
   cd AlzheimersDetection
   ```

2. (Recommended) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`), upload an MRI scan image, and view the predicted class.

## Tech Stack

- Python
- TensorFlow / Keras (MobileNetV2)
- Streamlit
- NumPy, Pillow, scikit-learn, Matplotlib

## Disclaimer

This project is for academic and research purposes only and is **not intended for clinical or diagnostic use**.

## Author

**Bhargavi Saraswat** — Final-year B.E. CSE, NMIT, Bengaluru