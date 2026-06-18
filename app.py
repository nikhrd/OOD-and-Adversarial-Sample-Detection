import streamlit as st
import torch
import pickle
import numpy as np
from PIL import Image
from torchvision import transforms

import config
from src.model import load_model
from src.hooks import register_hooks, captured_features
from src.mahalanobis import calculate_layer_scores, get_mahalanobis_score

# -------------------------------
# Load Model & Artifacts
# -------------------------------
@st.cache_resource
def load_all():
    model = load_model()
    register_hooks(model)

    with open(config.ARTIFACT_PATH, "rb") as f:
        data = pickle.load(f)

    stats = data["stats"]
    judge = data["judge"]

    return model, stats, judge


model, stats, judge = load_all()

# -------------------------------
# Image Transform
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------
# UI
# -------------------------------
st.title(" Adversarial & OOD Detection System")
st.write("Upload an image to check if it's **Clean / Adversarial / OOD**")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

# -------------------------------
# Prediction Logic
# -------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width="stretch")

    input_tensor = transform(image).unsqueeze(0).to(config.DEVICE)

    # Step 1: Mahalanobis perturbation
    perturbed = get_mahalanobis_score(input_tensor, model, stats)

    # Step 2: Feature scoring
    scores = calculate_layer_scores(perturbed, model, stats)

    # Step 3: Judge prediction
    prob = judge.predict_proba(scores)[0]
    prediction = judge.predict(scores)[0]

    # -------------------------------
    # Display Results
    # -------------------------------
    st.subheader("Results")

    if prediction == 1:
        st.success("Clean Image")
    else:
        st.error(" Adversarial / OOD Image")

    st.write(f"Confidence (Clean): {prob[1]:.4f}")
    st.write(f"Confidence (Adversarial/OOD): {prob[0]:.4f}")

    # -------------------------------
    # Extra Debug Info
    # -------------------------------
    with st.expander(" Debug Info"):
        st.write("Mahalanobis Scores:", scores)