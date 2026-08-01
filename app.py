# app.py
import io
import pickle
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

import config
from src.model import load_model
from src.hooks import register_hooks, captured_features
from src.mahalanobis import calculate_layer_scores, get_mahalanobis_score
from src.faiss_index import disambiguate_sample
from src.security import encrypt_image_bytes, log_event

st.set_page_config(page_title="Multi-Stage Secure AI Pipeline", layout="wide")

@st.cache_resource
def load_system_artifacts():
    model = load_model()
    register_hooks(model)

    with open(config.ARTIFACT_PATH, "rb") as f:
        artifacts = pickle.load(f)

    return (
        model, 
        artifacts["stats"], 
        artifacts["judge"], 
        artifacts["faiss_index"], 
        artifacts["faiss_threshold"]
    )

model, stats, judge, faiss_idx, faiss_threshold = load_system_artifacts()

transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Streamlit App Structure
st.title("🛡️ Multi-Stage OOD & Adversarial Detection Framework")
st.markdown("Unified framework for **Detection**, **FAISS Disambiguation**, and **AES Security Handling**.")

uploaded_file = st.file_uploader("Upload Image for Inspection", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    raw_bytes = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0).to(config.DEVICE)

    # Stage 1: Mahalanobis Scoring
    perturbed = get_mahalanobis_score(input_tensor, model, stats)
    layer_scores = calculate_layer_scores(perturbed, model, stats)

    # Stage 2: Classifier Stage
    probs = judge.predict_proba(layer_scores)[0]
    prediction = judge.predict(layer_scores)[0]

    with col2:
        st.subheader("Analysis & Security Actions")

        if prediction == 1:
            # CLEAN IMAGE LOGIC
            confidence = probs[1] * 100
            st.success(f"✅ **CLEAN IMAGE DETECTED** (Confidence: {confidence:.2f}%)")
            
            # Encrypt Image
            encrypted_data = encrypt_image_bytes(raw_bytes)
            log_event("CLEAN", probs[1], "Validated clean sample; encrypted successfully.")

            st.info("🔒 Image successfully encrypted using AES-Fernet standard.")
            st.download_button(
                label="📥 Download Encrypted Image (.bin)",
                data=encrypted_data,
                file_name="encrypted_image.bin",
                mime="application/octet-stream"
            )

        else:
            # SUSPICIOUS IMAGE LOGIC -> FAISS DISAMBIGUATION
            top_layer_feature = captured_features[-1]
            category, faiss_distance = disambiguate_sample(top_layer_feature, faiss_idx, faiss_threshold)
            
            confidence = probs[0] * 100
            
            if category == "Adversarial Attack":
                st.error(f"🚨 **ADVERSARIAL ATTACK DETECTED**")
            else:
                st.warning(f"⚠️ **OUT-OF-DISTRIBUTION (OOD) DETECTED**")

            st.write(f"Anomaly Detection Confidence: **{confidence:.2f}%**")
            st.write(f"FAISS Distance: `{faiss_distance:.4f}` (Threshold: `{faiss_threshold:.4f}`)")
            
            log_event(
                category.upper(), 
                probs[0], 
                f"FAISS Distance: {faiss_distance:.4f} vs Threshold: {faiss_threshold:.4f}"
            )
            st.error("⛔ Action Executed: Input rejected and logged to system failure file.")

    # Expandable Diagnostic View
    with st.expander("🔬 Diagnostics"):
        st.write("**Mahalanobis Multi-Layer Feature Scores:**", layer_scores)
        st.write("**FAISS Cutoff Threshold:**", faiss_threshold)