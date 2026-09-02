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


# --------------------------------------------------
# Streamlit Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Multi-Stage Secure AI Pipeline",
    layout="wide"
)


# --------------------------------------------------
# Load Model and Artifacts
# --------------------------------------------------

@st.cache_resource
def load_system():

    model = load_model()

    # Register feature extraction hooks
    hook_handles = register_hooks(model)

    # Load pre-computed artifacts
    with open(config.ARTIFACT_PATH, "rb") as file:
        artifacts = pickle.load(file)

    return (
        model,
        hook_handles,
        artifacts["stats"],
        artifacts["judge"],
        artifacts["faiss_index"],
        artifacts["faiss_threshold"]
    )


model, hook_handles, stats, judge, faiss_index, faiss_threshold = load_system()


# --------------------------------------------------
# Image Preprocessing
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize(
        (config.IMAGE_SIZE, config.IMAGE_SIZE)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# User Interface
# --------------------------------------------------

st.title("Multi-Stage OOD & Adversarial Detection Framework")

st.markdown(
    """
    **Detection → FAISS Disambiguation → AES Security Handling**

    Upload an image to determine whether it is a clean
    in-distribution sample, an adversarial attack, or an
    out-of-distribution sample.
    """
)


uploaded_file = st.file_uploader(
    "Upload Image for Inspection",
    type=["png", "jpg", "jpeg"]
)


# --------------------------------------------------
# Image Analysis
# --------------------------------------------------

if uploaded_file is not None:

    # Read uploaded image
    raw_bytes = uploaded_file.getvalue()

    image = Image.open(
        io.BytesIO(raw_bytes)
    ).convert("RGB")


    # --------------------------------------------------
    # Display Uploaded Image
    # --------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )


    # --------------------------------------------------
    # Preprocess Image
    # --------------------------------------------------

    input_tensor = transform(image)

    input_tensor = input_tensor.unsqueeze(0)

    input_tensor = input_tensor.to(config.DEVICE)


    # --------------------------------------------------
    # Stage 1: Mahalanobis Analysis
    # --------------------------------------------------

    with st.spinner("Analyzing image..."):

        perturbed = get_mahalanobis_score(
            input_tensor,
            model,
            stats
        )


        # Calculate multi-layer Mahalanobis scores
        layer_scores = calculate_layer_scores(
            perturbed,
            model,
            stats
        )


    # --------------------------------------------------
    # Stage 2: Machine Learning Judge
    # --------------------------------------------------

    probs = judge.predict_proba(layer_scores)[0]

    prediction = judge.predict(layer_scores)[0]


    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    with col2:

        st.subheader("Analysis & Security Actions")


        # --------------------------------------------------
        # CLEAN IMAGE
        # --------------------------------------------------

        if prediction == 1:

            confidence = probs[1] * 100

            st.success(
                f"**CLEAN IMAGE DETECTED**\n\n"
                f"Confidence: **{confidence:.2f}%**"
            )


            # Encrypt the original uploaded image
            encrypted_data = encrypt_image_bytes(
                raw_bytes
            )


            # Log successful validation
            log_event(
                "CLEAN",
                probs[1],
                "Validated clean sample; encrypted successfully."
            )


            st.info(
                "Image successfully encrypted using "
                "AES-Fernet encryption."
            )


            st.download_button(
                label="Download Encrypted Image (.bin)",
                data=encrypted_data,
                file_name="encrypted_image.bin",
                mime="application/octet-stream"
            )


        # --------------------------------------------------
        # SUSPICIOUS IMAGE
        # --------------------------------------------------

        else:

            # Ensure features are available
            if not captured_features:

                st.error(
                    "Feature extraction failed. "
                    "No features were captured by the model hooks."
                )

                st.stop()


            # Get final ResNet feature representation
            top_layer_feature = captured_features[-1]


            # --------------------------------------------------
            # Stage 3: FAISS Disambiguation
            # --------------------------------------------------

            category, faiss_distance = disambiguate_sample(
                top_layer_feature,
                faiss_index,
                faiss_threshold
            )


            confidence = probs[0] * 100


            # --------------------------------------------------
            # Display Classification
            # --------------------------------------------------

            if category == "Adversarial Attack":

                st.error(
                    "**ADVERSARIAL ATTACK DETECTED**"
                )

            else:

                st.warning(
                    "**OUT-OF-DISTRIBUTION (OOD) DETECTED**"
                )


            st.write(
                f"Anomaly Detection Confidence: "
                f"**{confidence:.2f}%**"
            )

            st.write(
                f"FAISS Distance: "
                f"`{faiss_distance:.4f}`"
            )

            st.write(
                f"FAISS Threshold: "
                f"`{faiss_threshold:.4f}`"
            )


            # --------------------------------------------------
            # Security Logging
            # --------------------------------------------------

            log_event(
                category.upper(),
                probs[0],
                (
                    f"FAISS Distance: "
                    f"{faiss_distance:.4f} "
                    f"vs Threshold: "
                    f"{faiss_threshold:.4f}"
                )
            )


            st.error(
                "Action Executed: "
                "Input rejected and logged to system failure file."
            )


    # --------------------------------------------------
    # Diagnostic Information
    # --------------------------------------------------

    with st.expander("Diagnostics"):

        st.write(
            "**Mahalanobis Multi-Layer Feature Scores:**"
        )

        st.write(layer_scores)


        st.write(
            "**FAISS Cutoff Threshold:**"
        )

        st.write(faiss_threshold)


        st.write(
            "**Captured Feature Layers:**"
        )

        st.write(len(captured_features))