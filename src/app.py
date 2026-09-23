import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

# Load environment variables
load_dotenv()

# --- Configuration ---
HF_REPO_ID = os.getenv("HF_REPO_ID", "musharraf-mac2/TB_detector_by_MSH_V1")
HF_FILENAME = os.getenv("HF_FILENAME", "baseline_densenet121_shenzhen.pth")
HF_TOKEN = os.getenv("HF_TOKEN")
CLASS_NAMES = ["Normal", "TB Positive"]

# --- Model Setup ---
@st.cache_resource
def load_model():
    """Loads the DenseNet121 model with the modified classifier from Hugging Face."""
    # Instantiate the standard DenseNet121 architecture
    model = models.densenet121(weights=None)

    # Modify the classifier to output 2 classes (Normal vs TB)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 2)

    # Download and load the saved weights from HF
    try:
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            token=HF_TOKEN
        )
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
    except Exception as e:
        st.error(f"Error loading model weights from Hugging Face: {e}")
        return None

    model.eval()
    return model

# --- Preprocessing ---
def preprocess_image(image):
    """Preprocesses the input PIL image for the DenseNet121 model."""
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # Ensure image is RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    return preprocess(image).unsqueeze(0)

# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="TB Detection System", page_icon="🏥")

    st.title("🏥 Tuberculosis (TB) Detection System")
    st.markdown("""
    Upload a chest X-ray image to predict whether it is **Normal** or **TB Positive**.
    *Note: This is a research tool and should not be used for clinical diagnosis.*
    """)

    # Load the model
    with st.spinner("Loading model..."):
        model = load_model()

    if model is None:
        st.stop()

    # File Uploader
    uploaded_file = st.file_uploader("Choose a Chest X-ray image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-ray", use_container_width=True)

        if st.button("Predict"):
            with st.spinner("Analyzing image..."):
                # Preprocess and predict
                input_tensor = preprocess_image(image)

                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                    confidence, predicted_idx = torch.max(probabilities, 0)

                result = CLASS_NAMES[predicted_idx]
                conf_score = confidence.item() * 100

                # Display Results
                if result == "Normal":
                    st.success(f"**Prediction: {result}**")
                else:
                    st.error(f"**Prediction: {result}**")

                st.metric("Confidence Score", f"{conf_score:.2f}%")

                # Progress bar for confidence
                st.progress(conf_score / 100)

if __name__ == "__main__":
    main()
