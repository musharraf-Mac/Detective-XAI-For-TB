import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from PIL import Image
from torchvision import transforms, models
from typing import cast
from torch.amp.autocast_mode import autocast

st.set_page_config(
    page_title="TB - Classifier - Model B",
    page_icon="🫁",
    layout="wide"
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../../models/best_model_b.pth"))
DEVICE = torch.device("cuda" if torch.cuda.is_available else "cpu")
USE_AMP = True  # Set to True to use Automatic Mixed Precision (AMP) for inference

CLASS_NAMES = ["Normal", "TB Positive", "Abnormal but not TB"]
CLASS_ICONS = ['✅', '🔴', '⚠️']

# Preprocess
val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Model
def create_model_b():
    """Recreate DenseNet-121 for 3-class classification."""
    model = models.densenet121(weights=None)
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential( # type: ignore
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 3)
    )
    return model

# load model
@st.cache_resource
def load_model():
    """Load DenseNet-121 with saved weights."""
    model = create_model_b().to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint

# Predict
def predict(image: Image.Image, model):
    """Predict class probabilities for a chest X-ray."""
    
    # Convert to Grayscale
    img = np.array(image.convert('L')) 
    img = np.stack([img]*3, axis=2)  # Convert to 3 channels
    
    # Apply transforms and cast to Tensor
    img_tensor = cast(torch.Tensor, val_transform(img))
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
    
    # Predict
    with torch.no_grad():
        with autocast("cuda", enabled=USE_AMP):
            outputs = model(img_tensor)
            probs = torch.softmax(outputs.float(), dim=1)[0].cpu().numpy()
            
        return probs
    
# UI
st.title("🫁 TB Classifier — Model B")
st.markdown(
    "Upload a **chest X-ray** to classify it into **Normal**, **TB Positive**, or **Abnormal (Not TB)**."
)
st.caption("⚠️ This model assumes the image is already a validated chest X-ray. Use Model A first if unsure.")

# Check model exists
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file not found at: `{MODEL_PATH}`")
    st.info("Train Model B first with: `python src/training_scripts/train_model_b_tb_det.py`")
    st.stop()

# Load model
with st.spinner("Loading model..."):
    model, checkpoint = load_model()

# Sidebar — model info
with st.sidebar:
    st.header("ℹ️ Model Info")
    st.markdown(f"**Architecture:** DenseNet-121")
    st.markdown(f"**Classes:** {len(CLASS_NAMES)}")
    st.markdown(f"**Device:** `{DEVICE}`")
    if 'epoch' in checkpoint:
        st.markdown(f"**Trained Epoch:** {checkpoint['epoch'] + 1}")
    if 'val_loss' in checkpoint:
        st.markdown(f"**Val Loss:** {checkpoint['val_loss']:.4f}")
    if 'val_acc' in checkpoint:
        st.markdown(f"**Val Acc:** {checkpoint['val_acc']:.4f}")
    if 'val_f1' in checkpoint:
        st.markdown(f"**Val F1:** {checkpoint['val_f1']:.4f}")
    if 'val_auc' in checkpoint:
        st.markdown(f"**Val AUC:** {checkpoint['val_auc']:.4f}")

# File uploader
uploaded_file = st.file_uploader(
    "Choose a chest X-ray image...",
    type=["png", "jpg", "jpeg", "bmp", "tiff"]
)

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert('RGB')
    
    # Two-column layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📷 Uploaded Image")
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("🔍 Prediction")
        
        with st.spinner("Analyzing..."):
            probs = predict(image, model)
        
        # Get predicted class
        predicted_idx = int(np.argmax(probs))
        predicted_class = CLASS_NAMES[predicted_idx]
        predicted_conf = probs[predicted_idx]
        icon = CLASS_ICONS[predicted_idx]
        
        # Result banner
        if predicted_idx == 0:
            st.success(f"{icon} **{predicted_class}** — Confidence: {predicted_conf * 100:.2f}%")
        elif predicted_idx == 1:
            st.error(f"{icon} **{predicted_class}** — Confidence: {predicted_conf * 100:.2f}%")
        else:
            st.warning(f"{icon} **{predicted_class}** — Confidence: {predicted_conf * 100:.2f}%")
        
        st.markdown("---")
        st.markdown("**Class Probabilities**")
        
        # Bar chart for each class
        for i, (name, icon) in enumerate(zip(CLASS_NAMES, CLASS_ICONS)):
            prob_pct = probs[i] * 100
            st.progress(
                float(probs[i]),
                text=f"{icon} {name}: {prob_pct:.2f}%"
            )
        
        st.markdown("---")
        
        # Clinical recommendation
        st.markdown("**📋 Recommendation**")
        if predicted_idx == 0:
            st.info("No TB detected. Routine follow-up as needed.")
        elif predicted_idx == 1:
            st.error(
                "TB detected. Recommend immediate sputum testing (GeneXpert / culture) "
                "and referral to a TB specialist."
            )
        else:
            st.warning(
                "Abnormal lung finding detected (not TB). Recommend referral for "
                "further diagnostic workup (CT, specialist consultation)."
            )

# Footer
st.markdown("---")
st.caption(
    "Model B — Stage 2 of the TB Detection Pipeline | "
    "⚠️ For research use only. Not a substitute for clinical judgment."
)
