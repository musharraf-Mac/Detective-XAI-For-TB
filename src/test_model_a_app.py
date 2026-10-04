import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from PIL import Image
from torchvision import transforms, models


# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Chest X-ray Detector",
    page_icon="🩻",
    layout="centered"
)

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/best_model_a.pth'))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# PREPROCESSING (must match training)
# ============================================================
val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ============================================================
# LOAD MODEL (cached)
# ============================================================
@st.cache_resource
def load_model():
    """Load DenseNet-121 with saved weights."""
    model = models.densenet121(weights=None)
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential( # type: ignore
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 2)
    )
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    return model


# ============================================================
# PREDICT
# ============================================================
def predict(image: Image.Image, model):
    """Predict if uploaded image is a chest X-ray."""
    # Convert to grayscale (like training)
    img = np.array(image.convert('L'))
    
    # Convert to 3-channel
    img = np.stack([img, img, img], axis=2)
    
    # Apply transforms and convert to Tensor
    img_tensor = torch.as_tensor(val_transform(img))
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
    
    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        chest_xray_prob = probs[0, 1].item()
        not_chest_xray_prob = probs[0, 0].item()
    
    return chest_xray_prob, not_chest_xray_prob


# ============================================================
# UI
# ============================================================
st.title("🩻 Chest X-ray Detector — Model A")
st.markdown("Upload any image to check if it is a **chest X-ray**.")

# Check model exists
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file not found at: `{MODEL_PATH}`")
    st.stop()

model = load_model()

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["png", "jpg", "jpeg", "bmp", "tiff"]
)

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert('RGB')
    
    # Show uploaded image
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Predict
    with st.spinner("Analyzing..."):
        cxr_prob, not_cxr_prob = predict(image, model)
    
    # Show result
    with col2:
        st.subheader("Prediction")
        
        if cxr_prob >= 0.5:
            st.success(f"✅ **Chest X-ray** ({cxr_prob * 100:.2f}%)")
        else:
            st.error(f"❌ **Not a Chest X-ray** ({not_cxr_prob * 100:.2f}%)")
        
        st.markdown("---")
        st.markdown("**Confidence Scores**")
        st.progress(cxr_prob, text=f"Chest X-ray: {cxr_prob * 100:.2f}%")
        st.progress(not_cxr_prob, text=f"Not Chest X-ray: {not_cxr_prob * 100:.2f}%")
        
        st.markdown("---")
        st.caption(f"Device: `{DEVICE}`")
        st.caption(f"Model: `best_model_a.pth`")


st.markdown("---")
st.caption("Model A — Stage 1 of the TB Detection Pipeline")