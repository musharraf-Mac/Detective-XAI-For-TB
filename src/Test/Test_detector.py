import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

st.set_page_config(page_title="TB Model B Tester", layout="wide")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# MUST match your label order (0, 1, 2) in the CSV
CLASS_NAMES = ["Normal", "TB", "Abnormal (not TB)"]
DEFAULT_B = "models/best_model_b.pth"
DEFAULT_A = "models/best_model_a.pth"

TF = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_densenet(num_classes):
    m = models.densenet121(weights=None)       # same head as your training script
    n = m.classifier.in_features
    # Use type ignore because Pylance thinks .classifier must be a Linear layer
    m.classifier = nn.Sequential(nn.Linear(n, 512), nn.ReLU(), # type: ignore
                                 nn.Dropout(0.3), nn.Linear(512, num_classes)) # type: ignore
    return m


class CAMWrapper(nn.Module):
    """DenseNet forward with a non-inplace ReLU so Grad-CAM hooks work cleanly."""
    def __init__(self, base):
        super().__init__()
        self.features = base.features
        self.classifier = base.classifier

    def forward(self, x):
        out = F.relu(self.features(x))
        out = F.adaptive_avg_pool2d(out, (1, 1)).flatten(1)
        return self.classifier(out)


@st.cache_resource
def load_model(path, num_classes):
    m = build_densenet(num_classes)
    try:
        ckpt = torch.load(path, map_location="cpu")
    except Exception:                                   # numpy scalars in your checkpoint
        ckpt = torch.load(path, map_location="cpu", weights_only=False)  # only for your own files
    m.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    return CAMWrapper(m).to(DEVICE).eval()


def read_gray(uploaded):
    data = np.frombuffer(uploaded.getvalue(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def predict(model, gray):
    # Create RGB version of grayscale image
    img_rgb = np.stack([gray] * 3, axis=2)
    # Apply the transformation pipeline (returns a torch.Tensor)
    x = TF(img_rgb)
    # Ensure x is a torch.Tensor to satisfy Pylance (type: ignore)
    x = x.unsqueeze(0).to(DEVICE) # type: ignore

    with torch.no_grad():
        # Get probabilities for the classes
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    return x, probs


def gradcam_overlay(model, x, gray, class_idx):
    # Grad-CAM expects an image in range [0, 1] with 3 channels
    rgb = cv2.resize(gray, (224, 224)).astype(np.float32) / 255.0
    rgb = np.stack([rgb] * 3, axis=-1)

    with GradCAM(model=model, target_layers=[model.features.norm5]) as cam:
        # cam() returns a list of heatmaps, we take the first one [0]
        heat = cam(input_tensor=x, targets=[ClassifierOutputTarget(class_idx)])[0] # type: ignore

    return show_cam_on_image(rgb, heat, use_rgb=True)


# ---------------- UI ----------------
st.title("TB Detection: Model B Tester (3-class)")
st.caption("Research prototype for testing only. Not a diagnostic tool.")

with st.sidebar:
    st.header("Settings")
    path_b = st.text_input("Model B checkpoint", DEFAULT_B)
    use_gate = st.checkbox("Run Model A gate first (is it a chest X-ray?)", value=False)
    path_a = st.text_input("Model A checkpoint", DEFAULT_A, disabled=not use_gate)
    gate_thr = st.slider("Gate: minimum P(chest X-ray)", 0.5, 0.99, 0.90, 0.01,
                         disabled=not use_gate)
    low_conf = st.slider("Low-confidence warning below", 0.34, 0.95, 0.60, 0.01)
    show_cam = st.checkbox("Show Grad-CAM", value=True)
    st.write(f"Device: **{DEVICE}**")

try:
    model_b = load_model(path_b, 3)
except Exception as e:
    st.error(f"Could not load Model B from `{path_b}`: {e}")
    st.stop()

gate = None
if use_gate:
    try:
        gate = load_model(path_a, 2)
    except Exception as e:
        st.error(f"Could not load Model A from `{path_a}`: {e}")
        st.stop()

files = st.file_uploader("Upload image(s)", type=["png", "jpg", "jpeg", "bmp"],
                         accept_multiple_files=True)

for f in files:
    st.divider()
    st.subheader(f.name)
    gray = read_gray(f)
    if gray is None:
        st.error("Could not read this file as an image.")
        continue

    if gate is not None:
        _, gp = predict(gate, gray)
        p_cxr = float(gp[1])                 # label 1 = chest X-ray in Model A
        if p_cxr < gate_thr:
            c1, c2 = st.columns(2)
            c1.image(gray, caption="Uploaded", width=300)
            c2.error(f"Please submit only a chest X-ray image. (P(chest X-ray) = {p_cxr:.3f})")
            continue

    x, probs = predict(model_b, gray)
    idx = int(probs.argmax())

    c1, c2, c3 = st.columns([1, 1, 1])
    c1.image(gray, caption="Uploaded", use_container_width=True)
    if show_cam:
        c2.image(gradcam_overlay(model_b, x, gray, idx),
                 caption=f"Grad-CAM for: {CLASS_NAMES[idx]}", use_container_width=True)
    with c3:
        st.metric("Prediction", CLASS_NAMES[idx], f"{probs[idx] * 100:.1f}% confidence")
        st.bar_chart(pd.DataFrame({"probability": probs}, index=CLASS_NAMES))
        if probs[idx] < low_conf:
            st.warning("Low confidence. Treat this prediction as unreliable.")
        if idx == 2:
            st.info("Abnormal but not TB. A disease-specific model is needed for the exact condition.")
