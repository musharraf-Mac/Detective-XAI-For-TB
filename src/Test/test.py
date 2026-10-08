import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import pandas as pd
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from safetensors.torch import load_file
import json

# Project-specific imports
from nlp_model.tb_utils import build_clinical_text

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES_B = ["Normal", "TB", "Abnormal (not TB)"]
# CLASS_NAMES_B = ["Normal", "TB Positive"]
# DEFAULT_B_PATH = "models/baseline_densenet121_shenzhen.pth"
DEFAULT_B_PATH = "models/best_model_b.pth"
DEFAULT_A_PATH = "models/best_model_a.pth"
NLP_MODEL_PATH = r"models\Sym_nlp.safetensors"
TOKENIZER_PATH = "emilyalsentzer/Bio_ClinicalBERT"
THRESHOLD_FILE = r"nlp_model\config\threshold.json"

# --- Vision Model Architecture ---
def build_densenet(num_classes):
    m = models.densenet121(weights=None)
    n = m.classifier.in_features
    m.classifier = nn.Sequential( # type: ignore
        nn.Linear(n, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    # m.classifier = nn.Linear(n, num_classes)
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

# --- Cached Loaders ---
@st.cache_resource
def load_vision_model(path, num_classes):
    m = build_densenet(num_classes)
    try:
        ckpt = torch.load(path, map_location="cpu")
    except Exception:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    m.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    return CAMWrapper(m).to(DEVICE).eval()

@st.cache_resource
def load_nlp_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(
            TOKENIZER_PATH,
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False
        )
        state_dict = load_file(NLP_MODEL_PATH)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading NLP model: {e}")
        return None, None

def get_nlp_threshold():
    try:
        with open(THRESHOLD_FILE, 'r') as f:
            data = json.load(f)
            return data.get("threshold", 0.85)
    except:
        return 0.85

# --- Preprocessing & Inference ---
TF = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def read_gray(uploaded):
    data = np.frombuffer(uploaded.getvalue(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

def predict_vision(model, gray):
    img_rgb = np.stack([gray] * 3, axis=2)
    x = TF(img_rgb)
    # Ensure x is a torch.Tensor to avoid Pylance NDArray error
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(np.array(x))
    x = x.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    return x, probs

def gradcam_overlay(model, x, gray, class_idx):
    rgb = cv2.resize(gray, (224, 224)).astype(np.float32) / 255.0
    rgb = np.stack([rgb] * 3, axis=-1)
    with GradCAM(model=model, target_layers=[model.features.norm5]) as cam:
        heat = cam(input_tensor=x, targets=[ClassifierOutputTarget(class_idx)])[0] # type: ignore
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)  # Normalize to [0, 1]
        
        heat_thresholded = np.where(heat > 0.5, heat, 0)  # Apply threshold
        
        colormap = cv2.applyColorMap(np.uint8(255 * heat_thresholded), cv2.COLORMAP_JET)  # create a color map  # type: ignore
        colormap = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB) 
        
        # Blend with original
    alpha = 0.4  # Transparency (0 = only original, 1 = only heatmap)
    overlay = (1 - alpha) * rgb + alpha * (colormap / 255.0)
    overlay = np.clip(overlay, 0, 1)        
    
    return (overlay * 255).astype(np.uint8), heat

# --- UI Layout ---
st.set_page_config(page_title="Detective XAI: Multi-modal TB Detection", layout="wide")

st.title("🕵️ Detective XAI: Multi-modal TB Detection")
st.markdown("Combining Chest X-ray Analysis and Clinical Symptom NLP for Tuberculosis Detection.")
st.caption("Research prototype for testing only. Not a diagnostic tool.")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Settings")
    gate_thr = st.slider("Model A Gate Threshold", 0.5, 0.99, 0.90, 0.01)
    nlp_thr = st.number_input("NLP Decision Threshold", 0.0, 1.0, 0.85, 0.01)
    st.divider()
    st.write(f"Compute Device: **{DEVICE}**")

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None

# --- Input Section ---
col_img, col_nlp = st.columns(2)

with col_img:
    st.subheader("🖼️ Imaging Data")
    uploaded_file = st.file_uploader("Upload Chest X-ray", type=["png", "jpg", "jpeg", "bmp"])
    if uploaded_file:
        gray_preview = read_gray(uploaded_file)
        if gray_preview is not None:
            st.image(gray_preview, caption="Uploaded Image", use_container_width=True)

with col_nlp:
    st.subheader("📝 Clinical Data")
    with st.expander("Enter Patient Symptoms", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", 0, 120, 34)
        with c2:
            sex = st.selectbox("Sex", ["M", "F"])
        with c3:
            bmi = st.number_input("BMI", 5.0, 60.0, 23.7)

        st.divider()
        hiv_status = st.selectbox("HIV Status", ["Positive", "Negative", "Unknown"])
        cough_dur = st.number_input("Cough Duration (Weeks)", 0, 52, 0)

        st.write("**Key Symptoms:**")
        symptoms = {
            "fever": st.checkbox("Fever"),
            "night_sweats": st.checkbox("Night Sweats"),
            "weight_loss": st.checkbox("Weight Loss"),
            "hemoptysis": st.checkbox("Hemoptysis (Blood in cough)"),
            "chest_pain": st.checkbox("Chest Pain"),
            "fatigue": st.checkbox("Fatigue"),
            "loss_of_appetite": st.checkbox("Loss of Appetite"),
        }
        free_text = st.text_area("Additional Clinical Notes (Optional)")

st.divider()

# --- Prediction Trigger ---
if st.button("🚀 Predict Probability", use_container_width=True):
    # Load Models
    with st.spinner("Loading models and analyzing..."):
        model_a = load_vision_model(DEFAULT_A_PATH, 2)
        model_b = load_vision_model(DEFAULT_B_PATH, 3)
        tokenizer, nlp_model = load_nlp_model()
        if tokenizer is None or nlp_model is None:
            st.error("Failed to load NLP model.")
            st.stop()

        vision_res = None
        nlp_res = None

        # 1. Vision Pipeline
        if uploaded_file:
            gray = read_gray(uploaded_file)
            # Gateway (Model A)
            _, gp = predict_vision(model_a, gray)
            p_cxr = float(gp[1])

            if p_cxr >= gate_thr:
                # Model B
                x, probs = predict_vision(model_b, gray)
                idx = int(probs.argmax())
                vision_res = {
                    "status": "Success",
                    "class": CLASS_NAMES_B[idx],
                    "conf": probs[idx],
                    "probs": probs,
                    "cam": gradcam_overlay(model_b, x, gray, idx),
                    "image": gray
                }
            else:
                vision_res = {
                    "status": "Blocked",
                    "message": f"Please upload a valid chest X-ray."
                }

        # 2. NLP Pipeline
        record = {
            "age_years": age,
            "sex": sex,
            "bmi": bmi,
            "hiv_status": 1 if hiv_status == "Positive" else (0 if hiv_status == "Negative" else None),
            "cough_duration_weeks": cough_dur,
            "cough_2weeks": 1 if cough_dur >= 2 else 0
        }
        for s, val in symptoms.items():
            record[s] = 1 if val else 0

        clinical_text = build_clinical_text(record, free_text=free_text)
        inputs = tokenizer(clinical_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = nlp_model(**inputs)
            probs_nlp = torch.nn.functional.softmax(outputs.logits, dim=-1)
            tb_prob = probs_nlp[0][1].item()

        nlp_res = {
            "text": clinical_text,
            "prob": tb_prob,
            "prediction": "TB Positive" if tb_prob >= nlp_thr else "Normal / Not TB"
        }

    st.session_state.results = {"vision": vision_res, "nlp": nlp_res}

# --- Results Section ---
if st.session_state.results:
    st.header("🎯 Diagnostic Results")
    res = st.session_state.results

    # Vision Result Display
    if res["vision"]:
        v = res["vision"]
        if v["status"] == "Success":
            c1, c2, c3 = st.columns([1, 1, 1])
            c1.image(v["image"], caption="Original X-ray", use_container_width=True)
            c2.image(v["cam"][0], caption=f"Grad-CAM: {v['class']}", use_container_width=True)
            with c3:
                st.metric("Vision Prediction", v["class"], f"{v['conf']*100:.1f}%")
                st.bar_chart(pd.DataFrame({"probability": v["probs"]}, index=CLASS_NAMES_B))
        else:
            st.error(v["message"])
    else:
        st.info("No image provided for analysis.")

    st.divider()

    # NLP Result Display
    if res["nlp"]:
        n = res["nlp"]
        st.subheader("Clinical Symptom Analysis")
        st.info(f"**Generated Clinical Sentence:**\n\n{n['text']}")

        col_m, col_p = st.columns([1, 2])
        with col_m:
            color = "red" if "Positive" in n["prediction"] else "green"
            st.markdown(f"### Prediction: <span style='color:{color}'>{n['prediction']}</span>", unsafe_allow_html=True)
            st.metric("P(TB) Probability", f"{n['prob']:.4f}")
        with col_p:
            st.write("Probability Gauge")
            st.progress(n["prob"])
    else:
        st.info("No clinical data provided for analysis.")
