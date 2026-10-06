import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nlp_model.tb_utils import build_clinical_text
import os
import json

# --- Configuration ---
MODEL_PATH = r"models\Sym_nlp.safetensors"
TOKENIZER_PATH = "emilyalsentzer/Bio_ClinicalBERT" # Using base tokenizer as specified in README
THRESHOLD_FILE = r"nlp_model\config\threshold.json"

@st.cache_resource
def load_nlp_model():
    """Loads the Bio_ClinicalBERT model from safetensors."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        # We load the model architecture and then the weights from safetensors
        model = AutoModelForSequenceClassification.from_pretrained(
            TOKENIZER_PATH,
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False
        )

        # Load the specific safetensors weights
        from safetensors.torch import load_file
        state_dict = load_file(MODEL_PATH)
        model.load_state_dict(state_dict)

        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading NLP model: {e}")
        return None, None

def get_threshold():
    """Retrieves the frozen threshold from config."""
    try:
        with open(THRESHOLD_FILE, 'r') as f:
            data = json.load(f)
            return data.get("threshold", 0.85) # Default to 0.85 from README
    except:
        return 0.85

def main():
    st.set_page_config(page_title="TB NLP Symptom Tester", page_icon="📝")

    st.title("📝 TB Symptom Analysis (NLP)")
    st.markdown("""
    This tool tests the **Bio_ClinicalBERT** model. It converts structured symptoms into
    a clinical sentence and predicts the probability of Tuberculosis.
    """)

    # Load Model & Tokenizer
    tokenizer, model = load_nlp_model()
    if model is None:
        st.stop()

    threshold = get_threshold()

    # --- Input Form ---
    with st.form("symptom_form"):
        st.subheader("Patient Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=34)
        with col2:
            sex = st.selectbox("Sex", ["M", "F"])
        with col3:
            bmi = st.number_input("BMI", min_value=5.0, max_value=60.0, value=23.7)

        st.divider()
        st.subheader("Clinical Symptoms")

        hiv_status = st.selectbox("HIV Status", ["Positive", "Negative", "Unknown"])
        cough_duration = st.number_input("Cough Duration (Weeks)", min_value=0, max_value=52, value=0)

        # Boolean symptoms
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

        submit = st.form_submit_button("Predict Probability")

    if submit:
        # 1. Prepare the record for tb_utils
        record = {
            "age_years": age,
            "sex": sex,
            "bmi": bmi,
            "hiv_status": 1 if hiv_status == "Positive" else (0 if hiv_status == "Negative" else None),
            "cough_duration_weeks": cough_duration,
            "cough_2weeks": 1 if cough_duration >= 2 else 0
        }
        # Add the boolean symptoms (True/False -> 1/0)
        for s, val in symptoms.items():
            record[s] = 1 if val else 0

        # 2. Build the clinical sentence
        clinical_text = build_clinical_text(record, free_text=free_text)

        st.info(f"**Generated Clinical Sentence:**\n\n{clinical_text}")

        # 3. Model Inference
        with st.spinner("Analyzing symptoms..."):
            inputs = tokenizer(clinical_text, return_tensors="pt", truncation=True, padding=True, max_length=512) # type: ignore 
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                # Apply softmax to get probabilities for the 2 classes [Not TB, TB]
                probs = torch.nn.functional.softmax(logits, dim=-1)
                tb_prob = probs[0][1].item()

        # 4. Results
        prediction = "TB Positive" if tb_prob >= threshold else "Normal / Not TB"
        color = "red" if prediction == "TB Positive" else "green"

        st.markdown(f"### Prediction: <span style='color:{color}'>{prediction}</span>", unsafe_allow_html=True)
        st.metric("P(TB) Probability", f"{tb_prob:.4f}")
        st.write(f"**Threshold used:** {threshold}")

        # Probability bar
        st.progress(tb_prob)

if __name__ == "__main__":
    main()
