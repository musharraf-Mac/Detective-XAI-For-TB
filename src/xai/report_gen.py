"""
Template-based clinical report generator.
Combines Grad-CAM findings, SHAP symptom attributions, and model predictions
into a natural-language clinical explanation.
"""
# SYMPTOM MAPPING

SYMPTOM_DESCRIPTIONS = {
    "cough_2weeks": "cough lasting more than 2 weeks",
    "hemoptysis": "coughing up blood (hemoptysis)",
    "fever": "fever",
    "night_sweats": "night sweats",
    "weight_loss": "unexplained weight loss",
    "chest_pain": "chest pain",
    "fatigue": "persistent fatigue",
    "loss_of_appetite": "loss of appetite",
}

# REGION MAPPING (from Grad-CAM)

def interpret_gradcam_region(heatmap_stats):
    """Convert Grad-CAM stats to clinical language."""
    region = heatmap_stats.get('region', 'unknown')
    concentration = heatmap_stats.get('concentration', 0)
    
    if 'upper' in region:
        clinical = "upper lung zones"
    elif 'middle' in region:
        clinical = "middle lung zones"
    elif 'lower' in region:
        clinical = "lower lung zones"
    else:
        clinical = "diffuse lung fields"
    
    if concentration > 0.3:
        quality = "The model's attention was diffuse, suggesting low confidence."
    elif concentration < 0.05:
        quality = "The model focused on a specific region."
    else:
        quality = "The model showed moderate focus."
    
    return clinical, quality

# REPORT GENERATOR

def generate_clinical_report(
    vision_prediction,      # "TB", "Normal", "Abnormal"
    vision_confidence,      # 0.0 - 1.0
    heatmap_stats,          # dict from analyze_heatmap()
    active_symptoms,        # list of symptom keys
    nlp_probability,        # P(TB) from NLP model
    nlp_threshold=0.85      # decision threshold
):
    """
    Generate a natural-language clinical report explaining the AI decision.
    """    
    report = []
    
    # ---- 1. Prediction Summary ----
    report.append("## Diagnostic Report\n")
    report.append(f"**AI Prediction:** {vision_prediction} ")
    report.append(f"(Confidence: {vision_confidence * 100:.1f}%)\n")
    
    # ---- 2. Visual Explanation (Grad-CAM) ----
    report.append("\n### What the X-ray Analysis Found")
    
    region, quality = interpret_gradcam_region(heatmap_stats)
    
    if vision_prediction == "TB":
        report.append(
            f"The AI model detected features consistent with **tuberculosis** "
            f"in the **{region}**. {quality}"
        )
        if 'upper' in region:
            report.append(
                "Upper lobe involvement is a **classic presentation of TB**."
            )
        else:
            report.append(
                f"⚠️ Note: TB typically presents in the **upper lobes**. "
                f"The focus on the {region} is atypical."
            )
    
    elif vision_prediction == "Normal":
        report.append(
            "The model found **no significant radiographic abnormalities**. "
            "Lung fields appear clear."
        )
        report.append(
            f"{quality} If symptoms are present, clinical correlation is recommended."
        )
    
    else:  # Abnormal
        report.append(
            f"The model detected **abnormal findings** in the **{region}** "
            f"that are **not consistent with TB**."
        )
        report.append(
            "This may indicate pneumonia, pleural effusion, or other pathology. "
            "Further diagnostic workup is recommended."
        )
    
    # ---- 3. Symptom Explanation (SHAP) ----
    report.append("\n### Clinical Symptoms Analysis")
    
    if active_symptoms:
        symptom_text = ", ".join([
            SYMPTOM_DESCRIPTIONS.get(s, s) for s in active_symptoms
        ])
        report.append(f"**Reported symptoms:** {symptom_text}")
        
        if nlp_probability >= nlp_threshold:
            report.append(
                f"\nThe clinical NLP model detected these symptoms as "
                f"**highly suggestive of TB** (P(TB) = {nlp_probability:.4f})."
            )
        else:
            report.append(
                f"\nThe clinical NLP model did not find strong TB signal "
                f"from symptoms alone (P(TB) = {nlp_probability:.4f})."
            )
    else:
        report.append("No clinical symptoms were reported.")
    
    # ---- 4. Combined Assessment ----
    report.append("\n### Combined Assessment")
    
    if vision_prediction == "TB" and nlp_probability >= nlp_threshold:
        report.append(
            "✅ **Both imaging and clinical data strongly support TB.** "
            "This is a high-confidence finding."
        )
    elif vision_prediction == "TB" and nlp_probability < nlp_threshold:
        report.append(
            "⚠️ **Imaging suggests TB, but symptoms do not correlate.** "
            "Consider re-evaluation or additional testing."
        )
    elif vision_prediction == "Normal" and nlp_probability >= nlp_threshold:
        report.append(
            "⚠️ **X-ray appears normal, but symptoms suggest TB.** "
            "Recommend sputum testing despite normal imaging."
        )
    elif vision_prediction == "Normal":
        report.append(
            "✅ **No evidence of TB from imaging or symptoms.** "
            "Routine follow-up recommended."
        )
    else:
        report.append(
            "**Abnormal findings detected.** "
            "Recommend referral for further diagnostic workup."
        )
    
    # ---- 5. Recommendation ----
    report.append("\n### Recommended Next Steps")
    
    if vision_prediction == "TB" or nlp_probability >= nlp_threshold:
        report.append(
            "1. **Sputum test (AFB smear and GeneXpert MTB/RIF)** "
            "to confirm TB and determine drug susceptibility.\n"
            "2. **Refer to a TB specialist** for treatment initiation.\n"
            "3. **Notify public health authorities** as required by law."
        )
    elif vision_prediction == "Normal":
        report.append(
            "1. **Routine follow-up** if symptoms persist.\n"
            "2. **Monitor symptoms** for 2-3 weeks."
        )
    else:
        report.append(
            "1. **Chest CT scan** for detailed evaluation.\n"
            "2. **Refer to pulmonologist** for further workup.\n"
            "3. **Rule out other infectious or inflammatory causes.**"
        )
    
    # ---- 6. Disclaimer ----
    report.append("\n---")
    report.append(
        "⚠️ **Disclaimer:** This is an AI-generated report for research "
        "purposes only. It is not a substitute for professional medical advice. "
        "Always consult a qualified clinician for diagnosis and treatment."
    )
    
    return "\n".join(report)