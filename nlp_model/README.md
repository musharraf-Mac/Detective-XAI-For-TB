# TB NLP / Symptom Model

Predicts **P(TB)**, the probability of tuberculosis, from structured symptoms. Each patient record is converted into a short clinical sentence, which a fine-tuned **Bio_ClinicalBERT** reads. This model is the NLP branch of the multimodal TB project. It will later be fused with the DenseNet-121 chest X-ray model.

> **Research prototype.**

## Contents

| Path | Description |
|---|---|
| `notebooks/` | Google Colab notebook: inspection, leakage audit, baselines, training, threshold selection, evaluation, saving |
| `tb_utils.py` | Builds the clinical sentence from a patient record (must match training) |
| `config/` | Preprocessing config, label mapping, frozen classification threshold |
| `results/` | Validation/test metrics and the patient IDs of each split |

## Trained model

The folder contains `model/`, `tokenizer/`, `tb_utils.py`, `threshold.json`, `preprocessing_config.json`, `label_mapping.json`, `metrics.json` and `splits.json`.

## Data

- Three CSV files (high, moderate and low TB burden), 10,000 rows each, 25 identical columns.
- Target: `true_tb_status` (0 = not TB, 1 = TB), a patient-level label. The file names "high/moderate/low burden" describe the setting, not the patient, and were **not** used to create labels.
- TB rate per file: about 35% (high), 17% (moderate), 6.4% (low); 19.6% overall.
- There is **no free-text column**. Text is generated from structured fields.
- Source of the data and of `true_tb_status`: `<TO BE CONFIRMED: real or simulated, how the label was determined>`.
- Raw CSV files are not stored in this repository.

## Method

1. **Inputs used:** age, sex, BMI, HIV status, cough duration, cough 2+ weeks, fever, night sweats, weight loss, hemoptysis, chest pain, fatigue, loss of appetite.
2. **Removed (leakage):** `tb_type`, `tb_classification`, `treatment_outcome`, `rifampicin_resistant`, `smear_result`, `xpert_mtb_detected`, `xpert_rif_resistance`, `cxr_result`, `cxr_abnormal`. These are lab tests, X-ray results, classifications or outcomes that appear at or after diagnosis. X-ray columns are also excluded so the X-ray branch is not double-counted in fusion. The derived `who_symptom_screen_count` was also removed.
3. **Not a feature:** the burden setting (file) and the `id`. The burden is used only for stratification and reporting.
4. **Split:** 70/15/15 train/validation/test, stratified and **grouped by patient `id`**. The same `id` appears in all three files, so it never crosses between splits.
5. **Text format:** for example, "34-year-old female. BMI 23.7. HIV negative. Cough for 7 weeks. Reports fever. Denies weight loss..." Present symptoms use "Reports", absent ones use "Denies", and unknown values use "not recorded". Negation words are kept and nothing is lower-cased.
6. **Model:** `emilyalsentzer/Bio_ClinicalBERT`, fine-tuned with class-weighted loss, early stopping on validation ROC-AUC, and the best checkpoint kept. Seed 42.
7. **Threshold:** chosen on the validation set only (best F1 among thresholds with validation sensitivity of at least 0.87). It was frozen at **0.85**, and the test set was used once.

## Results (internal held-out test set)

| Model | Threshold | Sensitivity | Specificity | Precision | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| TF-IDF + Logistic Regression | 0.70 | 0.910 | 0.980 | 0.915 | 0.913 | 0.993 |
| Structured Logistic Regression | 0.55 | 0.947 | 0.967 | 0.875 | 0.909 | 0.993 |
| Structured Gradient Boosting | 0.65 | 0.927 | 0.976 | 0.903 | 0.915 | 0.992 |
| **Bio_ClinicalBERT** | 0.85 | 0.895 | 0.983 | 0.926 | 0.910 | 0.993 |

Targets (F1 > 0.80, ROC-AUC > 0.85, sensitivity > 0.85) were met by all four models. See `results/metrics.json` for confidence intervals and the validation results.

## How to read these results

- The transformer performs **about the same as the simple baselines**. It reads the same yes/no symptoms, so it has no extra information to learn from. It would only show a benefit with real free-text clinical notes.
- Sensitivity and specificity depend on the threshold. They are not fixed properties of the model.
- Precision depends on how common TB is. It is lower in low-burden settings.
- Class weighting inflates the raw probabilities. Calibrate `P(TB)` before interpreting it as a literal risk, and before fusion.

## Usage

```python
from tb_utils import build_clinical_text
# After loading model and tokenizer (see the notebook, Step 16-17):
result = predict_tb_structured(age_years=34, sex="F", bmi=17.5, hiv_status=1,
                               cough_duration_weeks=4, fever=1, night_sweats=1,
                               weight_loss=1, hemoptysis=0, chest_pain=0,
                               fatigue=1, loss_of_appetite=1)
# returns tb_probability, prediction, threshold
```

Use structured input. Free-text sentences differ from the training format and give less reliable output.

## Interface for fusion

`get_tb_nlp_probability(...)` returns `{"tb_probability": float}`. To fuse with the DenseNet-121 output:

- Train the fusion model on patients that have both an X-ray and symptoms.
- Use predictions on data the NLP model did not train on (validation or out-of-fold).
- Calibrate the probabilities first, and choose a separate threshold for the fused model.

## Limitations

- The dataset may be simulated: a few basic symptoms separate TB from non-TB almost perfectly, and IDs and demographics repeat across files. Confirm its origin before drawing conclusions.
- The setting is confounded with patient mix (for example HIV prevalence about 27%, 11%, 4% across files).

## Reproducibility

Seeds are set for Python, NumPy and PyTorch. The exact patient IDs of each split are in `results/splits.json`. Library versions, GPU and hyperparameters are in `config/preprocessing_config.json`. GPU maths can cause small differences between runs.
