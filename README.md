# Detetive XAI for Tuberculosis

**Combining Clinical Data and Chest X-Ray Imaging with Validated XAI Methods for Resource-Constrained Healthcare Settings**

IT41043 — Intelligent Systems | Horizon Campus | Academic Year 2026 | Third Year, Second Semester

---

## Project Overview

This repository contains the implementation for our IT41043 group research project: a lightweight, multimodal, explainable AI framework for tuberculosis (TB) diagnosis. The system combines chest X-ray imaging with structured clinical data (symptoms, demographics, lab results) to improve diagnostic accuracy, while remaining efficient enough to run on low-spec hardware typical of resource-constrained healthcare settings.

**Core research contributions:**
1. Multimodal fusion of imaging (EfficientNet-B0) and clinical data (MLP) via late fusion
2. Dual explainability — Grad-CAM for imaging features, SHAP for clinical features
3. Validation of XAI outputs against radiologist-annotated TB lesion regions (IoU)
4. Efficiency benchmarking (model size, CPU-only inference time) for low-resource deployment
5. Generalisation testing on locally collected chest X-ray samples

---

## Team

| Name | Student ID | Github user name |
|---|---|---|
| MR. Abdur Rahman Musharraf | ITBIN-2313-0067 | Musharraf-mac |
| MMZ. Suzanee | ITBIN-2313-0113| Zaffraj-Suzanee |

**Module Coordinator:** Mr. Isuru Madusanka Samarappulige

---

## Model Architecture
![Model Architecture](<Multi - Model Architecture.drawio.svg>)


---

## Dataset

- **Public imaging data:** NIH ChestX-ray14, TBX11K (with radiologist-annotated bounding boxes)
- **Primary local data:** Anonymised chest X-ray and clinical data, collected via hospital visits and an anonymous submission form, subject to institutional ethics approval
- Full dataset description available in [`docs/milestone2_methodology.md`](docs/milestone2_methodology.md)

**Note:** Raw patient data is never committed to this public repository, in line with our ethics and data protection commitments. See `data/README.md` for details.

---

## Setup

```bash
git clone https://github.com/[your-username]/tb-multimodal-xai.git
cd tb-multimodal-xai
pip install -r requirements.txt
```

---

## Evaluation

- **Accuracy metrics:** AUC-ROC, Sensitivity, Specificity, F1-score
- **XAI validity:** IoU (Grad-CAM vs radiologist annotations, threshold ≥ 0.4)
- **Efficiency:** Model size, CPU-only inference time, peak RAM usage
- **Validation:** Stratified 5-fold cross-validation, Wilcoxon signed-rank test for significance

---

## Status

**Current stage:** Milestone 2 — Methodology and Data Description (Week 6)

- [x] Research gap and question finalised (Milestone 1)
- [x] Dataset sourcing plan finalised
- [x] System architecture designed
- [x] Data collection and preprocessing (in progress)
- [x] Model training
- [ ] XAI implementation and validation
- [ ] Final results and paper (Milestone 3–4)

---

## License

Academic project — Horizon Campus, IT41043. Not licensed for commercial use.

---

## Acknowledgements

- TorchXRayVision (Cohen et al., 2022) for pre-trained CXR models
- TBX11K dataset for radiologist-annotated TB lesion bounding boxes
- NIH Clinical Center for the ChestX-ray14 dataset
