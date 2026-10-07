import re, unicodedata
import pandas as pd

TEMPLATE_VERSION = "v1"

SYMPTOM_PHRASES = {
    "fever": "fever",
    "night_sweats": "night sweats",
    "weight_loss": "weight loss",
    "hemoptysis": "hemoptysis (coughing up blood)",
    "chest_pain": "chest pain",
    "fatigue": "fatigue",
    "loss_of_appetite": "loss of appetite",
}

def _missing(v):
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip() == ""
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False

def clean_text(s):
    """Light, safe cleaning. Keeps case and keeps negation words such as 'no', 'denies', 'without'."""
    if _missing(s):
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = re.sub(r"[\x00-\x1f\x7f]", " ", s)      # control characters
    s = re.sub(r"\s+", " ", s).strip()           # repeated spaces / newlines
    return s

def _flag(v):
    """Return True / False / None (unknown) from 1/0, yes/no, true/false ..."""
    if _missing(v):
        return None
    if isinstance(v, str):
        t = v.strip().lower()
        if t in ("1", "yes", "y", "true", "present", "positive"):
            return True
        if t in ("0", "no", "n", "false", "absent", "negative"):
            return False
        return None
    return bool(v)

def build_clinical_text(rec, free_text=None):
    """Turn one patient record (dict or pandas Series) into a short clinical sentence."""
    g = rec.get
    parts = []

    sex = g("sex")
    sex_word = None
    if not _missing(sex):
        sex_word = {"m": "male", "male": "male", "f": "female", "female": "female"}.get(str(sex).strip().lower())
    age = g("age_years")
    try:
        age_i = int(round(float(age))) if not _missing(age) else None
    except (TypeError, ValueError):
        age_i = None
    if age_i is not None:
        parts.append(f"{age_i}-year-old {sex_word or 'patient'}.")
    else:
        parts.append("Age not recorded.")
        if sex_word:
            parts.append(f"Sex: {sex_word}.")

    bmi = g("bmi")
    try:
        parts.append(f"BMI {float(bmi):.1f}." if not _missing(bmi) else "BMI not recorded.")
    except (TypeError, ValueError):
        parts.append("BMI not recorded.")

    hiv = _flag(g("hiv_status"))
    parts.append("HIV positive." if hiv is True else "HIV negative." if hiv is False else "HIV status not recorded.")

    dur, c2 = g("cough_duration_weeks"), _flag(g("cough_2weeks"))
    dur_val = None
    try:
        dur_val = float(dur) if not _missing(dur) else None
    except (TypeError, ValueError):
        dur_val = None
    if dur_val is not None and dur_val > 0:
        n = int(round(dur_val))
        parts.append(f"Cough for {n} week{'s' if n != 1 else ''}.")
    elif dur_val is not None:
        parts.append("No cough.")
    elif c2 is True:
        parts.append("Cough lasting 2 weeks or more.")
    elif c2 is False:
        parts.append("No cough lasting 2 weeks or more.")
    else:
        parts.append("Cough not recorded.")

    for key, phrase in SYMPTOM_PHRASES.items():
        f = _flag(g(key))
        if f is True:
            parts.append(f"Reports {phrase}.")
        elif f is False:
            parts.append(f"Denies {phrase}.")
        else:
            parts.append(f"{phrase[0].upper() + phrase[1:]} not recorded.")

    ft = clean_text(free_text)
    if ft:
        parts.append("Clinical notes: " + ft)
    return clean_text(" ".join(parts))