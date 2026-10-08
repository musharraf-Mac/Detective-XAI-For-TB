import shap
import torch
import numpy as np
from shap.maskers import Text


class ClinicalShapExplainer:
    """SHAP explainer for clinical text classification (TB vs Normal)."""
    
    def __init__(self, nlp_model, tokenizer):
        self.model = nlp_model
        self.tokenizer = tokenizer
        
        # Prediction function — must handle all input formats SHAP sends
        def predict_proba(texts):
            # ─── Normalize input to list[str] ───
            if isinstance(texts, str):
                texts = [texts]
            elif isinstance(texts, np.ndarray):
                texts = texts.tolist()
            
            # SHAP Text masker may pass tokens as list of strings
            # If elements are lists, join tokens into single strings
            cleaned = []
            for t in texts:
                if isinstance(t, list):
                    # Re-join token list into a sentence
                    cleaned.append(" ".join(str(tok) for tok in t))
                else:
                    cleaned.append(str(t))
            
            # Tokenize
            inputs = self.tokenizer(
                cleaned,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
            
            return probs.cpu().numpy()
        
        # Use Text masker
        masker = Text(tokenizer)
        self.explainer = shap.Explainer(predict_proba, masker)
    
    def explain(self, clinical_text):
        """Generate SHAP values for a single clinical text."""
        # Ensure input is a plain string
        if not isinstance(clinical_text, str):
            clinical_text = str(clinical_text)
        return self.explainer([clinical_text])
    
    def get_top_symptoms(self, shap_values, top_k=5):
        """Extract top-k symptoms driving the prediction."""
        values = shap_values.values[0]
        tokens = shap_values.data[0]
        
        # TB class = index 1
        tb_values = values[:, 1]
        
        token_contributions = list(zip(tokens, tb_values))
        sorted_contribs = sorted(
            token_contributions,
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return sorted_contribs[:top_k]


def format_shap_explanation(shap_values, class_name="TB"):
    """Format SHAP output as readable text."""
    values = shap_values.values[0]
    tokens = shap_values.data[0]
    tb_values = values[:, 1]
    
    positive_tokens = [tokens[i] for i in range(len(tokens)) if tb_values[i] > 0.01]
    negative_tokens = [tokens[i] for i in range(len(tokens)) if tb_values[i] < -0.01]
    
    exp = f"**Tokens supporting {class_name}:** {', '.join(positive_tokens[:10])}\n\n"
    exp += f"**Tokens against {class_name}:** {', '.join(negative_tokens[:10])}"
    return exp