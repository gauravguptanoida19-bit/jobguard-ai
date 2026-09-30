"""Explainability and human-readable reasoning modules."""
from src.explain.shap_explainer import compute_shap_explanations
from src.explain.reasons import generate_plain_english_reasons

__all__ = ["compute_shap_explanations", "generate_plain_english_reasons"]
