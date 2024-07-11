#!/usr/bin/env python
# coding: utf-8

import joblib
import numpy as np
from lime.lime_text import LimeTextExplainer

class DisasterCaptionClassifier:
    def __init__(self, model_path='F_text_model.pkl', classes=None):
        self.model = joblib.load(model_path)
        self.classes = classes or ["damaged_infra", "damaged_nature", "fire", "flood", "human_damage", "non_damage"]
        self.explainer = LimeTextExplainer()

    def predict_text(self, sentence):
        max_idx = np.argmax(self.model.predict_proba([sentence]))
        print(self.classes[max_idx])

    def lime_explainability(self, sentence, num_features=5):
        exp = self.explainer.explain_instance(sentence, self.model.predict_proba, num_features=num_features)
        return exp

    def save_explanation_to_html(self, explanation_html, filename="lime_explanation.html"):
        with open(filename, "w", encoding="utf-8") as file:
            file.write(explanation_html)
        print(f"Explanation saved to {filename}")

    def highlighted_text_html(self, exp):
        # Extract highlighted text
        html = exp.as_html()
        return f"<div>{html}</div>"

    def get_highlighted_text(self, sentence):
        exp = self.lime_explainability(sentence)
        return self.highlighted_text_html(exp)

# Example usage
if __name__ == "__main__":
    classifier = DisasterCaptionClassifier()

    # Example prediction
    sample_text = "All these fires around Washington sure do make for some great photos. Thank you to all wildland firefighters. #flyfishing #washington #explore #forrestfire"
    classifier.predict_text(sample_text)

    # Generate and save highlighted text HTML
    highlighted_html = classifier.get_highlighted_text(sample_text)
    classifier.save_explanation_to_html(highlighted_html, "highlighted_text.html") 