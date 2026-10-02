import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import joblib


# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Final hybrid model folder
MODEL_DIR = BASE_DIR / "models" / "final_hybrid_model"


# Load DistilBERT tokenizer
TOKENIZER = AutoTokenizer.from_pretrained(
    MODEL_DIR / "tokenizer"
)

# Load DistilBERT model
TEXT_MODEL = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
)

TEXT_MODEL.eval()


# Load the trained XGBoost classifier
model = joblib.load(
    MODEL_DIR / "hybrid_distilbert_behavior_xgboost.joblib"
)


# Load the classification threshold
with open(MODEL_DIR / "classification_threshold.txt", "r") as f:
    THRESHOLD = float(f.read().strip())


def predict(features):
    probability = model.predict_proba(features)[0][1]

    if probability >= THRESHOLD:
        prediction = "Potentially Fake"
    else:
        prediction = "Genuine"

    return prediction, probability
def get_text_embedding(review_text):
    inputs = TOKENIZER(
        review_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )

    with torch.no_grad():
        outputs = TEXT_MODEL.base_model(**inputs)

    hidden_states = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"].unsqueeze(-1)

    masked_embeddings = hidden_states * attention_mask

    embedding = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1)

    return embedding.cpu().numpy()
def build_hybrid_features(review_text, behavioral_features):
    text_embedding = get_text_embedding(review_text)

    behavioral_array = behavioral_features.to_numpy(
        dtype=np.float32
    )

    hybrid_features = np.concatenate(
        [text_embedding, behavioral_array],
        axis=1,
    )

    return hybrid_features