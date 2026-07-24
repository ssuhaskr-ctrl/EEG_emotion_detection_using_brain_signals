"""
predict_emotion.py

Load the trained CNN model and predict emotion from one EEG sample.
"""

import torch
import torch.nn.functional as F

from data.data_loader import load_all_subjects
from preprocessing.preprocessor import preprocess_dataset
from models.cnn_model import CNNModel
from config import EMOTION_LABELS

# ======================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "results/best_deep_model.pth"

emotion_map = EMOTION_LABELS

# ======================================================

print("=" * 60)
print("GAMEEMO EEG EMOTION DETECTION")
print("=" * 60)

print("\nLoading Dataset...\n")

X, y, subjects = load_all_subjects()

print("Dataset Shape :", X.shape)

print("\nPreprocessing Dataset...\n")

X, y, subjects, pp = preprocess_dataset(
    X,
    y,
    subjects,
    apply_ica=False
)

print("Processed Shape :", X.shape)

# ======================================================

sample_index = 100

sample = X[sample_index]

true_label = y[sample_index]

subject = subjects[sample_index]

sample_tensor = torch.tensor(
    sample,
    dtype=torch.float32
).unsqueeze(0)

sample_tensor = sample_tensor.to(DEVICE)

# ======================================================

print("\nLoading CNN Model...\n")

model = CNNModel()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.to(DEVICE)

model.eval()

print("Model Loaded Successfully")

# ======================================================

with torch.no_grad():

    output = model(sample_tensor)

    probability = F.softmax(output, dim=1)

    confidence, prediction = torch.max(
        probability,
        dim=1
    )

predicted_class = prediction.item()

confidence_score = confidence.item() * 100

# ======================================================

print()

print("=" * 60)

print("EMOTION DETECTION RESULT")

print("=" * 60)

print()
subject = subjects[sample_index]

print("Subject ID :", f"S{int(subject):02d}")
#print("Subject ID        :", subject)

print("Sample Index      :", sample_index)

print()

print("True Class        :", true_label)

print("Predicted Class   :", predicted_class)

print()

print("Detected Emotion  :", emotion_map[predicted_class])

print()

print("Confidence Score  : {:.2f}%".format(confidence_score))

print()

print("=" * 60)
