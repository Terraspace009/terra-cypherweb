import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
import numpy as np
import cv2

def load_model(path, device):
    """Load VGG13 model trained on grayscale (converted to RGB) with 7 outputs."""
    model = models.vgg13(weights=None)
    model.classifier[6] = nn.Linear(4096, 7)

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model

def preprocess_image(image, device):
    """Preprocess image (grayscale -> 3-channel, resize, normalize)."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rgb_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(rgb_image, (224, 224))
    tensor = torch.tensor(resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    tensor /= 255.0
    return tensor.to(device)

def predict_emotion_with_smoothing(model, image, labels, history, device, max_len=12):
    """Weighted smoothing for stable emotion predictions with boosting for weaker emotions."""
    input_tensor = preprocess_image(image, device)
    with torch.no_grad():
        output = model(input_tensor)
        logits = output.squeeze(0)

        # Boost weaker emotions
        SAD_INDEX = labels.index("sad")
        DISGUST_INDEX = labels.index("disgust")
        ANGRY_INDEX = labels.index("angry")
        FEAR_INDEX = labels.index("fear")
        NEUTRAL_INDEX = labels.index("neutral")

        logits[SAD_INDEX] *= 1.35
        logits[DISGUST_INDEX] *= 1.3
        logits[ANGRY_INDEX] *= 1.25
        logits[FEAR_INDEX] *= 1.15
        logits[NEUTRAL_INDEX] *= 0.85  # reduce Neutral dominance

        probs = F.softmax(logits, dim=0).cpu().numpy()

    history.append(probs)
    if len(history) > max_len:
        history.pop(0)

    weights = np.linspace(0.6, 1.0, len(history))
    smoothed_probs = np.average(history, axis=0, weights=weights)

    top_idx = int(np.argmax(smoothed_probs))
    confidence = float(smoothed_probs[top_idx])

    return labels[top_idx], confidence, smoothed_probs
