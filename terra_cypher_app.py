import os
import gdown

MODEL_PATH = "terra_emotion_model_vgg13.pt"
if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=17xJa95zg466gVIzIshKKwfld6Mfv20mR"
    gdown.download(url, MODEL_PATH, quiet=False)
