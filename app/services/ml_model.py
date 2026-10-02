import joblib
import os

model = None   

def load_model():
    global model

    if model is None:
        model_path = os.path.join("ml_model", "saved_model", "url_model.pkl")
        model = joblib.load(model_path)

    return model