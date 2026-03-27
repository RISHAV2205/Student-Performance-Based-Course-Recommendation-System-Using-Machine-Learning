import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../app/ml/model.joblib')

model = joblib.load(MODEL_PATH)

def predict(data):
    return model.predict([data])[0]