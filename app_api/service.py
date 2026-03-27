from .ml_model import predict

def recommend(data):
    pred = predict(data)

    if pred < 50:
        rec = "Basic Practice"
    elif pred < 75:
        rec = "Intermediate Problems"
    else:
        rec = "Advanced Challenges"

    return pred, rec