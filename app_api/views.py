import json
import joblib
import pandas as pd

from django.http import JsonResponse
from django.views import View
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt

# Load model once
model = joblib.load('app/ml/model.pkl')


@method_decorator(csrf_exempt, name='dispatch')
class PredictView(View):

    def post(self, request):
        try:
            data = json.loads(request.body)

            # Convert input to DataFrame
            input_df = pd.DataFrame([data])

            # Prediction (pipeline handles preprocessing)
            prediction = model.predict(input_df)

            return JsonResponse({
                "predicted_score": round(float(prediction[0]), 2)
            })

        except Exception as e:
            return JsonResponse({
                "error": str(e)
            })

    def get(self, request):
        return JsonResponse({
            "message": "Send POST request with student data"
        })