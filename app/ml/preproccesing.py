import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# MODELS
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

# METRICS
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# LOAD DATA
df = pd.read_csv(r"D:\academics\project\Performance_Based_Recommendation\app\ml\student_data.csv")

# FEATURES (IMPORTANT: include G1, G2 if available)
selected_features = [
    "studytime", "failures", "age", "goout",
    "traveltime", "Medu", "Fedu",
    "G1", "G2"   # 🔥 VERY IMPORTANT
]

X = df[selected_features]
y = df["G3"]

# PREPROCESSING
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), selected_features)
])

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MODELS DICTIONARY
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "SVR": SVR()
}

results = {}

# TRAIN & EVALUATE EACH MODEL
for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}

    print(f"\n🔹 {name}")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2 Score:", r2)

# SELECT BEST MODEL
best_model_name = max(results, key=lambda x: results[x]["R2"])
print("\n Best Model:", best_model_name)

# TRAIN BEST MODEL AGAIN & SAVE
best_model = models[best_model_name]

final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", best_model)
])

final_pipeline.fit(X_train, y_train)

joblib.dump(final_pipeline, "model.pkl")

print("✅ Best model saved successfully!")


# import matplotlib.pyplot as plt

# plt.scatter(y_test, y_pred)
# plt.xlabel("Actual Marks")
# plt.ylabel("Predicted Marks")
# plt.title("Actual vs Predicted")
# plt.show()


# residual plot
residuals = y_test - y_pred

import matplotlib.pyplot as plt

plt.scatter(y_pred, residuals)
plt.axhline(y=0, linestyle='--')
plt.xlabel("Predicted Marks")
plt.ylabel("Residuals (Error)")
plt.title("Residual Plot")
plt.show()