import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# STEP 2: LOAD DATA
df = pd.read_csv(r"D:\academics\project\Performance_Based_Recommendation\app\ml\student_data.csv")



# STEP 3: SELECT FEATURES (FINAL)
selected_features = [
    "studytime",
    "failures",
    "age",
    "goout",
    "traveltime",
    "Medu",
    "Fedu"
]

X = df[selected_features]
y = df["G3"]

# STEP 4: PREPROCESSING
# All are numeric → only scaling needed
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), selected_features)
])

# STEP 5: PIPELINE
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

# STEP 6: TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# STEP 7: TRAIN MODEL
pipeline.fit(X_train, y_train)

# STEP 8: SAVE MODEL
joblib.dump(pipeline, "model.pkl")

print("✅ Model trained & saved successfully!")