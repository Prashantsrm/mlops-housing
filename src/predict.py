import joblib
from sklearn.datasets import fetch_california_housing

model = joblib.load("src/model.joblib")
X, y = fetch_california_housing(return_X_y=True)
preds = model.predict(X)
print("R2 Score:", model.score(X, y))
print("Sample predictions:", preds[:5])

