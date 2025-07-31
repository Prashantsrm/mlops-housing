from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# Data load
X, y = fetch_california_housing(return_X_y=True)

# Model Training
model = LinearRegression()
model.fit(X, y)
preds = model.predict(X)

# Metrics
r2 = r2_score(y, preds)
mse = mean_squared_error(y, preds)
print(f"R2 score: {r2:.4f}, MSE: {mse:.4f}")

# Save model
joblib.dump(model, "src/model.joblib")

