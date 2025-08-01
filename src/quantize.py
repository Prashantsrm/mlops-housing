import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing

# Load model and parameters
model = joblib.load("src/model.joblib")
coefs, intercept = model.coef_, model.intercept_

joblib.dump((coefs, intercept), "src/unquant_params.joblib")

# Quantize to uint8
min_coef, max_coef = coefs.min(), coefs.max()
scale = 255 / (max_coef - min_coef)
coefs_q = ((coefs - min_coef) * scale).astype(np.uint8)
intercept_q = int((intercept - min_coef) * scale)
joblib.dump((coefs_q, intercept_q, min_coef, scale), "src/quant_params.joblib")

def dequant_predict(X):
    coefs_dq = coefs_q.astype(np.float32) / scale + min_coef
    intercept_dq = intercept_q / scale + min_coef
    return X @ coefs_dq + intercept_dq

# Inference check
X, y = fetch_california_housing(return_X_y=True)
preds = dequant_predict(X)
print("Sample dequantized predictions:", preds[:5])

