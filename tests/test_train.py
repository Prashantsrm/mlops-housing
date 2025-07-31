import pytest
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

def test_dataset_loads():
    X, y = fetch_california_housing(return_X_y=True)
    assert X.shape[0] > 0

def test_model_instance():
    model = LinearRegression()
    assert isinstance(model, LinearRegression)

def test_model_training():
    X, y = fetch_california_housing(return_X_y=True)
    model = LinearRegression().fit(X, y)
    assert hasattr(model, "coef_")

def test_r2_score_threshold():
    X, y = fetch_california_housing(return_X_y=True)
    model = LinearRegression().fit(X, y)
    r2 = model.score(X, y)
    assert r2 > 0.5

