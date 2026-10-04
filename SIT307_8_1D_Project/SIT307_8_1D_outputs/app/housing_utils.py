"""Shared transformations for the SIT307 housing notebook and application."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor, make_column_selector
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASIC_FEATURES = ["suburb", "bedrooms", "bathrooms", "parking", "land_size_m2"]
RAW_FEATURES = ["suburb", "bedrooms", "bathrooms", "parking", "land_size_m2",
                "sale_date", "unit_style_address"]
MODEL_NAMES = ["Ridge regression", "K-nearest neighbours", "Random forest"]

class PropertyFeatures(BaseEstimator, TransformerMixin):
    """Row-wise, target-free features; learned preprocessing stays in the pipeline."""
    def __init__(self, feature_set="basic"):
        self.feature_set = feature_set

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        d = X.copy()
        out = d[["suburb", "bedrooms", "bathrooms", "parking"]].copy()
        for c in ["bedrooms", "bathrooms", "parking"]:
            out[c] = pd.to_numeric(out[c], errors="raise").astype(float)
        land = pd.to_numeric(d["land_size_m2"], errors="raise").astype(float)
        if self.feature_set == "basic":
            out["land_size_m2"] = land
        elif self.feature_set == "engineered":
            out["log_land_size"] = np.log1p(land)
            out["bathrooms_per_bedroom"] = out["bathrooms"] / out["bedrooms"]
            # Fixed origin: no statistic is learned from validation dates.
            out["sale_time_years"] = (
                pd.to_datetime(d["sale_date"]) - pd.Timestamp("2024-01-01")
            ).dt.days / 365.25
            out["unit_style_address"] = pd.to_numeric(d["unit_style_address"]).astype(float)
        else:
            raise ValueError("feature_set must be 'basic' or 'engineered'")
        return out

def make_estimator(name, seed=42):
    estimators = {
        "Ridge regression": Ridge(),
        "K-nearest neighbours": KNeighborsRegressor(),
        "Random forest": RandomForestRegressor(n_estimators=120, random_state=seed, n_jobs=1),
    }
    preprocess = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), make_column_selector(dtype_include=np.number)),
        ("suburb", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["suburb"]),
    ])
    pipe = Pipeline([
        ("features", PropertyFeatures()),
        ("preprocess", preprocess),
        ("model", estimators[name]),
    ])
    # Train on log prices; predict and evaluate in the original AUD scale.
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)

def parameter_grid(name):
    common = {"regressor__features__feature_set": ["basic", "engineered"]}
    grids = {
        "Ridge regression": {"regressor__model__alpha": [0.1, 1.0, 10.0, 100.0]},
        "K-nearest neighbours": {
            "regressor__model__n_neighbors": [3, 5, 9],
            "regressor__model__weights": ["uniform", "distance"],
        },
        "Random forest": {
            "regressor__model__max_depth": [4, None],
            "regressor__model__min_samples_leaf": [2, 4],
        },
    }
    return {**common, **grids[name]}

def required_input_features(metadata):
    """Only request variables used by the selected feature transformation."""
    return BASIC_FEATURES if metadata["feature_set"] == "basic" else RAW_FEATURES


def validate_inputs(frame, metadata):
    """Validate predictive inputs and supply unused compatibility columns internally."""
    required = required_input_features(metadata)
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    d = frame[required].copy()
    if d.empty:
        raise ValueError("The input contains no properties.")
    if metadata["feature_set"] == "basic":
        # These fields are part of the training schema but ignored by basic Ridge.
        # Older CSVs remain accepted; their redundant fields cannot affect predictions.
        d["sale_date"] = metadata["date_max"]
        d["unit_style_address"] = 0
    d["suburb"] = d["suburb"].astype(str).str.strip()
    unknown = sorted(set(d["suburb"]) - set(metadata["suburbs"]))
    if unknown:
        raise ValueError("Unsupported suburb(s): " + ", ".join(unknown))
    for c in ["bedrooms", "bathrooms", "parking", "land_size_m2", "unit_style_address"]:
        d[c] = pd.to_numeric(d[c], errors="raise")
        observed = d[c].dropna().to_numpy(dtype=float)
        if not np.isfinite(observed).all():
            raise ValueError(f"{c}: values must be finite.")
    for c in ["bedrooms", "bathrooms", "parking", "unit_style_address"]:
        if d[c].isna().any() or (d[c] % 1 != 0).any():
            raise ValueError(f"{c}: enter a whole number for every property.")
    if (d["bedrooms"] < 1).any() or (d["bathrooms"] < 1).any() or (d["parking"] < 0).any():
        raise ValueError("Bedrooms and bathrooms must be positive; parking can be zero.")
    if (~d["unit_style_address"].isin([0, 1])).any():
        raise ValueError("unit_style_address must be 0 or 1.")
    if (d["land_size_m2"].dropna() <= 0).any():
        raise ValueError("Land size must be positive or blank when unknown.")
    dates = pd.to_datetime(d["sale_date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("Enter valid dates in YYYY-MM-DD format.")
    d["sale_date"] = dates.dt.strftime("%Y-%m-%d")
    return d[RAW_FEATURES]


def input_warnings(d, metadata):
    messages = []
    if d["land_size_m2"].isna().any():
        messages.append("Unknown land sizes will use the model's training-data imputation.")
    for c, limits in metadata["numeric_ranges"].items():
        if ((d[c] < limits["min"]) | (d[c] > limits["max"])).any():
            messages.append(f"{c}: some values are outside the observed training range.")
    if metadata["feature_set"] != "basic":
        dates = pd.to_datetime(d["sale_date"])
        if ((dates < pd.Timestamp(metadata["date_min"])) | (dates > pd.Timestamp(metadata["date_max"]))).any():
            messages.append("Some dates are outside the training period; future-market performance is untested.")
    return messages


def predict_properties(frame, model, metadata):
    """Use the same validated inference path for single-property and CSV requests."""
    clean = validate_inputs(frame, metadata)
    predicted = np.asarray(model.predict(clean), dtype=float)
    if not np.isfinite(predicted).all() or (predicted <= 0).any():
        raise ValueError("The model could not return a finite positive price for these inputs.")
    result = clean[required_input_features(metadata)].copy()
    result["predicted_sale_price_aud"] = np.round(predicted, 0)
    return result, input_warnings(clean, metadata)
