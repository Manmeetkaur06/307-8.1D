"""Local prototype. Run with: python -m streamlit run app.py"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
from housing_utils import required_input_features, predict_properties

st.set_page_config(page_title="Sydney property price explorer", layout="centered")
st.title("Sydney property price explorer")
st.caption("SIT307 8.1D | Mosman, Parramatta and Campbelltown")

@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / "housing_price_model.joblib")

try:
    metadata = json.loads((APP_DIR / "model_metadata.json").read_text())
    model = load_model()
except FileNotFoundError:
    st.error("Run the notebook's model-export section first, then restart the application.")
    st.stop()

st.write("Enter property details to estimate a sale price in Australian dollars.")
st.info(f"This prototype was trained on {metadata['n_properties']} recorded sold properties from Mosman, Parramatta and Campbelltown. Predictions are indicative and should not be treated as professional valuations.")
single, batch = st.tabs(["One property", "Upload CSV"])
required = required_input_features(metadata)

with single:
    with st.form("property_form"):
        suburb = st.selectbox("Suburb", metadata["suburbs"])
        a, b = st.columns(2)
        bedrooms = a.number_input("Bedrooms", min_value=1, max_value=20, value=3, step=1)
        bathrooms = b.number_input("Bathrooms", min_value=1, max_value=20, value=2, step=1)
        parking = a.number_input("Parking spaces", min_value=0, max_value=20, value=1, step=1)
        land_unknown = b.checkbox("Land size unknown", value=False)
        land = a.number_input("Land size (m²)", min_value=1.0, max_value=100000.0, value=500.0, step=10.0)
        st.caption("Tick Land size unknown if the property's usable land area is unavailable or only a shared-complex area is reported.")
        # Only appears if a future rerun actually selects engineered features.
        if metadata["feature_set"] != "basic":
            valuation_date = b.date_input("Valuation date", value=pd.Timestamp(metadata["date_max"]).date())
            unit_style = st.checkbox("Unit-style address", value=False)
        submitted = st.form_submit_button("Estimate sale price")
    if submitted:
        row = {"suburb": suburb, "bedrooms": bedrooms, "bathrooms": bathrooms,
               "parking": parking, "land_size_m2": np.nan if land_unknown else land}
        if metadata["feature_set"] != "basic":
            row.update(sale_date=str(valuation_date), unit_style_address=int(unit_style))
        try:
            result, warnings = predict_properties(pd.DataFrame([row]), model, metadata)
            for message in warnings:
                st.warning(message)
            st.metric("Estimated sale price", f"A${result.predicted_sale_price_aud.iloc[0]:,.0f}")
            st.caption(f"Model: {metadata['model_name']}. Cross-validated mean absolute error: A${metadata['cv_mae_aud']:,.0f}. This average error is not an individual prediction interval.")
            st.dataframe(result[required], hide_index=True)
        except (ValueError, TypeError, OverflowError) as exc:
            st.error(str(exc))

with batch:
    st.write("Upload a CSV using the template columns. Leave unknown land sizes blank.")
    template_row = {"suburb": "Parramatta", "bedrooms": 3, "bathrooms": 2,
                    "parking": 1, "land_size_m2": 500,
                    "sale_date": metadata["date_max"], "unit_style_address": 0}
    template = pd.DataFrame([template_row])[required]
    st.download_button("Download input template", template.to_csv(index=False),
                       "property_input_template.csv", "text/csv")
    upload = st.file_uploader("Property features", type=["csv"])
    if upload is not None:
        try:
            supplied = pd.read_csv(upload)
            result, warnings = predict_properties(supplied, model, metadata)
            for message in warnings:
                st.warning(message)
            st.dataframe(result, hide_index=True)
            st.download_button("Download predictions", result.to_csv(index=False),
                               "housing_predictions.csv", "text/csv")
        except (ValueError, TypeError, OverflowError, UnicodeDecodeError) as exc:
            st.error(str(exc))

with st.expander("Model and data limits"):
    st.write(f"Trained on {metadata['n_properties']} properties. Observed sales: {metadata['date_min']} to {metadata['date_max']}.")
    st.write("Cross-validation compares properties from these three sampled markets. It does not validate other suburbs or future price movements. Equal suburb counts do not represent Sydney's transaction mix. Source data and land-area uncertainty remain limitations.")
    st.write("Features do not include verified condition, floor area, views, frontage, renovation quality or development rights.")
