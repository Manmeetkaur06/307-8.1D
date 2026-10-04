SIT307 8.1D — Sydney housing project

1. Open SIT307_8_1D_Manmeet_Kaur.ipynb in Jupyter.
2. Keep sydney_housing_final_105_CLEANED.csv beside the notebook, or use its embedded fallback.
3. Restart the kernel and run all cells. No Google Drive mount is required.
4. Launch the application from the app directory:
   python -m pip install -r requirements.txt
   python -m streamlit run app.py

Measured selected model: Ridge regression
Pooled out-of-fold MAE: A$897,806.82
Pooled out-of-fold RMSE: A$2,306,068.40
Pooled out-of-fold R2: 0.705941
These are regression metrics, not a classification accuracy percentage.

The raw input CSV is unchanged. The modelling copy leaves land missing for the
seven supplied uncertain records and the 109 Victoria Road land/floor discrepancy.
Imputation is learned inside training folds. Source verification is partial.
Never substitute reported_land_size_m2 for the withheld land_size_m2 values.

The application asks for the five predictors used by the selected basic Ridge.
Date and address-style fields are supplied internally only for compatibility.
Inference checks and example predictions are saved with the analysis outputs.
The final project archive also includes the report and supplied app screenshots.
This local package does not create a hosted application or public sharing URL.

Contents: data, figures, tables, app source/module/model, run summary and requirements.
Keep the app module and model together. Retrain with this notebook if library
versions differ from those in model_metadata.json.
