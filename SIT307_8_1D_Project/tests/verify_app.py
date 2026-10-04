"""Run from the project folder: python tests/verify_app.py"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import joblib
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'SIT307_8_1D_outputs/app'
sys.path.insert(0,str(APP))
from housing_utils import BASIC_FEATURES, predict_properties

metadata=json.loads((APP/'model_metadata.json').read_text())
model=joblib.load(APP/'housing_price_model.joblib')
assert metadata['model_name']=='Ridge regression'
assert metadata['feature_set']=='basic'

checks=[]
app=AppTest.from_file(str(APP/'app.py'),default_timeout=30).run()
assert len(app.exception)==0
assert len(app.date_input)==0
assert [x.label for x in app.checkbox]==['Land size unknown']
checks.append('app starts with five predictive controls and no unused date or address controls')

app.selectbox[0].select('Campbelltown')
for control in app.number_input:
    if control.label=='Parking spaces':
        control.set_value(1)
app.button[0].click().run()
assert not app.exception and not app.error
assert app.metric[0].value=='A$947,325',app.metric[0].value
checks.append('single-property example agrees with supplied screenshot at A$947,325')

app.checkbox[0].check()
app.button[0].click().run()
assert not app.exception and not app.error
assert any('imputation' in x.value for x in app.warning)
checks.append('single-property missing land is accepted and explained')

examples=pd.read_csv(APP/'example_inputs.csv')
assert list(examples.columns)==BASIC_FEATURES
results,warnings=predict_properties(examples,model,metadata)
np.testing.assert_array_equal(results.predicted_sale_price_aud,[1685714,593857])
checks.append('five-column CSV examples agree with supplied batch screenshots')
assert any('imputation' in message for message in warnings)
checks.append('batch missing land triggers the imputation message')

legacy=examples.assign(sale_date='2026-09-24',unit_style_address=[0,1])
legacy_results,_=predict_properties(legacy,model,metadata)
pd.testing.assert_frame_equal(results,legacy_results)
checks.append('legacy seven-column CSV gives identical predictions')

single_results=[]
for i in range(len(examples)):
    one,_=predict_properties(examples.iloc[[i]],model,metadata)
    single_results.append(one.predicted_sale_price_aud.iloc[0])
np.testing.assert_array_equal(single_results,results.predicted_sale_price_aud)
checks.append('single-row and batch inference agree')

summary={'method':'Streamlit AppTest plus the shared CSV inference function',
         'streamlit_version':__import__('streamlit').__version__,
         'checks_passed':checks,'example_predictions_aud':results.predicted_sale_price_aud.tolist()}
(ROOT/'SIT307_8_1D_outputs/app_verification.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
