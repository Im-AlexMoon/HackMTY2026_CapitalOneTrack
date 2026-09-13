import json
from pathlib import Path
import pickle
import platform
from importlib.metadata import version

import numpy as np
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.gateway import inference_payload
from models.common.adapter import validate_row
from models.common.bundle import sha256_file
from models.common.contracts import ContractError, PredictionRequest
from models.common.service import Runner, create_app
from scripts.validate_model_bundle import validate
import pytest

ROOT=Path(__file__).resolve().parents[1]


def test_ready_bundles_still_require_explicit_trust():
    for name in ('onboarding','transaction','sequence'):
        c=TestClient(create_app(name,ROOT/'models'/name,False))
        assert c.get('/health').json()['ready'] is False
        assert c.get('/metadata').json()['manifest']['status']=='ready'
        body={'entity_id':'a','as_of':'2026-09-12T10:00:00Z'}
        assert c.post('/validate',json=body).json()['status']=='not_ready'
        prediction=c.post('/predict',json=body).json()
        assert prediction['risk_score'] is None and prediction['status']=='not_ready'
        assert c.post('/predict',json={**body,'as_of':'2026-09-12T10:00:00'}).status_code==422


def test_labels_and_future_events_do_not_reach_features():
    payload=PredictionRequest(entity_id='a',as_of='2026-09-12T10:00:00Z',application={'fraud_bool':1,'nested':{'label':1,'value':2}},events=[{'event_time':'2026-09-12T11:00:00Z','amount':10000},{'event_time':'2026-09-12T09:00:00Z','amount':2,'isFraud':1}]).safe_payload()
    assert payload['application']=={'nested':{'value':2}}
    assert len(payload['events'])==1 and 'isFraud' not in payload['events'][0]
    cleaned = inference_payload({'fraud_type':'account_takeover','ground_truth':1,'amount':12,'nested':{'fraud_pattern':'ring','merchant':'m'}})
    assert cleaned == {'amount':12,'nested':{'merchant':'m'}}


def test_demo_requires_synthetic_source_and_sequence_history():
    runner=Runner('sequence',ROOT/'models/sequence',demo=True)
    request=PredictionRequest(entity_id='a',as_of='2026-09-12T10:00:00Z',context={'source':'synthetic'})
    assert runner.predict(request).status=='insufficient_history'
    request.context={'source':'dataset'}
    assert runner.predict(request).status=='error'


def test_frozen_pipeline_parity_and_stale_verification(tmp_path):
    directory=tmp_path/'bundle';directory.mkdir();(directory/'tests').mkdir()
    model=Pipeline([('scaler',StandardScaler()),('classifier',LogisticRegression(random_state=42))]).fit([[1.],[2.],[8.],[10.]],[0,0,1,1])
    artifact=directory/'model.pickle';artifact.write_bytes(pickle.dumps(model))
    (directory/'adapter.py').write_text('from models.common.adapter import ArtifactAdapter\nAdapter = ArtifactAdapter\n')
    (directory/'pipeline.py').write_text('def build_features(payload):\n    return {"amount": payload["application"]["amount"]}\n')
    (directory/'requirements.lock').write_text(f'scikit-learn=={version("scikit-learn")}\n')
    contract={'status':'ready','input_layout':'tabular','features':[{'name':'amount','dtype':'number','nullable':False}], 'feature_order':['amount']}
    manifest={'status':'candidate','model':'onboarding','model_version':'fixture-v1','preprocessing_version':'fixture-v1','dataset':'test fixture','dataset_provenance':{'version':'fixture-v1','source_url':'https://example.test/fixture','files':[{'name':'fixture.csv','sha256':'0'*64}],'split_reference':'fixture split membership'},'target':'label','artifact_format':'pickle','input_format':'numpy','preprocessing_mode':'embedded_pipeline','environment':{'python':platform.python_version(),'libraries':{'scikit-learn':version('scikit-learn')}},'artifacts':[{'path':'model.pickle','sha256':sha256_file(artifact),'role':'model'}],'output':{'method':'predict_proba','positive_class':1,'normalization':{'kind':'identity'}}}
    for name,value in [('manifest.json',manifest),('feature_contract.json',contract)]:
        (directory/name).write_text(json.dumps(value))
    cases=[{'input':{'entity_id':'a','as_of':'2026-09-12T10:00:00Z','application':{'amount':amount}},'expected':{'risk_score':float(model.predict_proba([[amount]])[0,1]),'model_vector':model[:-1].transform([[amount]]).tolist()}} for amount in (3.,9.)]
    (directory/'tests/golden_cases.json').write_text(json.dumps({'cases':cases}))
    assert validate(directory,trusted=True,write=True)['cases']==2
    assert Runner('onboarding',directory,trusted=True).ready
    assert not Runner('onboarding',directory,trusted=False).ready
    (directory/'pipeline.py').write_text((directory/'pipeline.py').read_text()+'\n# Changed preprocessing\n')
    assert not Runner('onboarding',directory,trusted=True).ready


def test_unknown_categories_rejected():
    contract={'features':[{'name':'type','dtype':'string','nullable':False,'categories':['A'],'unknown_category':'reject'}],'feature_order':['type']}
    with pytest.raises(ContractError):
        validate_row({'type':'Z'},contract)
    with pytest.raises(ContractError):
        validate_row({},contract)
