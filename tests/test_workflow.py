import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app import db
from app.gateway import ModelGateway
from app.main import create_app
from models.common.contracts import PredictionRequest
from models.common.service import Runner

ROOT = Path(__file__).resolve().parents[1]


def client_for(demo=True, database_url='sqlite:///:memory:', unavailable=()):
    runners = {name: Runner(name, ROOT/'models'/name, demo=demo) for name in ('onboarding','transaction','sequence')}
    async def handler(request):
        name = request.url.host
        if name in unavailable:
            return httpx.Response(503)
        if request.url.path == '/metadata':
            return httpx.Response(200, json=runners[name].metadata())
        payload = json.loads(request.content)
        assert 'fraud_bool' not in payload['application']
        return httpx.Response(200, json=runners[name].predict(PredictionRequest.model_validate(payload)).model_dump())
    gateway = ModelGateway({name:f'http://{name}' for name in runners}, httpx.MockTransport(handler), demo_mode=demo)
    return TestClient(create_app(database_url, gateway, demo))


def step(client, scenario='sleeper_bustout'):
    response = client.post(f'/api/scenarios/{scenario}/step')
    assert response.status_code == 200, response.text
    return response.json()


def test_sleeper_rejects_and_alerts_before_exit_with_audit_and_review():
    with client_for() as c:
        state = c.get('/api/state').json()
        assert len(state['accounts']) == 5
        screening_seen = False
        while state['scenario']['cursor'] < state['scenario']['total_events'] - 1:
            state = step(c)
            screening_seen |= any(a['status']=='screening' for a in state['accounts'])
        sleeper = next(a for a in state['accounts'] if a['id']=='acct_sleeper')
        assert screening_seen
        assert any(a['status']=='rejected' for a in state['accounts'])
        assert sleeper['status']=='review_required'
        assert len(state['alerts'])==1
        assert all(e['type']!='cash_out' for e in sleeper['timeline'])
        alert = state['alerts'][0]
        response = c.patch('/api/alerts/'+alert['id'],json={'status':'escalated','notes':'Repeated outflows; escalate to review team.'})
        assert response.status_code==200
        assert next(a for a in response.json()['accounts'] if a['id']=='acct_sleeper')['status']=='escalated'
        state = step(c)
        assert len(state['alerts'])==1
        assert state['scenario']['status']=='completed'
        with c.app.state.service.sessions() as session:
            for table in (db.Application,db.Account,db.LedgerEvent,db.FeatureSnapshot,db.ModelInvocation,db.RiskScore,db.Alert,db.AnalystReview,db.ModelBundle,db.ScenarioRun,db.EntityLink):
                assert session.scalar(select(func.count()).select_from(table))>0, table.__tablename__


def test_reset_reproduces_scores_without_deleting_audit():
    with client_for() as c:
        first = step(c)
        run = first['scenario']['run_id']
        c.post('/api/scenarios/sleeper_bustout/reset')
        second = step(c)
        assert first['accounts'][0]['risk_history']==second['accounts'][0]['risk_history']
        assert second['scenario']['run_id']!=run
        with c.app.state.service.sessions() as session:
            assert session.scalar(select(func.count()).select_from(db.ScenarioRun))==2


def test_legitimate_and_false_positive_scenarios():
    with client_for() as c:
        for scenario in ('legitimate','false_positive','identity_farm'):
            state=c.post(f'/api/scenarios/{scenario}/reset').json()
            while state['scenario']['status']!='completed':
                state=step(c,scenario)
            if scenario=='legitimate':
                assert not state['alerts']
            elif scenario=='false_positive':
                alert=state['alerts'][0]
                response=c.patch('/api/alerts/'+alert['id'],json={'status':'dismissed','notes':'Verified planned purchases.'})
                assert response.status_code==200
            else:
                assert sum(a['status']=='rejected' for a in state['accounts'])==3


def test_unavailable_models_stay_pending_and_external_disables_synthetic_replay():
    with client_for(demo=False) as c:
        state=c.get('/api/state').json()
        assert state['mode']=='external' and state['accounts']==[]
        assert c.post('/api/scenarios/sleeper_bustout/start').status_code==409
        response=c.post('/api/applications',json={'application_id':'a','account_id':'a','alias':'Applicant','event_time':'2026-09-12T10:00:00Z','raw_payload':{'fraud_bool':1}})
        account=response.json()['accounts'][0]
        assert account['status']=='screening' and account['risk_score'] is None
    with client_for(unavailable=('transaction','sequence')) as c:
        state=step(c)
        assert state['accounts'][0]['risk_score'] is None
        assert not state['alerts']


def test_model_readiness_is_explicit_and_refreshable():
    with client_for() as c:
        response = c.get('/api/models/status')
        assert response.status_code == 200
        readiness = response.json()
        assert readiness['mode'] == 'demo'
        assert readiness['all_ready'] is False
        assert {item['model'] for item in readiness['models']} == {'onboarding', 'transaction', 'sequence'}
        assert all(item['bundle_status'] == 'ready' for item in readiness['models'])
        assert all(item['ready_for_external'] is False for item in readiness['models'])
        assert all('manifest' not in item and 'feature_contract' not in item for item in readiness['models'])


def test_idempotency_validation_and_threshold_preserve_decisions():
    with client_for() as c:
        body={'event_id':'test','account_id':'acct_ava','event_time':'2026-09-12T10:00:00Z','type':'purchase','amount':10}
        first=c.post('/api/events',json=body)
        second=c.post('/api/events',json=body)
        assert first.status_code==second.status_code==200
        assert first.json()['accounts']==second.json()['accounts']
        assert c.post('/api/events',json={**body,'amount':20}).status_code==409
        assert c.post('/api/events',json={**body,'event_id':'new','event_time':'2026-09-12T08:00:00Z'}).status_code==409
        assert c.patch('/api/policies/default/threshold',json={'threshold':1.2}).status_code==422
        assert c.patch('/api/scenarios/sleeper_bustout/speed',json={'speed':7}).status_code==422
        assert c.post('/api/events',json={**body,'amount':-1}).status_code==422
        assert c.post('/api/events',json={**body,'event_time':'2026-09-12T10:00:00'}).status_code==422


def test_restart_restores_state(tmp_path):
    url=f'sqlite:///{tmp_path / "audit.db"}'
    with client_for(database_url=url) as c:
        state=step(c)
    with client_for(database_url=url) as c:
        restored=c.get('/api/state').json()
        assert restored['accounts']==state['accounts']
        assert restored['scenario']['cursor']==1
        assert restored['scenario']['status']=='paused'


def test_lowering_threshold_creates_one_case_and_keeps_admission_history():
    with client_for() as c:
        state=c.post('/api/scenarios/identity_farm/reset').json()
        while state['scenario']['status']!='completed':
            state=step(c,'identity_farm')
        rejected=[a['id'] for a in state['accounts'] if a['status']=='rejected']
        state=c.patch('/api/policies/default/threshold',json={'threshold':0.05,'onboarding_threshold':1.0}).json()
        assert rejected==[a['id'] for a in state['accounts'] if a['status']=='rejected']
        count=len(state['alerts'])
        assert count>0
        state=c.patch('/api/policies/default/threshold',json={'threshold':0.05}).json()
        assert len(state['alerts'])==count
