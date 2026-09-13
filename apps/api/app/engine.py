"""Transactional replay and audit service. Only runners execute model code."""
import asyncio
import copy
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException
from sqlalchemy import select

from . import db
from .contracts import ApplicationInput, EventInput
from .scenarios import SCENARIOS, initial_accounts, events_for, stamp


def now():
    return datetime.now(timezone.utc).isoformat()


class Engine:
    def __init__(self, sessions, gateway, demo, config_dir):
        self.sessions, self.gateway, self.demo = sessions, gateway, demo
        self.lock = asyncio.Lock()
        self.revision = 0
        self.data = None
        self.config_dir = Path(config_dir)
        self.fusion = json.loads((self.config_dir / 'fusion_policy.json').read_text())
        if self.fusion['tau'] <= 0 or any(v < 0 for v in self.fusion['quality'].values()):
            raise ValueError('Invalid fusion policy')

    def key(self, identity):
        return f"{self.data['scenario']['run_id']}:{identity}"

    def record(self, session, table, identity=None, **fields):
        row = table(id=self.key(identity or uuid.uuid4().hex), run_id=self.data['scenario']['run_id'], **fields)
        session.merge(row)

    def save(self, session):
        for account in self.data['accounts']:
            self.record(session, db.Account, account['id'], payload=account)
        self.record(session, db.ScenarioRun, 'state', payload=copy.deepcopy(self.data))
        session.merge(db.Policy(id='default', payload=self.data['policy']))

    async def initialize(self):
        # Model containers, especially TensorFlow, can take longer than the API
        # process to import. Avoid freezing a transient startup race into UI state.
        await self.gateway.metadata(attempts=20, delay=0.5)
        with self.sessions() as session:
            rows = session.scalars(select(db.ScenarioRun)).all()
            latest = max(rows, key=lambda row: row.payload.get('saved_at', ''), default=None)
            if latest and latest.payload.get('mode') == ('demo' if self.demo else 'external'):
                self.data = copy.deepcopy(latest.payload)
                self.data['scenario']['status'] = 'paused'
        if self.data is None:
            await self.mutate('reset', 'sleeper_bustout')

    def snapshot(self):
        result = copy.deepcopy(self.data)
        result['model_status'] = self.gateway.readiness()['models']
        result['revision'] = self.revision
        return result

    def account(self, account_id):
        account = next((a for a in self.data['accounts'] if a['id'] == account_id), None)
        if account is None:
            raise HTTPException(404, 'Account not found')
        return account

    def schedule(self, scenario_id):
        result = []
        for item in events_for(scenario_id):
            result.append(item)
            if item['kind'] == 'application':
                result.append({'kind': 'screen', 'account_id': item['account_id'], 'event_time': item['event_time']})
        return result

    async def reset(self, session, scenario_id):
        selected = next((s for s in SCENARIOS if s['id'] == scenario_id), None)
        if selected is None:
            raise HTTPException(404, 'Scenario not found')
        policy = self.data['policy'] if self.data else {'id': 'default', **json.loads((self.config_dir / 'thresholds.json').read_text())}
        self.data = {'scenario': {**selected, 'run_id': uuid.uuid4().hex, 'status': 'paused', 'cursor': 0,
                     'total_events': len(self.schedule(scenario_id)) if self.demo else 0, 'speed': 1,
                     'event_time': stamp(0), 'source': 'synthetic' if self.demo else 'external'},
                     'scenarios': SCENARIOS if self.demo else [], 'policy': policy, 'accounts': [], 'alerts': [],
                     'mode': 'demo' if self.demo else 'external', 'saved_at': now()}
        if self.demo:
            for item in initial_accounts():
                await self.application(session, {k: v for k, v in item.items() if k != 'kind'})
        for name, metadata in self.gateway.statuses.items():
            self.record(session, db.ModelBundle, name, payload=metadata)

    async def score(self, session, account, name, as_of):
        portfolio_events = [
            {**event, "account_id": candidate["id"]}
            for candidate in self.data["accounts"]
            for event in candidate["timeline"]
        ]
        prediction = await self.gateway.predict(name, {'entity_id': account['id'], 'as_of': as_of,
            'application': account['application'], 'events': account['timeline'],
            'context': {'scenario_id': self.data['scenario']['id'], 'source': 'synthetic' if self.demo else 'external',
                        'portfolio_events': portfolio_events}})
        snapshot_id = uuid.uuid4().hex
        self.record(session, db.FeatureSnapshot, snapshot_id, account_id=account['id'], model=name,
                    preprocessing_version=prediction['preprocessing_version'], contract_hash=prediction['contract_hash'], payload=prediction['features'])
        prediction['feature_snapshot_id'] = self.key(snapshot_id)
        self.record(session, db.ModelInvocation, account_id=account['id'], model=name, risk_score=prediction['risk_score'],
                    payload={**prediction, 'as_of': as_of, 'observed_at': now()})
        account['signals'][name] = prediction
        return prediction['risk_score']

    async def application(self, session, body, screen=True):
        item = ApplicationInput.model_validate(body).model_dump(mode='json')
        existing = session.get(db.Application, self.key(item['application_id']))
        if existing:
            if existing.payload != item:
                raise HTTPException(409, 'Application id already exists with different content')
            return
        if any(a['id'] == item['account_id'] for a in self.data['accounts']):
            raise HTTPException(409, 'Account already exists')
        initial = item['raw_payload'].get('initial_balance', 0)
        if not isinstance(initial, (int, float)) or not math.isfinite(initial) or initial < 0:
            raise HTTPException(422, 'initial_balance must be finite and nonnegative')
        account = {'id': item['account_id'], 'alias': item['alias'], 'status': 'screening', 'risk_score': None,
            'application_risk': None, 'last_event_at': item['event_time'], 'transaction_count': 0, 'balance': initial,
            'trend': 0, 'signals': {'onboarding': None, 'transaction': None, 'sequence': None}, 'timeline': [],
            'risk_history': [], 'observations': [], 'application': item['raw_payload'], 'currency': None}
        self.data['accounts'].append(account)
        self.record(session, db.Application, item['application_id'], account_id=account['id'], raw_payload=item['raw_payload'], labels=item['labels'], payload=item)
        if item['raw_payload'].get('device_id'):
            self.record(session, db.EntityLink, account_id=account['id'], payload={'type': 'device', 'value': item['raw_payload']['device_id'], 'synthetic': self.demo})
        if screen:
            await self.screen(session, account, item['event_time'])

    async def screen(self, session, account, event_time):
        if account['status'] != 'screening':
            return
        risk = await self.score(session, account, 'onboarding', event_time)
        account['risk_score'] = account['application_risk'] = risk
        if risk is not None:
            account['status'] = 'rejected' if risk >= self.data['policy']['onboarding_threshold'] else 'monitoring'
        self.history(session, account, event_time)

    def history(self, session, account, event_time):
        point = {'event_time': event_time, 'risk_score': account['risk_score'], 'threshold': self.data['policy']['threshold'],
                 'signals': {name: value['risk_score'] if value else None for name, value in account['signals'].items()}}
        account['risk_history'].append(point)
        account['observations'] = list(dict.fromkeys(reason['description'] for signal in account['signals'].values() if signal
            for reason in signal['reason_codes'] if reason['code'] != 'SYNTHETIC_DEMO'))
        self.record(session, db.RiskScore, account_id=account['id'], risk_score=account['risk_score'], payload=point)

    def fuse(self, account):
        available = [(self.fusion['quality'].get(name, 1), value['risk_score']) for name, value in account['signals'].items()
                     if name != 'onboarding' and value and value['risk_score'] is not None]
        denominator = sum(weight for weight, _ in available)
        if not denominator:
            return None if account['transaction_count'] else account['application_risk']
        behavior = sum(w * risk for w, risk in available) / denominator
        if account['application_risk'] is None:
            return round(behavior, 6)
        weight = math.exp(-account['transaction_count'] / self.fusion['tau'])
        return round(weight * account['application_risk'] + (1 - weight) * behavior, 6)

    def alert(self, session, account, event_time):
        if account['status'] in {'screening', 'rejected'} or account['risk_score'] is None:
            return
        threshold = self.data['policy']['threshold']
        above = account['risk_score'] >= threshold
        previously_above = account.get('above_threshold', False)
        account['above_threshold'] = above
        unresolved = any(a['account_id'] == account['id'] and a['status'] in {'open', 'escalated'} for a in self.data['alerts'])
        if above and not previously_above and not unresolved:
            alert = {'id': uuid.uuid4().hex, 'account_id': account['id'], 'created_at': event_time, 'status': 'open',
                     'risk_score': account['risk_score'], 'threshold': threshold, 'notes': '',
                     'reason_codes': [r for v in account['signals'].values() if v for r in v['reason_codes'] if r['code'] != 'SYNTHETIC_DEMO']}
            self.data['alerts'].append(alert)
            self.record(session, db.Alert, alert['id'], account_id=account['id'], payload=alert)
            account['status'] = 'review_required'

    async def event(self, session, body):
        item = EventInput.model_validate(body).model_dump(mode='json')
        existing = session.get(db.LedgerEvent, self.key(item['event_id']))
        if existing:
            if existing.payload['input'] != item:
                raise HTTPException(409, 'Event id already exists with different content')
            return
        account = self.account(item['account_id'])
        if account['status'] in {'screening', 'rejected'}:
            raise HTTPException(409, 'Account is not admitted for monitoring')
        if datetime.fromisoformat(item['event_time']) < datetime.fromisoformat(account['last_event_at']):
            raise HTTPException(409, 'Out-of-order event; submit in event_time order')
        if account['currency'] and account['currency'] != item['currency']:
            raise HTTPException(422, 'Currency conversion must happen in the client adapter')
        account['currency'] = item['currency']
        account['balance'] = round(account['balance'] + item['amount'] * (1 if item['type'] == 'deposit' else -1), 2)
        account['transaction_count'] += 1
        event = {k: v for k, v in item.items() if k != 'labels'}
        event.update(id=item['event_id'], balance=account['balance'], sequence_number=account['transaction_count'])
        account['timeline'].append(event)
        account['last_event_at'] = item['event_time']
        self.record(session, db.LedgerEvent, item['event_id'], account_id=account['id'], event_time=item['event_time'],
                    sequence_number=account['transaction_count'], raw_payload=item['raw_payload'], labels=item['labels'],
                    payload={'input': item, 'observed_at': now(), 'event': event})
        previous = account['risk_score']
        for name in ('transaction', 'sequence'):
            await self.score(session, account, name, item['event_time'])
        account['risk_score'] = self.fuse(account)
        account['trend'] = round(account['risk_score'] - previous, 6) if previous is not None and account['risk_score'] is not None else 0
        self.history(session, account, item['event_time'])
        self.alert(session, account, item['event_time'])

    async def mutate(self, action, scenario_id=None, body=None):
        async with self.lock:
            before = copy.deepcopy(self.data)
            try:
                with self.sessions.begin() as session:
                    if action == 'reset':
                        await self.reset(session, scenario_id)
                    else:
                        scenario = self.data['scenario']
                        if scenario_id and scenario['id'] != scenario_id:
                            raise HTTPException(409, 'Scenario is not active; reset it first')
                        if action in {'step', 'start'} and not self.demo:
                            raise HTTPException(409, 'External mode accepts application/event API ingestion; synthetic replay is disabled')
                        if action == 'step' and scenario['cursor'] < scenario['total_events']:
                            item = self.schedule(scenario['id'])[scenario['cursor']]
                            data = {k: v for k, v in item.items() if k != 'kind'}
                            if item['kind'] == 'application':
                                await self.application(session, data, screen=False)
                            elif item['kind'] == 'screen':
                                await self.screen(session, self.account(item['account_id']), item['event_time'])
                            else:
                                await self.event(session, data)
                            scenario['cursor'] += 1
                            scenario['event_time'] = item['event_time']
                            if scenario['cursor'] == scenario['total_events']:
                                scenario['status'] = 'completed'
                        elif action in {'start', 'pause'}:
                            if scenario['cursor'] < scenario['total_events']:
                                scenario['status'] = 'running' if action == 'start' else 'paused'
                        elif action == 'speed':
                            scenario['speed'] = body['speed']
                        elif action == 'threshold':
                            self.data['policy'].update({k: v for k, v in body.items() if v is not None})
                            for account in self.data['accounts']:
                                self.alert(session, account, scenario['event_time'])
                        elif action == 'review':
                            alert = next((a for a in self.data['alerts'] if a['id'] == body['id']), None)
                            if not alert:
                                raise HTTPException(404, 'Alert not found')
                            if alert['status'] != 'open':
                                raise HTTPException(409, 'Alert already reviewed')
                            alert.update(status=body['status'], notes=body['notes'])
                            self.account(alert['account_id'])['status'] = 'escalated' if body['status'] == 'escalated' else 'monitoring'
                            self.record(session, db.Alert, alert['id'], account_id=alert['account_id'], payload=alert)
                            self.record(session, db.AnalystReview, alert_id=alert['id'], payload={**body, 'observed_at': now()})
                        elif action == 'application':
                            await self.application(session, body)
                        elif action == 'event':
                            await self.event(session, body)
                        elif action == 'rescreen':
                            account = self.account(body['account_id'])
                            await self.screen(session, account, account['last_event_at'])
                    self.data['saved_at'] = now()
                    self.save(session)
                self.revision += 1
                return self.snapshot()
            except Exception:
                self.data = before
                raise

    async def autoplay(self):
        while True:
            await asyncio.sleep(2 / self.data['scenario']['speed'])
            if self.data['scenario']['status'] == 'running':
                try:
                    await self.mutate('step', self.data['scenario']['id'])
                except Exception:
                    await self.mutate('pause', self.data['scenario']['id'])
