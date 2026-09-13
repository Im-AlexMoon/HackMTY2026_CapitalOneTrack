import asyncio
import json
import os
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .contracts import ApplicationInput, EventInput, ReviewInput, SpeedInput, ThresholdInput
from .db import Base, connect
from .engine import Engine
from .gateway import ModelGateway


def create_app(database_url=None, gateway=None, demo=None):
    demo = demo if demo is not None else os.getenv('DEMO_MODE', 'false').lower() == 'true'
    engine, sessions = connect(database_url or os.getenv('DATABASE_URL', 'sqlite:///./firstwatch.db'))
    urls = {name: os.getenv(f'{name.upper()}_RUNNER_URL', f'http://127.0.0.1:{8101 + i}')
            for i, name in enumerate(('onboarding', 'transaction', 'sequence'))}
    gateway = gateway or ModelGateway(urls, demo_mode=demo)
    config = os.getenv('CONFIG_DIR') or str(Path(__file__).resolve().parents[3] / 'configs')
    service = Engine(sessions, gateway, demo, config)

    @asynccontextmanager
    async def lifespan(app):
        Base.metadata.create_all(engine)
        await service.initialize()
        task = asyncio.create_task(service.autoplay())
        yield
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        await gateway.close()
        engine.dispose()

    app = FastAPI(title='FirstWatch API', version='1.0.0', lifespan=lifespan)
    app.state.service = service
    app.add_middleware(CORSMiddleware, allow_origins=os.getenv('CORS_ORIGINS', 'http://localhost:3000,http://127.0.0.1:3000').split(','),
                       allow_methods=['GET', 'POST', 'PATCH'], allow_headers=['Content-Type'])

    @app.get('/health')
    async def health():
        readiness = gateway.readiness()
        return {'status': 'ok', 'mode': readiness['mode'], 'models_ready': readiness['all_ready']}

    @app.get('/api/models/status')
    async def model_status(refresh: bool = True):
        if refresh:
            await gateway.metadata()
        return gateway.readiness()

    @app.get('/api/state')
    async def state():
        return service.snapshot()

    @app.get('/api/accounts')
    async def accounts():
        return service.snapshot()['accounts']

    @app.get('/api/accounts/{account_id}')
    async def account(account_id: str):
        return service.account(account_id)

    @app.get('/api/accounts/{account_id}/timeline')
    async def timeline(account_id: str):
        return service.account(account_id)['timeline']

    @app.get('/api/alerts')
    async def alerts():
        return service.snapshot()['alerts']

    @app.post('/api/scenarios/{scenario_id}/{action}')
    async def control(scenario_id: str, action: str):
        if action not in {'start', 'pause', 'step', 'reset'}:
            raise HTTPException(404, 'Unknown control')
        return await service.mutate(action, scenario_id)

    @app.patch('/api/scenarios/{scenario_id}/speed')
    async def speed(scenario_id: str, body: SpeedInput):
        return await service.mutate('speed', scenario_id, body.model_dump())

    @app.patch('/api/policies/{policy_id}/threshold')
    async def threshold(policy_id: str, body: ThresholdInput):
        if policy_id != 'default':
            raise HTTPException(404, 'Policy not found')
        return await service.mutate('threshold', body=body.model_dump())

    @app.patch('/api/alerts/{alert_id}')
    async def review(alert_id: str, body: ReviewInput):
        return await service.mutate('review', body={'id': alert_id, **body.model_dump()})

    @app.post('/api/applications')
    async def application(body: ApplicationInput):
        return await service.mutate('application', body=body.model_dump(mode='json'))

    @app.post('/api/events')
    async def event(body: EventInput):
        return await service.mutate('event', body=body.model_dump(mode='json'))

    @app.post('/api/accounts/{account_id}/rescreen')
    async def rescreen(account_id: str):
        return await service.mutate('rescreen', body={'account_id': account_id})

    @app.get('/api/scenarios/{scenario_id}/stream')
    async def stream(scenario_id: str, request: Request):
        async def events():
            revision = -1
            while not await request.is_disconnected():
                if revision != service.revision:
                    revision = service.revision
                    yield f'event: state\ndata: {json.dumps(service.snapshot(), allow_nan=False)}\n\n'
                else:
                    yield ': heartbeat\n\n'
                await asyncio.sleep(1)
        return StreamingResponse(events(), media_type='text/event-stream', headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

    return app


app = create_app()
