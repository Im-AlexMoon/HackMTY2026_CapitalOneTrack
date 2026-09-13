"""Run with python -m uvicorn models.common.app:app --port 8101."""

from models.common.service import create_app

app = create_app()
