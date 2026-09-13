ARG PYTHON_VERSION=3.12.3
FROM python:${PYTHON_VERSION}-slim
WORKDIR /app
COPY models/onboarding/requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir -r /tmp/requirements.lock
COPY models /app/models
COPY scripts/validate_model_bundle.py /app/scripts/validate_model_bundle.py
ENV MODEL_KIND=onboarding MODEL_DIR=/app/models/onboarding PYTHONDONTWRITEBYTECODE=1
RUN useradd --create-home runner
USER runner
EXPOSE 8000
CMD ["uvicorn", "models.common.app:app", "--host", "0.0.0.0", "--port", "8000"]
