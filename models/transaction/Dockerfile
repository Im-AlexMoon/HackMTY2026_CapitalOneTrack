ARG PYTHON_VERSION=3.13.15
FROM python:${PYTHON_VERSION}-slim
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY models/transaction/requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir -r /tmp/requirements.lock
COPY models /app/models
COPY scripts/validate_model_bundle.py /app/scripts/validate_model_bundle.py
ENV MODEL_KIND=transaction MODEL_DIR=/app/models/transaction PYTHONDONTWRITEBYTECODE=1
RUN useradd --create-home runner
USER runner
EXPOSE 8000
CMD ["uvicorn", "models.common.app:app", "--host", "0.0.0.0", "--port", "8000"]
