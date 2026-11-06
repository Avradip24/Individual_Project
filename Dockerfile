# SmartSpend demonstrator. Model files are NOT baked in: mount them at /models
# (see docker-compose.yml). requirements-demo.txt pins the versions the models were
# trained with (scikit-learn must match the pickled models).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MODELS_DIR=/models \
    PRELOAD_MODELS=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app

# CPU-only PyTorch (the default Linux wheel pulls ~2 GB of CUDA libraries); everything else from PyPI.
COPY requirements-demo.txt .
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r requirements-demo.txt

COPY src ./src

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.demo.api:app", "--host", "0.0.0.0", "--port", "8000"]