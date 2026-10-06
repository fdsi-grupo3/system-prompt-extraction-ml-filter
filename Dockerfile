# Imagen de inferencia: solo ONNX Runtime + tokenizer (sin PyTorch, ~400 MB menos).
# Requiere haber entrenado y exportado el modelo antes:  models/leakfilter-v1/{model.int8.onnx, tokenizer*, filter_config.json}
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    LEAKFILTER_MODEL_DIR=/app/models/leakfilter-v1 LEAKFILTER_BACKEND=onnx

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY leakfilter/__init__.py leakfilter/preprocessing.py leakfilter/inference.py leakfilter/
COPY service/ service/
COPY data/seed/prompts.json data/seed/prompts.json
COPY models/leakfilter-v1/model.int8.onnx models/leakfilter-v1/filter_config.json models/leakfilter-v1/
COPY models/leakfilter-v1/tokenizer.json models/leakfilter-v1/tokenizer_config.json models/leakfilter-v1/config.json models/leakfilter-v1/

RUN useradd --create-home filter && chown -R filter /app
USER filter
EXPOSE 8001
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
