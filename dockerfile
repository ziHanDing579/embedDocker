FROM python:3.12-slim

# Unbuffered stdout: Python buffers when stdout isn't a TTY, so your logs
# would reach `kubectl logs` in delayed chunks, or not at all if the
# container dies with the buffer unflushed.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies before code. This layer is ~2.5 GB and changes rarely;
# copying source first would invalidate it on every edit.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Only the two files model.py actually opens.
COPY model/tokenizer.json   model/tokenizer.json
COPY model/onnx/model.onnx  model/onnx/model.onnx
COPY otel_setup.py model.py main.py ./

ENV MODEL_DIR=/app/model

RUN useradd -m -u 1000 app && chown -R app:app /app
USER app

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]