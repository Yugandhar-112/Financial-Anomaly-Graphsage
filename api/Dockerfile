# Build context must be the REPO ROOT (not api/), since we need src/models/
# and checkpoints/ alongside api/ itself:
#   docker build -f api/Dockerfile -t fraud-api .
#   docker run -p 7860:7860 fraud-api
#
# Port 7860 matches the Hugging Face Spaces Docker SDK convention (step 7).

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt /app/api/requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.2.2 \
    && pip install --no-cache-dir -r /app/api/requirements.txt

# Only what inference actually needs at runtime: the model classes (not the
# full training pipeline) and the trained checkpoints + feature scaler.
COPY src/models /app/src/models
COPY api /app/api
COPY checkpoints /app/checkpoints

WORKDIR /app/api
EXPOSE 7860
ENV FRONTEND_ORIGIN="*"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]