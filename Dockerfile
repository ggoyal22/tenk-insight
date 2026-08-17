FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces requires a non-root user with UID 1000
RUN useradd -m -u 1000 user

WORKDIR /app

COPY requirements.txt .

# Install CPU-only torch first — avoids pulling the CUDA build (~2GB larger)
# which is unnecessary since HF free tier is CPU-only. requirements.txt acts as a
# constraints file so the version comes from there and the next step finds the
# requirement already satisfied rather than refetching torch from PyPI.
RUN pip install --no-cache-dir torch \
    --index-url https://download.pytorch.org/whl/cpu \
    -c requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# Pre-download embedding and reranker models at build time so cold starts
# load from disk (~30s) rather than downloading at runtime (~5min).
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-large-en-v1.5'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')" \
    && chown -R user:user /app/.cache

COPY --chown=user:user . .

USER user

EXPOSE 7860

HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health

ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
# Suppress tokenizers parallelism warning in a single-process container
ENV TOKENIZERS_PARALLELISM=false

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
