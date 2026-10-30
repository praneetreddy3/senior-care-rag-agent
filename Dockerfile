FROM python:3.11-slim

# Hugging Face Spaces runs containers as user 1000, so build and run as that user.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR $HOME/app

# CPU-only torch first: avoids pulling ~2 GB of CUDA libraries we never use.
RUN pip install --no-cache-dir --user torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu

# Install deps before copying code so this layer is cached between builds
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

# Build the vector index at image build time so startup is fast
# (the embedding model is downloaded and cached here too)
RUN python -m scripts.build_index

# Hugging Face Spaces expects the app on port 7860
EXPOSE 7860

# Default: Streamlit UI (used by the Hugging Face Space).
# To serve the FastAPI microservice instead:
#   docker run -p 8000:8000 <image> uvicorn api:app --host 0.0.0.0 --port 8000
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
