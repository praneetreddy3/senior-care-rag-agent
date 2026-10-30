FROM python:3.11-slim

# Run as a non-root user (safer default for containers)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR $HOME/app

# Install deps before copying code so this layer is cached between builds
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

# Build the vector index at image build time so startup is fast
# (the embedding model is downloaded and cached here too)
RUN python -m scripts.build_index

EXPOSE 8501

# Default: Streamlit UI.
# To serve the FastAPI microservice instead:
#   docker run -p 8000:8000 <image> uvicorn api:app --host 0.0.0.0 --port 8000
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
