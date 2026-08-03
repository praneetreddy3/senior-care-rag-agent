FROM python:3.11-slim

WORKDIR /app

# Install deps first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the vector index at image build time so startup is fast
RUN python -m scripts.build_index

EXPOSE 8000
# Default: serve the FastAPI microservice. Override CMD to run Streamlit or MCP.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
