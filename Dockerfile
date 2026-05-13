FROM python:3.12-slim

WORKDIR /app

# Install deps first (layer cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model weights and tokenizer
COPY models/best_model/ models/best_model/

# Copy API code
COPY api/ api/

# App Runner / Cloud Run default port
ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
