# 1. Base image Python
FROM python:3.10-slim

# 2. Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# 3. Set working directory
WORKDIR /app

# 4. Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Copy the entire project code
COPY . .

# 6. Expose ports: 8000 for FastAPI, 8501 for Streamlit Dashboard
EXPOSE 8000
EXPOSE 8501

# 7. Default command (Run FastAPI Server)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
