# Use a minimalist Python base image
FROM python:3.13-alpine

# Set the working directory
WORKDIR /code

# System deps for some Python wheels (chromadb, pypdf, python-docx)
RUN apk add --no-cache \
  build-base \
  && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY ./app ./app

EXPOSE 4000

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "4000"]
